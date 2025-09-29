import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32


class ROS2Infrastructure(Node):
    """
    Generic ROS2 infrastructure layer for handling ROS2 communication.

    This class provides fundamental ROS2 operations:
    - Generic topic subscription and publishing
    - Message conversion utilities
    - Thread-safe data access
    - ROS2 lifecycle management
    """

    def __init__(self, node_name: str = 'ros2_infrastructure'):
        """
        Initialize generic ROS2 infrastructure.

        Args:
            node_name: Name for the ROS2 node
        """
        # Initialize ROS2 node
        super().__init__(node_name)

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Data storage for different topics
        self.data_storage = {}
        self.data_locks = {}
        self.data_callbacks = {}
        self.subs = []

        # Setup default QoS profile
        self.default_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        self.get_logger().info('Generic ROS2 Infrastructure initialized successfully')

    def create_subscriber(self,
                        topic: str,
                        msg_type: Any,
                        callback: Optional[Callable] = None,
                        qos_profile: Optional[QoSProfile] = None) -> str:
        """
        Create a subscriber for a specific topic.

        Args:
            topic: Topic name to subscribe to
            msg_type: ROS2 message type
            callback: Optional callback function for message processing
            qos_profile: QoS profile for the subscriber

        Returns:
            Subscription ID
        """
        if qos_profile is None:
            qos_profile = self.default_qos

        # Create topic-specific data storage
        topic_key = self._get_topic_key(topic)
        self.data_storage[topic_key] = None
        self.data_locks[topic_key] = threading.Lock()

        # Store user callback only if no callback is provided as parameter
        if callback and topic_key not in self.data_callbacks:
            self.data_callbacks[topic_key] = callback

        self.get_logger().info(f"Creating subscriber for topic: {topic} (key: {topic_key})")

        # Create generic callback wrapper - fix lambda closure by binding parameters at creation time
        subscriber = self.create_subscription(
            msg_type,
            topic,
            lambda msg, topic=topic, callback=callback: self._handle_message(topic, msg, callback),
            qos_profile
        )

        self.get_logger().info(f'Subscriber created successfully for topic: {topic}')

        if subscriber:
            self.get_logger().info(f'Subscriber object is valid: {type(subscriber)}')

        return f"sub_{topic_key}"

    def add_publisher(self,
                         topic: str,
                         msg_type: Any,
                         qos_profile: Optional[QoSProfile] = None):
        """
        Create a publisher for a specific topic.

        Args:
            topic: Topic name to publish to
            msg_type: ROS2 message type
            qos_profile: QoS profile for the publisher

        Returns:
            Publisher object
        """
        if qos_profile is None:
            qos_profile = self.default_qos

        return super().create_publisher(msg_type, topic, qos_profile)

    def _get_topic_key(self, topic: str) -> str:
        """Convert topic name to internal key."""
        return topic.replace('/', '_').strip('_')

    def _handle_message(self, topic: str, msg, user_callback: Optional[Callable]):
        """
        Handle incoming ROS2 messages.

        Args:
            topic: Topic name
            msg: ROS2 message
            user_callback: Optional user-provided callback
        """
        self.get_logger().info(f'_handle_message called for topic: {topic}')
        topic_key = self._get_topic_key(topic)
        self.get_logger().info(f"Processing message for topic: {topic} (key: {topic_key})")
        try:
            # Convert message to Python data structure
            processed_data = self._convert_message(msg)

            # Store data thread-safely
            with self.data_locks[topic_key]:
                self.data_storage[topic_key] = processed_data

            # Call user callback if provided (prioritize parameter callback over stored callback)
            if user_callback:
                user_callback(topic, processed_data)
            elif topic_key in self.data_callbacks and self.data_callbacks[topic_key]:
                self.data_callbacks[topic_key](topic, processed_data)

        except Exception as e:
            self.get_logger().error(f'Error handling message from {topic}: {e}')

    def _convert_message(self, msg) -> Dict[str, Any]:
        """
        Convert ROS2 message to Python data structure.

        Args:
            msg: ROS2 message

        Returns:
            Dictionary with converted data
        """
        self.get_logger().info(f"converting message: {msg.data=}")
        # Handle Image messages
        if hasattr(msg, 'encoding') and hasattr(msg, 'data'):
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            return {
                'data': cv_image,
                'header': msg.header,
                'encoding': msg.encoding,
                'height': msg.height,
                'width': msg.width,
                'type': 'image'
            }

        # Handle JointState messages
        elif hasattr(msg, 'position'):
            return {
                'position': np.array(msg.position) if msg.position else np.array([]),
                'velocity': np.array(msg.velocity) if msg.velocity else np.array([]),
                'effort': np.array(msg.effort) if msg.effort else np.array([]),
                'header': msg.header,
                'name': msg.name,
                'type': 'joint_state'
            }

        # Handle Float32 messages
        elif hasattr(msg, 'data'):
            return {
                'data': float(msg.data),
                'type': 'float32'
            }

        # Handle Twist messages
        elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
            return {
                'linear': {
                    'x': msg.linear.x,
                    'y': msg.linear.y,
                    'z': msg.linear.z
                },
                'angular': {
                    'x': msg.angular.x,
                    'y': msg.angular.y,
                    'z': msg.angular.z
                },
                'type': 'twist'
            }

        # Generic fallback
        else:
            return {
                'message': msg,
                'type': 'unknown'
            }

    def get_data(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Args:
            topic: Topic name

        Returns:
            Latest data from the topic or None if not available
        """
        topic_key = self._get_topic_key(topic)

        if topic_key not in self.data_storage:
            return None

        with self.data_locks[topic_key]:
            data = self.data_storage[topic_key]
            return data.copy() if data is not None else None

    def publish_message(self, topic: str, msg_data: Dict[str, Any]):
        """
        Publish message to a topic.

        Args:
            topic: Topic name
            msg_data: Dictionary with message data
        """
        try:
            # Create appropriate message type based on data
            msg = self._create_message(msg_data)

            # Get or create publisher
            topic_key = self._get_topic_key(topic)
            publisher_attr = f"publisher_{topic_key}"

            if not hasattr(self, publisher_attr):
                # Determine message type from data
                msg_type = self._get_message_type(msg_data)
                publisher = self.create_publisher(topic, msg_type)
                setattr(self, publisher_attr, publisher)
            else:
                publisher = getattr(self, publisher_attr)

            # Publish message
            publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing to {topic}: {e}')

    def _create_message(self, msg_data: Dict[str, Any]):
        """Create ROS2 message from data dictionary."""
        msg_type = msg_data.get('type', 'unknown')

        if msg_type == 'twist':
            msg = Twist()
            if 'linear' in msg_data:
                msg.linear.x = msg_data['linear'].get('x', 0.0)
                msg.linear.y = msg_data['linear'].get('y', 0.0)
                msg.linear.z = msg_data['linear'].get('z', 0.0)
            if 'angular' in msg_data:
                msg.angular.x = msg_data['angular'].get('x', 0.0)
                msg.angular.y = msg_data['angular'].get('y', 0.0)
                msg.angular.z = msg_data['angular'].get('z', 0.0)
            return msg

        elif msg_type == 'float32':
            from std_msgs.msg import Float32
            msg = Float32()
            msg.data = msg_data.get('data', 0.0)
            return msg

        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    def _get_message_type(self, msg_data: Dict[str, Any]):
        """Get ROS2 message type from data dictionary."""
        msg_type = msg_data.get('type', 'unknown')

        if msg_type == 'twist':
            return Twist
        elif msg_type == 'float32':
            from std_msgs.msg import Float32
            return Float32
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    def register_callback(self, topic: str, callback: Callable):
        """
        Register callback for topic data updates.

        Args:
            topic: Topic name
            callback: Callback function (topic, data) -> None
        """
        topic_key = self._get_topic_key(topic)
        self.data_callbacks[topic_key] = callback

    def wait_for_data(self, topics: Union[str, list], timeout: float = 5.0) -> bool:
        """
        Wait for data to arrive on specified topics.

        Args:
            topics: Single topic name or list of topic names
            timeout: Timeout in seconds

        Returns:
            True if all data received, False on timeout
        """
        if isinstance(topics, str):
            topics = [topics]

        start_time = time.time()

        while time.time() - start_time < timeout:
            all_received = True
            for topic in topics:
                topic_key = self._get_topic_key(topic)
                if topic_key not in self.data_storage or self.data_storage[topic_key] is None:
                    all_received = False
                    break

            if all_received:
                return True

            time.sleep(0.1)
            # No need to call spin_once here since the executor is running in a separate thread

        return False

    def get_all_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data from all subscribed topics.

        Returns:
            Dictionary with all topic data
        """
        result = {}
        for topic_key in self.data_storage:
            with self.data_locks[topic_key]:
                data = self.data_storage[topic_key]
                result[topic_key] = data.copy() if data is not None else None
        return result

    def is_data_available(self, topics: Union[str, list]) -> bool:
        """
        Check if data is available for specified topics.

        Args:
            topics: Single topic name or list of topic names

        Returns:
            True if all specified data is available
        """
        if isinstance(topics, str):
            topics = [topics]

        for topic in topics:
            topic_key = self._get_topic_key(topic)
            if topic_key not in self.data_storage or self.data_storage[topic_key] is None:
                return False
        return True

    def get_subscribed_topics(self) -> list:
        """Get list of all subscribed topics."""
        return list(self.data_storage.keys())

    def shutdown(self):
        """Shutdown ROS2 infrastructure."""
        try:
            # Set shutdown flag to signal spin thread
            self._should_shutdown = True

            # Send final zero command if we have action publisher
            for attr_name in dir(self):
                if attr_name.startswith('publisher_'):
                    publisher = getattr(self, attr_name)
                    if hasattr(publisher, 'publish'):
                        # Try to publish zero command
                        try:
                            if 'cmd_vel' in attr_name:
                                zero_msg = Twist()
                                publisher.publish(zero_msg)
                        except:
                            pass

            # Destroy ROS2 node
            self.destroy_node()

            self.get_logger().info('Generic ROS2 Infrastructure shutdown successfully')

        except Exception as e:
            self.get_logger().error(f'Error shutting down infrastructure: {e}')


class ROS2Manager:
    """
    Manager class for handling ROS2 infrastructure lifecycle.
    """

    def __init__(self):
        rclpy.init()

        self.infrastructure = None
        self.spin_thread = None
        self.executor = None
        self.is_initialized = False
        self._should_shutdown = False

    def initialize(self, node_name: str = 'ros2_infrastructure') -> ROS2Infrastructure:
        """Initialize generic ROS2 infrastructure in a separate thread."""
        if not self.is_initialized:
            self.infrastructure = ROS2Infrastructure(node_name)

            # Create executor and add node
            self.executor = MultiThreadedExecutor()
            self.infrastructure.get_logger().info('Adding node to executor...')
            self.executor.add_node(self.infrastructure)
            self.infrastructure.get_logger().info('Node added to executor successfully')

            # Start spinning in a separate thread
            self.infrastructure.get_logger().info('Starting spin thread...')
            self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self.spin_thread.start()

            # Give some time for thread to start
            import time
            time.sleep(0.1)

            self.is_initialized = True
            self.infrastructure.get_logger().info('ROS2 Manager initialization completed')

        return self.infrastructure

    def _spin_loop(self):
        """Spin loop running in separate thread."""
        try:
            self.infrastructure.get_logger().info('Spin loop started')
            while not self._should_shutdown and rclpy.ok():
                self.executor.spin_once(timeout_sec=0.1)
            self.infrastructure.get_logger().info('Spin loop stopped')
        except KeyboardInterrupt:
            self.infrastructure.get_logger().info('Keyboard interrupt, shutting down.')
        finally:
            if self.infrastructure:
                self.infrastructure.destroy_node()

    def shutdown(self):
        """Shutdown ROS2 infrastructure."""
        self._should_shutdown = True

        if self.spin_thread and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)

        if self.infrastructure:
            self.infrastructure.shutdown()

        if rclpy.ok():
            rclpy.shutdown()

        self.is_initialized = False
