import time
import numpy as np
import transforms3d


from umi.common.pose_util import mat_to_rot6d, pose_to_mat
from typing import Dict, Optional, Tuple, Any
from diffusion_policy.infrastructure.ros2_infrastructure import ROS2Manager
from cv_bridge import CvBridge
from diffusion_policy.model.common.rotation_transformer import transform_rotation
from scipy.spatial.transform import Rotation
from loguru import logger
from ikpy.chain import Chain



class ROS2Environment:
    """
    ROS2 environment interface for robotics tasks.

    This environment provides a clean interface that separates
    environment logic from ROS2 communication infrastructure.
    """

    def __init__(self,
                 urdf_path: str,
                 rgb_topic: str = '/rgb',
                 joint_states_topic: str = '/eef_states',
                 gripper_topic: str = '/gripper_width',
                 action_topic: str = '/joint_states',
                 image_shape: Tuple[int, int, int] = (3, 224, 224),
                 timeout: float = 5.,
                 n_obs_steps: int = 2,
                 manager: Optional[ROS2Manager] = None):
        """
        Initialize ROS2 environment.

        Args:
            rgb_topic: Topic name for RGB camera images
            joint_states_topic: Topic name for robot joint states
            gripper_topic: Topic name for gripper state
            action_topic: Topic name for publishing actions
            image_shape: Expected shape of RGB images (C, H, W)
            timeout: Timeout for waiting for sensor data
            n_obs_steps: Number of observation steps to stack for diffusion policy
            manager: Optional ROS2Manager instance (creates new one if None)
        """
        # Store environment parameters
        self.rgb_topic = rgb_topic
        self.joint_states_topic = joint_states_topic
        self.gripper_topic = gripper_topic
        self.action_topic = action_topic
        self.image_shape = image_shape
        self.timeout = timeout
        self.n_obs_steps = n_obs_steps
        self.robot_chain = self.load_robot_model(urdf_path)

        # Initialize neutral joint configuration for Franka Panda
        self.neutral_joint_angles = [
            0.0,    # panda_joint1
            -0.785, # panda_joint2
            0.0,    # panda_joint3
            -2.356, # panda_joint4
            0.0,    # panda_joint5
            1.571,  # panda_joint6
            0.785,  # panda_joint7
        ]

        # Initialize last_joint_angles to match full chain length (including fixed joints)
        self.last_joint_angles = [0.0] * len(self.robot_chain.links)

        # Initialize observation history for stacking
        self.obs_history = []

        # Initialize ROS2 manager and infrastructure
        self.manager = manager or ROS2Manager()
        self.infrastructure = self.manager.initialize(node_name='ros2_environment')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Set up subscriptions for required topics
        self._setup_subscriptions()

        # Wait for initial sensor data
        if not self._wait_for_initial_data(timeout):
            raise TimeoutError(f"Timeout waiting for sensor data after {timeout} seconds")

        print('ROS2 Environment initialized successfully')

    def _setup_subscriptions(self):
        """Set up subscriptions for all required topics."""
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32, Float64
        from geometry_msgs.msg import Pose


        # Create subscribers for required topics with custom conversion callback
        self.infrastructure.create_subscriber(self.rgb_topic, Image, self._convert_message)
        self.infrastructure.create_subscriber(self.joint_states_topic, Pose, self._convert_message)
        self.infrastructure.create_subscriber(self.gripper_topic, Float64, self._convert_message)

        print(f"Subscriptions created for: {self.rgb_topic}, {self.joint_states_topic}, {self.gripper_topic}")

    def _wait_for_initial_data(self, timeout: float) -> bool:
        """Wait for initial data from all subscribed topics."""
        required_topics = [self.rgb_topic, self.joint_states_topic, self.gripper_topic]
        return self.infrastructure.wait_for_data(required_topics, timeout)


    def load_robot_model(self, urdf_path):
        """Loads the robot kinematics chain from a URDF file using IKPy."""
        try:
            logger.info(f"Loading robot URDF from '{urdf_path}'...")

            # Active joints mask for Franka Panda (7 revolute joints only)
            # This matches the working validation script - fixed joints are handled automatically
            self.active_joints_mask = [False, True, True, True, True, True, True, True, False]

            robot_chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=["panda_link0"],
                active_links_mask=self.active_joints_mask
            )

            logger.info(f"Robot chain loaded successfully")
            logger.info(f"Number of links in chain: {len(robot_chain.links)}")
            logger.info(f"Active joints count: {sum(self.active_joints_mask)}")
            logger.info(f"Expected joint angles length: {sum(self.active_joints_mask)}")

            return robot_chain

        except Exception as e:
            logger.error(f"Failed to load robot model: {e}")
            raise

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            Initial observation after reset
        """
        try:
            # Send zero command to stop robot
            zero_action = np.zeros(6)
            self.step(zero_action)

            # Wait for new sensor data
            time.sleep(1.0)

            # Wait for new initial data
            if not self._wait_for_initial_data(self.timeout):
                raise RuntimeError("Timeout waiting for sensor data after reset")

            print('Environment reset successfully')

            # Return initial observation
            return self.get_obs(n_steps=self.n_obs_steps)

        except Exception as e:
            print(f'Error resetting environment: {e}')
            raise

    def step(self, action: np.ndarray):
        """
        Execute action in the environment.

        Args:
            action: Action array, (Ta, 10), to execute. Format: [x,y,z, rot_6d(6), gripper]

        Returns:
            Tuple of (observation, reward, done, info).
        """
        try:
            # Validate action shape
            if len(action.shape) != 2 or action.shape[1] != 10:
                raise ValueError(f"Expected action shape [N, 10], got {action.shape}")

            current_ik_guess = self.last_joint_angles

            for robot_step in range(action.shape[0]):
                action_position = action[robot_step, 0:3]

                action_rotation_6d = action[robot_step, 3:9].reshape(1, 6) # Keep as 2D for compatibility
                action_rotation = transform_rotation(
                    action_rotation_6d, from_rep='rotation_6d', to_rep='quaternion'
                )

                # Extract gripper width (element 9, 10th element)
                gripper_width = action[robot_step, 9] # Single value for gripper
                rot = Rotation.from_quat(action_rotation[0]) # Use first (and only) rotation
                target_orientation_matrix = rot.as_matrix()

                # Prepare initial position for IK solver
                if len(current_ik_guess) < len(self.robot_chain.links):
                    initial_ik_position = list(current_ik_guess) + [0.0] * (len(self.robot_chain.links) - len(current_ik_guess))
                else:
                    initial_ik_position = current_ik_guess

                joint_angles = self.robot_chain.inverse_kinematics(
                    target_position=action_position,
                    target_orientation=target_orientation_matrix,
                    orientation_mode="all",
                    initial_position=initial_ik_position,
                )

                # Extract active joint angles
                active_joint_angles = [joint_angles[i] for i, active in enumerate(self.active_joints_mask) if active]

                # Update tracking variables
                current_ik_guess = joint_angles
                self.last_joint_angles = active_joint_angles

                # Publish joint states for robot arm
                self._publish_joint_states(active_joint_angles)

                # Publish gripper command
                self._publish_gripper_command(gripper_width)


            # Get new observation
            obs = self.get_obs()

            reward = 0.0
            done = False
            info = {}

            return obs, reward, done, info

        except Exception as e:
            print(f'Error executing action: {e}')
            # Return current observation even if action fails
            return self.get_obs(), 0.0, False, {'error': str(e)}

    def get_obs(self, n_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get current observation from the environment with optional stacking.

        Args:
            n_steps: Number of steps to stack (defaults to self.n_obs_steps)

        Returns:
            Dictionary containing stacked observations with shapes:
            - camera0_rgb: RGB images [n_steps, 3, 224, 224]
            - robot0_eef_pos: End-effector positions [n_steps, 3]
            - robot0_eef_rot_axis_angle: Rotation in axis-angle format [n_steps, 6]
            - robot0_gripper_width: Gripper widths [n_steps, 1]
            - robot0_eef_rot_axis_angle_wrt_start: Rotation relative to start [n_steps, 6]
        """
        # Get raw data from subscribed topics
        raw_data = self._get_raw_sensor_data()

        # Check if all required data is available
        if not self._is_data_available(raw_data):
            raise RuntimeError("Not all sensor data available for observation")

        # Process current observation
        current_obs = self._process_raw_observations(raw_data)

        # Add to observation history
        self.obs_history.append(current_obs)

        # Use provided n_steps or default to self.n_obs_steps
        if n_steps is None:
            n_steps = self.n_obs_steps

        return self._stack_last_n_obs(self.obs_history, n_steps)

    def _publish_joint_states(self, joint_states: np.ndarray):
        """Publish joint states to the action topic."""
        from sensor_msgs.msg import JointState

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.infrastructure.get_clock().now().to_msg()
        joint_state_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        joint_state_msg.position = joint_states.tolist()
        self.infrastructure.publish_message(self.action_topic, joint_state_msg, JointState)

    def _publish_gripper_command(self, gripper_width: float):
        """Publish gripper command to the gripper topic."""
        from std_msgs.msg import Float64

        gripper_msg = Float64()
        gripper_msg.data = float(gripper_width)

        # Use the gripper_topic for publishing commands
        # Note: In a real setup, you might want a separate command topic
        self.infrastructure.publish_message('/gripper_command', gripper_msg, Float64)

    def _convert_message(self, topic: str, raw_data: Dict[str, Any]):
        """
        Convert ROS2 message to environment-specific data structure.

        Args:
            topic: Topic name
            raw_data: Raw data from infrastructure
        """
        try:
            msg = raw_data.get('raw_message')
            if msg is None: return

            # Convert message based on topic
            if topic == self.rgb_topic:
                # Handle Image messages
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                converted_data = {
                    'data': cv_image,
                    'header': msg.header,
                    'encoding': msg.encoding,
                    'height': msg.height,
                    'width': msg.width,
                    'type': 'image'
                }
            elif topic == self.joint_states_topic:
                # Handle geometry_msgs/Pose messages
                converted_data = {
                    'position': np.array([msg.position.x, msg.position.y, msg.position.z]),
                    'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]),
                    'header': msg.header if hasattr(msg, 'header') else None,
                    'type': 'pose'
                }
            elif topic == self.gripper_topic:
                # Handle Float64 messages
                converted_data = {
                    'data': float(msg.data),
                    'type': 'float64'
                }
            else:
                # Generic fallback
                converted_data = {
                    'message': msg,
                    'type': 'unknown'
                }

            # Store the converted data back to infrastructure
            topic_key = self.infrastructure._get_topic_key(topic)
            with self.infrastructure.data_locks[topic_key]:
                self.infrastructure.data_storage[topic_key] = converted_data

        except Exception as e:
            print(f'Error converting message from {topic}: {e}')

    def _get_raw_sensor_data(self) -> Dict[str, any]:
        """
        Get raw sensor data from subscribed topics.

        Returns:
            Dictionary containing raw sensor data with the following structure:

            'rgb_image': Dict with keys:
                - 'data': np.ndarray of shape [224, 224, 3] containing RGB image data
                - 'header': ROS2 message header with timestamp and frame info
                - 'encoding': str, image encoding (e.g., 'rgb8')
                - 'height': int, image height in pixels
                - 'width': int, image width in pixels
                - 'type': str, message type identifier ('image')

            'joint_states': Dict with keys:
                - 'position': np.ndarray of shape [3], end-effector position [x, y, z]
                - 'orientation': np.ndarray of shape [4], quaternion orientation [x, y, z, w]
                - 'header': ROS2 message header or None
                - 'type': str, message type identifier ('pose')

            'gripper_width': Dict with keys:
                - 'data': float, gripper width in meters
                - 'type': str, message type identifier ('float64')
        """
        
        return {
            'rgb_image': self.infrastructure.get_data(self.rgb_topic),
            'joint_states': self.infrastructure.get_data(self.joint_states_topic),
            'gripper_width': self.infrastructure.get_data(self.gripper_topic)
        }

    def _is_data_available(self, raw_data: Dict[str, any]) -> bool:
        """Check if all required sensor data is available."""
        return all(data is not None for data in raw_data.values())

    def _process_raw_observations(self, raw_data: Dict[str, any]) -> Dict[str, np.ndarray]:
        """
        Process raw sensor data into structured observations.
        """
        # Extract RGB image
        assert "rgb_image" in raw_data
        assert "joint_states" in raw_data
        assert "gripper_width" in raw_data

        rgb_obs = raw_data['rgb_image']['data']

        # Ensure RGB is in correct format (H, W, 3) and normalized to [0, 1]
        if rgb_obs.dtype == np.uint8:
            rgb_obs = rgb_obs.astype(np.float32) / 255.0

        # Extract end-effector position
        joint_states = raw_data['joint_states']
        agent_pos = joint_states['position'] # an np.array (x,y,z)
        agent_ori = joint_states['orientation'] # an np.array (x,y,z,w)

        # Quaternion to 6D conversion
        rot_matrix = transforms3d.quaternions.quat2mat(agent_ori)
        rot_6d = mat_to_rot6d(rot_matrix)

        # Store initial pose for relative calculation (on first call)
        if not hasattr(self, 'start_pose_mat'): 
            self.start_pose_mat = rot_matrix

        # Current pose matrix (need proper axis-angle conversion)
        start_rot_inverse = self.start_pose_mat.T
        relative_rot_matrix = start_rot_inverse @ rot_matrix
        relative_rot_6d = mat_to_rot6d(relative_rot_matrix)

        return {
            "camera0_rgb": rgb_obs,
            "robot0_eef_pos": agent_pos,
            "robot0_eef_rot_axis_angle": rot_6d,
            "robot0_gripper_width": np.array([raw_data['gripper_width']['data']]),
            "robot0_eef_rot_axis_angle_wrt_start": relative_rot_6d
        }

    def _extract_euler_angles(self, joint_states: Dict) -> Optional[np.ndarray]:
        """Extract Euler angles from joint states."""
        if joint_states is None:
            return None

        # This is a simplified implementation
        # In practice, you would compute this from forward kinematics
        positions = joint_states['position']

        # Assuming the last 3-4 joint values relate to orientation
        # This needs to be adapted to your specific robot configuration
        if len(positions) >= 3:
            # Extract orientation-related joints
            orientation_joints = positions[-3:]
            return orientation_joints

        return np.zeros(3)

    def _euler_to_rotation_6d(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to 6D rotation representation."""
        # Convert Euler angles to rotation matrix
        rot_matrix = transforms3d.euler.euler2mat(
            euler_angles[0], euler_angles[1], euler_angles[2]
        )

        # Convert to 6D representation (first two columns of rotation matrix)
        rotation_6d = rot_matrix[:, :2].flatten()

        return rotation_6d

    def render(self, mode='human'):
        """
        Render the environment (optional implementation).

        Args:
            mode: Rendering mode
        """
        # Optional: implement visualization if needed
        pass

    def close(self):
        """Clean up environment resources."""
        try:
            # Send final zero command to stop robot
            zero_action = np.zeros(6)
            self.step(zero_action)

            # Shutdown infrastructure through manager
            if self.manager:
                self.manager.shutdown()

            print('ROS2 Environment closed successfully')

        except Exception as e:
            print(f'Error closing environment: {e}')

    def seed(self, seed=None):
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        # Optional: implement seed setting if needed
        pass

    @property
    def observation_space(self):
        """Return observation space information."""
        return self.get_observation_shapes()

    @property
    def action_space(self):
        """Return action space information."""
        # This should be customized based on your specific robot
        # For now, return a generic action space
        return {
            'shape': (6,),  # 6DOF action space
            'low': -1.0,
            'high': 1.0
        }

    def is_ready(self) -> bool:
        """Check if environment is ready (all sensor data available)."""
        raw_data = self._get_raw_sensor_data()
        return self._is_data_available(raw_data)

    def get_observation_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get information about observation shapes.

        Returns:
            Dictionary mapping observation names to their shapes
        """
        return {
            "camera0_rgb": (self.n_obs_steps,) + self.image_shape,
            "robot0_eef_pos": (self.n_obs_steps, 3),
            "robot0_eef_quat": (self.n_obs_steps, 6),
            "robot0_gripper_width": (self.n_obs_steps, 1),
        }

    def _stack_last_n_obs(self, obs_history: list, n_steps: int) -> Dict[str, np.ndarray]:
        """
        Stack the last n observations from history.

        Args:
            obs_history: List of observation dictionaries
            n_steps: Number of steps to stack

        Returns:
            Dictionary with stacked observations
        """
        assert len(obs_history) > 0, "Observation history cannot be empty"

        result = dict()
        for key in obs_history[-1].keys():
            result[key] = self._stack_array_last_n(
                [obs[key] for obs in obs_history],
                n_steps
            )
        return result

    def _stack_array_last_n(self, array_list: list, n_steps: int) -> np.ndarray:
        """
        Stack last n arrays from a list.

        Args:
            array_list: List of numpy arrays
            n_steps: Number of steps to stack

        Returns:
            Stacked array with shape (n_steps,) + array_shape
        """
        assert len(array_list) > 0, "Array list cannot be empty"

        result = np.zeros((n_steps,) + array_list[-1].shape,
                         dtype=array_list[-1].dtype)
        start_idx = -min(n_steps, len(array_list))
        result[start_idx:] = np.array(array_list[start_idx:])
        if n_steps > len(array_list):
            # pad with the oldest available observation
            result[:start_idx] = array_list[start_idx]
        return result

    def __del__(self):
        """Cleanup on destruction."""
        self.close()

