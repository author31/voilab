import numpy as np
import time
import transforms3d
from typing import Dict, Optional, Tuple, Any
from diffusion_policy.infrastructure.ros2_infrastructure import ROS2Infrastructure, ROS2Manager


class ROS2Environment:
    """
    ROS2 environment interface for robotics tasks.

    This environment provides a clean interface that separates
    environment logic from ROS2 communication infrastructure.
    """

    def __init__(self,
                 rgb_topic: str = '/rgb',
                 joint_states_topic: str = '/joint_states',
                 gripper_topic: str = '/gripper',
                 action_topic: str = '/cmd_vel',
                 image_shape: Tuple[int, int, int] = (3, 224, 224),
                 timeout: float = 5.0,
                 real_world: bool = False,
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
            real_world: Whether this is a real-world or simulation environment
            manager: Optional ROS2Manager instance (creates new one if None)
        """
        # Store environment parameters
        self.rgb_topic = rgb_topic
        self.joint_states_topic = joint_states_topic
        self.gripper_topic = gripper_topic
        self.action_topic = action_topic
        self.image_shape = image_shape
        self.timeout = timeout
        self.real_world = real_world

        # Initialize ROS2 manager and infrastructure
        self.manager = manager or ROS2Manager()
        self.infrastructure = self.manager.initialize(node_name='ros2_environment')

        # Set up subscriptions for required topics
        self._setup_subscriptions()

        # Wait for initial sensor data
        if not self._wait_for_initial_data(timeout):
            raise TimeoutError(f"Timeout waiting for sensor data after {timeout} seconds")

        print('ROS2 Environment initialized successfully')

    def _setup_subscriptions(self):
        """Set up subscriptions for all required topics."""
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float32

        # Create subscribers for required topics
        self.infrastructure.create_subscriber(self.rgb_topic, Image)
        self.infrastructure.create_subscriber(self.joint_states_topic, JointState)
        self.infrastructure.create_subscriber(self.gripper_topic, Float32)

        print(f"Subscriptions created for: {self.rgb_topic}, {self.joint_states_topic}, {self.gripper_topic}")

    def _wait_for_initial_data(self, timeout: float) -> bool:
        """Wait for initial data from all subscribed topics."""
        required_topics = [self.rgb_topic, self.joint_states_topic, self.gripper_topic]
        return self.infrastructure.wait_for_data(required_topics, timeout)

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
            return self.get_obs()

        except Exception as e:
            print(f'Error resetting environment: {e}')
            raise

    def step(self, action: np.ndarray):
        """
        Execute action in the environment.

        Args:
            action: Action array to execute (format depends on robot configuration)

        Returns:
            Tuple of (observation, reward, done, info) - matching gym interface
        """
        try:
            # Publish action through infrastructure
            self._publish_action(action)

            # Get new observation
            obs = self.get_obs()

            # For now, return dummy reward/done/info
            # These can be customized based on specific task requirements
            reward = 0.0
            done = False
            info = {}

            return obs, reward, done, info

        except Exception as e:
            print(f'Error executing action: {e}')
            # Return current observation even if action fails
            return self.get_obs(), 0.0, False, {'error': str(e)}

    def get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get current observation from the environment.

        Returns:
            Dictionary containing:
            - camera0_rgb: RGB image [3, 224, 224]
            - robot0_eef_pos: End-effector position [3]
            - robot0_eef_rot_axis_angle: Rotation in axis-angle format [6]
            - robot0_gripper_width: Gripper width [1]
            - robot0_eef_rot_axis_angle_wrt_start: Rotation relative to start [6]
        """
        # Get raw data from subscribed topics
        raw_data = self._get_raw_sensor_data()

        # Check if all required data is available
        if not self._is_data_available(raw_data):
            raise RuntimeError("Not all sensor data available for observation")

        return self._process_raw_observations(raw_data)

    def _publish_action(self, action: np.ndarray):
        """Publish action to the action topic."""
        from geometry_msgs.msg import Twist

        # Convert action to Twist message
        twist_msg = Twist()
        if len(action) >= 3:
            twist_msg.linear.x = float(action[0])
            twist_msg.linear.y = float(action[1])
            twist_msg.linear.z = float(action[2])
        if len(action) >= 6:
            twist_msg.angular.x = float(action[3])
            twist_msg.angular.y = float(action[4])
            twist_msg.angular.z = float(action[5])

        # Publish using infrastructure
        self.infrastructure.publish_message(self.action_topic, {
            'type': 'twist',
            'linear': {
                'x': twist_msg.linear.x,
                'y': twist_msg.linear.y,
                'z': twist_msg.linear.z
            },
            'angular': {
                'x': twist_msg.angular.x,
                'y': twist_msg.angular.y,
                'z': twist_msg.angular.z
            }
        })

    def _get_raw_sensor_data(self) -> Dict[str, any]:
        """Get raw sensor data from subscribed topics."""
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

        Args:
            raw_data: Raw data from infrastructure

        Returns:
            Processed observation dictionary
        """
        # Extract RGB image
        rgb_obs = raw_data['rgb_image']

        # Extract end-effector position (simplified)
        joint_positions = raw_data['joint_states']['position']
        if len(joint_positions) >= 3:
            hand_pos = joint_positions[:3]  # Simplified assumption
        else:
            hand_pos = np.zeros(3)

        # Extract current rotation
        current_euler = self._extract_euler_angles(raw_data['joint_states'])
        rot_6d = self._euler_to_rotation_6d(current_euler)

        # Extract gripper width
        gripper_width = np.array([raw_data['gripper_width']])

        # Compute rotation relative to start
        if raw_data['start_euler'] is not None:
            # Compute relative rotation
            start_rot_matrix = transforms3d.euler.euler2mat(
                raw_data['start_euler'][0], raw_data['start_euler'][1], raw_data['start_euler'][2]
            )
            current_rot_matrix = transforms3d.euler.euler2mat(
                current_euler[0], current_euler[1], current_euler[2]
            )

            relative_rot_matrix = current_rot_matrix @ np.linalg.inv(start_rot_matrix)
            rot_wrt_start_6d = relative_rot_matrix[:, :2].flatten()
        else:
            rot_wrt_start_6d = np.zeros(6)

        return {
            "camera0_rgb": rgb_obs,
            "robot0_eef_pos": hand_pos,
            "robot0_eef_rot_axis_angle": rot_6d,
            "robot0_gripper_width": gripper_width,
            "robot0_eef_rot_axis_angle_wrt_start": rot_wrt_start_6d
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
            "camera0_rgb": self.image_shape,
            "robot0_eef_pos": (3,),
            "robot0_eef_rot_axis_angle": (6,),
            "robot0_gripper_width": (1,),
            "robot0_eef_rot_axis_angle_wrt_start": (6,)
        }

    def __del__(self):
        """Cleanup on destruction."""
        self.close()


class ROS2EnvironmentFactory:
    """
    Factory class for creating ROS2 environments with common configurations.
    """

    @staticmethod
    def create_default_environment(real_world: bool = False) -> ROS2Environment:
        """
        Create environment with default configuration.

        Args:
            real_world: Whether this is a real-world or simulation environment

        Returns:
            Configured ROS2Environment instance
        """
        return ROS2Environment(real_world=real_world)

    @staticmethod
    def create_custom_environment(rgb_topic: str = '/rgb',
                                joint_states_topic: str = '/joint_states',
                                gripper_topic: str = '/gripper',
                                action_topic: str = '/cmd_vel',
                                image_shape: Tuple[int, int, int] = (3, 224, 224),
                                real_world: bool = False) -> ROS2Environment:
        """
        Create environment with custom topic configuration.

        Args:
            rgb_topic: RGB camera topic
            joint_states_topic: Joint states topic
            gripper_topic: Gripper state topic
            action_topic: Action command topic
            image_shape: Expected image shape
            real_world: Real-world flag

        Returns:
            Configured ROS2Environment instance
        """
        return ROS2Environment(
            rgb_topic=rgb_topic,
            joint_states_topic=joint_states_topic,
            gripper_topic=gripper_topic,
            action_topic=action_topic,
            image_shape=image_shape,
            real_world=real_world
        )

    @staticmethod
    def create_franka_environment(real_world: bool = False) -> ROS2Environment:
        """
        Create environment pre-configured for Franka robot.

        Args:
            real_world: Whether this is a real-world or simulation environment

        Returns:
            Configured ROS2Environment instance
        """
        return ROS2Environment(
            rgb_topic='/camera/color/image_raw',
            joint_states_topic='/joint_states',
            gripper_topic='/gripper_width',
            action_topic='/cartesian_velocity_controller/cmd_vel',
            image_shape=(3, 224, 224),
            real_world=real_world
        )

    @staticmethod
    def create_ur5_environment(real_world: bool = False) -> ROS2Environment:
        """
        Create environment pre-configured for UR5 robot.

        Args:
            real_world: Whether this is a real-world or simulation environment

        Returns:
            Configured ROS2Environment instance
        """
        return ROS2Environment(
            rgb_topic='/rgb/image',
            joint_states_topic='/ur5/joint_states',
            gripper_topic='/ur5/gripper_width',
            action_topic='/ur5/cmd_vel',
            image_shape=(3, 224, 224),
            real_world=real_world
        )


# Example usage and testing
def main():
    """Example usage of ROS2Environment."""
    try:
        # Create environment using factory
        env = ROS2EnvironmentFactory.create_default_environment(real_world=False)

        # Test environment interface
        print("Testing ROS2 Environment...")

        # Reset environment
        obs = env.reset()
        print("Reset successful. Observation shapes:")
        for key, value in obs.items():
            print(f"  {key}: {value.shape}")

        # Test step
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move forward
        obs, reward, done, info = env.step(action)
        print(f"Step successful. Reward: {reward}, Done: {done}")

        # Test observation shapes
        shapes = env.get_observation_shapes()
        print("\nObservation space information:")
        for key, shape in shapes.items():
            print(f"  {key}: {shape}")

        # Check if ready
        print(f"\nEnvironment ready: {env.is_ready()}")

        # Clean up
        env.close()
        print("Environment test completed successfully!")

    except Exception as e:
        print(f"Error in environment test: {e}")


if __name__ == '__main__':
    main()