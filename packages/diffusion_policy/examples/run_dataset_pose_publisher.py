#!/usr/bin/env python3

"""
Simplified script to publish pose information from dataset.zarr.zip via ROS2.
Uses inverse kinematics with ikpy to convert end-effector poses to realistic joint angles.

Dataset structure:
     ├── camera0_rgb (T, H, W, 3) uint8
     ├── robot0_demo_end_pose (T, 6)
     ├── robot0_demo_start_pose (T, 6)
     ├── robot0_eef_pos (T, 3)
     ├── robot0_eef_rot_axis_angle (T, 3)
     └── robot0_gripper_width (T, 1)

Published to /joint_command (sensor_msgs/JointState):
    - header: timestamp and frame_id
    - name: actual robot joint names from URDF
    - position: joint angles solved via IK from EE poses
    - velocity: empty list
    - effort: empty list
"""

import argparse
import time
import sys
import os
import rclpy
from rclpy.node import Node
import numpy as np
from pathlib import Path
import zarr
import zipfile
from scipy.spatial.transform import Rotation
from ikpy.chain import Chain
from sensor_msgs.msg import JointState


class SimpleDatasetPosePublisher(Node):
    """Simplified ROS2 Node to publish joint states from dataset using IKPy solver."""

    def __init__(self, dataset_path, urdf_path, topic, publish_rate):
        super().__init__('simple_dataset_pose_publisher')

        self.dataset_path = dataset_path
        self.urdf_path = urdf_path
        self.topic = topic
        self.publish_rate = publish_rate

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(JointState, topic, 10)

        # Load dataset and robot model
        self.data = self.load_dataset(dataset_path)
        self.robot_chain = self.load_robot_model(urdf_path)

        # Get joint names from the robot model - 7 arm joints
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        self.get_logger().info(f'Active joints ({len(self.joint_names)}): {self.joint_names}')

        # Playback state
        self.current_step = 0
        self.total_steps = len(self.data['robot0_eef_pos'])
        self.loop_enabled = False
        self.episode_delay = 2.0

        # IK solver state
        self.neutral_joint_angles = self.get_neutral_joint_config()
        self.last_joint_angles = None

        self.get_logger().info(f'Simple Dataset Pose Publisher started')
        self.get_logger().info(f'Publishing to topic: {topic}')
        self.get_logger().info(f'Dataset contains {self.total_steps} timesteps')

    def get_neutral_joint_config(self):
        """Get a neutral joint configuration for the Franka Panda robot."""
        # 7 Franka arm joints - all values must be within URDF bounds
        # panda_joint4 has restricted range: (-3.0718, -0.0698), so -2.356 is valid
        return [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    def load_dataset(self, zip_path):
        """Extracts and loads the Zarr dataset from a zip file."""
        try:
            self.get_logger().info(f"Loading dataset from '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_path = os.path.splitext(zip_path)[0]
                zip_ref.extractall(extract_path)

                zarr_path = os.path.join(extract_path, 'data')
                data = zarr.open(store=zarr.DirectoryStore(zarr_path), mode='r')

                self.get_logger().info(f"Dataset loaded successfully")
                return data
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            raise

    def load_robot_model(self, urdf_path):
        """Loads the robot kinematics chain from a URDF file using IKPy."""
        try:
            self.get_logger().info(f"Loading robot URDF from '{urdf_path}'...")

            # Active joints mask for Franka Panda (7 revolute joints only)
            # This matches the working validation script - fixed joints are handled automatically
            self.active_joints_mask = [False, True, True, True, True, True, True, True, False]

            robot_chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=["panda_link0"],
                active_links_mask=self.active_joints_mask
            )

            # Debug: Log chain information
            self.get_logger().info(f"Robot chain loaded successfully")
            self.get_logger().info(f"Number of links in chain: {len(robot_chain.links)}")
            self.get_logger().info(f"Active joints count: {sum(self.active_joints_mask)}")
            self.get_logger().info(f"Expected joint angles length: {sum(self.active_joints_mask)}")

            # Debug: Log joint bounds for each link
            for i, link in enumerate(robot_chain.links):
                if hasattr(link, 'bounds') and link.bounds is not None:
                    self.get_logger().info(f"Joint {i} ({link.name}) bounds: {link.bounds}")
                else:
                    self.get_logger().info(f"Joint {i} ({link.name}) bounds: None")

            return robot_chain

        except Exception as e:
            self.get_logger().error(f"Failed to load robot model: {e}")
            raise

    def solve_ik(self, target_pos, target_rot):
        """
        Solve inverse kinematics for target pose using IKPy.

        Args:
            target_pos: Target position [x, y, z]
            target_rot: Target rotation as axis-angle [rx, ry, rz]

        Returns:
            list of joint angles for active joints, or None if IK fails
        """
        try:
            # Convert axis-angle to rotation matrix
            rot = Rotation.from_rotvec(target_rot)
            target_orientation_matrix = rot.as_matrix()

            # Use last successful pose as initial guess, otherwise neutral
            initial_position = self.last_joint_angles if self.last_joint_angles is not None else self.neutral_joint_angles

            # Debug: Log initial position info
            self.get_logger().debug(f"Initial position length: {len(initial_position)}")
            self.get_logger().debug(f"Initial position: {initial_position}")

            # Ensure initial position matches expected length for IKPy
            # We need to match the full chain length (9 links), but only provide values for active joints
            full_initial_position = [0.0] * len(self.robot_chain.links)

            # Map our 7 active joints to the correct positions in the full chain
            active_indices = [i for i, active in enumerate(self.active_joints_mask) if active]
            for i, active_idx in enumerate(active_indices):
                if i < len(initial_position):
                    full_initial_position[active_idx] = initial_position[i]

            # Debug: Check initial position against bounds
            self.get_logger().debug(f"Full initial position for IK: {full_initial_position}")
            for i, (angle, link) in enumerate(zip(full_initial_position, self.robot_chain.links)):
                if hasattr(link, 'bounds') and link.bounds is not None:
                    min_angle, max_angle = link.bounds
                    if angle < min_angle or angle > max_angle:
                        self.get_logger().warn(f"Initial position {i} ({angle}) outside bounds for {link.name}: {link.bounds}")

            # transform the target_pos's base to "panda_joint0" base
            initial_position = full_initial_position

            # Solve IK using IKPy
            calculated_joints = self.robot_chain.inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_orientation_matrix,
                orientation_mode="all",
                initial_position=initial_position
            )

            # Filter to get only active joint angles (excluding base and end joints)
            active_joint_angles = [calculated_joints[i] for i, active in enumerate(self.active_joints_mask) if active]

            # Debug: Log results
            self.get_logger().debug(f"IK returned {len(calculated_joints)} joint angles")
            self.get_logger().debug(f"Filtered to {len(active_joint_angles)} active joints: {active_joint_angles}")

            # Validate joint angles
            if self.validate_joint_angles(active_joint_angles):
                self.last_joint_angles = active_joint_angles  # Store for next iteration
                return active_joint_angles
            else:
                self.get_logger().warn(f"Joint angles validation failed: {active_joint_angles}")
                return None

        except Exception as e:
            self.get_logger().warn(f"IK solver failed: {e}")
            self.get_logger().debug(f"Target position: {target_pos}")
            self.get_logger().debug(f"Target rotation: {target_rot}")
            return None

    def validate_joint_angles(self, joint_angles):
        """Validate that joint angles are within reasonable limits for Franka Panda."""
        if joint_angles is None or len(joint_angles) == 0:
            return False

        # Franka Panda joint limits from URDF (radians)
        joint_limits = [
            (-2.8973, 2.8973),   # panda_joint1
            (-1.7628, 1.7628),   # panda_joint2
            (-2.8973, 2.8973),   # panda_joint3
            (-3.0718, -0.0698),  # panda_joint4
            (-2.8973, 2.8973),   # panda_joint5
            (-0.0175, 3.7525),   # panda_joint6
            (-2.8973, 2.8973),   # panda_joint7
        ]

        for i, angle in enumerate(joint_angles):
            if i >= len(joint_limits):
                break
            min_angle, max_angle = joint_limits[i]
            if angle < min_angle - 0.1 or angle > max_angle + 0.1:  # Small tolerance
                return False

        # Check for NaN or infinite values
        if any(not np.isfinite(angle) for angle in joint_angles):
            return False

        return True

    def publish_step(self):
        """Publish a single timestep from the dataset."""
        if self.current_step >= self.total_steps:
            if self.loop_enabled:
                self.current_step = 0
                self.get_logger().info("Looping back to first timestep...")
                self.last_joint_angles = None  # Reset IK state
            else:
                self.get_logger().info("All timesteps published successfully")
                return False

        try:
            # Extract pose data from dataset
            target_pos = self.data['robot0_eef_pos'][self.current_step]
            target_rot = self.data['robot0_eef_rot_axis_angle'][self.current_step]
            gripper_width = self.data['robot0_gripper_width'][self.current_step][0]

            # Solve IK to get joint angles
            joint_angles = self.solve_ik(target_pos, target_rot)

            if joint_angles is None:
                self.get_logger().warn(f"Skipping step {self.current_step} due to IK failure")
                self.current_step += 1
                return True

            # Add gripper joint values
            final_joint_angles = list(joint_angles)
            finger_positions = [gripper_width, gripper_width]  # Both fingers same width
            final_joint_angles.extend(finger_positions)

            # Convert numpy types to native Python floats for ROS2
            final_joint_angles = [float(angle) for angle in final_joint_angles]

            # Create and publish JointState message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "panda_hand"

            # Use all joint names (arm + gripper) for publishing
            all_joint_names = self.joint_names + self.gripper_joint_names
            msg.name = all_joint_names
            msg.position = final_joint_angles
            msg.velocity = []
            msg.effort = []

            self.publisher_.publish(msg)

            # Log progress occasionally
            if self.current_step % 100 == 0:
                self.get_logger().info(f"Published step {self.current_step}/{self.total_steps}")

        except Exception as e:
            self.get_logger().error(f"Error publishing step {self.current_step}: {e}")

        # Move to next step
        self.current_step += 1
        return True

    def run_publishing_loop(self):
        """Simple publishing loop without threading."""
        try:
            publish_interval = 1.0 / self.publish_rate

            while rclpy.ok():
                if not self.publish_step():
                    if self.loop_enabled:
                        time.sleep(self.episode_delay)
                        continue
                    else:
                        break

                time.sleep(publish_interval)

        except Exception as e:
            self.get_logger().error(f"Error in publishing loop: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Publish joint states from dataset.zarr.zip using inverse kinematics with IKPy'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset.zarr.zip file'
    )
    parser.add_argument(
        '--urdf_path',
        type=str,
        default=None,
        help='Path to robot URDF file (default: assets/franka_panda/franka_panda.urdf)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='/joint_command',
        help='ROS2 topic to publish joint states to (default: /joint_command)'
    )
    parser.add_argument(
        '--publish_rate',
        type=float,
        default=10.0,
        help='Publishing rate in Hz (default: 10.0)'
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop through timesteps continuously'
    )
    parser.add_argument(
        '--episode_delay',
        type=float,
        default=2.0,
        help='Delay between loops in seconds (default: 2.0)'
    )

    return parser.parse_args()


def main():
    """Main function to load dataset and publish joint states using IKPy."""
    args = parse_args()

    # Determine URDF path
    if args.urdf_path is None:
        script_dir = Path(__file__).parent.parent.parent.parent
        urdf_path = script_dir / 'assets' / 'franka_panda' / 'franka_panda_umi.urdf'
    else:
        urdf_path = Path(args.urdf_path)

    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")
    print(f"Using URDF: {urdf_path}")

    try:
        # Initialize ROS2
        rclpy.init()

        # Create publisher node
        publisher_node = SimpleDatasetPosePublisher(
            str(dataset_path), str(urdf_path), args.topic, args.publish_rate
        )
        publisher_node.loop_enabled = args.loop
        publisher_node.episode_delay = args.episode_delay

        try:
            # Run publishing loop
            publisher_node.run_publishing_loop()
        except KeyboardInterrupt:
            publisher_node.get_logger().info("Publishing interrupted by user")
        finally:
            # Cleanup
            publisher_node.destroy_node()
            rclpy.shutdown()
            print("Shutdown complete")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
