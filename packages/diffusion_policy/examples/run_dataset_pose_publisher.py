#!/usr/bin/env python3

"""
Simplified script to publish pose information from dataset.zarr.zip via ROS2.
Uses inverse kinematics with cuRobo to convert end-effector poses to realistic joint angles.

Dataset structure:
    data/
     ├── camera0_rgb (T, H, W, 3) uint8
     ├── robot0_demo_end_pose (T, 6)
     ├── robot0_demo_start_pose (T, 6)
     ├── robot0_eef_pos (T, 3)
     ├── robot0_eef_rot_axis_angle (T, 3)
     └── robot0_gripper_width (T, 1)
    meta/
     └── attrs: episode_ends (array of episode end indices)

Episode Handling:
    - The script respects episode boundaries from meta/episode_ends
    - At the end of each episode, the robot resets to neutral position
    - Episode transitions are separated by a configurable delay (default: 2.0s)
    - Supports looping through all episodes continuously

Coordinate Systems and Transformations:
    UMI Dataset Collection:
    - Dataset poses are collected using the Universal Manipulation Interface (UMI) method
    - UMI uses ORB_SLAM3 visual-inertial SLAM for tracking the handheld gripper
    - The SLAM system creates an arbitrary, gravity-aligned world frame per demonstration
    - An ArUco tag is used to establish a consistent reference frame (tx_slam_tag.json)
    - Final dataset stores end-effector poses in ArUco tag coordinate frame
    
    Isaac Sim Replay:
    - cuRobo IK solver expects poses in robot base frame (panda_link0)
    - This script transforms poses from tag frame to robot base frame
    
    Transformation Configuration (priority order):
    1. --tx_slam_tag_path: Load calibration from UMI's tx_slam_tag.json file
    2. --tag_to_robot_pos/quat: Manually specify tag position/orientation in robot frame
    3. --use_identity_transform: Assume tag frame = robot base frame (default)
    
    The default behavior (identity transform) assumes the ArUco tag coordinate system
    coincides with the robot base frame, meaning poses can be used directly.

Published to /joint_command (sensor_msgs/JointState):
    - header: timestamp and frame_id
    - name: actual robot joint names from URDF
    - position: joint angles solved via IK from EE poses
    - velocity: empty list
    - effort: empty list
"""
import torch
import argparse
import time
import sys
import os
import rclpy
from rclpy.node import Node
from pathlib import Path
import zarr
import zipfile
from sensor_msgs.msg import JointState as RosJointState
from diffusion_policy.model.common.rotation_transformer import transform_rotation
from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation

# cuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)


class ReplayBufferPosePublisher(Node):
    """ROS2 Node to publish joint states from replay buffer using cuRobo solver."""

    def __init__(self, dataset_path, urdf_path, topic, publish_rate, voxel_size=0.005,
                 tx_slam_tag_path=None, tag_to_robot_pos=None, tag_to_robot_quat=None, 
                 use_identity_transform=True):
        super().__init__('replay_buffer_pose_publisher')
        
        self.tensor_args = TensorDeviceType()
        self.voxel_size = voxel_size
        self.dataset_path = dataset_path
        self.urdf_path = urdf_path
        self.topic = topic
        self.publish_rate = publish_rate

        # Compute coordinate transformation from ArUco tag frame to robot base frame
        self.H_robot_tag = self._compute_transform(
            tx_slam_tag_path, tag_to_robot_pos, tag_to_robot_quat, use_identity_transform
        )
        self.get_logger().info(f"Using tag-to-robot transformation:\n{self.H_robot_tag}")

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(RosJointState, topic, 10)

        # Load dataset and robot model
        self.data, self.meta = self.load_dataset(dataset_path)

        # Get joint names from the robot model - 7 arm joints
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        # Parse episode information
        self.episode_ends = list(self.meta.episode_ends)
        self.episode_starts = [0] + self.episode_ends[:-1]
        self.num_episodes = len(self.episode_ends)
        self.get_logger().info(f"Dataset contains {self.num_episodes} episodes")
        self.get_logger().info(f"Episode boundaries: {self.episode_ends}")

        # Playback state
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_step = 0
        self.episode_end_step = self.episode_ends[0] if self.episode_ends else 0
        self.total_steps = len(self.data['robot0_eef_pos'])
        self.loop_enabled = False
        self.episode_delay = 2.0

        # Trajectory publishing state
        self.current_trajectory = []
        self.current_trajectory_idx = 0

        # IK solver state
        self.neutral_joint_angles = self.get_neutral_joint_config()
        self.last_joint_angles = self.neutral_joint_angles.copy()
        
        self.init_motion_gen_config()   # Initialize motion gen config

    def init_motion_gen_config(self, robot_cfg: str = "franka.yml", trajopt_tsteps: int = 32):
        self.get_logger().info(f'Initializing motion gen config')

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            trajopt_tsteps=trajopt_tsteps,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=False,
            num_trajopt_seeds=2,
            num_graph_seeds=2,  
            evaluate_interpolated_trajectory=True,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        self.get_logger().info(f'Motion gen config initialized')

    def _compute_transform(self, tx_slam_tag_path, tag_to_robot_pos, tag_to_robot_quat, 
                          use_identity_transform):
        """
        Compute the transformation matrix from tag frame to robot base frame.
        
        Priority: tx_slam_tag_path > manual pos/quat > identity
        
        Args:
            tx_slam_tag_path: Path to tx_slam_tag.json file
            tag_to_robot_pos: Position [x, y, z] of tag in robot frame
            tag_to_robot_quat: Quaternion [x, y, z, w] of tag in robot frame
            use_identity_transform: Whether to use identity transform
            
        Returns:
            H_robot_tag: 4x4 transformation matrix (robot frame expressed in tag frame)
        """
        import json
        
        H_tag_to_robot = np.eye(4)
        
        # Priority 1: Load from tx_slam_tag.json if provided
        if tx_slam_tag_path is not None:
            try:
                with open(tx_slam_tag_path, 'r') as f:
                    data = json.load(f)
                    tx_slam_tag = np.array(data['tx_slam_tag'])
                    # tx_slam_tag transforms from tag to SLAM frame
                    # We need the inverse: SLAM to tag
                    H_tag_to_robot = np.linalg.inv(tx_slam_tag)
                    self.get_logger().info(f"Loaded transformation from {tx_slam_tag_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load {tx_slam_tag_path}: {e}")
                raise
        
        # Priority 2: Use manual position and quaternion
        elif not use_identity_transform and tag_to_robot_pos is not None and tag_to_robot_quat is not None:
            H_tag_to_robot[:3, 3] = tag_to_robot_pos
            H_tag_to_robot[:3, :3] = Rotation.from_quat(tag_to_robot_quat).as_matrix()
            self.get_logger().info("Using manually specified tag-to-robot transformation")
        
        # Priority 3: Use identity transform (default)
        else:
            self.get_logger().info("Using identity transform (tag frame = robot base frame)")
        
        # Compute the inverse: robot frame expressed in tag frame
        H_robot_tag = np.linalg.inv(H_tag_to_robot)
        return H_robot_tag

    def get_neutral_joint_config(self):
        """Get a neutral joint configuration for the Franka Panda robot."""
        # 7 Franka arm joints - all values must be within URDF bounds
        # panda_joint4 has restricted range: (-3.0718, -0.0698), so -2.356 is valid
        return [0.0, 0.0002, 0.0, -0.0698, 0.0, 0.0005, 0.0]
    
    def go_to_home(self):
        """Go to home position."""
        self.last_joint_angles = self.neutral_joint_angles.copy()
        goal = RosJointState()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        goal.name = self.joint_names
        goal.position = self.neutral_joint_angles
        goal.velocity = []
        goal.effort = []
        self.publisher_.publish(goal)
        self.get_logger().info(f"Published home position")

    def load_dataset(self, zip_path):
        """Extracts and loads the Zarr dataset from a zip file.
        
        Returns:
            Tuple[zarr.Group, zarr.Group]: (data group, meta group)
        """
        try:
            self.get_logger().info(f"Loading dataset from '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_path = os.path.splitext(zip_path)[0]
                zip_ref.extractall(extract_path)

                zarr_root = zarr.open(store=zarr.DirectoryStore(extract_path), mode='r')
                data = zarr_root['data']
                meta = zarr_root['meta']

                self.get_logger().info(f"Dataset loaded successfully")
                return data, meta
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            raise

    def plan_motion(self, start_state: List[float], target_pos: np.ndarray, target_quat: np.ndarray) -> Tuple[bool, List[List[float]]]:
        """
        Plan motion to target end-effector pose using cuRobo motion generation.
        
        The target pose is provided in the ArUco tag coordinate frame (from the UMI dataset)
        and is automatically transformed to the robot base frame using the configured
        transformation (H_robot_tag) before being passed to the IK solver.

        Args:
            start_state: Start joint state [q1, q2, q3, q4, q5, q6, q7]
            target_pos: Target EE position [x, y, z] in tag frame (numpy array or list)
            target_quat: Target EE quaternion [x, y, z, w] in tag frame, scipy format (numpy array or list)

        Returns:
            Tuple[bool, List[List[float]]]: (success, trajectory_waypoints)
                - success: True if motion planning succeeded, False otherwise
                - trajectory_waypoints: List of joint configurations forming the trajectory
        """
        # Convert numpy arrays to lists if needed
        if isinstance(target_pos, np.ndarray):
            target_pos = target_pos.tolist()
        if isinstance(target_quat, np.ndarray):
            target_quat = target_quat.tolist()
        
        # Construct target pose in tag frame (from dataset)
        H_tag_to_eef = np.eye(4)
        H_tag_to_eef[:3, 3] = target_pos
        H_tag_to_eef[:3, :3] = Rotation.from_quat(target_quat).as_matrix()
        
        # Transform target pose from tag frame to robot base frame
        # H_robot_tag transforms robot frame to tag frame (computed in __init__)
        # We want: H_robot_to_eef = H_robot_tag @ H_tag_to_eef
        H_robot_to_eef = self.H_robot_tag @ H_tag_to_eef
        new_pos_R_E = H_robot_to_eef[:3, 3]
        new_quat_xyzw = Rotation.from_matrix(H_robot_to_eef[:3, :3]).as_quat()
        curobo_quat = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]

        # Ensure start_state is a list with correct length (7 joints for Franka)
        if len(start_state) > 7:
            start_state = start_state[:7]

        joint_state = CuroboJointState.from_position(
            position=self.tensor_args.to_device(start_state).unsqueeze(0),
            joint_names=self.joint_names
        )
        ik_goal = Pose(
            position=self.tensor_args.to_device(new_pos_R_E),
            quaternion=self.tensor_args.to_device(curobo_quat)
        )
        
        result = self.motion_gen.plan_single(
            joint_state, 
            ik_goal, 
            MotionGenPlanConfig(max_attempts=3, enable_finetune_trajopt=True)
        )
        
        success = bool(result.success.item())
        if success:
            result_trajectory = result.get_interpolated_plan().position.tolist()
            self.get_logger().info(f"Motion planning succeeded with {len(result_trajectory)} waypoints")
        else:
            result_trajectory = [start_state]
            self.get_logger().warn(f"Motion planning failed for position {target_pos}")
            
        return success, result_trajectory

    def plan_next_trajectory(self):
        """Plan the trajectory for the next dataset step, respecting episode boundaries."""
        # Check if we've reached the end of the current episode
        if self.current_step >= self.episode_end_step:
            self.get_logger().info(f"Episode {self.current_episode} complete at step {self.current_step}")
            
            # Reset to neutral position at end of episode
            self.last_joint_angles = self.neutral_joint_angles.copy()
            
            # Move to next episode
            self.current_episode += 1
            if self.current_episode >= self.num_episodes:
                if self.loop_enabled:
                    self.get_logger().info("All episodes complete, looping back to first episode...")
                    self.current_episode = 0
                    self.current_step = 0
                    self.episode_start_step = 0
                    self.episode_end_step = self.episode_ends[0]
                else:
                    self.get_logger().info("All episodes published successfully")
                    return False
            else:
                # Update episode boundaries
                self.episode_start_step = self.episode_ends[self.current_episode - 1]
                self.episode_end_step = self.episode_ends[self.current_episode]
                self.current_step = self.episode_start_step
                self.get_logger().info(
                    f"Starting episode {self.current_episode} "
                    f"(steps {self.episode_start_step} to {self.episode_end_step})"
                )

        try:
            # Extract pose data from dataset
            target_pos = np.array(self.data['robot0_eef_pos'][self.current_step])
            target_rot_axis_angle = np.array(self.data['robot0_eef_rot_axis_angle'][self.current_step])
            gripper_width = self.data['robot0_gripper_width'][self.current_step][0]

            self.get_logger().info(
                f"Episode {self.current_episode}, Step {self.current_step} "
                f"({self.current_step - self.episode_start_step + 1}/"
                f"{self.episode_end_step - self.episode_start_step}): "
                f"Dataset position: {target_pos}"
            )

            # Convert axis-angle to quaternion, scipy format [x, y, z, w]
            target_quat = transform_rotation(
                target_rot_axis_angle, from_rep='axis_angle', to_rep='quaternion'
            )
            
            success, trajectories = self.plan_motion(self.last_joint_angles, target_pos, target_quat)
            if not success:
                self.get_logger().warn(f"Skipping step {self.current_step} due to motion planning failure")
                self.current_step += 1
                return True
            
            # Update last joint angles to the final waypoint of the trajectory
            self.last_joint_angles = trajectories[-1]
            self.current_step += 1
            self.current_trajectory = trajectories
            self.current_trajectory_idx = 0
            return True
        except Exception as e:
            self.get_logger().error(f"Error planning step {self.current_step}: {e}")
            return False

    def publish_trajectory_waypoint(self):
        """Publish the next waypoint in the current trajectory."""
        if self.current_trajectory_idx >= len(self.current_trajectory):
            return False
        
        waypoint = self.current_trajectory[self.current_trajectory_idx]
        goal = RosJointState()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'panda_link0'
        goal.name = self.joint_names
        goal.position = waypoint[:7]
        goal.velocity = []
        goal.effort = []
        self.publisher_.publish(goal)
        
        self.get_logger().debug(
            f"Episode {self.current_episode}: Published waypoint "
            f"{self.current_trajectory_idx + 1}/{len(self.current_trajectory)} "
            f"for step {self.current_step - 1}"
        )
        
        self.current_trajectory_idx += 1
        return True

    def run_publishing_loop(self):
        """Publishing loop with incremental trajectory waypoint publishing."""
        try:
            publish_interval = 1.0 / self.publish_rate
            
            # Plan first trajectory
            if not self.plan_next_trajectory():
                self.get_logger().info("No data to publish")
                return

            previous_episode = self.current_episode
            
            while rclpy.ok():
                # Publish next waypoint in trajectory
                result = self.publish_trajectory_waypoint()
                if not result:
                    # Current trajectory finished, plan next one
                    if not self.plan_next_trajectory():
                        # No more data to publish
                        break
                    
                    # Check if we moved to a new episode
                    if self.current_episode != previous_episode:
                        self.get_logger().info(
                            f"Episode transition: {previous_episode} -> {self.current_episode}. "
                            f"Waiting {self.episode_delay}s..."
                        )
                        time.sleep(self.episode_delay)
                        previous_episode = self.current_episode
                        self.go_to_home()
                    
                    continue

                time.sleep(publish_interval)

        except Exception as e:
            self.get_logger().error(f"Error in publishing loop: {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Publish joint states from dataset.zarr.zip using inverse kinematics with cuRobo'
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
        default=100.0,
        help='Publishing rate in Hz (default: 60.0)'
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop through all episodes continuously'
    )
    parser.add_argument(
        '--episode_delay',
        type=float,
        default=2.0,
        help='Delay between episodes in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--tx_slam_tag_path',
        type=str,
        default=None,
        help='Path to tx_slam_tag.json calibration file from UMI data collection'
    )
    parser.add_argument(
        '--tag_to_robot_pos',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help='Position of ArUco tag relative to robot base [x, y, z] in meters (default: [0, 0, 0])'
    )
    parser.add_argument(
        '--tag_to_robot_quat',
        type=float,
        nargs=4,
        default=[0.0, 0.0, 0.0, 1.0],
        help='Orientation of ArUco tag relative to robot base as quaternion [x, y, z, w] (default: [0, 0, 0, 1])'
    )
    parser.add_argument(
        '--use_identity_transform',
        action='store_true',
        default=True,
        help='Use identity transform (no coordinate transformation, assumes tag frame = robot base frame)'
    )
    return parser.parse_args()


def main():
    """Main function to load dataset and publish joint states using cuRobo."""
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
        publisher_node = ReplayBufferPosePublisher(
            str(dataset_path), str(urdf_path), args.topic, args.publish_rate,
            tx_slam_tag_path=args.tx_slam_tag_path,
            tag_to_robot_pos=args.tag_to_robot_pos,
            tag_to_robot_quat=args.tag_to_robot_quat,
            use_identity_transform=args.use_identity_transform
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