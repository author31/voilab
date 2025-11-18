#!/usr/bin/env python3

"""
Simplified script to publish pose information from dataset.zarr.zip via ROS2.
Uses inverse kinematics with ikpy to convert end-effector poses to realistic joint angles.

Dataset structure:
     ├── camera0_rgb (T, H, W, 3) uint8
     ├── robot0_demo_end_pose (T, 6)
     ├── robot0_demo_start_pose (T, 6)
     ├── robot0_eef_pos (T, 3)  # POSITIONS IN ARUCO TAG COORDINATE FRAME
     ├── robot0_eef_rot_axis_angle (T, 3)  # ORIENTATIONS IN ARUCO TAG COORDINATE FRAME
     └── robot0_gripper_width (T, 1)

Coordinate Systems:
    - Dataset poses are in ArUco tag coordinate frame (from dataset_planning.py)
    - IK solver expects poses in robot base coordinate frame (panda_link0)
    - This script handles the coordinate transformation automatically

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
import numpy as np
from pathlib import Path
import zarr
import zipfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState as RosJointState

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.geom.types import WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


class ReplayBufferPosePublisher(Node):
    """ROS2 Node to publish joint states from replay buffer using cuRobo's IKSolver."""

    def __init__(self, dataset_path, urdf_path, topic, publish_rate, coord_transform_path=None, start_episode=0):
        super().__init__('replay_buffer_pose_publisher')

        self.tensor_args = TensorDeviceType()
        self.dataset_path = dataset_path
        self.urdf_path = urdf_path
        self.topic = topic
        self.publish_rate = publish_rate
        self.coord_transform_path = coord_transform_path
        self.start_episode = start_episode

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(RosJointState, topic, 10)

        # Load coordinate transformation if provided
        self.coord_transform = self.get_transform_from_base_to_tag(y_offset=0.5)

        # Load dataset and robot model
        self.data, self.meta, self.episode_ends = self.load_dataset(dataset_path)
        # Get joint names from the robot model - 7 arm joints
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]

        self.get_logger().info(f'Active joints ({len(self.joint_names)}): {self.joint_names}')

        # Episode-based playback state
        self.total_episodes = len(self.episode_ends)
        self.loop_enabled = False
        self.episode_delay = 2.0

        # Validate start episode
        if self.start_episode < 0 or self.start_episode >= self.total_episodes:
            raise ValueError(f"Invalid start episode {self.start_episode}. Must be between 0 and {self.total_episodes - 1}")

        self.current_episode = self.start_episode
        self.current_episode_step = 0

        # Calculate episode boundaries
        self.episode_starts = [0] + self.episode_ends[:-1].tolist()
        self.episode_lengths = [end - start for start, end in zip(self.episode_starts, self.episode_ends)]

        # IK solver
        # self.ik_solver = self.init_ik_solver()

        # MotionGen
        self.init_motion_gen()
        self.last_joint_angles = None

    def get_transform_from_base_to_tag(self, y_offset=0.5):
        """Get the transformation matrix from robot base frame to ArUco tag frame.
        Assume the ArUco tag is 10 units along the y-axis from the robot base frame.
        Args:
            y_offset: The offset along the y-axis in the robot base frame.
        Returns:
            The transformation matrix from robot base frame to ArUco tag frame.
        """
        H_base_aruco = np.eye(4)
        H_base_aruco[1, 3] = y_offset # move 10 units along y-axis
        return H_base_aruco

    def init_ik_solver(self):
        """Initialize a cuRobo IKSolver for the Franka Panda robot."""
        self.get_logger().info('Initializing IK solver')

        config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        kinematics_cfg = config_file["robot_cfg"]["kinematics"]

        robot_cfg = RobotConfig.from_basic(
            kinematics_cfg["urdf_path"],
            kinematics_cfg["base_link"],
            kinematics_cfg["ee_link"],
            self.tensor_args,
        )

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )

        ik_solver = IKSolver(ik_config)
        return ik_solver

    def init_motion_gen(self):
        """Initialize a cuRobo MotionGen for the Franka Panda robot."""
        self.get_logger().info('Initializing MotionGen')

        world_model = WorldConfig.from_dict(
            {
                "blox": {
                    "world": {
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                        "integrator_type": "occupancy",
                        "voxel_size": 0.005,
                    }
                }
            }
        )
        robot_path = "franka.yml"
        trajopt_tsteps = 16

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_path,
            None,
            self.tensor_args,
            trajopt_tsteps=trajopt_tsteps,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=True,
            num_trajopt_seeds=2,
            num_graph_seeds=2,
            evaluate_interpolated_trajectory=True,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()

    def get_neutral_joint_angles(self):
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

                root = zarr.open(store=zarr.DirectoryStore(extract_path), mode='r')

                # Load both data and meta groups
                data = root['data']
                meta = root['meta']

                # Extract episode ends information
                assert 'episode_ends' in meta
                episode_ends = meta.episode_ends

                self.get_logger().info(f"Dataset loaded successfully")
                self.get_logger().info(f"Found {len(episode_ends)} episodes")
                return data, meta, episode_ends
                
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            raise

    def transform_pose_to_robot_frame(self, pos, rot_axis_angle):
        """
        Transform pose from ArUco tag frame to robot base frame.

        Args:
            pos: Position in ArUco tag frame [x, y, z]
            rot_axis_angle: Rotation in ArUco tag frame [rx, ry, rz] (axis-angle)

        Returns:
            tuple: (pos_robot, rot_robot) in robot base frame
        """
        try:
            # Convert axis-angle to rotation matrix
            rot = Rotation.from_rotvec(rot_axis_angle)
            rot_matrix = rot.as_matrix()

            # Create 4x4 transformation matrix for the pose
            pose_tag = np.zeros((4, 4))
            pose_tag[:3, :3] = rot_matrix
            pose_tag[:3, 3] = pos
            pose_tag[3, 3] = 1.0

            # Apply coordinate transformation: pose_robot = tx_robot_tag @ pose_tag
            pose_robot = self.coord_transform @ pose_tag

            # Extract position and rotation from transformed pose
            pos_robot = pose_robot[:3, 3]
            rot_matrix_robot = pose_robot[:3, :3]

            # Convert rotation matrix back to axis-angle
            rot_robot = Rotation.from_matrix(rot_matrix_robot).as_rotvec()

            return pos_robot, rot_robot

        except Exception as e:
            self.get_logger().error(f"Failed to transform pose: {e}")
            return pos, rot_axis_angle  # Return original pose if transformation fails

    def solve_ik(self, target_pos, target_rot):
        """
        Solve inverse kinematics for target pose using cuRobo's IKSolver.

        Args:
            target_pos: Target position [x, y, z] (will be transformed to robot frame if needed)
            target_rot: Target rotation as axis-angle [rx, ry, rz] (will be transformed to robot frame if needed)

        Returns:
            list of joint angles for active joints, or None if IK fails
        """
        try:
            transformed_pos, transformed_rot = self.transform_pose_to_robot_frame(target_pos, target_rot)

            quat = Rotation.from_rotvec(transformed_rot).as_quat()

            goal = Pose(
                position=self.tensor_args.to_device(
                    torch.tensor(transformed_pos, dtype=torch.float32).unsqueeze(0)
                ),
                quaternion=self.tensor_args.to_device(
                    torch.tensor(quat, dtype=torch.float32).unsqueeze(0)
                ),
            )

            result = self.ik_solver.solve_batch(goal)
            success = bool(result.success.view(-1)[0].item())
            if not success:
                return None

            joint_tensor = result.solution[0].squeeze(0)
            joint_angles = joint_tensor.detach().cpu().numpy().tolist()
            return joint_angles

        except Exception as e:
            self.get_logger().warn(f"IK solver failed: {e}")
            self.get_logger().debug(f"Original target position (ArUco frame): {target_pos}")
            self.get_logger().debug(f"Original target rotation (ArUco frame): {target_rot}")
            return None
    
    def solve_ik_with_motion_gen(self, target_pos, target_rot):
        """
        Solve inverse kinematics for target pose using cuRobo's MotionGen.
        """
        try:
            transformed_pos, transformed_rot = self.transform_pose_to_robot_frame(target_pos, target_rot)
            quat = Rotation.from_rotvec(transformed_rot).as_quat()
            curobo_quat = [quat[3], quat[0], quat[1], quat[2]]
            goal = Pose(
                position=self.tensor_args.to_device(
                    torch.tensor(transformed_pos, dtype=torch.float32).unsqueeze(0)
                ),
                quaternion=self.tensor_args.to_device(
                    torch.tensor(curobo_quat, dtype=torch.float32).unsqueeze(0)
                ),
            )

            initial_position = self.last_joint_angles if self.last_joint_angles is not None else self.get_neutral_joint_angles()

            start_joint_state = CuroboJointState.from_position(
                position=self.tensor_args.to_device(initial_position).unsqueeze(0),
                joint_names=[
                    "panda_link1", "panda_link2", "panda_link3", "panda_link4",
                    "panda_link5", "panda_link6", "panda_link7"
                ]
            )

            result = self.motion_gen.plan_single(
                start_joint_state, 
                goal, 
                MotionGenPlanConfig(max_attempts=3, enable_finetune_trajopt=True)
            )
            success = bool(result.success.view(-1)[0].item())
            if not success:
                return None

            trajectories = result.get_interpolated_plan().position.tolist()
            return trajectories

        except Exception as e:
            self.get_logger().warn(f"IK solver failed: {e}")
            self.get_logger().debug(f"Original target position (ArUco frame): {target_pos}")
            self.get_logger().debug(f"Original target rotation (ArUco frame): {target_rot}")
            return None

    def publish_episode_step(self):
        """Publish a single timestep within the current episode."""
        episode_start = self.episode_starts[self.current_episode]
        episode_end = self.episode_ends[self.current_episode]
        global_step = episode_start + self.current_episode_step

        try:
            # Extract pose data from dataset (in ArUco tag coordinate frame)
            target_pos = self.data['robot0_eef_pos'][global_step]
            target_rot = self.data['robot0_eef_rot_axis_angle'][global_step]
            gripper_width = self.data['robot0_gripper_width'][global_step][0]

            # Solve IK to get joint angles (handles coordinate transformation internally)
            trajectories = self.solve_ik_with_motion_gen(target_pos, target_rot)

            if trajectories is None:
                self.get_logger().warn(f"Skipping episode {self.current_episode} step {self.current_episode_step} due to IK failure")
                self.current_episode_step += 1
                return True            

            for trajectory in trajectories:
                msg = RosJointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "panda_link0"
                msg.name = self.joint_names
                msg.position = [float(angle) for angle in trajectory]
                msg.velocity = []
                msg.effort = []
                self.publisher_.publish(msg)
                time.sleep(0.01)
                self.last_joint_angles = trajectory

            # Log progress occasionally
            if self.current_episode_step % 50 == 0:
                self.get_logger().info(f"Episode {self.current_episode + 1}/{self.total_episodes}, step {self.current_episode_step + 1}/{self.episode_lengths[self.current_episode]}")

        except Exception as e:
            self.get_logger().error(f"Error publishing episode {self.current_episode} step {self.current_episode_step}: {e}")

        # Move to next step within episode
        self.current_episode_step += 1

        # Check if episode is complete
        if self.current_episode_step >= self.episode_lengths[self.current_episode]:
            self.get_logger().info(f"Episode {self.current_episode + 1} completed successfully")
            return False

        return True

    def run_publishing_loop(self):
        """Simple publishing loop without threading."""
        try:
            publish_interval = 1.0 / self.publish_rate

            while rclpy.ok():
                # Track if we just completed an episode (before calling publish_episode_step)
                episode_starting = self.current_episode_step == 0 and self.current_episode > 0

                if not self.publish_episode_step():
                    if self.loop_enabled:
                        time.sleep(self.episode_delay)
                        continue
                    else:
                        break

                # Add delay between episodes (when we just started a new episode)
                if episode_starting:
                    time.sleep(self.episode_delay)

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
    parser.add_argument(
        '--episode',
        type=int,
        default=0,
        help='Episode number to start replay from (0-indexed, default: 0)'
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
        publisher_node = ReplayBufferPosePublisher(
            str(dataset_path), str(urdf_path), args.topic, args.publish_rate,
            start_episode=args.episode
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
