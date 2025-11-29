"""
UMI Dataset Trajectory Replay via ROS2 with CuRobo IK
======================================================

This script replays UMI-collected trajectories on a Franka Panda robot by:
1. Loading end-effector poses from the dataset (in ArUco tag frame)
2. Transforming them to robot base frame using FK-based calibration
3. Solving IK with CuRobo MotionGen
4. Publishing joint commands to /joint_command

Dataset Structure (from UMI):
    dataset.zarr.zip/
     ├── camera0_rgb (T, H, W, 3) uint8
     ├── robot0_demo_end_pose (T, 6)
     ├── robot0_demo_start_pose (T, 6)
     ├── robot0_eef_pos (T, 3)           # TCP position in ArUco tag frame
     ├── robot0_eef_rot_axis_angle (T, 3) # TCP orientation (axis-angle) in ArUco tag frame
     └── robot0_gripper_width (T, 1)

Coordinate Transformation Chain:
    The dataset poses (T_tag_eef) are in ArUco tag frame. To replay on the robot,
    we need to transform them to robot base frame (panda_link0).

    Final transform: T_base_eef = T_base_tag @ T_tag_eef

    Where T_base_tag is computed as:
        T_base_tag = T_base_gopro @ T_gopro_tag

    Components:
        - T_base_gopro: FK from panda_link0 to gopro_link at calibration joint config
                        Computed via: T_base_umi_tcp @ T_umi_tcp_gopro
                        (since ee_link is configured as umi_tcp in CuRobo)
        
        - T_gopro_tag:  ArUco tag pose in GoPro camera frame
                        Loaded from: demos/mapping/tag_detection.pkl
                        Contains tvec (position) and rvec (axis-angle rotation)
        
        - T_tag_eef:    End-effector (TCP) pose from dataset
                        robot0_eef_pos + robot0_eef_rot_axis_angle

    URDF Fixed Transforms (from panda_link7):
        - panda_link7 -> gopro_link:  xyz=[0, 0, 0.107]
        - panda_link7 -> umi_tcp:     xyz=[0, 0.086, 0.327]

Usage:
    IMPORTANT: Robot must be in CALIBRATION POSE when starting!
    The calibration pose is the same pose used when the ArUco tag was detected.

    python umi_ros2_ik_publisher.py --session_dir /path/to/session --episode 0

Published to /joint_command (sensor_msgs/JointState):
    - header: timestamp and frame_id (panda_link0)
    - name: panda_joint1-7 + panda_finger_joint1-2
    - position: joint angles from IK + gripper positions
    - velocity: empty list
    - effort: empty list
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import zarr
import zipfile
import os
import torch
import argparse
from scipy.spatial.transform import Rotation as R
import pickle

# CuRobo Imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.robot import JointState as CuroboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


class UmiPosePublisher(Node):
    def __init__(self, session_dir, episode_idx=0):
        super().__init__('umi_pose_publisher')
        
        self.publisher_ = self.create_publisher(JointState, '/joint_command', 10)
        self.timer = self.create_timer(0.008, self.timer_callback)  # 125Hz
        
        # Subscribe to current joint states
        self.current_joint_states = None
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        
        # 1. Load Data
        self.session_dir = session_dir
        self.data, self.meta, self.episode_ends = self.load_dataset(self.session_dir)
        self.episode_idx = episode_idx
        self.setup_episode_indices()
        
        # 2. Setup CuRobo
        self.setup_curobo()
        
        # 3. Calibration will be computed when first joint states are received
        # The robot must be in the calibration pose when starting!
        self.T_base_tag = None
        self.calibration_complete = False
        
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        
        self.current_step = self.start_idx
        self.last_joint_angles = None
        self.get_logger().info("Initialization Complete. Waiting for joint states to compute calibration...")
        self.get_logger().warn("IMPORTANT: Robot must be in CALIBRATION POSE when starting!")

    def setup_episode_indices(self):
        """Calculates start and end indices for the requested episode."""
        if self.episode_idx == 0:
            self.start_idx = 0
            self.end_idx = self.episode_ends[0]
        else:
            self.start_idx = self.episode_ends[self.episode_idx - 1]
            self.end_idx = self.episode_ends[self.episode_idx]
            
        self.get_logger().info(f"Replaying Episode {self.episode_idx}: Steps {self.start_idx} to {self.end_idx}")

    def load_tag_poses(self, session_dir: str, frame_idx: int = 0, tag_id: int = 13):
        """Loads the tag poses from the session directory."""
        tag_poses_path = os.path.join(session_dir, 'demos', 'mapping', 'tag_detection.pkl')

        if not os.path.exists(tag_poses_path):
            self.get_logger().error(f"Tag poses file not found at '{tag_poses_path}'")
            raise FileNotFoundError(f"Tag poses file not found at '{tag_poses_path}'")

        self.get_logger().info(f"Loading tag poses from '{tag_poses_path}' for frame {frame_idx} and tag {tag_id}...")

        with open(tag_poses_path, 'rb') as f:
            tag_poses = pickle.load(f)
        
        target_tag = tag_poses[frame_idx]['tag_dict'][tag_id]
        assert "rvec" in target_tag and "tvec" in target_tag, "Tag poses must contain rvec and tvec"
        return target_tag["tvec"], target_tag["rvec"]

    def load_dataset(self, session_dir):
        """Extracts and loads the Zarr dataset from a zip file."""
        try:
            self.get_logger().info(f"Loading dataset from '{session_dir}'...")
            zarr_zip_path = os.path.join(session_dir, 'dataset.zarr.zip')
            if not os.path.exists(zarr_zip_path):
                self.get_logger().error(f"Zarr zip file not found at '{zarr_zip_path}'")
                raise FileNotFoundError(f"Zarr zip file not found at '{zarr_zip_path}'")

            extract_path = os.path.splitext(zarr_zip_path)[0]
            if not os.path.exists(extract_path):
                with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

            root = zarr.open(store=zarr.DirectoryStore(extract_path), mode='r')
            data = root['data']
            meta = root['meta']
            episode_ends = meta['episode_ends'][:] # Load into memory

            self.get_logger().info(f"Dataset loaded. Found {len(episode_ends)} episodes.")
            return data, meta, episode_ends

        except Exception as e:
            self.get_logger().error(f"Failed to load dataset: {e}")
            raise

    def setup_curobo(self):
        """Initialize CuRobo MotionGen."""
        self.get_logger().info('Initializing MotionGen...')
        
        # Use standard Franka config provided by CuRobo
        robot_file = "/workspace/voilab/assets/curobo/franka_umi.yaml"
        # robot_file = "franka.yml"
        
        self.tensor_args = TensorDeviceType()
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            None,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=True,
            num_trajopt_seeds=12, 
            num_graph_seeds=12,
            interpolation_dt=0.01,
            evaluate_interpolated_trajectory=True,
        )
        
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        
        # Store kinematics for FK computation
        self.kinematics = self.motion_gen.kinematics
        
        # Precompute fixed transforms from URDF
        # panda_link7 -> gopro_link: xyz="0 0 0.107" rpy="0 0 0"
        # panda_link7 -> umi_tcp: xyz="0 0.086 0.327" rpy="0 0 0"
        self.T_link7_gopro = np.eye(4)
        self.T_link7_gopro[:3, 3] = [0, 0, 0.107]
        
        self.T_link7_umi_tcp = np.eye(4)
        self.T_link7_umi_tcp[:3, 3] = [0, 0.086, 0.327]
        
        # T_umi_tcp_gopro = inv(T_link7_umi_tcp) @ T_link7_gopro
        self.T_umi_tcp_gopro = np.linalg.inv(self.T_link7_umi_tcp) @ self.T_link7_gopro

    def get_matrix_from_pose(self, pos, rot_vec):
        """Convert position and rotation vector (axis-angle) to 4x4 Matrix."""
        T = np.eye(4)
        T[:3, 3] = pos
        # Handle zero rotation case
        if np.allclose(rot_vec, 0):
            T[:3, :3] = np.eye(3)
        else:
            T[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
        return T

    def compute_fk_to_gopro(self, joint_states):
        """
        Compute FK from panda_link0 to gopro_link using CuRobo kinematics.
        
        Args:
            joint_states: numpy array of 7 joint angles
            
        Returns:
            T_base_gopro: 4x4 transformation matrix from base to gopro_link
        """
        # Convert joint states to tensor
        q = torch.tensor([joint_states.tolist()], dtype=torch.float32, device=self.tensor_args.device)
        
        # Get FK to umi_tcp (the configured ee_link)
        state = self.kinematics.get_state(q)
        
        # Extract position and quaternion (CuRobo format: w, x, y, z)
        ee_pos = state.ee_pose.position.cpu().squeeze().numpy()
        ee_quat = state.ee_pose.quaternion.cpu().squeeze().numpy()  # [w, x, y, z]
        
        # Convert to scipy quaternion format (x, y, z, w)
        quat_scipy = [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]
        rot_mat = R.from_quat(quat_scipy).as_matrix()
        
        # Build T_base_umi_tcp
        T_base_umi_tcp = np.eye(4)
        T_base_umi_tcp[:3, :3] = rot_mat
        T_base_umi_tcp[:3, 3] = ee_pos
        
        # Compute T_base_gopro = T_base_umi_tcp @ T_umi_tcp_gopro
        T_base_gopro = T_base_umi_tcp @ self.T_umi_tcp_gopro
        
        return T_base_gopro

    def joint_states_callback(self, msg):
        """Callback to store current joint states from /joint_states topic."""
        joint_order = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                       'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        
        # Extract joint positions in the correct order
        positions = []
        for joint_name in joint_order:
            if joint_name not in msg.name:
                self.get_logger().error(f"Joint {joint_name} not found in /joint_states message")
                return
            idx = msg.name.index(joint_name)
            positions.append(msg.position[idx])
        
        self.current_joint_states = np.array(positions, dtype=np.float32)

    def get_current_joint_states(self):
        """Get current joint states from /joint_states topic, or fallback to neutral."""
        if self.current_joint_states is None:
            self.get_logger().error("No joint states received yet")
            return None

        return self.current_joint_states

    def compute_calibration_transform(self, joint_states, tag_pos_in_cam, tag_rot_in_cam):
        """
        Computes T_base_tag - the transformation from ArUco tag frame to robot base frame.
        
        Chain: T_base_tag = T_base_gopro @ T_gopro_tag
        
        Where:
        - T_base_gopro: FK from panda_link0 to gopro_link at the given joint configuration
        - T_gopro_tag: Hardcoded ArUco tag pose in GoPro camera frame (from calibration)
        
        Args:
            joint_states: numpy array of 7 joint angles (calibration pose)
            
        Returns:
            T_base_tag: 4x4 transformation matrix from ArUco tag to robot base
        """
        self.get_logger().info("Computing calibration transform (T_base_tag) from FK...")
        
        # 1. Compute T_base_gopro via FK at the current (calibration) joint configuration
        T_base_gopro = self.compute_fk_to_gopro(joint_states)
        
        # 2. Hardcoded T_gopro_tag (ArUco tag pose in GoPro camera frame)
        # These values are from the calibration measurement
        T_gopro_tag = self.get_matrix_from_pose(tag_pos_in_cam, tag_rot_in_cam)
        
        # 3. Chain: T_base_tag = T_base_gopro @ T_gopro_tag
        T_base_tag = T_base_gopro @ T_gopro_tag
        
        self.get_logger().info(f"T_base_tag computed from joint states: {joint_states.tolist()}")
        
        return T_base_tag

    def solve_ik(self, target_pose_matrix):
        """
        Solves IK for a target 4x4 matrix in Base Frame.
        Returns: Joint angles (numpy array)
        """
        # Convert 4x4 Matrix to CuRobo Pose (Pos + Quaternion)
        pos = target_pose_matrix[:3, 3].tolist()
        rot_mat = target_pose_matrix[:3, :3].tolist()
        
        # Scipy to quat (x,y,z,w)
        quat_scipy = R.from_matrix(rot_mat).as_quat() 
        # To CuRobo (w,x,y,z)
        quat_curobo = torch.tensor([[quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]], 
                                  dtype=torch.float32, device=self.tensor_args.device)
        pos_curobo = torch.tensor([pos], dtype=torch.float32, device=self.tensor_args.device)
        pose = Pose(position=pos_curobo, quaternion=quat_curobo)
        
        curr_joint_states = self.get_current_joint_states()
        if curr_joint_states is None:
            self.get_logger().error("No current joint states available")
            return None

        start_joint_state = CuroboJointState.from_position(
            position=self.tensor_args.to_device(curr_joint_states).unsqueeze(0),
            joint_names=self.joint_names
        )

        # Solve IK
        result = self.motion_gen.plan_single(
            start_joint_state,
            pose,
            MotionGenPlanConfig(max_attempts=3, enable_finetune_trajopt=True)
        )

        success = bool(result.success.view(-1)[0].item())
        if not success:
            self.get_logger().error(f"MotionGen plan failed: {result}")
            return None

        last_joint_angle = result.optimized_plan.position.tolist()
        # last_joint_angle = result.interpolated_plan.position.cpu().numpy().tolist()
        self.last_joint_angles = last_joint_angle
        return last_joint_angle


    def timer_callback(self):
        # Wait for joint states to be available
        if self.current_joint_states is None:
            self.get_logger().info("Waiting for joint states...")
            return
        
        # Compute calibration on first joint states received
        # The robot must be in calibration pose at this moment!
        if not self.calibration_complete:
            self.get_logger().info("First joint states received. Computing calibration...")
            tag_pos_in_cam, tag_rot_in_cam = self.load_tag_poses(self.session_dir, 0, 13)
            self.T_base_tag = self.compute_calibration_transform(self.current_joint_states, tag_pos_in_cam, tag_rot_in_cam)
            self.calibration_complete = True
            self.get_logger().info("Calibration complete. Starting trajectory replay...")
            return

        if self.current_step == self.end_idx:
            self.get_logger().info("Episode finished.")
            rclpy.shutdown()
            exit(0)

        # 1. Read Data (In ArUco Tag Frame)
        # UMI format: robot0_eef_pos is (T, 3), rot is (T, 3) axis-angle
        pos_in_tag = self.data['robot0_eef_pos'][self.current_step]
        rot_in_tag = self.data['robot0_eef_rot_axis_angle'][self.current_step]
        gripper_width = self.data['robot0_gripper_width'][self.current_step]

        # 2. Construct T_tag_eef
        T_tag_eef = self.get_matrix_from_pose(pos_in_tag, rot_in_tag)

        # 3. Transform to Robot Base Frame
        # T_base_eef = T_base_tag * T_tag_eef
        T_base_eef = self.T_base_tag @ T_tag_eef

        # 4. Solve IK
        joint_angles = self.solve_ik(T_base_eef)

        if joint_angles is not None:
            # 5. Publish
            for waypoint in joint_angles:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "panda_link0"
                
                # Map specific Franka joint names
                msg.name = [
                    'panda_joint1', 'panda_joint2', 'panda_joint3', 
                    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
                    'panda_finger_joint1', 'panda_finger_joint2'
                ]
                
                # Combine 7DOF arm + 2 Gripper fingers
                # Gripper mapping: dataset width is total width (0 to 0.08)
                # URDF mimic joints go 0 to 0.04 each. So divide by 2.
                finger_pos = max(0.0, min(0.04, gripper_width[0]))
                
                # Flatten array
                joints_list = waypoint + [finger_pos, finger_pos]
                msg.position = [float(x) for x in joints_list]
                self.publisher_.publish(msg)

        self.current_step += 1
        if self.current_step % 10 == 0:
            self.get_logger().info(f"Published step: {self.current_step}/{self.end_idx}")
            self.get_logger().info(f"Current joint states: {self.get_current_joint_states()}")
        
        

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Publish UMI Dataset via ROS2')
    parser.add_argument('--session_dir', type=str, required=True, help='Path to session directory')
    parser.add_argument('--episode', type=int, default=0, help='Episode index')
    args = parser.parse_args()

    node = UmiPosePublisher(args.session_dir, args.episode)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()