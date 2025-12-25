"""
UMI Dataset Trajectory Replay via ROS2 with IKPy
=================================================

This script replays UMI-collected trajectories on a Franka Panda robot by:
1. Loading end-effector poses from the dataset (in ArUco tag frame)
2. Transforming them to robot base frame using FK-based calibration
3. Solving IK with IKPy
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
                        (since ee_link is configured as umi_tcp in IKPy chain)
        
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
import argparse
from scipy.spatial.transform import Rotation as R
import pickle

# IKPy Import
from ikpy.chain import Chain


class UmiPosePublisher(Node):
    def __init__(self, session_dir, episode_idx=0):
        super().__init__('umi_pose_publisher')
        
        self.publisher_ = self.create_publisher(JointState, '/joint_command', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 125Hz
        
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
        
        # 2. Setup IKPy
        self.setup_ikpy()
        
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

    def setup_ikpy(self):
        """Initialize IKPy robot chain."""
        self.get_logger().info('Initializing IKPy chain...')
        
        # URDF file path for Franka Panda with UMI gripper
        urdf_path = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf"
        
        # Active links mask for the kinematic chain:
        # [base_origin, joint1, joint2, joint3, joint4, joint5, joint6, joint7, umi_tcp_joint]
        # First and last are fixed/origin links, middle 7 are the active revolute joints
        self.active_links_mask = [False, True, True, True, True, True, True, True, False]
        
        # Create kinematic chain from URDF
        # Chain goes from panda_link0 to umi_tcp
        self.robot_chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["panda_link0"],
            active_links_mask=self.active_links_mask,
            name="panda_umi"
        )
        
        self.get_logger().info(f'IKPy chain initialized with {len(self.robot_chain.links)} links')
        
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
        Compute FK from panda_link0 to gopro_link using IKPy kinematics.
        
        Args:
            joint_states: numpy array of 7 joint angles
            
        Returns:
            T_base_gopro: 4x4 transformation matrix from base to gopro_link
        """
        # IKPy expects full joint list including inactive joints
        # Format: [0, joint1, joint2, joint3, joint4, joint5, joint6, joint7, 0]
        full_joints = np.zeros(len(self.robot_chain.links))
        full_joints[1:8] = joint_states  # Set active joints (indices 1-7)
        
        # Get FK to umi_tcp (the end of the chain)
        # forward_kinematics returns a 4x4 transformation matrix
        T_base_umi_tcp = self.robot_chain.forward_kinematics(full_joints)
        
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
        Solves IK for a target 4x4 matrix in Base Frame using IKPy.
        
        Args:
            target_pose_matrix: 4x4 transformation matrix for target pose
            
        Returns: 
            Joint angles as a list of 7 values (panda_joint1-7), or None if failed
        """
        curr_joint_states = self.get_current_joint_states()
        if curr_joint_states is None:
            self.get_logger().error("No current joint states available")
            return None

        # Build initial position for IK solver (full joint list including inactive joints)
        # Format: [0, joint1, joint2, joint3, joint4, joint5, joint6, joint7, 0]
        initial_position = np.zeros(len(self.robot_chain.links))
        initial_position[1:8] = curr_joint_states  # Set active joints

        # Solve IK using IKPy
        # inverse_kinematics_frame takes a 4x4 transformation matrix directly
        try:
            calculated_joints = self.robot_chain.inverse_kinematics_frame(
                target=target_pose_matrix,
                initial_position=initial_position
            )
        except Exception as e:
            self.get_logger().error(f"IK solver failed: {e}")
            return None

        # Extract active joints (indices 1-7 are the 7 Panda joints)
        # Convert to list - handles both numpy array and list returns from ikpy
        joint_angles = list(calculated_joints[1:8])
        
        self.last_joint_angles = joint_angles
        return joint_angles


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
            # 5. Publish single joint configuration (IKPy returns single config, not trajectory)
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
            
            # Combine arm joints and gripper
            joints_list = list(joint_angles) + [finger_pos, finger_pos]
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
