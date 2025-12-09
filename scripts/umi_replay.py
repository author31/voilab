import os
import zipfile
import pickle
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

class UMIReplay:
    """
    ROS2-free trajectory replay for UMI datasets using Isaac Sim Lula Kinematics.
    """
    
    # Calibration pose joint angles (radians)
    CALIBRATION_JOINTS = np.array([
        0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741
    ])
    
    def __init__(
        self,
        panda_articulation,
        session_dir: str,
        episode_idx: int = 0,
        lula_solver=None,
        art_kine_solver=None,
        tag_frame_idx: int = 0,
        tag_id: int = 13
    ):
        self.panda = panda_articulation
        self.session_dir = session_dir
        self.episode_idx = episode_idx
        self.lula_solver = lula_solver
        self.art_kine_solver = art_kine_solver
        self.tag_frame_idx = tag_frame_idx
        self.tag_id = tag_id
        
        # Fixed transforms from URDF (panda_link7 to child frames)
        self.T_link7_gopro = np.eye(4)
        self.T_link7_gopro[:3, 3] = [0, 0, 0.107]
        
        self.T_link7_umi_tcp = np.eye(4)
        self.T_link7_umi_tcp[:3, 3] = [0, 0.086, 0.327]
        
        # T_umi_tcp_gopro = inv(T_link7_umi_tcp) @ T_link7_gopro
        self.T_umi_tcp_gopro = np.linalg.inv(self.T_link7_umi_tcp) @ self.T_link7_gopro
        
        # Load dataset
        self.data, self.meta, self.episode_ends = self._load_dataset()
        self.start_idx, self.end_idx = self._get_episode_bounds()
        self.current_step = self.start_idx
        
        # Load tag calibration pose
        tvec, rvec = self._load_tag_pose()
        self.T_gopro_tag = self._pose_to_matrix(tvec, R.from_rotvec(rvec).as_matrix())
        
        # Calibration transform
        self.T_base_tag = None
        self.calibrated = False
        
        print(f"[UMIReplay] Initialized. Robot must be in calibration pose.")

    def _load_dataset(self):
        zarr_zip_path = os.path.join(self.session_dir, 'dataset.zarr.zip')
        if not os.path.exists(zarr_zip_path):
            raise FileNotFoundError(f"Zarr zip file not found at '{zarr_zip_path}'")
        
        extract_path = os.path.splitext(zarr_zip_path)[0]
        if not os.path.exists(extract_path):
            print(f"[UMIReplay] Extracting dataset to {extract_path}...")
            with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        
        root = zarr.open(store=zarr.DirectoryStore(extract_path), mode='r')
        data = root['data']
        meta = root['meta']
        episode_ends = meta['episode_ends'][:]
        return data, meta, episode_ends

    def _get_episode_bounds(self):
        if self.episode_idx == 0:
            start_idx = 0
            end_idx = self.episode_ends[0]
        else:
            if self.episode_idx >= len(self.episode_ends):
                raise ValueError(f"Episode {self.episode_idx} invalid.")
            start_idx = self.episode_ends[self.episode_idx - 1]
            end_idx = self.episode_ends[self.episode_idx]
        return int(start_idx), int(end_idx)

    def _load_tag_pose(self):
        tag_poses_path = os.path.join(self.session_dir, 'demos', 'mapping', 'tag_detection.pkl')
        if not os.path.exists(tag_poses_path):
            raise FileNotFoundError(f"Tag poses file not found at '{tag_poses_path}'")
        
        with open(tag_poses_path, 'rb') as f:
            tag_poses = pickle.load(f)
        
        target_tag = tag_poses[self.tag_frame_idx]['tag_dict'][self.tag_id]
        return target_tag["tvec"], target_tag["rvec"]

    def _pose_to_matrix(self, pos, rot):
        T = np.eye(4)
        T[:3, 3] = np.asarray(pos).flatten()
        rot = np.asarray(rot)
        if rot.shape == (3,):
            if np.allclose(rot, 0):
                T[:3, :3] = np.eye(3)
            else:
                T[:3, :3] = R.from_rotvec(rot).as_matrix()
        else:
            T[:3, :3] = rot
        return T

    def _compute_fk_to_gopro(self):
        """Compute FK from panda_link0 to gopro_link using Lula."""
        # Note: art_kine_solver calculates pose in World Frame if lula base pose is set,
        # or Robot Frame if not. We handle base pose in calibrate().
        
        ee_pose = self.art_kine_solver.compute_end_effector_pose()
        position = ee_pose[0]
        rot_matrix = ee_pose[1]
        quat_wxyz = R.from_matrix(rot_matrix).as_quat()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T_base_umi_tcp = np.eye(4)
        T_base_umi_tcp[:3, 3] = position
        T_base_umi_tcp[:3, :3] = R.from_quat(quat_xyzw).as_matrix()

        T_base_gopro = T_base_umi_tcp @ self.T_umi_tcp_gopro
        return T_base_gopro

    def calibrate(self):
        """
        Compute T_base_tag calibration transform.
        CRITICAL: Updates Lula with current robot base pose before FK.
        """
        if self.calibrated:
            return
        
        print("[UMIReplay] Computing calibration transform (T_base_tag)...")
        
        # 1. Update Lula with the current World Pose of the robot base
        # This ensures the FK computation includes the robot's placement in the scene
        robot_pos, robot_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos,
            robot_orientation=robot_quat
        )
        
        # 2. Compute T_base_gopro via FK
        T_base_gopro = self._compute_fk_to_gopro()
        
        # 3. Chain: T_base_tag = T_base_gopro @ T_gopro_tag
        self.T_base_tag = T_base_gopro @ self.T_gopro_tag
        
        self.calibrated = True
        print(f"[UMIReplay] Calibration complete.")
        print(f"            Robot Base: {robot_pos}")
        print(f"            Tag Transform established.")

    def update(self):
        if not self.calibrated:
            self.calibrate()
            return True
        
        if self.current_step >= self.end_idx:
            print(f"[UMIReplay] Episode {self.episode_idx} finished.")
            return False
        
        pos_in_tag = self.data['robot0_eef_pos'][self.current_step]
        rot_in_tag = self.data['robot0_eef_rot_axis_angle'][self.current_step]
        gripper_width = float(self.data['robot0_gripper_width'][self.current_step][0])
        
        T_tag_eef = self._pose_to_matrix(pos_in_tag, rot_in_tag)
        T_base_eef = self.T_base_tag @ T_tag_eef
        
        target_pos = T_base_eef[:3, 3]
        target_rot = R.from_matrix(T_base_eef[:3, :3])
        target_quat_xyzw = target_rot.as_quat()
        target_quat_wxyz = np.array([
            target_quat_xyzw[3], target_quat_xyzw[0], 
            target_quat_xyzw[1], target_quat_xyzw[2]
        ])
        
        # Update robot base pose for IK (Redundant if robot doesn't move, but safe)
        robot_pos, robot_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos,
            robot_orientation=robot_quat
        )
        
        action, success = self.art_kine_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat_wxyz
        )
        if success:
            print(f"[UMIReplay] IK solution: {action}")
            self.panda.set_joint_positions(np.array([*action.joint_positions, 0.05, 0.05]))
            self._set_gripper(gripper_width)
        else:
            print(f"[UMIReplay] IK failed at step {self.current_step}")
        
        if self.current_step % 50 == 0:
            progress = (self.current_step - self.start_idx) / (self.end_idx - self.start_idx) * 100
            print(f"[UMIReplay] Step {self.current_step}/{self.end_idx} ({progress:.1f}%)")
        
        self.current_step += 1
        return True

    def _set_gripper(self, width: float):
        finger_pos = np.clip(width / 2.0, 0.0, 0.04)
        idx1 = self.panda.get_dof_index("panda_finger_joint1")
        idx2 = self.panda.get_dof_index("panda_finger_joint2")
        
        if idx1 is not None and idx2 is not None:
            self.panda.set_joint_positions(
                positions=np.array([finger_pos, finger_pos]),
                joint_indices=np.array([idx1, idx2])
            )

    def reset(self, episode_idx: int = None):
        if episode_idx is not None:
            self.episode_idx = episode_idx
            self.start_idx, self.end_idx = self._get_episode_bounds()
        self.current_step = self.start_idx
        self.calibrated = False
        self.T_base_tag = None