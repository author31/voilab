import os
import zipfile
import pickle
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from isaacsim.core.utils.types import ArticulationAction

class UMIReplay:
    """
    Trajectory replay for UMI datasets using Isaac Sim Lula Kinematics.
    """
    
    
    def __init__(
        self,
        panda_articulation,
        T_base_tag,
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
        
        # T_base_tag: ArUco tag pose in robot base frame.
        self.T_base_tag = T_base_tag
        
        # Load dataset
        self.data, self.meta, self.episode_ends = self._load_dataset()
        self.start_idx, self.end_idx = self._get_episode_bounds()
        self.current_step = self.start_idx
        
        self.calibrated = False
        
        # Waypoint storage: list of [target_pos, target_rot] pairs
        # target_pos: np.array of shape (3,) - x, y, z position
        # target_rot: scipy.spatial.transform.Rotation object
        self.waypoints = []
        
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

    def calibrate(self):
        """
        Compute T_base_tag calibration transform.
        CRITICAL: Updates Lula with current robot base pose before FK.
        """
        if self.calibrated:
            return
        
        robot_pos, robot_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(
            robot_position=robot_pos,
            robot_orientation=robot_quat
        )
        
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
        
        # Transform EEF pose from tag frame to base frame
        T_tag_eef = self._pose_to_matrix(pos_in_tag, rot_in_tag)
        T_base_eef = self.T_base_tag @ T_tag_eef
        
        target_pos = T_base_eef[:3, 3]
        target_rot = R.from_matrix(T_base_eef[:3, :3])
        
        # Store waypoint [target_pos, target_rot]
        self.waypoints.append([target_pos.copy(), target_rot])
        
        target_quat_xyzw = target_rot.as_quat()
        target_quat_wxyz = np.array([
            target_quat_xyzw[3], target_quat_xyzw[0], 
            target_quat_xyzw[1], target_quat_xyzw[2]
        ])
        
        # Update robot base pose for IK
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
        self.waypoints = []

    def visualize_waypoints(
        self,
        show_orientation: bool = True,
        orientation_scale: float = 0.02,
        marker_size: int = 10,
        title: str = None,
        save_path: str = None,
        figsize: tuple = (10, 8),
        dpi: int = 150
    ):
        """
        Visualize stored waypoints using Matplotlib 3D scatter plot.
        
        Args:
            show_orientation: If True, show orientation arrows at each waypoint
            orientation_scale: Length scale for orientation arrows
            marker_size: Size of waypoint markers
            title: Custom title for the plot
            save_path: If provided, save the figure as PNG to this path
            figsize: Figure size as (width, height) in inches
            dpi: Resolution for saved PNG
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not self.waypoints:
            print("[UMIReplay] No waypoints to visualize. Run update() first.")
            return None
        
        # Extract positions and rotations
        positions = np.array([wp[0] for wp in self.waypoints])
        rotations = [wp[1] for wp in self.waypoints]
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        num_waypoints = len(self.waypoints)
        step_indices = np.arange(num_waypoints)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory line
        ax.plot(x, y, z, color='gray', alpha=0.5, linewidth=1, label='Trajectory')
        
        # Scatter plot with color gradient
        scatter = ax.scatter(
            x, y, z,
            c=step_indices,
            cmap='viridis',
            s=marker_size,
            alpha=0.8
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Step')
        
        # Add orientation arrows if requested
        if show_orientation:
            # Sample waypoints for orientation visualization
            sample_interval = max(1, num_waypoints // 50)
            sampled_indices = list(range(0, num_waypoints, sample_interval))
            if (num_waypoints - 1) not in sampled_indices:
                sampled_indices.append(num_waypoints - 1)
            
            colors = ['red', 'green', 'blue']
            labels = ['X-axis', 'Y-axis', 'Z-axis']
            
            for axis_idx in range(3):
                for i, idx in enumerate(sampled_indices):
                    pos = positions[idx]
                    rot = rotations[idx]
                    
                    # Get the axis direction in world frame
                    local_axis = np.zeros(3)
                    local_axis[axis_idx] = 1.0
                    world_axis = rot.apply(local_axis) * orientation_scale
                    
                    ax.quiver(
                        pos[0], pos[1], pos[2],
                        world_axis[0], world_axis[1], world_axis[2],
                        color=colors[axis_idx],
                        alpha=0.6,
                        arrow_length_ratio=0.3,
                        linewidth=1
                    )
        
        # Mark start and end points
        ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, marker='D', label='Start', zorder=5)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100, marker='s', label='End', zorder=5)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        plot_title = title or f'UMI Replay Waypoints - Episode {self.episode_idx} ({num_waypoints} waypoints)'
        ax.set_title(plot_title)
        
        ax.legend(loc='upper left')
        
        # Equal aspect ratio
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"[UMIReplay] Waypoint visualization saved to {save_path}")
        
        plt.show()
        return fig

    def clear_waypoints(self):
        """Clear all stored waypoints."""
        self.waypoints = []
        print("[UMIReplay] Waypoints cleared.")