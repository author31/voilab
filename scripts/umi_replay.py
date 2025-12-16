"""
UMI Replay - Pure functions for UMI dataset trajectory replay.

This module provides stateless functions for:
- Loading UMI zarr datasets
- Computing episode bounds
- Transforming poses from tag frame to robot base frame
- Setting gripper positions
- Visualizing waypoints
"""

import os
import zipfile
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_umi_dataset(session_dir: str):
    """
    Load UMI dataset from zarr zip file.
    
    Args:
        session_dir: Path to session directory containing dataset.zarr.zip
        
    Returns:
        tuple: (data, meta, episode_ends)
            - data: zarr data group
            - meta: zarr meta group  
            - episode_ends: numpy array of episode end indices
    """
    zarr_zip_path = os.path.join(session_dir, 'dataset.zarr.zip')
    if not os.path.exists(zarr_zip_path):
        raise FileNotFoundError(f"Zarr zip file not found at '{zarr_zip_path}'")
    
    extract_path = os.path.splitext(zarr_zip_path)[0]
    if not os.path.exists(extract_path):
        print(f"[umi_replay] Extracting dataset to {extract_path}...")
        with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    root = zarr.open(store=zarr.DirectoryStore(extract_path), mode='r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    return data, meta, episode_ends


def get_episode_bounds(episode_ends: np.ndarray, episode_idx: int):
    """
    Get start and end indices for a specific episode.
    
    Args:
        episode_ends: Array of episode end indices
        episode_idx: Episode index (0-based)
        
    Returns:
        tuple: (start_idx, end_idx)
    """
    if episode_idx == 0:
        start_idx = 0
        end_idx = episode_ends[0]
    else:
        if episode_idx >= len(episode_ends):
            raise ValueError(f"Episode {episode_idx} invalid. Max episode: {len(episode_ends) - 1}")
        start_idx = episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]
    
    return int(start_idx), int(end_idx)


def pose_to_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Convert position and rotation to 4x4 transformation matrix.
    
    Args:
        pos: Position vector [x, y, z]
        rot: Rotation - either rotation vector (3,) or rotation matrix (3,3)
        
    Returns:
        4x4 transformation matrix
    """
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


def compute_replay_step(data, step_idx: int, T_base_tag: np.ndarray):
    """
    Compute IK target for a single replay step.
    
    Transforms end-effector pose from tag frame to robot base frame.
    
    Args:
        data: zarr data group containing robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_gripper_width
        step_idx: Current step index in the dataset
        T_base_tag: 4x4 transform from robot base to ArUco tag frame
        
    Returns:
        tuple: (target_pos, target_rot, target_quat_wxyz, gripper_width)
            - target_pos: np.array (3,) position in base frame
            - target_rot: scipy Rotation object
            - target_quat_wxyz: np.array (4,) quaternion in WXYZ format
            - gripper_width: float gripper width in meters
    """
    pos_in_tag = data['robot0_eef_pos'][step_idx]
    rot_in_tag = data['robot0_eef_rot_axis_angle'][step_idx]
    gripper_width = float(data['robot0_gripper_width'][step_idx][0])
    
    # # Rotate tag frame 90 degrees around z axis
    pos_in_tag = R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix() @ pos_in_tag

    # Transform EEF pose from tag frame to base frame
    T_tag_eef = pose_to_matrix(pos_in_tag, rot_in_tag)
    T_base_eef = T_base_tag @ T_tag_eef

    target_pos = T_base_eef[:3, 3]
    target_rot = R.from_matrix(T_base_eef[:3, :3])
    
    # Convert to quaternion WXYZ format for Isaac Sim
    target_quat_xyzw = target_rot.as_quat()
    target_quat_wxyz = np.array([
        target_quat_xyzw[3], target_quat_xyzw[0],
        target_quat_xyzw[1], target_quat_xyzw[2]
    ])
    # Manual offset removed – using the transformed pose directly
    
    # Debug logging for trajectory replay
    print(f"[umi_replay] Step {step_idx} | Gripper width: {gripper_width:.4f} m | Target quaternion (WXYZ): {target_quat_wxyz}")
    return target_pos, target_rot, target_quat_wxyz, gripper_width


def set_gripper_width(panda, width: float, threshold: float = 0.04, step: float = 0.08):
    """
    Threshold-based gripper control with gradual movement.
    
    Args:
        panda: Robot articulation
        width: Input gripper width from dataset
        threshold: Threshold to determine open (>=) or closed (<)
        step: Amount to change finger position per call (0.0-1.0)
    """
    target_pos = 1.0 if width >= threshold else 0.0
    
    idx1 = panda.get_dof_index("panda_finger_joint1")
    idx2 = panda.get_dof_index("panda_finger_joint2")
    
    if idx1 is not None and idx2 is not None:
        # Get current finger position
        current_positions = panda.get_joint_positions(joint_indices=np.array([idx1, idx2]))
        current_pos = current_positions[0]
        
        # Move gradually toward target
        if current_pos < target_pos:
            finger_pos = min(current_pos + step, target_pos)
        elif current_pos > target_pos:
            finger_pos = max(current_pos - step, target_pos)
        else:
            finger_pos = target_pos
        
        panda.set_joint_positions(
            positions=np.array([finger_pos, finger_pos]),
            joint_indices=np.array([idx1, idx2])
        )


def visualize_waypoints(
    waypoints: list,
    episode_idx: int = 0,
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
        waypoints: List of (target_pos, target_rot) tuples
            - target_pos: np.array (3,) position
            - target_rot: scipy Rotation object
        episode_idx: Episode index for title
        show_orientation: If True, show orientation arrows at each waypoint
        orientation_scale: Length scale for orientation arrows
        marker_size: Size of waypoint markers
        title: Custom title for the plot
        save_path: If provided, save the figure as PNG to this path
        figsize: Figure size as (width, height) in inches
        dpi: Resolution for saved PNG
        
    Returns:
        matplotlib.figure.Figure: The generated figure, or None if no waypoints
    """
    if not waypoints:
        print("[umi_replay] No waypoints to visualize.")
        return None
    
    # Extract positions and rotations
    positions = np.array([wp[0] for wp in waypoints])
    rotations = [wp[1] for wp in waypoints]
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    num_waypoints = len(waypoints)
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
        
        for axis_idx in range(3):
            for idx in sampled_indices:
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
    
    plot_title = title or f'UMI Replay Waypoints - Episode {episode_idx} ({num_waypoints} waypoints)'
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
        print(f"[umi_replay] Waypoint visualization saved to {save_path}")
    
    plt.show()
    return fig


# ----------------------------------------------------------------------
# Helper: linear Cartesian interpolation (used for the intervention phase)
# ----------------------------------------------------------------------
def linear_cartesian_path(start_pos, start_rot, goal_pos, goal_rot, step_size=0.02):
    """
    Generate a list of (pos, rot) way‑points that linearly interpolate
    between two poses in Cartesian space.

    Parameters
    ----------
    start_pos, goal_pos : np.ndarray (3,)
        Start / goal positions in the same frame.
    start_rot, goal_rot : scipy.spatial.transform.Rotation
        Start / goal orientations.
    step_size : float
        Approximate translation distance per generated waypoint (meters).

    Returns
    -------
    List[Tuple[np.ndarray, Rotation]]
    """
    vec = goal_pos - start_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return [(goal_pos, goal_rot)]

    n_steps = max(1, int(np.ceil(dist / step_size)))
    positions = [start_pos + (i / n_steps) * vec for i in range(1, n_steps + 1)]

    # Spherical linear interpolation for orientation
    slerp = R.slerp(0, 1, [start_rot, goal_rot])
    rotations = [slerp(i / n_steps) for i in range(1, n_steps + 1)]

    return list(zip(positions, rotations))
