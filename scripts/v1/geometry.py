"""
Pure-numpy pose math for the v1 config-based Isaac Sim pipeline.

This module is IMPORT-SAFE: it has no ``isaacsim`` dependency, so it can be
imported anywhere (before or after ``SimulationApp``). It is the single home for
the pose conversions and the linear Cartesian planner that were previously
duplicated byte-for-byte in ``base_config`` and ``simple_room_config``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def wxyz_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> 6D representation (first two columns)."""
    if len(matrix.shape) == 2:
        return np.concatenate([matrix[:, 0], matrix[:, 1]])
    return np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=-1)


def rot6d_to_matrix(rot6d):
    """6D rotation representation -> 3x3 matrix via Gram-Schmidt."""
    is_batch = len(rot6d.shape) == 2
    if not is_batch:
        rot6d = rot6d[None, :]

    x_raw = rot6d[:, 0:3]
    y_raw = rot6d[:, 3:6]

    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)

    matrix = np.stack([x, y, z], axis=-1)
    if not is_batch:
        matrix = matrix[0]
    return matrix


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Pose [x,y,z, rx,ry,rz] (axis-angle) -> 4x4 matrix."""
    is_batch = len(pose.shape) == 2
    if not is_batch:
        pose = pose[None, :]

    pos = pose[:, :3]
    rot_mat = R.from_rotvec(pose[:, 3:6]).as_matrix()

    mat = np.zeros((pose.shape[0], 4, 4))
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = pos
    mat[:, 3, 3] = 1.0

    if not is_batch:
        mat = mat[0]
    return mat


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """4x4 matrix -> 10D pose [x,y,z, rot6d(6)]."""
    is_batch = len(mat.shape) == 3
    if not is_batch:
        mat = mat[None, :]

    pos = mat[:, :3, 3]
    rot6d = matrix_to_rot6d(mat[:, :3, :3])
    pose10d = np.concatenate([pos, rot6d], axis=-1)

    if not is_batch:
        pose10d = pose10d[0]
    return pose10d


def plan_line_cartesian(
    p_start: np.ndarray,
    q_start_wxyz: np.ndarray,
    p_goal: np.ndarray,
    q_goal_wxyz: np.ndarray,
    step_m: float = 0.005,
):
    """Linear Cartesian trajectory with SLERP orientation interpolation."""
    p_start = np.asarray(p_start, dtype=float)
    p_goal = np.asarray(p_goal, dtype=float)

    dist = np.linalg.norm(p_goal - p_start)
    n_steps = max(2, int(np.ceil(dist / step_m)))

    positions = np.linspace(p_start, p_goal, n_steps)

    q0_xyzw = wxyz_to_xyzw(np.asarray(q_start_wxyz, dtype=float))
    q1_xyzw = wxyz_to_xyzw(np.asarray(q_goal_wxyz, dtype=float))

    key_rots = R.from_quat([q0_xyzw, q1_xyzw])
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(np.linspace(0.0, 1.0, n_steps))
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.array([xyzw_to_wxyz(q) for q in quats_xyzw])

    return [np.concatenate([p, q_wxyz]) for p, q_wxyz in zip(positions, quats_wxyz)]
