"""
Pose conversion utilities for Isaac Sim object spawning.
Converts rotation vectors (rvec) and translation vectors (tvec) to formats usable by Isaac Sim.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def rvec_to_quaternion_wxyz(rvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector (Rodrigues/axis-angle) to quaternion WXYZ format.

    The rotation vector format (used in OpenCV and the reconstruction system):
    - The direction of the vector is the axis of rotation
    - The magnitude (length) is the rotation angle in radians
    - Example: [0, 0, 1.5708] = 90° rotation around Z-axis

    Args:
        rvec: Rotation vector [x, y, z] where magnitude = angle (radians)

    Returns:
        np.ndarray: Quaternion in WXYZ format [w, x, y, z]
    """
    rvec = np.array(rvec)

    # Handle zero rotation (identity)
    if np.allclose(rvec, 0):
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Convert to rotation matrix using scipy
    rot = R.from_rotvec(rvec)

    # Get quaternion in XYZW format (scipy default)
    quat_xyzw = rot.as_quat()

    # Convert to WXYZ format (Isaac Sim convention)
    quat_wxyz = np.array([
        quat_xyzw[3],  # w
        quat_xyzw[0],  # x
        quat_xyzw[1],  # y
        quat_xyzw[2]   # z
    ])

    return quat_wxyz


def quaternion_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from XYZW to WXYZ format.

    Args:
        quat: Quaternion in XYZW format [x, y, z, w]

    Returns:
        np.ndarray: Quaternion in WXYZ format [w, x, y, z]
    """
    quat = np.array(quat)
    if len(quat) != 4:
        raise ValueError(f"Invalid quaternion length: {len(quat)}, expected 4")
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def validate_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a quaternion.

    Args:
        quat: Quaternion in WXYZ format [w, x, y, z]

    Returns:
        np.ndarray: Normalized quaternion in WXYZ format
    """
    quat = np.array(quat)

    if len(quat) != 4:
        raise ValueError(f"Invalid quaternion length: {len(quat)}, expected 4")

    # Calculate magnitude
    norm = np.linalg.norm(quat)

    if norm < 0.001:
        # Nearly zero quaternion, return identity
        print("WARNING: Near-zero quaternion detected, using identity rotation")
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Normalize
    return quat / norm


def rvec_tvec_to_transform_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to 4x4 transform matrix.

    Args:
        rvec: Rotation vector [x, y, z]
        tvec: Translation vector [x, y, z]

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    rvec = np.array(rvec)
    tvec = np.array(tvec)

    # Convert rvec to rotation matrix
    rot = R.from_rotvec(rvec)
    rot_matrix = rot.as_matrix()

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = tvec

    return T


def transform_tvec_to_world_frame(tvec: np.ndarray, world_frame: np.ndarray) -> np.ndarray:
    """
    Transform translation vector from camera frame to world frame.

    This is a placeholder; the actual transformation depends on the camera calibration.
    Users should replace with their own transformation logic.

    Args:
        tvec: Translation vector in camera frame [x, y, z]
        world_frame: 4x4 transformation matrix from camera to world (or similar)

    Returns:
        np.ndarray: Transformed translation vector in world frame
    """
    # If world_frame is a 4x4 matrix, we can apply it to a homogeneous vector
    assert world_frame.shape == (4, 4)

    vec_hom = np.array([tvec[0], tvec[1], tvec[2], 1.0])
    transformed = world_frame @ vec_hom
    return transformed[:3]