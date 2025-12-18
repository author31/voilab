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


def pose_to_transform_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous transformation matrix from position and quaternion.
    
    Args:
        position: Translation vector [x, y, z]
        quat_wxyz: Quaternion in WXYZ format [w, x, y, z]
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    # Convert WXYZ to XYZW for scipy
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = position
    return T


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


def set_prim_scale(prim, scale: float | np.ndarray | list) -> None:
    """
    Set the local scale of a prim in Isaac Sim.

    Args:
        prim: Isaac Sim prim object (SingleXFormPrim, XFormPrim, etc.)
        scale: Scale factor - either a single float for uniform scaling,
               or array [sx, sy, sz] for non-uniform scaling
    """
    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale, scale])
    else:
        scale = np.array(scale)

    prim.set_local_scale(scale)


def compute_optimal_cup_grasp(object_pose: np.ndarray, cup_dimensions: dict = None, tilt_angle_deg: float = 15.0):
    """
    Compute optimal grasp orientation and position for picking up a cup.

    Args:
        object_pose: 4x4 transformation matrix of the cup in robot base frame
        cup_dimensions: Optional dict with 'height', 'diameter', 'wall_thickness'
        tilt_angle_deg: Tilt angle in degrees for grasp orientation (default: 15°)

    Returns:
        dict: {
            'grasp_position': np.ndarray (3,) optimal grasp position,
            'grasp_rotation': scipy Rotation object,
            'grasp_quaternion_wxyz': np.ndarray (4,) quaternion,
            'approach_direction': np.ndarray (3,) unit vector for approach
        }
    """
    # Validate transformation matrix
    if not validate_transformation_matrix(object_pose):
        print("[ERROR] Invalid object_pose matrix, using identity")
        object_pose = np.eye(4)

    # Default cup dimensions if not provided
    if cup_dimensions is None:
        breakpoint()
        cup_dimensions = {
            'height': 0.10,  # 10cm
            'diameter': 0.08,  # 8cm
            'wall_thickness': 0.003  # 3mm
        }

    # Extract object position and orientation
    obj_pos = object_pose[:3, 3]
    obj_rot_matrix = object_pose[:3, :3]
    obj_rot = R.from_matrix(obj_rot_matrix)

    # Grasp at 2/3 of cup height for stability
    grasp_height = cup_dimensions['height'] * 0.66
    grasp_pos_local = np.array([0, 0, grasp_height])

    # Transform to world coordinates
    grasp_pos = obj_pos + obj_rot_matrix @ grasp_pos_local

    # Add small random perturbation to avoid systematic errors
    # This helps when the exact same grasp consistently fails
    if abs(tilt_angle_deg) < 1:  # For straight-down grasps, add small random offset
        random_offset = np.random.uniform(-0.005, 0.005, 3)  # ±5mm random offset
        grasp_pos += random_offset
        print(f"[Debug] Added random offset to grasp position: {random_offset}")

    # Validate grasp is reachable and aligned
    # Ensure gripper will approach from side, not from directly above center
    approach_vector = grasp_pos - obj_pos
    approach_vector[2] = 0  # Remove z-component
    if np.linalg.norm(approach_vector) < 0.05:  # If too centered, offset slightly
        offset_dir = np.array([1, 0, 0])  # Default to +x direction
        offset_dir_world = obj_rot_matrix @ offset_dir
        grasp_pos += offset_dir_world * 0.02  # Offset 2cm to the side
        print(f"[Debug] Adjusted grasp position to avoid center approach")

    # Approach from top with slight tilt (15 degrees) for stability
    # Tilt direction: towards robot (use provided tilt angle)
    tilt_angle = np.radians(tilt_angle_deg)

    # Create grasp rotation:
    # 1. Start with object orientation (aligned with cup)
    grasp_rot = obj_rot

    # 2. Add tilt around local x-axis (pitch)
    tilt_rot = R.from_euler('x', tilt_angle)
    grasp_rot = grasp_rot * tilt_rot

    # Approach direction (mostly downward with slight forward component)
    approach_dir = grasp_rot.apply([0, 0, -1])

    # Convert to quaternion WXYZ
    grasp_quat_xyzw = grasp_rot.as_quat()
    grasp_quat_wxyz = np.array([
        grasp_quat_xyzw[3], grasp_quat_xyzw[0],
        grasp_quat_xyzw[1], grasp_quat_xyzw[2]
    ])

    return {
        'grasp_position': grasp_pos,
        'grasp_rotation': grasp_rot,
        'grasp_quaternion_wxyz': grasp_quat_wxyz,
        'approach_direction': approach_dir,
        'grasp_width': cup_dimensions['diameter'] - 0.005  # 5mm smaller for grip
    }


def compute_pregrasp_pose(grasp_info: dict, approach_distance: float = 0.1):
    """
    Compute pre-grasp pose for approach trajectory.

    Args:
        grasp_info: Output from compute_optimal_cup_grasp
        approach_distance: Distance from grasp to pre-grasp (meters)

    Returns:
        tuple: (position, quaternion_wxyz) for pre-grasp pose
    """
    # Move back along approach direction
    grasp_pos = grasp_info['grasp_position']
    approach_dir = grasp_info['approach_direction']

    pregrasp_pos = grasp_pos - approach_dir * approach_distance

    # Keep same orientation as grasp
    pregrasp_quat = grasp_info['grasp_quaternion_wxyz']

    return pregrasp_pos, pregrasp_quat

def compute_stacking_place_pose(bottom_cup_pose: np.ndarray, cup_height: float = 0.10):
    """
    Compute pose for placing a cup on top of another cup.

    Args:
        bottom_cup_pose: 4x4 transformation matrix of bottom cup
        cup_height: Height of the cup to be placed

    Returns:
        tuple: (position, rotation_matrix) for placement
    """
    # Get bottom cup position
    bottom_pos = bottom_cup_pose[:3, 3]

    # Place on top with small clearance (1cm)
    clearance = 0.01
    place_pos = bottom_pos + np.array([0, 0, cup_height + clearance])

    # Keep same orientation as bottom cup
    place_rot = bottom_cup_pose[:3, :3]

    return place_pos, place_rot


def detect_cup_type(object_name: str, dimensions: dict = None):
    """
    Detect cup type from object name or dimensions.

    Args:
        object_name: Name from object_poses.json
        dimensions: Optional dict with object dimensions

    Returns:
        str: 'small_cup', 'large_cup', or 'unknown'
    """
    # Check by name first
    if 'blue' in object_name.lower():
        return 'small_cup'  # Assuming blue cups are smaller
    elif 'pink' in object_name.lower() or 'white' in object_name.lower():
        return 'large_cup'  # Assuming pink/white cups are larger

    # Check by dimensions if available
    if dimensions:
        diameter = dimensions.get('diameter', 0)
        if diameter < 0.07:  # < 7cm
            return 'small_cup'
        elif diameter > 0.09:  # > 9cm
            return 'large_cup'

    return 'unknown'


def validate_transformation_matrix(T: np.ndarray) -> bool:
    """
    Validate a 4x4 transformation matrix.

    Args:
        T: 4x4 transformation matrix to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if T.shape != (4, 4):
        print(f"[ERROR] Invalid matrix shape: {T.shape}, expected (4, 4)")
        return False

    # Check bottom row
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        print(f"[ERROR] Invalid bottom row: {T[3, :]}")
        return False

    # Check rotation matrix properties
    R_mat = T[:3, :3]
    det = np.linalg.det(R_mat)

    if abs(det - 1.0) > 0.01:
        print(f"[ERROR] Invalid rotation matrix determinant: {det}")
        print(f"[ERROR] Rotation matrix:\n{R_mat}")
        return False

    # Check orthogonality
    RTR = R_mat.T @ R_mat
    if not np.allclose(RTR, np.eye(3), atol=0.01):
        print(f"[ERROR] Rotation matrix not orthogonal")
        return False

    return True


def gf_matrix4d_to_numpy(matrix) -> np.ndarray:
    """
    Converts a Gf.Matrix4d (row-major) to a row-major NumPy array.
    
    Parameters:
    matrix (Gf.Matrix4d): The input Gf.Matrix4d to convert.
    
    Returns:
    np.ndarray: A 4x4 NumPy array in row-major order.
    """
    # Extract the matrix elements row by row
    data = [[matrix[i][j] for j in range(4)] for i in range(4)]
    # Create a NumPy array from the data (NumPy uses row-major by default)
    return np.array(data, dtype=np.float64).T

