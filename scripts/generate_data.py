"""
Isaac Sim Workspace Launcher with UMI Trajectory Replay.

This script initializes the Isaac Sim environment, loads the robot and scene,
and replays UMI dataset trajectories. All business logic is exposed here.

Architecture:
- One simulation_app instance per episode
- Explicit state management (no hidden class state)
- Pure function calls to umi_replay module
"""

import os
import json
import registry
import argparse
import numpy as np
import time
import sys
from simulation_dataset_accumulator import SimulationDatasetAccumulator
import zarr
from zarr.storage import ZipStore
from numcodecs import Blosc
import cv2
from utils import get_object_pose
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["kitchen", "dining-room", "living-room"], required=True)
parser.add_argument("--session_dir", type=str, default=None)
parser.add_argument("--x_offset", type=float, default=0.1, help="X-axis offset for coordinate calibration (meters)")
parser.add_argument("--y_offset", type=float, default=0.15, help="Y-axis offset for coordinate calibration (meters)")
parser.add_argument("--z_offset", type=float, default=-0.07, help="Z-axis offset for coordinate calibration (meters)")
parser.add_argument("--episode", type=int, default=0, help="Episode index (optional)")

args = parser.parse_args()

from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.robot_motion.motion_generation"]
}
simulation_app = SimulationApp(config)

import omni.usd
from isaacsim.util.debug_draw import _debug_draw
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    ArticulationTrajectory
)
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import RigidPrim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R, Slerp
from object_loader import load_object_transforms_from_json, map_object_name_to_asset
import utils
import lula
from pxr import UsdPhysics
from umi_replay import set_gripper_width
from motion_plan import PickPlace, TeleopEEController



assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("[Main] ERROR: Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

enable_extension("isaacsim.robot_motion.motion_generation")

# --- Configuration ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm_v2.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"
ASSETS_DIR = "/workspace/voilab/assets/CADs"

# CORRECTED: Use these paths in the solver initialization
LULA_ROBOT_DESCRIPTION_PATH = "/workspace/voilab/assets/lula/frank_umi_descriptor.yaml"
LULA_URDF_PATH = "/workspace/voilab/assets/franka_panda/franka_panda_umi-isaacsim.urdf"
PANDA0_PATH = FRANKA_PANDA_PRIM_PATH + "/panda/panda_link0"
LEFT_PATH = FRANKA_PANDA_PRIM_PATH + "/panda/panda_leftfinger"
RIGHT_PATH = FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger"


DEBUG_DRAW = _debug_draw.acquire_debug_draw_interface()


# Helper functions

def random_yaw_from_ref(q_ref_wxyz, yaw_range_deg=90):
    """
    q_ref_wxyz: quaternion gốc (wxyz)
    yaw_range_deg: random yaw trong [-yaw_range_deg, +yaw_range_deg]
    """
    # random yaw offset
    yaw_offset = np.random.uniform(-yaw_range_deg, yaw_range_deg)

    # ref rotation
    R_ref = R.from_quat([
        q_ref_wxyz[1],
        q_ref_wxyz[2],
        q_ref_wxyz[3],
        q_ref_wxyz[0],
    ])  # xyzw

    # yaw rotation (around Z)
    R_yaw = R.from_euler('z', yaw_offset, degrees=True)

    # apply yaw (world Z)
    R_new = R_yaw * R_ref

    # back to wxyz
    q_new_xyzw = R_new.as_quat()
    q_new_wxyz = np.array([
        q_new_xyzw[3],
        q_new_xyzw[0],
        q_new_xyzw[1],
        q_new_xyzw[2],
    ])

    return q_new_wxyz, yaw_offset

def get_isaac_camera_intrinsics(camera):
    """
    Isaac Sim camera → returns intrinsic matrix K
    """
    # Use getters
    W, H = camera.get_resolution()  # returns (width, height)
    h_fov_rad = np.deg2rad(camera.get_horizontal_fov())  # in radians

    # Compute fx, fy
    fx = W / (2 * np.tan(h_fov_rad / 2))
    fy = fx  # assuming square pixels

    # Principal point (center)
    cx = W / 2
    cy = H / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ], dtype=np.float32)

    return K



def quat_wxyz_to_rpy_deg(quat_wxyz):
    # đổi wxyz → xyzw cho scipy
    quat_xyzw = [
        quat_wxyz[1],
        quat_wxyz[2],
        quat_wxyz[3],
        quat_wxyz[0],
    ]
    rpy = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
    return rpy  # roll, pitch, yaw (deg)

def calculate_camera_orientation(eye_pos, target_pos, up_axis=np.array([0, 0, 1])):
    """
    Helper function.
    Computes the quaternion (WXYZ) for a Robotics Camera (Look=X, Up=Z)
    at eye_pos looking at target_pos.
    """
    eye_pos = np.array(eye_pos)
    target_pos = np.array(target_pos)
    
    # 1. Forward Vector (Camera X-axis)
    fwd = target_pos - eye_pos
    fwd = fwd / np.linalg.norm(fwd)
    
    # 2. Right Vector
    right = np.cross(fwd, up_axis)
    
    # Handle degenerate case (looking straight up/down)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0, 1, 0])
        
    right = right / np.linalg.norm(right)
    
    # 3. Up Vector (Camera Z-axis)
    z_axis = np.cross(right, fwd)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Y_axis (Camera Left)
    y_axis = np.cross(z_axis, fwd)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Basis: X=fwd, Y=y_axis, Z=z_axis
    R_matrix = np.column_stack((fwd, y_axis, z_axis))
    
    # Convert to quaternion (Scalar-Last xyzw -> WXYZ for Isaac Sim)
    quat_xyzw = R.from_matrix(R_matrix).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    return quat_wxyz


def get_T_world_base() -> np.ndarray:
    time = Usd.TimeCode.Default()
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(time)

    base_prim = stage.GetPrimAtPath(FRANKA_PANDA_PRIM_PATH)
    T_gf = cache.GetLocalToWorldTransform(base_prim)

    return utils.gf_matrix4d_to_numpy(T_gf)


def get_T_world_aruco(aruco_tag_pose: dict) -> np.ndarray:
    aruco_translation = np.array(aruco_tag_pose['translation'])
    aruco_quat_wxyz = np.array(aruco_tag_pose['rotation_quat'])
    aruco_quat_xyzw = np.array([aruco_quat_wxyz[1], aruco_quat_wxyz[2], aruco_quat_wxyz[3], aruco_quat_wxyz[0]])
    
    T_world_aruco = np.eye(4)
    T_world_aruco[:3, 3] = aruco_translation
    T_world_aruco[:3, :3] = R.from_quat(aruco_quat_xyzw).as_matrix()
    return T_world_aruco


def calibrate_robot_base(panda, lula_solver):
    """
    Update Lula solver with current robot base pose.
    Must be called before computing IK.
    
    Args:
        panda: Panda articulation object
        lula_solver: LulaKinematicsSolver instance
    """
    from isaacsim.core.prims import SingleXFormPrim
    import isaacsim.core.utils.prims as prims_utils
    
    # Try both common paths for Franka link0
    base_link_path = "/World/Franka/panda/panda_link0"
    if not prims_utils.is_prim_path_valid(base_link_path):
        base_link_path = f"{panda.prim_path}/panda/panda_link0"
        
    base_link_prim = SingleXFormPrim(base_link_path)
    robot_pos, robot_quat = base_link_prim.get_world_pose()
    
    lula_solver.set_robot_base_pose(
        robot_position=robot_pos,
        robot_orientation=robot_quat
    )


def apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz):
    """
    Compute and apply IK solution for target pose.
    
    Args:
        panda: Panda articulation object
        art_kine_solver: ArticulationKinematicsSolver instance
        target_pos: Target position (3,)
        target_quat_wxyz: Target orientation as quaternion WXYZ (4,)
        step_idx: Current step index (for logging)
        
    Returns:
        bool: True if IK succeeded
    """
    action, success = art_kine_solver.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_quat_wxyz
    )

    if success:
        panda.set_joint_positions(action.joint_positions, np.arange(7))
        return True

    return False


class RigidPrimManager:
    def __init__(self):
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def get(self, prim_path):
        if prim_path not in self._cache:
            prim = RigidPrim(
                prim_path,
                name=prim_path.replace("/", "_")
            )
            prim.initialize()
            self._cache[prim_path] = prim
        return self._cache[prim_path]


def make_get_object_world_pose(prim_mgr):
    def _get_object_world_pose(prim_path):
        prim = prim_mgr.get(prim_path)
        pos_batch, quat_wxyz_batch = prim.get_world_poses()
        pos = pos_batch[0]
        quat_wxyz = quat_wxyz_batch[0]

        quat_xyzw = np.array([
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = pos
        return T
    return _get_object_world_pose


def get_object_world_size(object_prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Object prim not found: {object_prim_path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    return prim_range.GetSize()


# ----------------------------------------------------------------------
# IsaacSim Trajectory Generation Helper Functions
# ----------------------------------------------------------------------


def get_end_effector_pose(panda, lula_solver, art_kine_solver) -> np.ndarray:
    calibrate_robot_base(panda, lula_solver)
    ee_pos, ee_rot_matrix = art_kine_solver.compute_end_effector_pose()
    eef_rot = R.from_matrix(ee_rot_matrix[:3, :3]).as_rotvec()
    return np.concatenate([ee_pos.astype(np.float64), eef_rot.astype(np.float64)])


def get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver):
    calibrate_robot_base(panda, lula_solver)

    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
    quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)


def save_multi_episode_dataset(output_path: str, episodes: list[dict]) -> None:
    """
     edited to save multi-view images
    """

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = ZipStore(output_path, mode="w")
    root = zarr.group(store)
    data = root.create_group("data")

    first_ep = episodes[0]
    
    # --- 1. Save Multi-view RGB Images ---
    camera_rgb_keys = [k for k in first_ep.keys() if "_rgb" in k]
    ref_camera_key = camera_rgb_keys[0] # Use for length calculation
    
    for key in camera_rgb_keys:
        rgb_data = np.concatenate([ep[key] for ep in episodes], axis=0).astype(np.uint8)
        data.create_dataset(key, data=rgb_data, compressor=compressor)

    # --- 2. Save Extrinsics (NEW) ---
    camera_ext_keys = [k for k in first_ep.keys() if "_extrinsics" in k]
    for key in camera_ext_keys:
        ext_data = np.concatenate([ep[key] for ep in episodes], axis=0).astype(np.float32)      # 4x4 matrices
        data.create_dataset(key, data=ext_data, compressor=compressor)

    # --- 3. Save Robot Data ---
    demo_start = np.concatenate([ep["demo_start"] for ep in episodes], axis=0).astype(np.float64)
    demo_end = np.concatenate([ep["demo_end"] for ep in episodes], axis=0).astype(np.float64)
    eef_pos = np.concatenate([ep["eef_pos"] for ep in episodes], axis=0).astype(np.float32)
    eef_rot = np.concatenate([ep["eef_rot"] for ep in episodes], axis=0).astype(np.float32)
    gripper = np.concatenate([ep["gripper"] for ep in episodes], axis=0).astype(np.float32)

    data.create_dataset("robot0_demo_start_pose", data=demo_start, compressor=compressor)
    data.create_dataset("robot0_demo_end_pose", data=demo_end, compressor=compressor)
    data.create_dataset("robot0_eef_pos", data=eef_pos, compressor=compressor)
    data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot, compressor=compressor)
    data.create_dataset("robot0_gripper_width", data=gripper, compressor=compressor)

    # Lengths
    episode_lengths = [len(ep[ref_camera_key]) for ep in episodes]
    episode_ends = np.cumsum(episode_lengths)
    
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=episode_ends)
    store.close()
    print("[SAVE] replay_dataset.zarr.zip saved at:", output_path)


def _load_progress(session_dir: str) -> set[int]:
    progress_path = os.path.join(session_dir, ".previous_progress.json")
    print(progress_path)
    if not os.path.exists(progress_path):
        return set()
    try:
        with open(progress_path, "r") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[Main] WARNING: Failed to read progress file: {exc}")
        return set()
    completed = payload.get("completed_episodes", [])
    return set(int(x) for x in completed)


def _save_progress(session_dir: str, completed: set[int]) -> None:
    progress_path = os.path.join(session_dir, ".previous_progress.json")
    payload = {"completed_episodes": sorted(completed)}
    with open(progress_path, "w") as f:
        json.dump(payload, f, indent=2)


def _normalize_object_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def step_world_and_record(
    world,
    cameras,
    panda,
    lula_solver,
    art_kine_solver,
    rgb_dict_list,
    extrinsics_dict_list,
    eef_pos_list,
    eef_rot_list,
    gripper_list,
    joint_pos_list=None, # <--- NEW
    render=True,
    sleep_dt=0.01,
):
    """
    Khanh: modified to return multi-view images & camera extrinsics.
    """
    world.step(render=render)
    time.sleep(sleep_dt)

    # RGB & Extrinsics: Capture from all cameras
    frame_dict = {}
    ext_dict = {}
    
    for cam_key, cam_obj in cameras.items():
        # 1. Get RGB
        img = cam_obj.get_rgb()
        if img is not None:
            frame_dict[cam_key] = img
            
        # 2. Get Extrinsics (Camera-to-World Matrix)
        # Note: get_world_pose returns tuple (position, orientation_wxyz)
        pos, quat_wxyz = cam_obj.get_world_pose()
        
        # Convert to 4x4 Homogeneous Matrix
        T = np.eye(4)
        T[:3, 3] = pos
        # Convert WXYZ to XYZW for scipy
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        
        ext_dict[cam_key] = T

    # Only append if we got data
    if frame_dict:
        rgb_dict_list.append(frame_dict)
        extrinsics_dict_list.append(ext_dict) # <--- NEW

    # End-effector pose
    eef_pose6d = get_end_effector_pose(panda, lula_solver, art_kine_solver)
    eef_pos_list.append(eef_pose6d[:3])
    eef_rot_list.append(eef_pose6d[3:])

    # Gripper & Joint Positions
    joint_pos = panda.get_joint_positions()
    gripper_width = joint_pos[-2] + joint_pos[-1]
    gripper_list.append([gripper_width])
    
    if joint_pos_list is not None:
        joint_pos_list.append(joint_pos)

    return eef_pose6d

def _set_fixed_objects_for_episode(cfg, object_prims):
    if cfg.get("environment_vars", {}).get("SCENE_CONFIG") != "living_scene":
        return

    fixed = cfg.get("environment_vars", {}).get("FIXED_OBJECTS", [])
    stage = omni.usd.get_context().get_stage()

    for item in fixed:
        name = _normalize_object_name(item["name"])
        prim = object_prims.get(name)
        if prim is None:
            continue
        pos = np.array(item["position"], dtype=np.float64)
        quat_wxyz = np.array(item["rotation_quat_wxyz"], dtype=np.float64)
        prim.set_world_pose(position=pos, orientation=quat_wxyz)
        prim_usd = stage.GetPrimAtPath(prim.prim_path)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim_usd)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        rigid_api.CreateKinematicEnabledAttr(True)

def wxyz_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

def xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def plan_line_cartesian(
    p_start: np.ndarray,
    q_start_wxyz: np.ndarray,
    p_goal: np.ndarray,
    q_goal_wxyz: np.ndarray,
    step_m: float = 0.005,
    ):
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

def save_episode_video(rgb_seq, episode_idx, session_dir, fps=30):
    """
    rgb_seq: (T, H, W, 3), uint8 or float
    """
    assert rgb_seq.ndim == 4, "Expected (T, H, W, 3)"

    vis_dir = os.path.join(session_dir, "videos")
    os.makedirs(vis_dir, exist_ok=True)

    h, w = rgb_seq.shape[1:3]
    out_path = os.path.join(vis_dir, f"episode_{episode_idx:04d}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in rgb_seq:
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"[Video] Saved episode video: {out_path}")

from pxr import Usd, UsdPhysics, PhysxSchema

def debug_usd_physics(prim):
    print("\n========== USD / PhysX DEBUG ==========")
    print("Prim path:", prim.GetPath())
    print("Prim type:", prim.GetTypeName())
    print("Is valid:", prim.IsValid())
    print("Is active:", prim.IsActive())

    stage = prim.GetStage()
    rb = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    print("Has RigidBodyAPI:", rb is not None)

    if rb:
        print("  rigidBodyEnabled:",
              rb.GetRigidBodyEnabledAttr().Get())
        print("  kinematicEnabled:",
              rb.GetKinematicEnabledAttr().Get())

    stage = prim.GetStage()
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Get(
        stage,
        prim.GetPath()
    )

    print("Has PhysxRigidBodyAPI:", physx_rb is not None)

    if physx_rb:
        print("  disableGravity:",
              physx_rb.GetDisableGravityAttr().Get())
        print("  linearDamping:",
              physx_rb.GetLinearDampingAttr().Get())
        print("  angularDamping:",
              physx_rb.GetAngularDampingAttr().Get())

def debug_dynamic_state(prim):
    stage = prim.GetStage()
    rb = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    print("\n========== DYNAMIC STATE ==========")

    if not rb:
        print("❌ No RigidBodyAPI → cannot be dynamic")
        return

    enabled = rb.GetRigidBodyEnabledAttr().Get()
    kinematic = rb.GetKinematicEnabledAttr().Get()

    print("RigidBody enabled:", enabled)
    print("Kinematic:", kinematic)

    if not enabled:
        print("❌ RigidBody is DISABLED")

    if kinematic:
        print("⚠️ Object is KINEMATIC (cannot be grasped by force)")

from pxr import UsdGeom

def debug_collision(stage, prim):
    print("\n========== COLLISION DEBUG ==========")
    collision_found = False

    for child in Usd.PrimRange(prim):
        stage = child.GetStage()
        col = UsdPhysics.CollisionAPI.Get(
            stage,
            child.GetPath()
        )

        if col:
            collision_found = True
            print("Collision on:", child.GetPath())
            print("  enabled:",
                  col.GetCollisionEnabledAttr().Get())

            mesh = UsdGeom.Mesh(child)
            if mesh:
                print("  collider mesh:", child.GetTypeName())

    if not collision_found:
        print("❌ NO COLLISION FOUND → fingers pass through")

def debug_mass_inertia(prim):
    print("\n========== MASS / INERTIA ==========")

    stage = prim.GetStage()
    mass_api = UsdPhysics.MassAPI.Get(
        stage,
        prim.GetPath()
    )

    if not mass_api:
        print("❌ No MassAPI → PhysX auto mass (danger)")
        return

    print("Mass:", mass_api.GetMassAttr().Get())
    print("Density:", mass_api.GetDensityAttr().Get())
    print("CenterOfMass:", mass_api.GetCenterOfMassAttr().Get())
    print("DiagonalInertia:",
          mass_api.GetDiagonalInertiaAttr().Get())

    mass = mass_api.GetMassAttr().Get()
    if mass is None or mass <= 0:
        print("❌ INVALID MASS")

from pxr import UsdShade

def debug_physics_material(stage, prim):
    print("\n========== PHYSICS MATERIAL ==========")

    for child in Usd.PrimRange(prim):
        mat = UsdShade.Material.Get(stage, child.GetPath())
        if mat:
            print("Material found:", child.GetPath())

    stage = prim.GetStage()
    physx_mat = PhysxSchema.PhysxMaterialAPI.Get(
        stage,
        prim.GetPath()
    )
    if not physx_mat:
        print("❌ No PhysxMaterialAPI → friction default (usually low)")
        return

    print("Static friction:",
          physx_mat.GetStaticFrictionAttr().Get())
    print("Dynamic friction:",
          physx_mat.GetDynamicFrictionAttr().Get())
    print("Restitution:",
          physx_mat.GetRestitutionAttr().Get())

from pxr import UsdPhysics

def find_rigid_body_prim(root_prim):
    for p in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI.Get(
            p.GetStage(), p.GetPath()
        ):
            return p
    return None

from pxr import Usd, UsdPhysics, UsdShade, Sdf

def fix_rigid_object_physx(prim, static_friction=0.8, dynamic_friction=0.6, restitution=0.0):
    """
    Fix a rigid object in Isaac Sim PhysX:
    - Assigns proper PhysX material (friction, restitution)
    - Binds material to all collider shapes under the prim
    """

    stage = prim.GetStage()
    if stage is None:
        print(f"[PhysXFix] ERROR: Prim {prim.GetPath()} has no stage")
        return

    # --- Step 1: Create or get a Material prim ---
    material_path = Sdf.Path(f"/World/Materials/{prim.GetName()}_physx_material")
    if not stage.GetPrimAtPath(material_path):
        mat_prim = UsdShade.Material.Define(stage, material_path)
    else:
        mat_prim = UsdShade.Material(stage.GetPrimAtPath(material_path))

    # --- Step 2: Apply MaterialAPI and set friction / restitution ---
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(restitution)

    # --- Step 3: Bind this material to all collider shapes under the prim ---
    def bind_material_to_colliders(obj_prim):
        for child in obj_prim.GetAllChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI):
                binding_api = UsdShade.MaterialBindingAPI.Apply(child)
                binding_api.Bind(mat_prim, UsdShade.Tokens.weakerThanDescendants, "physics")
            # 遞迴綁定
            bind_material_to_colliders(child)

    bind_material_to_colliders(prim)

    print(f"[PhysXFix] Applied PhysX Material to {prim.GetPath()}: "
          f"static_friction={static_friction}, dynamic_friction={dynamic_friction}, restitution={restitution}")

def main():
    """Main entry point."""
    print(f"[Main] Starting with task: {args.task}")
    
    # --- Load registry configuration ---
    registry_class = registry.get_task_registry(args.task)
    if not registry_class.validate_environment():
        print(f"[Main] WARNING: Registry validation failed")

    cfg = registry_class.get_config()
    assert cfg.get("aruco_tag_pose") is not None, "Aruco tag pose is required"
    assert cfg.get("franka_pose") is not None, "Franka pose is required"
    assert cfg.get("camera_pose") is not None, "Camera pose is required"
    is_episode_completed = registry_class.is_episode_completed

    print(f"[Main] Configuration: {cfg}")
    franka_pose = cfg.get("franka_pose", {})
    franka_translation = franka_pose.get("translation", [0, 0, 0])
    franka_rotation = franka_pose.get("rotation_quat", [0, 0, 0, 1])
    aruco_tag_pose = cfg.get("aruco_tag_pose", {})
    camera_translation = cfg.get("camera_pose", {}).get("translation", [0, 0, 0])

    # --- Setup scene and world ---
    stage_utils.open_stage(BASE_SCENE_FP)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- Setup robot ---
    robot = stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    robot_xform = SingleXFormPrim(prim_path=FRANKA_PANDA_PRIM_PATH)

    # Configure Franka gripper
    gripper = ParallelGripper(
        end_effector_prim_path=FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.04, 0.04]),
        joint_closed_positions=np.array([0.0, 0.0]),  # 🔥關鍵
        action_deltas=np.array([0.005, 0.005]),
    )


    # Create SingleManipulator and add to world scene
    panda = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PANDA_PRIM_PATH,
            name="my_franka",
            end_effector_prim_path=FRANKA_PANDA_PRIM_PATH + "/panda/panda_rightfinger",
            gripper=gripper,
        )
    )
    panda.gripper.set_default_state(panda.gripper.joint_opened_positions)
    # Set robot position after world reset
    robot_xform.set_local_pose(
        translation=np.array(franka_translation) / stage_utils.get_stage_units(),
        orientation=np.array(franka_rotation)
    )
    set_camera_view(camera_translation, franka_translation)
    # camera = Camera(
    #     prim_path=f"{GOPRO_PRIM_PATH}/Camera",
    #     name="gopro_camera",
    #     resolution=(224,224)
    # )
    # camera.initialize()

    # --- 在創建 panda 後 ---
    # 1️⃣ 創建 controller
    controller = ArticulationController()

    # 2️⃣ 初始化 controller，使用 SingleManipulator 的 articulation view
    controller.initialize(panda._articulation_view)

    # 3️⃣ 將 controller 保存到 panda 物件或其他 class
    panda.controller = controller

        #################### START MULTI-CAM ####################
    cameras = {}
    
    # 1. Wrist Camera: original camera, attached to the robot arm.
    cameras["wrist"] = Camera(
        prim_path=f"{GOPRO_PRIM_PATH}/Camera",
        name="wrist_camera",
        resolution=(224, 224)
    )

    # 2. Top-Down Camera: camera from BEV.
    """     
    top_cam_pos = np.array(franka_translation) + np.array([0.6, 0.0, 1.8])
    top_cam_target = np.array(franka_translation) + np.array([0.6, 0.0, 0.0])
    top_cam_quat = calculate_camera_orientation(top_cam_pos, top_cam_target, up_axis=np.array([1, 0, 0]))

    cameras["top"] = Camera(
        prim_path="/World/TopCamera",
        name="top_camera",
        position=top_cam_pos,
        orientation=top_cam_quat,
        resolution=(224, 224)
    )
    """
    # 3. Angled Camera: camera from side-view, overlooking the whole workspace
    angle_cam_pos = np.array(franka_translation) + np.array([1.6, -2.0, 1.3])
    
    angle_cam_target = np.array(franka_translation) + np.array([0.3, 0.0, 0.2])
    angle_cam_quat = calculate_camera_orientation(
        angle_cam_pos, 
        angle_cam_target, 
        up_axis=np.array([0, 0, 1])
    )

    cameras["angle"] = Camera(
        prim_path="/World/AngleCamera",
        name="angle_camera",
        position=angle_cam_pos,
        orientation=angle_cam_quat,
        resolution=(224, 224)
    )

    for cam in cameras.values():
        cam.initialize()
    #################### END MULTI-CAM ####################
    

    world.reset()
    prim_mgr = RigidPrimManager()

    lula_solver = None
    art_kine_solver = None
    object_prims = {}
    object_poses_path = None

    if args.session_dir is None:
        print("[Main] ERROR: session_dir is required for multi-episode replay.")
        simulation_app.close()
        return

    object_poses_path = os.path.join(args.session_dir, 'demos', 'mapping', 'object_poses.json')
    print(f"[Main] Looking for object poses at: {object_poses_path}")

    preload_objects = cfg.get("environment_vars", {}).get("PRELOAD_OBJECTS", [])
    preload_by_name = {}
    for entry in preload_objects:
        assert isinstance(entry, dict), f"PRELOAD_OBJECTS entry must be a dict: {entry}"
        raw_name = entry.get("name")
        asset_filename = entry.get("assets")
        prim_path = entry.get("prim_path")
        assert raw_name, f"Missing name for PRELOAD_OBJECTS entry: {entry}"
        assert asset_filename, f"Missing assets for PRELOAD_OBJECTS entry: {entry}"
        assert prim_path, f"Missing prim_path for PRELOAD_OBJECTS entry: {entry}"

        object_name = _normalize_object_name(raw_name)
        preload_by_name[object_name] = entry
        if object_name in object_prims:
            continue

        full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {raw_name}")
            continue

        try:
            stage_utils.add_reference_to_stage(
                usd_path=full_asset_path,
                prim_path=prim_path
            )
        except Exception as e:
            print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
            continue

        if "knife" in prim_path.lower():
            q_ref = np.array([0.707, 0.0, 0.0, 0.707])
            q_rand, yaw = random_yaw_from_ref(q_ref)
        else:
            q_ref = np.array([1, 0.0, 0.0, 0.0])
            q_rand, yaw = random_yaw_from_ref(q_ref)

        obj_prim = SingleXFormPrim(prim_path=prim_path, name=object_name, orientation=q_rand)
        world.scene.add(obj_prim)
        object_prims[object_name] = obj_prim
        print(f"[ObjectLoader] Preloaded {raw_name} as {prim_path}")

    # Initialize kinematics solvers
    print(f"[Main] Initializing Kinematics with UMI config...")
    lula_solver = LulaKinematicsSolver(
        robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
        urdf_path=LULA_URDF_PATH
    )

    art_kine_solver = ArticulationKinematicsSolver(
        panda,
        kinematics_solver=lula_solver,
        end_effector_frame_name="umi_tcp"
    )
    with open(object_poses_path, "r") as f:
        object_pose_records = json.load(f)
    
    # [Data Generation] Multiply episodes to get multiple robot starts per object layout
    N_REPEATS = 2
    total_episodes = len(object_pose_records) * N_REPEATS
    print(f"[Main] Replay initialized for {total_episodes} episodes ({len(object_pose_records)} layouts * {N_REPEATS} repeats).")


    # --- Main simulation loop ---
    print("[Main] Starting simulation loop...")

    completed_episodes = _load_progress(args.session_dir)
    episodes_to_run = [ep for ep in range(total_episodes) if ep not in completed_episodes]
    collected_episodes = []

    camera_intrinsics = {}
    for key, cam in cameras.items():
        K = get_isaac_camera_intrinsics(cam)
        camera_intrinsics[key] = K
        print(f"[Intrinsic] {key}:\n{K}\n")
    for episode_idx in episodes_to_run:
        if not simulation_app.is_running():
            break

        print(f"[Main] Starting episode {episode_idx}")
        world.reset()
        prim_mgr.clear()

        # Environment setup
        robot_xform.set_local_pose(
            translation=np.array(franka_translation) / stage_utils.get_stage_units(),
            orientation=np.array(franka_rotation)
        )
        set_camera_view(camera_translation, franka_translation)

        # Object configuration
        if object_poses_path and os.path.exists(object_poses_path):
            # Use modulo to reuse object layouts
            object_transforms = load_object_transforms_from_json(
                object_poses_path,
                episode_index=episode_idx % len(object_pose_records),
                aruco_tag_pose=aruco_tag_pose,

                cfg=cfg,
            )

            if len(object_transforms) == 0:
                print(f"[ObjectLoader] Skipping episode: {episode_idx} as objects are not constructed successfully.")
                continue

            for obj in object_transforms:
                object_name = _normalize_object_name(obj["object_name"])
                if object_name == "plate":
                    continue
                if object_name not in object_prims:
                    preload_entry = preload_by_name.get(object_name)
                    assert preload_entry, f"Object {object_name} missing from PRELOAD_OBJECTS"
                    asset_filename = preload_entry["assets"]
                    prim_path = preload_entry["prim_path"]
                    full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
                    if not os.path.exists(full_asset_path):
                        print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
                        continue

                    try:
                        stage_utils.add_reference_to_stage(
                            usd_path=full_asset_path,
                            prim_path=prim_path
                        )
                    except Exception as e:
                        print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
                        continue

                    obj_prim = RigidPrim(prim_path, object_name)
                    obj_prim.initialize()
                    object_prims[object_name] = obj_prim

                obj_prim = object_prims[object_name]
                # fix_rigid_object_physx(
                #     prim,
                #     static_friction=0.8,
                #     dynamic_friction=0.6,
                #     restitution=0.0
                # )

                # debug_usd_physics(prim)
                # debug_dynamic_state(prim)
                # debug_collision(stage, prim)
                # debug_mass_inertia(prim)
                # debug_physics_material(stage, prim)

                # rb_prim = find_rigid_body_prim(prim)
                # if rb_prim is None:
                #     print("❌ No RigidBody found under", prim.GetPath())
                # else:
                #     print("✅ RigidBody found at", rb_prim.GetPath())
                #     debug_usd_physics(rb_prim)

                # if "knife" in prim_path.lower()():
                #     obj_prim.set_world_pose(position=np.array([1.98354244, 4.99211439, 0.7627228]))
                # else:
                obj_pos = np.array(obj["position"], dtype=np.float64)

                # 預設 orientation
                orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                if args.task == "kitchen":
                    obj_pos[0] -= 0.05  # 🔥微調位置，讓物體不會太靠近牆面
                else:obj_pos[2] -= 0.25 # 🔥微調高度，避免物體懸空或穿地面
                if "fork" in object_name.lower():
                    orientation = np.array([0.707, 0.0, 0.0, 0.707])
                    print(f"[ObjectLoader] Positioned {object_name} with fork-specific orientation at {obj_pos}")
                elif "knife" in object_name.lower():
                    orientation = np.array([0.707, 0.0, 0.0, -0.707])
                    print(f"[ObjectLoader] Positioned {object_name} with knife-specific orientation at {obj_pos}")
                elif "blue_block" in object_name.lower():
                    orientation = np.array([0.707107, 0.707107, 0, 0])  # 45 degree yaw
                    print(f"[ObjectLoader] Positioned {object_name} with block-specific orientation at {obj_pos}")
                elif "red_block" in object_name.lower():
                    orientation = np.array([0.0677732, -0.7038514, -0.0677732, -0.7038514])  # 45 degree yaw
                    print(f"[ObjectLoader] Positioned {object_name} with block-specific orientation at {obj_pos}")
                elif "green_block" in object_name.lower():
                    orientation = np.array([0.5, 0.5, 0.5, -0.5])  # 45 degree yaw
                    print(f"[ObjectLoader] Positioned {object_name} with block-specific orientation at {obj_pos}")
                # 最後一次性設置
                obj_prim.set_world_pose(position=obj_pos, orientation=orientation)


        curr_pos, _ = get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver)

        #fork_origin, rot_quat_wxyz1 = get_object_pose("/World/fork")
        #knife_origin, rot_quat_wxyz2 = get_object_pose("/World/knife")

        #fork_rpy = quat_wxyz_to_rpy_deg(rot_quat_wxyz1)
        #knife_rpy = quat_wxyz_to_rpy_deg(rot_quat_wxyz2)

        #print("fork origin:", fork_origin)
        #print("fork RPY (deg): roll={:.2f}, pitch={:.2f}, yaw={:.2f}".format(*fork_rpy))

        #print("knife origin:", knife_origin)
        #print("knife RPY (deg): roll={:.2f}, pitch={:.2f}, yaw={:.2f}".format(*knife_rpy))

        teleop = TeleopEEController()
        get_object_world_pose = make_get_object_world_pose(prim_mgr)
        pickplace = PickPlace(
            get_end_effector_pose_fn=get_end_effector_pos_quat_wxyz,
            get_object_world_pose_fn=get_object_world_pose,
            apply_ik_solution_fn=apply_ik_solution,
            plan_line_cartesian_fn=plan_line_cartesian,
            teleop=teleop,
            world=world,
            task=args.task,
        )

        if args.task=="kitchen":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif args.task=="dining-room":
            INIT_EE_POS = curr_pos + np.array([-0.16, 0., 0.13])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        elif args.task=="living-room":
            INIT_EE_POS = curr_pos + np.array([-0.1, 0.2, 0.20])
            INIT_EE_QUAT_WXYZ = np.array([0.0081739, -0.9366365, 0.350194, 0.0030561])
        else:
            raise RuntimeError(f"Unknown task, expected one of 'kitchen', 'dining-room', 'living-room', got {args.task}")
        
        # [Data Augmentation] Add random noise to initial EE pose
        # Position noise: +/- 5cm
        pos_noise = np.random.uniform(-0.2, 0.2, size=3)
        #INIT_EE_POS = INIT_EE_POS + pos_noise

        # Rotation noise: +/- 5 degrees around random axis
        # Convert WXYZ -> SciPy (which uses x,y,z,w scalar_last usually, but we check input)
        # R.from_quat expects (x, y, z, w). INIT_EE_QUAT_WXYZ is w,x,y,z.
        w, x, y, z = INIT_EE_QUAT_WXYZ
        base_rot = R.from_quat([x, y, z, w])
        
        # Random axis-angle rotation
        rand_axis = np.random.normal(size=3)
        rand_axis /= np.linalg.norm(rand_axis)
        rand_angle_deg = np.random.uniform(-10, 10)
        rand_rot = R.from_rotvec(rand_axis * np.deg2rad(rand_angle_deg))
        
        # Apply noise
        final_rot = rand_rot * base_rot
        fx, fy, fz, fw = final_rot.as_quat() # x,y,z,w
        INIT_EE_QUAT_WXYZ = np.array([fw, fx, fy, fz]) # Back to w,x,y,z
        
        print(f"[Init] Random Noise Applied. Pos Noise: {pos_noise}, Angle Noise (deg): {rand_angle_deg:.2f}")

        
        # Motion planner initialization
        motion_planner = registry.get_motion_planner(
            args.task,
            cfg,
            get_object_world_pose_fn=get_object_world_pose,
            pickplace=pickplace,
        )

        # Initialize end-effector pose
        calibrate_robot_base(panda, lula_solver)
        success = apply_ik_solution(
            panda,
            art_kine_solver,
            INIT_EE_POS,
            INIT_EE_QUAT_WXYZ,
        )

        if not success:
            print("[Init] WARNING: Failed to apply EE initial pose")
        print("Gripper after open command:", gripper.get_joint_positions())
        for _ in range(30):
            world.step(render=True)

        rgb_dict_list = []
        extrinsics_dict_list = []
        eef_pos_list = []
        eef_rot_list = []
        gripper_list = []
        joint_pos_list = [] # <--- NEW
        eef_pose6d = None
        episode_start_pose = None
        episode_end_pose = None
        # --- 在 main 裡面，迴圈開始前建立目錄 ---
        debug_dir = os.path.join(args.session_dir, "debug_views")
        os.makedirs(debug_dir, exist_ok=True)
        while simulation_app.is_running():

            # Predefine motion planning to collect data
            motion_planner.step(panda, lula_solver, art_kine_solver)
            eef_pose6d = step_world_and_record(
                world,
                cameras,
                panda,
                lula_solver,
                art_kine_solver,
                rgb_dict_list,
                extrinsics_dict_list,
                eef_pos_list,
                eef_rot_list,
                gripper_list,
                joint_pos_list=joint_pos_list, # <--- NEW
                render=True,
            )
            # --- 新增：即時視覺化監看 ---
            """
            
            if rgb_dict_list:
                # 取得最後一幀的影像字典
                # 假設 rgb_dict_list 存儲的是每一幀各相機的圖像
                # 如果 step_world_and_record 直接回傳 rgb_dict 會更方便
                last_frame_wrist = rgb_dict_list[-1]["wrist"]
                last_frame_top = rgb_dict_list[-1]["top"]
                last_frame_angle = rgb_dict_list[-1]["angle"]

                # 將 RGB 轉為 BGR (OpenCV 格式)
                img_wrist = cv2.cvtColor(last_frame_wrist, cv2.COLOR_RGB2BGR)
                img_top = cv2.cvtColor(last_frame_top, cv2.COLOR_RGB2BGR)
                img_angle = cv2.cvtColor(last_frame_angle, cv2.COLOR_RGB2BGR)

                # 水平拼接三張影像 (Hconcat)
                # 因為你的分辨率都是 224x224，拼接後會是 672x224
                combined_view = cv2.hconcat([img_wrist, img_top, img_angle])

                # 加入文字標籤 (選配)
                cv2.putText(combined_view, f"Wrist | Top | Angle - Ep: {episode_idx}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 顯示視窗
                cv2.imshow("Multi-Camera Monitor", combined_view)
                cv2.waitKey(1) # 給予 1ms 時間刷新視窗
            """
            if episode_start_pose is None:
                episode_start_pose = eef_pose6d.copy()

            if motion_planner.is_done():
                episode_end_pose = eef_pose6d.copy()
                print("[Main] Motion plan finished")
                break

        if episode_end_pose is None and eef_pos_list:
            episode_end_pose = np.concatenate([eef_pos_list[-1], eef_rot_list[-1]])

        if not rgb_dict_list:
            print(f"[Main] WARNING: No frames captured for episode {episode_idx}")
            continue

        demo_start_list = np.repeat(episode_start_pose[None, :], len(rgb_dict_list), axis=0)
        demo_end_list = np.repeat(episode_end_pose[None, :], len(rgb_dict_list), axis=0)
        episode_record = {
            "episode_idx": episode_idx,
            "eef_pos": np.stack(eef_pos_list, 0),
            "eef_rot": np.stack(eef_rot_list, 0),
            "gripper": np.stack(gripper_list, 0),
            "joint_pos": np.stack(joint_pos_list, 0), # <--- NEW
            "demo_start": demo_start_list,
            "demo_end": demo_end_list,
        }

        # Save RGBs and Extrinsics for each camera
        for cam_key in cameras.keys():
            # RGB
            frames = [step[cam_key] for step in rgb_dict_list]
            episode_record[f"{cam_key}_rgb"] = np.stack(frames, 0)
            
            # Extrinsics (NEW)
            exts = [step[cam_key] for step in extrinsics_dict_list]
            episode_record[f"{cam_key}_extrinsics"] = np.stack(exts, 0)
        

        episode_success = is_episode_completed(episode_record)
        episode_record["success"] = episode_success

        episode_success = is_episode_completed(episode_record)
        episode_record["success"] = episode_success

        if episode_success:
            # 設定影片屬性 (寬度 = 224 * 3, 高度 = 224)
            video_path = os.path.join(debug_dir, f"episode_{episode_idx:04d}_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (224 * 2, 224))

            for frame_data in rgb_dict_list:
                w = cv2.cvtColor(frame_data["wrist"], cv2.COLOR_RGB2BGR)
                #t = cv2.cvtColor(frame_data["top"], cv2.COLOR_RGB2BGR)
                a = cv2.cvtColor(frame_data["angle"], cv2.COLOR_RGB2BGR)
                
                combined = cv2.hconcat([w, a])
                out.write(combined)
            
            out.release()
            print("[Main] Task success")
        else:
            print("[Main] Task fail")

        collected_episodes.append(episode_record)

        if episode_success:
            completed_episodes.add(episode_idx)
            _save_progress(args.session_dir, completed_episodes)

    successful_episodes = [
        ep for ep in collected_episodes
        if ep.get("success", False)
    ]
    print(f"[Main] Total successful trials collected: {len(successful_episodes)}")

    if successful_episodes:
        # 1️⃣ 先存「這一輪新產生的 dataset」
        new_dataset_path = os.path.join(
            args.session_dir, "simulation_dataset.zarr.zip"
        )
        save_multi_episode_dataset(new_dataset_path, successful_episodes)

        # 2️⃣ 再丟進「長期累積的 merged dataset」
        merged_dataset_path = os.path.join(
            args.session_dir, "simulation_dataset_merged.zarr.zip"
        )

        merger = SimulationDatasetAccumulator(
            merged_path=merged_dataset_path
        )

        merger.update_with(
            new_dataset_path=new_dataset_path
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
