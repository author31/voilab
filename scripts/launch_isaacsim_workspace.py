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

from numpy.random import beta
import registry
import argparse
import numpy as np
import time
import sys
import zarr
from zarr.storage import ZipStore
from numcodecs import Blosc

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["kitchen", "dining-table", "living-room"], required=True)
parser.add_argument("--session_dir", type=str, default=None)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--x_offset", type=float, default=0.1, help="X-axis offset for coordinate calibration (meters)")
parser.add_argument("--y_offset", type=float, default=0.15, help="Y-axis offset for coordinate calibration (meters)")
parser.add_argument("--z_offset", type=float, default=-0.07, help="Z-axis offset for coordinate calibration (meters)")
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
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

from scipy.spatial.transform import Rotation as R
from object_loader import load_object_transforms_from_json, map_object_name_to_asset
import utils
import lula

# Import pure functions from umi_replay
from umi_replay import (
    load_umi_dataset,
    get_episode_bounds,
    compute_replay_step,
    set_gripper_width,
    visualize_waypoints,
    linear_cartesian_path,
)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("[Main] ERROR: Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

enable_extension("isaacsim.robot_motion.motion_generation")

# --- Configuration ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm.usd"
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
def draw_coordinate_frame(
    target_pos: np.ndarray,
    target_rot,
    axis_length: float = 0.1,
    line_width: float = 3.0,
    draw_interface=None
):
    """
    Draw a 3D coordinate frame using debug draw lines.
    
    Args:
        target_pos: np.array (3,) position in base frame
        target_rot: scipy Rotation object representing orientation
        axis_length: Length of each axis in meters (default: 0.1m)
        line_width: Width of the drawn lines (default: 3.0)
        draw_interface: Debug draw interface (if None, will acquire)
    """
    # Acquire debug draw interface if not provided
    if draw_interface is None:
        draw_interface = _debug_draw.acquire_debug_draw_interface()
    
    # Get rotation matrix from scipy Rotation object
    rot_matrix = target_rot.as_matrix()  # Shape: (3, 3)
    
    # Define axis directions in local frame
    x_axis = np.array([axis_length, 0.0, 0.0])
    y_axis = np.array([0.0, axis_length, 0.0])
    z_axis = np.array([0.0, 0.0, axis_length])
    
    # Rotate axes to world frame
    x_axis_world = rot_matrix @ x_axis
    y_axis_world = rot_matrix @ y_axis
    z_axis_world = rot_matrix @ z_axis
    
    # Calculate endpoint positions
    x_end = target_pos + x_axis_world
    y_end = target_pos + y_axis_world
    z_end = target_pos + z_axis_world
    
    # Convert to tuples for draw_lines
    origin = tuple(target_pos)
    x_end_tuple = tuple(x_end)
    y_end_tuple = tuple(y_end)
    z_end_tuple = tuple(z_end)
    
    # Define colors (Red=X, Green=Y, Blue=Z)
    red = (1.0, 0.0, 0.0, 1.0)
    green = (0.0, 1.0, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)
    
    # Draw three axes
    draw_interface.draw_lines([origin], [x_end_tuple], [red], [line_width])
    draw_interface.draw_lines([origin], [y_end_tuple], [green], [line_width])
    draw_interface.draw_lines([origin], [z_end_tuple], [blue], [line_width])
    
    return draw_interface


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


def get_T_base_tag(aruco_tag_pose: dict) -> np.ndarray:
    """
    Compute ArUco tag pose relative to robot base frame.
    T_base_tag = inv(T_world_base) @ T_world_tag
    """
    T_world_base = get_T_world_base()
    
    aruco_translation = np.array(aruco_tag_pose['translation'])
    aruco_quat_wxyz = np.array(aruco_tag_pose['rotation_quat'])
    aruco_quat_xyzw = np.array([aruco_quat_wxyz[1], aruco_quat_wxyz[2], aruco_quat_wxyz[3], aruco_quat_wxyz[0]])
    
    T_world_aruco = np.eye(4)
    T_world_aruco[:3, 3] = aruco_translation
    T_world_aruco[:3, :3] = R.from_quat(aruco_quat_xyzw).as_matrix()

    T_base_tag = np.linalg.inv(T_world_base) @ T_world_aruco
    return T_base_tag


def calibrate_robot_base(panda, lula_solver):
    """
    Update Lula solver with current robot base pose.
    Must be called before computing IK.
    
    Args:
        panda: Panda articulation object
        lula_solver: LulaKinematicsSolver instance
    """
    robot_pos, robot_quat = panda.get_world_pose()
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


def get_object_world_pose(object_prim_path: str) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Object prim not found: {object_prim_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    T_gf = cache.GetLocalToWorldTransform(prim)
    return utils.gf_matrix4d_to_numpy(T_gf)


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

def create_lula_pose(position: np.ndarray, rotation: R) -> lula.Pose3:
    """
    Create Lula Pose3 from position and scipy Rotation.

    Args:
        position: np.array (3,) position in world frame
        rotation: scipy Rotation object representing orientation

    Returns:
        lula.Pose3 object for trajectory generation
    """
    quat_xyzw = rotation.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return lula.Pose3(lula.Rotation3(*quat_wxyz), position)


def calculate_side_grasp_rotation(approach_dir):
    """
    Calculate rotation matrix for side grasp approach.
    Ensures the gripper's Z-axis (approach direction) points towards the object.
    
    Args:
        approach_dir: Normalized direction vector from gripper to object
    
    Returns:
        scipy.spatial.transform.Rotation object
    """
    # Normalize approach direction
    z_axis = approach_dir / np.linalg.norm(approach_dir)
    
    # Choose a temporary up vector (world Z-axis)
    world_up = np.array([0.0, 0.0, 1.0])
    
    # If approach direction is nearly vertical, use Y-axis as reference
    if np.abs(np.dot(z_axis, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    
    # Calculate X-axis (perpendicular to both Z and world up)
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Calculate Y-axis (perpendicular to both X and Z)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Construct rotation matrix [x_axis, y_axis, z_axis]
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    return R.from_matrix(rot_matrix)


def execute_task_space_trajectory(
    panda,
    taskspace_trajectory_generator,
    task_space_spec,
    duration: float = 5.0,
    phase_name: str = "Trajectory"
) -> bool:
    """Refined execution with Lula-specific error checks."""
    try:
        # Generate the trajectory; returns None if unreachable [web:1]
        trajectory = taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            task_space_spec, "umi_tcp"
        )
        
        if trajectory is None:
            print(f"[Trajectory] {phase_name} failed: Pose unreachable by Lula.")
            return False

        physics_dt = 1/60 # Match physics rate for ArticulationTrajectory [web:1]
        articulation_trajectory = ArticulationTrajectory(panda, trajectory, physics_dt)
        
        # Get action sequence for deterministic execution
        actions = articulation_trajectory.get_action_sequence()
        for action in actions:
            panda.apply_action(action)
            # Ensure the simulation steps forward to process the action
            # The world.step() happens in the main loop, so we yield here if using async
            # In this sync script, we rely on the loop structure
        return True

    except Exception as e:
        print(f"[Trajectory] Error in {phase_name}: {e}")
        return False



def initialize_trajectory_generator(panda):
    """
    Initialize LulaTaskSpaceTrajectoryGenerator with existing robot config and articulation.

    Args:
        panda: Panda articulation object

    Returns:
        LulaTaskSpaceTrajectoryGenerator instance
    """
    print("[Init] Initializing LulaTaskSpaceTrajectoryGenerator...")
    try:
        taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
            urdf_path=LULA_URDF_PATH,
        )
        print("[Init] LulaTaskSpaceTrajectoryGenerator initialized successfully")
        return taskspace_trajectory_generator
    except Exception as e:
        print(f"[Init] Error initializing trajectory generator: {e}")
        return None


def create_replay_state(session_dir: str, episode_idx: int, cfg: dict):
    """
    Initialize simplified replay state for an episode.

    Args:
        session_dir: Path to session directory
        episode_idx: Episode index to replay

    Returns:
        dict: Replay state with essential tracking variables only
    """
    data, meta, episode_ends = load_umi_dataset(session_dir)
    start_idx, end_idx = get_episode_bounds(episode_ends, episode_idx)

    return {
        "data": data,
        "meta": meta,
        "episode_ends": episode_ends,
        "episode_idx": episode_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "current_step": start_idx,
        "calibrated": False,
        "waypoints": [],
        "finished": False,
        # ---- Intervention helpers ----
        "intervention_needed": False,
        "intervention_step": None,
        "saved_ee_pose": None,
    }


def step_replay(replay_state: dict, panda, lula_solver, art_kine_solver, T_world_aruco: np.ndarray, cfg: dict):
    # ---------- Calibration ----------
    if not replay_state["calibrated"]:
        calibrate_robot_base(panda, lula_solver)
        replay_state["calibrated"] = True
        return True

    # ---------- Finished? ----------
    if replay_state["current_step"] >= replay_state["end_idx"]:
        replay_state["finished"] = True
        print(f"[Main] Episode {replay_state['episode_idx']} finished.")
        return False

    # ---------- Intervention flag ----------
    if replay_state["intervention_needed"]:
        # The main loop will call `run_intervention` – we just return.
        return True

    # ---------- Normal replay ----------
    step_idx = replay_state["current_step"]
    data = replay_state["data"]

    # Prepare coordinate offsets
    offsets = {
        'x': args.x_offset,
        'y': args.y_offset,
        'z': args.z_offset
    }

    target_pos, target_rot, target_quat_wxyz, gripper_width = compute_replay_step(
        data, step_idx, T_world_aruco, offsets
    )

    target_obj_path = cfg["environment_vars"]["TARGET_OBJECT_PATH"]
    T_world_obj = get_object_world_pose(target_obj_path)
    obj_pos = T_world_obj[:3, 3]
    dist_to_obj = np.linalg.norm(target_pos - obj_pos)
    print(f"[Main] Distance to object {target_obj_path}: {dist_to_obj}")

    # If we are still farther than 10 cm, keep following the recorded demo
    if dist_to_obj > 0:
        draw_coordinate_frame(target_pos, target_rot, axis_length=0.05, draw_interface=DEBUG_DRAW)
        calibrate_robot_base(panda, lula_solver)
        success = apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz)

        if success:
            set_gripper_width(panda, gripper_width, threshold=0.05, step=0.05)
            replay_state["waypoints"].append((target_pos.copy(), target_rot))
            replay_state["current_step"] += 1
        else:
            print(f"[Main] IK failed at step {step_idx}, skipping...")
            replay_state["current_step"] += 1

        if step_idx % 100 == 0:
            progress = (step_idx - replay_state["start_idx"]) / (replay_state["end_idx"] - replay_state["start_idx"]) * 100
            gripper_state = "Open" if gripper_width > 0.05 else "Closed"
            print(f"[Main] Step {step_idx} ({progress:.1f}%) | Gripper: {gripper_state}")

        return True

    # replay_state["current_step"] += 1
    print(f"[Main] *** Intervention triggered at step {step_idx} (dist={dist_to_obj:.3f} m) ***")
    replay_state["intervention_needed"] = True
    replay_state["intervention_step"] = step_idx
    replay_state["saved_ee_pose"] = (target_pos, target_rot)
    return True


# ----------------------------------------------------------------------
# Intervention phase – straight‑line Cartesian approach + grasp
# ----------------------------------------------------------------------
def run_intervention_with_retry(replay_state, panda, lula_solver, art_kine_solver, cfg, T_base_tag: np.ndarray, taskspace_trajectory_generator, max_attempts: int = 3):
    """
    Executes improved grasping with optimal orientation and multi-angle retry logic.
    Called only when replay_state['intervention_needed'] is True.
    """
    for attempt in range(max_attempts):
        try:
            success = run_intervention(replay_state, panda, art_kine_solver, cfg, taskspace_trajectory_generator)

            if success:
                print(f"[Intervention] Attempt {attempt + 1} completed successfully")
                return True
            else:
                print(f"[Intervention] Attempt {attempt + 1} failed to grasp")

        except Exception as e:
            print(f"[Intervention] Attempt {attempt + 1} failed with error: {e}")

        # Reset for next attempt
        if attempt < max_attempts - 1:
            print(f"[Intervention] Resetting for retry...")
            # Open gripper fully and move back slightly
            set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)
            time.sleep(0.5)

    print(f"[Intervention] All {max_attempts} attempts failed")
    return False



def draw_grasp_debug_visualization(pregrasp_pos, pregrasp_rot, grasp_pos, grasp_rot, object_name):
    """
    Draw visual debugging for grasp poses to help verify alignment.

    Args:
        pregrasp_pos: Pre-grasp position (3,)
        pregrasp_rot: Pre-grasp orientation (scipy Rotation)
        grasp_pos: Final grasp position (3,)
        grasp_rot: Final grasp orientation (scipy Rotation)
        object_name: Name of target object (for debugging)
    """
    print(f"\n[Debug] Visualizing grasp poses for {object_name}:")
    print(f"  Pre-grasp position: {pregrasp_pos}")
    print(f"  Final grasp position: {grasp_pos}")
    print(f"  Approach distance: {np.linalg.norm(grasp_pos - pregrasp_pos):.3f}m")

    # Draw pre-grasp pose with yellow coordinate frame
    draw_coordinate_frame(pregrasp_pos, pregrasp_rot, axis_length=0.05, draw_interface=DEBUG_DRAW)

    # Draw final grasp pose with orange coordinate frame (scaled slightly smaller)
    draw_coordinate_frame(grasp_pos, grasp_rot, axis_length=0.03, draw_interface=DEBUG_DRAW)

    # Draw approach direction arrow
    approach_dir = grasp_pos - pregrasp_pos
    approach_dir_norm = approach_dir / np.linalg.norm(approach_dir)
    arrow_end = pregrasp_pos + approach_dir_norm * 0.1  # 10cm arrow

    # Draw arrow line in purple
    purple = (0.5, 0.0, 0.5, 1.0)
    DEBUG_DRAW.draw_lines(
        [tuple(pregrasp_pos)],
        [tuple(arrow_end)],
        [purple],
        [5.0]  # Line width
    )

    # Add a small delay to visualize
    time.sleep(0.5)


def run_intervention(replay_state, panda, art_kine_solver, cfg, taskspace_trajectory_generator):
    """Refined 4-phase grasp logic with proper end-effector orientation towards object."""
    
    # Get current gripper pose and target object pose
    cur_pos, cur_rot = panda.gripper.get_world_pose()
    target_obj_prim_path = cfg["environment_vars"]["TARGET_OBJECT_PATH"]
    T_world_obj = get_object_world_pose(target_obj_prim_path)
    obj_size = get_object_world_size(target_obj_prim_path)
    height = obj_size[1]/2
    
    # Target is the CENTER of the cup
    cup_center = T_world_obj[:3, 3]
    cup_center[2] -= height
    draw_coordinate_frame(cup_center, R.from_euler("xyz", [0,0,0], degrees=True), axis_length=0.1, draw_interface=DEBUG_DRAW)
    
    # Calculate approach direction: vector FROM current position TO cup center
    approach_vector = cup_center - cur_pos
    approach_distance = np.linalg.norm(approach_vector)
    
    if approach_distance < 1e-6:
        print("[Main] Error: Gripper already at target position")
        return False
    
    # Normalize to get unit direction vector pointing at cup center
    approach_dir = approach_vector / approach_distance
    
    # Calculate grasp rotation that points the gripper towards the cup center
    grasp_rot = calculate_side_grasp_rotation(approach_dir)
    grasp_rot_xyzw = grasp_rot.as_quat()
    grasp_rot_wxyz = grasp_rot_xyzw[[3, 0, 1, 2]]
    
    # Define phase positions with clear offsets FROM cup center
    pregrasp_offset = 0.08   # 8cm away from cup center (pointing at it)
    approach_offset = 0.03   # 3cm away from cup center
    lift_height = 0.20       # Height to lift after grasp

    
    # Phase 1: PreGrasp - EEF 8cm away from cup center, pointing directly at it
    pregrasp_pos = cup_center - approach_dir * pregrasp_offset

    draw_coordinate_frame(pregrasp_pos, grasp_rot, axis_length=0.1, draw_interface=DEBUG_DRAW)
    
    # Phase 2: Approach - Move closer to 3cm from cup center along same direction
    approach_pos = cup_center - approach_dir * approach_offset
    
    # Phase 3: Grasp - Contact position (same as approach, but close gripper)
    grasp_pos = approach_pos.copy()
    
    # Phase 4: Lift - Move straight up from grasp position
    lift_pos = grasp_pos + np.array([0.0, 0.0, lift_height])
    
    # Define all phases with names and target poses
    phases = [
        ("PreGrasp", pregrasp_pos, grasp_rot_wxyz, False),   # 8cm from cup center, pointing at it
        ("Approach", approach_pos, grasp_rot_wxyz, False),   # 3cm from cup center
        ("Grasp", grasp_pos, grasp_rot_wxyz, True),          # Close gripper
        ("Lift", lift_pos, grasp_rot_wxyz, False),           # Lift object
    ]
    
    print(f"[Main] Cup center position: {cup_center}")
    print(f"[Main] Approach direction (gripper→cup): {approach_dir}")
    print(f"[Main] PreGrasp distance from cup center: {pregrasp_offset}m (8cm)")
    
    # Execute each phase
    for name, pos, rot, should_close_gripper in phases:
        print(f"\n[Main] === Phase: {name} ===")
        print(f"[Main] Target pos: {pos}")
        print(f"[Main] Target rot (wxyz): {rot}")
        
        # Attempt IK solution
        success = apply_ik_solution(panda, art_kine_solver, pos, rot)
        print(f"[Main] IK success: {success}")
        
        if not success:
            print(f"[Main] Warning: {name} phase failed, continuing...")
            continue
        
        # Allow simulation to settle
        for _ in range(10):
            simulation_app.update()
        
        # Close gripper during grasp phase
        if should_close_gripper:
            print("[Main] Closing gripper...")
            set_gripper_width(panda, width=0.0, threshold=0.02, step=0.005)
            print("[Main] Gripper closed")

        time.sleep(1)
    
    # Mark intervention as complete
    replay_state["finished"] = True
    print("\n[Main] Intervention complete!")
    return True


def get_end_effector_pose(panda, lula_solver, art_kine_solver) -> np.ndarray:
    base_pos, base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(
        robot_position=base_pos,
        robot_orientation=base_quat,
    )
    ee_pos, ee_rot_matrix = art_kine_solver.compute_end_effector_pose()
    eef_rot = R.from_matrix(ee_rot_matrix[:3, :3]).as_rotvec()
    return np.concatenate([ee_pos.astype(np.float64), eef_rot.astype(np.float64)])


def save_dataset(
    output_path: str,
    rgb_list,
    eef_pos_list,
    eef_rot_list,
    gripper_list,
    demo_start_list,
    demo_end_list,
):
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = ZipStore(output_path, mode="w")
    root = zarr.group(store)
    data = root.create_group("data")

    data.create_dataset(
        "camera0_rgb",
        data=np.stack(rgb_list, 0).astype(np.uint8),
        compressor=compressor,
    )
    data.create_dataset(
        "robot0_demo_start_pose",
        data=np.stack(demo_start_list, 0).astype(np.float64),
        compressor=compressor,
    )
    data.create_dataset(
        "robot0_demo_end_pose",
        data=np.stack(demo_end_list, 0).astype(np.float64),
        compressor=compressor,
    )
    data.create_dataset(
        "robot0_eef_pos",
        data=np.stack(eef_pos_list, 0).astype(np.float32),
        compressor=compressor,
    )
    data.create_dataset(
        "robot0_eef_rot_axis_angle",
        data=np.stack(eef_rot_list, 0).astype(np.float32),
        compressor=compressor,
    )
    data.create_dataset(
        "robot0_gripper_width",
        data=np.stack(gripper_list, 0).astype(np.float32),
        compressor=compressor,
    )
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([len(rgb_list)]))
    store.close()
    print("[SAVE] replay_dataset.zarr.zip saved at:", output_path)


def save_multi_episode_dataset(output_path: str, episodes: list[dict]) -> None:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = ZipStore(output_path, mode="w")
    root = zarr.group(store)
    data = root.create_group("data")

    rgb = np.concatenate([ep["rgb"] for ep in episodes], axis=0).astype(np.uint8)
    demo_start = np.concatenate([ep["demo_start"] for ep in episodes], axis=0).astype(np.float64)
    demo_end = np.concatenate([ep["demo_end"] for ep in episodes], axis=0).astype(np.float64)
    eef_pos = np.concatenate([ep["eef_pos"] for ep in episodes], axis=0).astype(np.float32)
    eef_rot = np.concatenate([ep["eef_rot"] for ep in episodes], axis=0).astype(np.float32)
    gripper = np.concatenate([ep["gripper"] for ep in episodes], axis=0).astype(np.float32)

    data.create_dataset("camera0_rgb", data=rgb, compressor=compressor)
    data.create_dataset("robot0_demo_start_pose", data=demo_start, compressor=compressor)
    data.create_dataset("robot0_demo_end_pose", data=demo_end, compressor=compressor)
    data.create_dataset("robot0_eef_pos", data=eef_pos, compressor=compressor)
    data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot, compressor=compressor)
    data.create_dataset("robot0_gripper_width", data=gripper, compressor=compressor)

    episode_lengths = [len(ep["rgb"]) for ep in episodes]
    episode_ends = np.cumsum(episode_lengths)
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=episode_ends)
    store.close()
    print("[SAVE] replay_dataset.zarr.zip saved at:", output_path)


def is_episode_completed(episode_record: dict) -> bool:
    return True


def _load_progress(session_dir: str) -> set[int]:
    progress_path = os.path.join(session_dir, ".previous_progress.json")
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
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
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
    camera = Camera(
        prim_path=f"{GOPRO_PRIM_PATH}/Camera",
        name="gopro_camera",
    )
    camera.initialize()
    world.reset()

    # --- Initialize replay state ---
    replay_state = None
    lula_solver = None
    art_kine_solver = None
    taskspace_trajectory_generator = None
    T_base_tag = None
    T_world_aruco = None
    object_prims = {}
    object_poses_path = None

    if args.session_dir is None:
        print("[Main] ERROR: session_dir is required for multi-episode replay.")
        simulation_app.close()
        return

    object_poses_path = os.path.join(args.session_dir, 'demos', 'mapping', 'object_poses.json')
    print(f"[Main] Looking for object poses at: {object_poses_path}")
    preload_objects = cfg.get("environment_vars", {}).get("PRELOAD_OBJECTS", [])
    for entry in preload_objects:
        if isinstance(entry, str):
            raw_name = entry
            asset_filename = map_object_name_to_asset(raw_name)
            prim_path = f"/World/{_normalize_object_name(raw_name)}"
        elif isinstance(entry, dict):
            raw_name = entry.get("name", "")
            asset_filename = entry.get("assets") or map_object_name_to_asset(raw_name)
            prim_path = entry.get("prim_path", f"/World/{_normalize_object_name(raw_name)}")
        else:
            continue

        object_name = _normalize_object_name(raw_name)
        if not raw_name or not asset_filename:
            print(f"[ObjectLoader] WARNING: Invalid PRELOAD_OBJECTS entry: {entry}")
            continue
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

        obj_prim = SingleXFormPrim(prim_path=prim_path, name=object_name)
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

    # Compute transform from robot base to ArUco tag
    # T_base_tag = get_T_base_tag(aruco_tag_pose)
    T_world_aruco = get_T_world_aruco(aruco_tag_pose)

    data, meta, episode_ends = load_umi_dataset(args.session_dir)
    total_episodes = len(episode_ends)
    print(f"[Main] Replay initialized for {total_episodes} episodes.")

    # --- Main simulation loop ---
    print("[Main] Starting simulation loop...")

    completed_episodes = _load_progress(args.session_dir)
    episodes_to_run = [ep for ep in range(total_episodes) if ep not in completed_episodes]
    collected_episodes = []

    for episode_idx in episodes_to_run:
        if not simulation_app.is_running():
            break

        print(f"[Main] Starting episode {episode_idx}")
        world.reset()
        robot_xform.set_local_pose(
            translation=np.array(franka_translation) / stage_utils.get_stage_units(),
            orientation=np.array(franka_rotation)
        )
        set_camera_view(camera_translation, franka_translation)
        if object_poses_path and os.path.exists(object_poses_path):
            object_transforms = load_object_transforms_from_json(
                object_poses_path,
                episode_index=episode_idx,
                aruco_tag_pose=aruco_tag_pose,
                cfg=cfg,
            )

            if len(object_transforms) == 0:
                print(f"[ObjectLoader] Skipping episode: {episode_idx} as objects are not constructed successfully.")
                continue

            for obj in object_transforms:
                object_name = _normalize_object_name(obj["object_name"])
                if object_name not in object_prims:
                    asset_filename = map_object_name_to_asset(object_name)
                    full_asset_path = os.path.join(ASSETS_DIR, asset_filename)
                    if not os.path.exists(full_asset_path):
                        print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
                        continue

                    prim_path = f"/World/{object_name}"
                    try:
                        stage_utils.add_reference_to_stage(
                            usd_path=full_asset_path,
                            prim_path=prim_path
                        )
                    except Exception as e:
                        print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
                        continue

                    obj_prim = SingleXFormPrim(prim_path=prim_path, name=object_name)
                    world.scene.add(obj_prim)
                    object_prims[object_name] = obj_prim

                obj_prim = object_prims[object_name]
                obj_prim.set_world_pose(position=obj["position"])
                print(f"[ObjectLoader] Positioned {object_name} at {obj['position']}")

        replay_state = create_replay_state(args.session_dir, episode_idx, cfg)
        print(f"[Main] Replay initialized. Episode {episode_idx}: steps {replay_state['start_idx']} to {replay_state['end_idx']}")

        rgb_list = []
        eef_pos_list = []
        eef_rot_list = []
        gripper_list = []
        episode_start_pose = None
        episode_end_pose = None

        while simulation_app.is_running():
            world.step(render=True)
            time.sleep(0.01)
            set_gripper_width(panda, 0.04)

            if replay_state is not None and not replay_state["finished"]:
                if replay_state["intervention_needed"]:
                    run_intervention_with_retry(replay_state, panda, lula_solver, art_kine_solver, cfg, T_base_tag, taskspace_trajectory_generator)
                else:
                    step_replay(
                        replay_state, panda, lula_solver, art_kine_solver, T_world_aruco, cfg
                    )
                    if replay_state["finished"]:
                        print("[Main] Replay finished. Visualizing waypoints...")
                        visualize_waypoints(
                            replay_state["waypoints"],
                            episode_idx=replay_state["episode_idx"],
                            show_orientation=True,
                            orientation_scale=0.02,
                            save_path=os.path.join(args.session_dir, 'waypoints.png'),
                            dpi=150
                        )

            img = camera.get_rgb()
            if img is not None:
                rgb_list.append(img)

            eef_pose6d = get_end_effector_pose(panda, lula_solver, art_kine_solver)
            eef_pos_list.append(eef_pose6d[:3])
            eef_rot_list.append(eef_pose6d[3:])

            joint_pos = panda.get_joint_positions()
            gripper_width = joint_pos[-2] + joint_pos[-1]
            gripper_list.append([gripper_width])

            if episode_start_pose is None:
                episode_start_pose = eef_pose6d.copy()

            if replay_state["finished"]:
                episode_end_pose = eef_pose6d.copy()
                break

        if episode_end_pose is None and eef_pos_list:
            episode_end_pose = np.concatenate([eef_pos_list[-1], eef_rot_list[-1]])

        if not rgb_list:
            print(f"[Main] WARNING: No frames captured for episode {episode_idx}")
            continue

        demo_start_list = np.repeat(episode_start_pose[None, :], len(rgb_list), axis=0)
        demo_end_list = np.repeat(episode_end_pose[None, :], len(rgb_list), axis=0)
        episode_record = {
            "episode_idx": episode_idx,
            "rgb": np.stack(rgb_list, 0),
            "eef_pos": np.stack(eef_pos_list, 0),
            "eef_rot": np.stack(eef_rot_list, 0),
            "gripper": np.stack(gripper_list, 0),
            "demo_start": demo_start_list,
            "demo_end": demo_end_list,
        }
        collected_episodes.append(episode_record)

        if is_episode_completed(episode_record):
            completed_episodes.add(episode_idx)
            _save_progress(args.session_dir, completed_episodes)

    successful_episodes = [ep for ep in collected_episodes if is_episode_completed(ep)]
    if successful_episodes:
        output_zarr = os.path.join(args.session_dir, "simulation_dataset.zarr.zip")
        save_multi_episode_dataset(output_zarr, successful_episodes)

    simulation_app.close()


if __name__ == "__main__":
    main()
