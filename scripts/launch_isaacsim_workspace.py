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

from llvmlite.binding.transforms import passmanagers
import registry
import argparse
import numpy as np
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["kitchen", "dining-table", "living-room"], required=True)
parser.add_argument("--session_dir", type=str, default=None)
parser.add_argument("--episode", type=int, default=0)
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
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

from scipy.spatial.transform import Rotation as R
from object_loader import load_objects_from_json
import utils

# Import pure functions from umi_replay
from umi_replay import (
    load_umi_dataset,
    get_episode_bounds,
    compute_replay_step,
    set_gripper_width,
    visualize_waypoints,
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

PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2"
]

DRIVE_CONFIGS = {
    "panda_joint1": {"stiffness": 80.0, "damping": 4.0, "pos": 0.0},
    "panda_joint2": {"stiffness": 80.0, "damping": 4.0, "pos": -0.569},
    "panda_joint3": {"stiffness": 80.0, "damping": 4.0, "pos": 0.0},
    "panda_joint4": {"stiffness": 80.0, "damping": 4.0, "pos": -2.810},
    "panda_joint5": {"stiffness": 80.0, "damping": 4.0, "pos": 0.0},
    "panda_joint6": {"stiffness": 80.0, "damping": 4.0, "pos": 3.037},
    "panda_joint7": {"stiffness": 80.0, "damping": 4.0, "pos": 0.741},
    "panda_finger_joint1": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
    "panda_finger_joint2": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
}

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


def get_T_base_tag(aruco_tag_pose: dict) -> np.ndarray:
    """
    Compute ArUco tag pose relative to robot base frame.
    T_base_tag = inv(T_world_base) @ T_world_tag
    """
    time = Usd.TimeCode.Default()
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(time)

    base_prim = stage.GetPrimAtPath(FRANKA_PANDA_PRIM_PATH)
    T_W_base_gf = cache.GetLocalToWorldTransform(base_prim)
    
    T_W_base = np.array(T_W_base_gf)
    
    aruco_translation = np.array(aruco_tag_pose['translation'])
    aruco_quat_wxyz = np.array(aruco_tag_pose['rotation_quat'])
    aruco_quat_xyzw = np.array([aruco_quat_wxyz[1], aruco_quat_wxyz[2], aruco_quat_wxyz[3], aruco_quat_wxyz[0]])
    
    T_W_aruco = np.eye(4)
    T_W_aruco[:3, 3] = aruco_translation
    T_W_aruco[:3, :3] = R.from_quat(aruco_quat_xyzw).as_matrix()
    
    # Proper matrix multiplication with numpy
    T_base_tag = np.linalg.inv(T_W_base) @ T_W_aruco
    
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


def apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz, step_idx):
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
        joint_positions = np.array([*action.joint_positions, 0.0, 0.0])
        panda.set_joint_positions(joint_positions)
        return True

    return False

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
    }

def step_replay(replay_state: dict, panda, lula_solver, art_kine_solver, T_base_tag: np.ndarray):
    """
    Simplified step replay function with direct gripper control.

    Args:
        replay_state: Dictionary containing replay state
        panda: Robot articulation
        lula_solver: Lula kinematics solver
        art_kine_solver: Articulation kinematics solver
        T_base_tag: Transform from base to tag frame

    Returns:
        bool: True if replay should continue, False if finished
    """
    # Calibrate robot base on first run
    if not replay_state["calibrated"]:
        calibrate_robot_base(panda, lula_solver)
        replay_state["calibrated"] = True
        return True

    # Check if episode finished
    if replay_state["current_step"] >= replay_state["end_idx"]:
        replay_state["finished"] = True
        print(f"[Main] Episode {replay_state['episode_idx']} finished.")
        return False

    global DEBUG_DRAW
    # Get current trajectory target
    step_idx = replay_state["current_step"]
    data = replay_state["data"]
    target_pos, target_rot, target_quat_wxyz, gripper_width = compute_replay_step(
        data, step_idx, T_base_tag
    )

    draw_coordinate_frame(target_pos, target_rot, axis_length=0.05, draw_interface=DEBUG_DRAW)

    # Apply IK for trajectory following
    calibrate_robot_base(panda, lula_solver)
    success = apply_ik_solution(panda, art_kine_solver, target_pos, target_quat_wxyz, step_idx)

    if success:
        # Direct gripper control based on dataset
        set_gripper_width(panda, gripper_width, threshold=0.05, step=0.05)

        # Store waypoint for visualization
        replay_state["waypoints"].append((target_pos.copy(), target_rot))

        # Always advance to next step
        replay_state["current_step"] += 1
    else:
        print(f"[Main] IK failed at step {step_idx}, skipping...")
        replay_state["current_step"] += 1

    # Progress logging
    if step_idx % 100 == 0:
        progress = (step_idx - replay_state["start_idx"]) / (replay_state["end_idx"] - replay_state["start_idx"]) * 100
        gripper_state = "Open" if gripper_width > 0.05 else "Closed"
        print(f"[Main] Step {step_idx} ({progress:.1f}%) | Gripper: {gripper_state}")

    return True

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
    world.reset()

    # Set robot position after world reset
    robot_xform.set_local_pose(
        translation=np.array(franka_translation) / stage_utils.get_stage_units(),
        orientation=np.array(franka_rotation)
    )
    set_camera_view(camera_translation, franka_translation)

    # --- Initialize replay state ---
    replay_state = None
    lula_solver = None
    art_kine_solver = None
    T_base_tag = None
    
    if args.session_dir:
        # Load objects from session
        object_poses_path = os.path.join(args.session_dir, 'demos', 'mapping', 'object_poses.json')
        print(f"[Main] Looking for object poses at: {object_poses_path}")
        if os.path.exists(object_poses_path):
            print(f"[Main] Loading objects from: {object_poses_path}")
            load_objects_from_json(
                object_poses_path,
                ASSETS_DIR,
                world,
                episode_index=args.episode,
                aruco_tag_pose=aruco_tag_pose,
                cfg=cfg
            )

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
        T_base_tag = get_T_base_tag(aruco_tag_pose)
        
        # Create replay state for the episode
        replay_state = create_replay_state(args.session_dir, args.episode, cfg)
        print(f"[Main] Replay initialized. Episode {args.episode}: steps {replay_state['start_idx']} to {replay_state['end_idx']}")

    # --- Main simulation loop ---
    print("[Main] Starting simulation loop...")
    
    while simulation_app.is_running():
        world.step(render=True)
        time.sleep(0.01)
        
        # Process replay if active
        if replay_state is not None and not replay_state["finished"]:
            should_continue = step_replay(
                replay_state, panda, lula_solver, art_kine_solver, T_base_tag
            )

            if not should_continue:
                # Episode finished - visualize waypoints
                print("[Main] Replay finished. Visualizing waypoints...")
                visualize_waypoints(
                    replay_state["waypoints"],
                    episode_idx=replay_state["episode_idx"],
                    show_orientation=True,
                    orientation_scale=0.02,
                    save_path=os.path.join(args.session_dir, 'waypoints.png'),
                    dpi=150
                )

    simulation_app.close()


if __name__ == "__main__":
    main()
