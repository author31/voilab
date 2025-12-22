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
from motion_plan import KitchenMotionPlanner, DiningRoomMotionPlanner, LivingRoomMotionPlanner
from pxr import UsdPhysics
from umi_replay import set_gripper_width

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


def get_end_effector_pose(panda, lula_solver, art_kine_solver) -> np.ndarray:
    base_pos, base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(
        robot_position=base_pos,
        robot_orientation=base_quat,
    )
    ee_pos, ee_rot_matrix = art_kine_solver.compute_end_effector_pose()
    eef_rot = R.from_matrix(ee_rot_matrix[:3, :3]).as_rotvec()
    return np.concatenate([ee_pos.astype(np.float64), eef_rot.astype(np.float64)])


def get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver):
    base_pos, base_quat = panda.get_world_pose()
    lula_solver.set_robot_base_pose(robot_position=base_pos, robot_orientation=base_quat)

    ee_pos, ee_T = art_kine_solver.compute_end_effector_pose()  # ee_T[:3,:3] rotation
    quat_xyzw = R.from_matrix(ee_T[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return ee_pos.astype(np.float64), quat_wxyz.astype(np.float64)


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

def step_world_and_record(
    world,
    camera,
    panda,
    lula_solver,
    art_kine_solver,
    rgb_list,
    eef_pos_list,
    eef_rot_list,
    gripper_list,
    render=True,
    sleep_dt=0.01,
    ):
    world.step(render=render)
    time.sleep(sleep_dt)

    # RGB
    img = camera.get_rgb()
    if img is not None:
        rgb_list.append(img)

    # End-effector pose
    eef_pose6d = get_end_effector_pose(panda, lula_solver, art_kine_solver)
    eef_pos_list.append(eef_pose6d[:3])
    eef_rot_list.append(eef_pose6d[3:])

    # Gripper
    joint_pos = panda.get_joint_positions()
    gripper_width = joint_pos[-2] + joint_pos[-1]
    gripper_list.append([gripper_width])

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

    with open(object_poses_path, "r") as f:
        object_pose_records = json.load(f)
    total_episodes = len(object_pose_records)
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

        # Environment setup
        robot_xform.set_local_pose(
            translation=np.array(franka_translation) / stage_utils.get_stage_units(),
            orientation=np.array(franka_rotation)
        )
        set_camera_view(camera_translation, franka_translation)
        _set_fixed_objects_for_episode(cfg, object_prims)

        # Object configuration
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

        # Make simulation settle
        for _ in range(50):
            set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)
            world.step(render=True)
            time.sleep(1 / 60)

        # Motion planner initialization
        if args.task=="kitchen":
            motion_planner = KitchenMotionPlanner(
                cfg,
                get_end_effector_pose_fn=get_end_effector_pos_quat_wxyz,
                get_object_world_pose_fn=get_object_world_pose,
                apply_ik_solution_fn=apply_ik_solution,
            )
        elif args.task=="dining-room":
            motion_planner = DiningRoomMotionPlanner(cfg)
        elif args.task=="living-room":
            motion_planner = LivingRoomMotionPlanner(cfg)
        else:
            print(f"[Error] Wrong task name.")
        
        rgb_list = []
        eef_pos_list = []
        eef_rot_list = []
        gripper_list = []
        episode_start_pose = None
        episode_end_pose = None

        while simulation_app.is_running():

            # Predefine motion planning to collect data
            motion_planner.step(panda, lula_solver, art_kine_solver)

            eef_pose6d = step_world_and_record(
                world,
                camera,
                panda,
                lula_solver,
                art_kine_solver,
                rgb_list,
                eef_pos_list,
                eef_rot_list,
                gripper_list,
                render=True,
            )

            if episode_start_pose is None:
                episode_start_pose = eef_pose6d.copy()
            
            if motion_planner.is_done():
                episode_end_pose = eef_pose6d.copy()
                print(f"[Main] Motion plan finished")
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
    print(f"[Main] Total successful trials collected: {len(successful_episodes)}")
    if successful_episodes:
        output_zarr = os.path.join(args.session_dir, "simulation_dataset.zarr.zip")
        save_multi_episode_dataset(output_zarr, successful_episodes)

    simulation_app.close()


if __name__ == "__main__":
    main()
