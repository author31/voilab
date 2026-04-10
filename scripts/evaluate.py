"""
Isaac Sim Workspace Launcher with UMI Trajectory Replay.

This script initializes the Isaac Sim environment, loads the robot and scene,
and replays UMI dataset trajectories. All business logic is exposed here.

Architecture:
- One simulation_app instance per episode
- Explicit state management (no hidden class state)
- Pure function calls to umi_replay module
"""

import sys
sys.path.append(".venv/lib/python3.10/site-packages")
sys.path.append("packages/diffusion_policy/src")
sys.path.append("packages/umi/src")

import os
import json
import registry
import argparse
import numpy as np
import time
import zarr
from zarr.storage import ZipStore
from numcodecs import Blosc

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--render", action='store_true')
parser.add_argument("--policy_control", action='store_true')
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_length", type=int, default=300)
parser.add_argument("--session_dir", type=str, default=None)
parser.add_argument("--task", type=str, choices=["kitchen", "dining-room", "living-room"], default="kitchen")
parser.add_argument("--x_offset", type=float, default=0.1, help="X-axis offset for coordinate calibration (meters)")
parser.add_argument("--y_offset", type=float, default=0.15, help="Y-axis offset for coordinate calibration (meters)")
parser.add_argument("--z_offset", type=float, default=-0.07, help="Z-axis offset for coordinate calibration (meters)")
args = parser.parse_args()

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_umi_action, get_real_umi_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from umi.real_world.real_inference_util import get_real_umi_action, get_real_umi_obs_dict

import torch
import dill
import hydra
# exit()

from isaacsim.simulation_app import SimulationApp

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
from isaacsim.core.prims import RigidPrim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

from scipy.spatial.transform import Rotation as R, Slerp
from object_loader import load_object_transforms_from_json, map_object_name_to_asset
import utils
import lula
from pxr import UsdPhysics
from umi_replay import set_gripper_width
from motion_plan import PickPlace



assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("[Main] ERROR: Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

enable_extension("isaacsim.robot_motion.motion_generation")

# --- Configuration ---
BASE_SCENE_FP = "assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "assets/franka_panda/franka_panda_arm.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"
ASSETS_DIR = "assets/CADs"

# CORRECTED: Use these paths in the solver initialization
LULA_ROBOT_DESCRIPTION_PATH = "assets/lula/frank_umi_descriptor.yaml"
LULA_URDF_PATH = "assets/franka_panda/franka_panda_umi-isaacsim.urdf"
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
    world.step(render=True)
    if render:
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

    if args.session_dir is not None:
        object_poses_path = os.path.join(args.session_dir, 'demos', 'mapping', 'object_poses.json')
    else:
        object_poses_path = None

    print(f"[Main] Configuration: {cfg}")
    # load checkpoint
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill)
    omega_cfg = payload['cfg']
    instance = hydra.utils.get_class(omega_cfg._target_)
    workspace: BaseWorkspace = instance(omega_cfg, output_dir='./data/eval_outputs')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if omega_cfg.training.use_ema:
        policy = workspace.ema_model

    policy.cuda()
    policy.eval()

    franka_pose = omega_cfg.get("franka_pose", {})
    franka_translation = franka_pose.get("translation", [0, 0, 0])
    franka_rotation = franka_pose.get("rotation_quat", [0, 0, 0, 1])
    aruco_tag_pose = omega_cfg.get("aruco_tag_pose", {})
    camera_translation = omega_cfg.get("camera_pose", {}).get("translation", [0, 0, 0])

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
        resolution=(224,224)
    )
    camera.initialize()
    world.reset()
    prim_mgr = RigidPrimManager()

    lula_solver = None
    art_kine_solver = None
    object_prims = {}

    # input("Press Enter to continue...")

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

        obj_prim = SingleXFormPrim(prim_path=prim_path, name=object_name, orientation=entry.get("quat_wxyz", np.array([1,0,0,0])))
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

    total_episodes = args.num_episodes

    # --- Main simulation loop ---
    print("[Main] Starting simulation loop...")

    collected_episodes = []

    for episode_idx in range(total_episodes):
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
            print("Loading object poses from:", object_poses_path)
            object_transforms = load_object_transforms_from_json(
                object_poses_path,
                episode_index=episode_idx,
                aruco_tag_pose=aruco_tag_pose,
                cfg=cfg,
            )
        else:
            def sample_random_pos(range_xy=(0.15, 0.15), height=1.05):
                x_offset = np.random.uniform(-range_xy[0], range_xy[0])
                y_offset = np.random.uniform(-range_xy[1], range_xy[1])
                return [5 + x_offset, 2.7 + y_offset, height]

            blue_cup_pos = sample_random_pos()
            pink_cup_pos = sample_random_pos()
            while np.linalg.norm(np.array(blue_cup_pos) - np.array(pink_cup_pos)) < 0.1:
                blue_cup_pos = sample_random_pos()
                pink_cup_pos = sample_random_pos()

            object_transforms = [
                dict(object_name="blue_cup", position=blue_cup_pos),
                dict(object_name="pink_cup", position=pink_cup_pos),
            ]

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
            obj_pos = np.array(obj["position"], dtype=np.float64)
            obj_prim.set_world_pose(position=obj_pos)
            print(f"[ObjectLoader] Positioned {object_name} at {obj_pos}")

        # Make simulation settle
        # for _ in range(10000):
        for _ in range(100):
            set_gripper_width(panda, width=0.1, threshold=0.0, step=0.05)
            world.step(render=False)
            # world.step(render=args.render)
            # if args.render:
            #     time.sleep(1 / 60)

        curr_pos, _ = get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver)
        get_object_world_pose = make_get_object_world_pose(prim_mgr)
        pickplace = PickPlace(
            get_end_effector_pose_fn=get_end_effector_pos_quat_wxyz,
            get_object_world_pose_fn=get_object_world_pose,
            apply_ik_solution_fn=apply_ik_solution,
            plan_line_cartesian_fn=plan_line_cartesian,
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
        
        rgb_list = []
        eef_pos_list = []
        eef_rot_list = []
        gripper_list = []
        eef_pose6d = None
        episode_start_pose = None
        episode_end_pose = None
        actions = None
        action_t = 0

        motion_planner.step(panda, lula_solver, art_kine_solver)

        while simulation_app.is_running():

            # # Predefine motion planning to collect data
            # if not args.policy_control:
            #     motion_planner.step(panda, lula_solver, art_kine_solver)

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
                render=args.render,
            )
            if episode_start_pose is None:
                episode_start_pose = eef_pose6d.copy()

            if motion_planner.is_done():
                episode_end_pose = eef_pose6d.copy()
                print("[Main] Motion plan finished")
                break

            t = len(rgb_list)
            if t > args.max_length:
                print(f"[Main] Reached max episode length: {args.max_length}")
                break

            if t == 4 or (t > 0 and t % 45 == 0):
                obs_dict = dict(
                    camera0_rgb=np.stack([rgb_list[-4], rgb_list[-1]]),
                    robot0_eef_pos=np.stack([eef_pos_list[-4], eef_pos_list[-1]]),
                    robot0_eef_rot_axis_angle=np.stack([eef_rot_list[-4], eef_rot_list[-1]]),
                    robot0_gripper_width=np.stack([gripper_list[-4], gripper_list[-1]]),
                )
                umi_obs_dict = get_real_umi_obs_dict(
                    obs_dict, omega_cfg.task.shape_meta, omega_cfg.task.pose_repr.obs_pose_repr,
                    episode_start_pose=[episode_start_pose]
                )
                umi_obs_dict = dict_apply(umi_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).cuda())
                # print(umi_obs_dict)
                with torch.no_grad():
                    umi_action = policy.predict_action(umi_obs_dict)['action']
                # print(umi_action)
                actions = get_real_umi_action(
                    umi_action.squeeze(0).cpu().numpy(),
                    obs_dict,
                    action_pose_repr=omega_cfg.task.pose_repr.action_pose_repr,
                )
                action_t = 1
                # print(action)

            if actions is not None:
                action_idx = action_t // 3
                DEBUG_DRAW.clear_points()
                draw_actions = actions[action_idx:, :3].tolist()
                DEBUG_DRAW.draw_points(draw_actions, [(1, 0, 0, 1)] * len(draw_actions), [10] * len(draw_actions))
                DEBUG_DRAW.draw_points(eef_pos_list[-1:], [(0, 1, 0, 1)], [10])

                if args.policy_control:
                    action = actions[action_idx]

                    gripper_open_rate = (action[-1] - pickplace.gripper_close_thresh) / (pickplace.gripper_open_thresh - pickplace.gripper_close_thresh)
                    # print(gripper_open_rate, action[-1])
                    if not pickplace.attached:

                        ee_pos, ee_quat = get_end_effector_pos_quat_wxyz(panda, lula_solver, art_kine_solver)
                        target_T_ee_to_obj = np.array(
                            [[0.66619042, -0.74381219, 0.05416401, 0.00814615],
                             [-0.74120343, -0.66838736, -0.06225608, -0.0895842],
                             [0.08250937, 0.00132786, -0.9965894, -0.12217499],
                             [0., 0., 0., 1.]]
                        )

                        quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
                        T_ee = np.eye(4)
                        T_ee[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
                        T_ee[:3, 3] = ee_pos

                        T_obj = pickplace.get_obj_pose(pickplace.attached_object_path)
                        T_ee_to_obj = np.linalg.inv(T_ee) @ np.array(T_obj)

                        # print(np.linalg.norm(target_T_ee_to_obj[:3, 3] - T_ee_to_obj[:3, 3]))
                        if np.linalg.norm(target_T_ee_to_obj[:3, 3] - T_ee_to_obj[:3, 3]) < 0.03:
                            pickplace.T_ee_to_obj = target_T_ee_to_obj
                            pickplace.attached = True
                            print('Start syncing attached object...')

                    pickplace._sync_attached_object(panda, lula_solver, art_kine_solver)

                    target_rot = utils.rvec_to_quaternion_wxyz(action[3:6])
                    # target_rot = pickplace.grasp_quat
                    set_gripper_width(panda, action[-1])
                    apply_ik_solution(panda, art_kine_solver, action[:3], target_rot)
                    action_t += 1
            #     # print(action_idx, action[:, 3])
            #     # exit()

            if is_episode_completed(None):
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

        episode_success = is_episode_completed(episode_record)
        episode_record["success"] = episode_success

        if episode_success:
            print("[Main] Task success")
        else:
            print("[Main] Task fail")

        collected_episodes.append(episode_record)

    successful_episodes = [
        ep for ep in collected_episodes
        if ep.get("success", False)
    ]
    print(f"[Main] Total successful trials collected: {len(successful_episodes)}")
    simulation_app.close()


if __name__ == "__main__":
    main()
