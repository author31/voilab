import os
import registry
import argparse
import numpy as np
import json
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

import isaacsim.core.utils.stage as stage_utils

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from umi_replay import UMIReplay
from object_loader import load_objects_from_json
from utils import set_prim_scale
from isaacsim.storage.native import get_assets_root_path

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

def setup_robot_drives(panda):
    joint_indices = []
    kps = []
    kds = []
    default_positions = []

    for joint_name in PANDA_ARM_JOINTS:
        dof_index = panda.get_dof_index(joint_name)
        if dof_index is not None:
            cfg = DRIVE_CONFIGS.get(joint_name)
            if not cfg: continue
            joint_indices.append(dof_index)
            kps.append(cfg["stiffness"])
            kds.append(cfg["damping"])
            default_positions.append(cfg["pos"])

    kps_array = np.array([kps], dtype=np.float32)
    kds_array = np.array([kds], dtype=np.float32)
    indices_array = np.array(joint_indices, dtype=np.int32)
    positions_array = np.array([default_positions], dtype=np.float32)
    # panda.set_gains(kps=kps_array, kds=kds_array, joint_indices=indices_array)
    panda.set_joint_positions(positions=positions_array, joint_indices=indices_array)
    return indices_array, positions_array

def load_robotic_arm(prim_path: str, name: str):
    pass


def main():
    print(f"[Main] Starting with task: {args.task}")
    
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
    franka_rotation = franka_pose.get("rotation_quat", [0,0,0,1])
    aruco_tag_pose = cfg.get("aruco_tag_pose", {})
    
    camera_translation = cfg.get("camera_pose", {}).get("translation", [0, 0, 0])

    stage_utils.open_stage(BASE_SCENE_FP)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    robot = stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    robot_xform = SingleXFormPrim(prim_path=FRANKA_PANDA_PRIM_PATH)
    robot_xform.set_world_pose(position=np.array(franka_translation), orientation=np.array(franka_rotation))

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
    world.reset()

    # Set robot position after world reset
    panda.set_world_pose(
        position=np.array(franka_translation) / stage_utils.get_stage_units(),
        orientation=np.array(franka_rotation)
    )
    set_camera_view(camera_translation, franka_translation)
    
    # Load objects from object_poses.json if session_dir provided
    if args.session_dir:
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
        else:
            print(f"[Main] INFO: object_poses.json not found at {object_poses_path}")
            print(f"[Main] Continuing without spawned objects")

    
    # Initialize replay
    replay = None
    if args.session_dir:
        print(f"[Main] Initializing Kinematics with UMI config...")
        lula = LulaKinematicsSolver(
            robot_description_path=LULA_ROBOT_DESCRIPTION_PATH,
            urdf_path=LULA_URDF_PATH
        )

        art_kine_solver = ArticulationKinematicsSolver(
            panda,
            kinematics_solver=lula,
            end_effector_frame_name="umi_tcp" 
        )

        replay = UMIReplay(panda, args.session_dir, args.episode, lula, art_kine_solver)

    print("Starting simulation loop...")
    

    while simulation_app.is_running():
        world.step(render=True)
        if replay is not None:
            if not replay.update():
                print("[Main] Replay finished. Continuing simulation...")
                replay = None

    simulation_app.close()

if __name__ == "__main__":
    main()