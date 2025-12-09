import registry
import argparse
import numpy as np

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

from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.utils.viewports import set_camera_view
from umi_replay import UMIReplay


enable_extension("isaacsim.robot_motion.motion_generation")

# --- Configuration ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/Collected_franka-umi-scene/panda.usd"
FRANKA_PANDA_PRIM_PATH = "/World/panda"

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

def update_simulation(num_steps=5):
    for _ in range(num_steps):
        simulation_app.update()

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

def main():
    print(f"[Main] Starting with task: {args.task}")
    
    registry_class = registry.get_task_registry(args.task)
    if not registry_class.validate_environment():
        print(f"[Main] WARNING: Registry validation failed")
    
    cfg = registry_class.get_config()
    franka_translation = cfg.get("franka_pose", {}).get("translation", [0, 0, 0])

    camera_translation = cfg.get("camera_pose", {}).get("translation", [0, 0, 0])

    stage_utils.open_stage(BASE_SCENE_FP)
    world = World()
    stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)

    panda = SingleArticulation(FRANKA_PANDA_PRIM_PATH, name="panda_robot")
    world.scene.add(panda)

    set_camera_view(camera_translation, franka_translation)
    world.reset()

    print(f"[Main] Setting robot world pose to: {franka_translation}")
    panda.set_world_pose(position=franka_translation)
    setup_robot_drives(panda)

    print("[Main] Settling robot into calibration pose (200 steps)...")
    for _ in range(200):
        world.step(render=True)

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
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()

        if replay is not None:
            if not replay.update():
                print("[Main] Replay finished. Continuing simulation...")
                replay = None
        
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()