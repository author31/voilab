import registry
import argparse
import numpy as np
import os
import imageio
import zarr
from zarr.storage import ZipStore
from numcodecs import Blosc
from scipy.spatial.transform import Rotation as R

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
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import rot_matrix_to_quat
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

def get_end_effector_pose(panda, lula, art_kine_solver): 

    # Run fk to obtain end-effector pose 
    base_pos, base_quat = panda.get_world_pose() 
    lula.set_robot_base_pose( robot_position=base_pos, 
                             robot_orientation=base_quat) 
    ee_pos, ee_rot_matrix = art_kine_solver.compute_end_effector_pose() 
    eef_rot = R.from_matrix(ee_rot_matrix[:3, :3]).as_rotvec()

    return np.concatenate([ee_pos.astype(np.float64), eef_rot.astype(np.float64)])

def save_dataset(output_path,
                  rgb_list,
                  eef_pos_list,
                  eef_rot_list,
                  gripper_list,
                  demo_start_list,
                  demo_end_list,):
    
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    store = ZipStore(output_path, mode="w")
    root = zarr.group(store)
    data = root.create_group("data")

    data.create_dataset(
        "camera0_rgb",
        data=np.stack(rgb_list, 0).astype(np.uint8),
        compressor=compressor
    )
    data.create_dataset(
        "robot0_demo_start_pose",
        data=np.stack(demo_start_list, 0).astype(np.float64),
        compressor=compressor
    )
    data.create_dataset(
        "robot0_demo_end_pose",
        data=np.stack(demo_end_list, 0).astype(np.float64),
        compressor=compressor
    )
    data.create_dataset(
        "robot0_eef_pos",
        data=np.stack(eef_pos_list, 0).astype(np.float32),
        compressor=compressor
    )
    data.create_dataset(
        "robot0_eef_rot_axis_angle",
        data=np.stack(eef_rot_list, 0).astype(np.float32),
        compressor=compressor
    )
    data.create_dataset(
        "robot0_gripper_width",
        data=np.stack(gripper_list, 0).astype(np.float32),
        compressor=compressor
    )
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([len(rgb_list)]))
    store.close()
    print("[SAVE] replay_dataset.zarr.zip saved at:", output_path)

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
    camera = Camera(
        prim_path="/World/panda/panda/panda_link7/gopro_link/Camera",
        name="gopro_camera",
    )
    camera.initialize()

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

    # Prepare buffers for dataset
    rgb_list = []
    eef_pos_list = []
    eef_rot_list = []
    gripper_list = []
    episode_start_pose = None
    episode_end_pose = None

    while simulation_app.is_running():

        if replay is None:
            break

        world.step(render=True)

        # Capture rgb image
        img = camera.get_rgb()
        rgb_list.append(img)

        # Capture end-effector pos and rot
        eef_pose6d = get_end_effector_pose(panda, lula, art_kine_solver)
        eef_pos_list.append(eef_pose6d[:3])
        eef_rot_list.append(eef_pose6d[3:])

        # Capture gripper width
        joint_pos = panda.get_joint_positions()
        gripper_width = joint_pos[-2] + joint_pos[-1]
        gripper_list.append([gripper_width])

        if episode_start_pose is None:
            episode_start_pose = eef_pose6d.copy()

        if not replay.update():
            print("[Main] Replay finished. Exiting…")
            episode_end_pose = eef_pose6d.copy()
            replay = None
            break

        world.step(render=True)

    # Complete episode start and end pose
    demo_start_list = np.repeat(episode_start_pose[None, :], len(rgb_list), axis=0)
    demo_end_list = np.repeat(episode_end_pose[None, :], len(rgb_list), axis=0)

    # Save to session_dir
    output_zarr = os.path.join(args.session_dir, "simulation_dataset.zarr.zip")
    save_dataset(output_zarr,
                rgb_list,
                eef_pos_list,
                eef_rot_list,
                gripper_list,
                demo_start_list,
                demo_end_list)

    simulation_app.close()

if __name__ == "__main__":
    main()