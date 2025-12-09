import registry
import argparse
import numpy as np
import json
from pathlib import Path
from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.ros2.bridge"]
}
simulation_app = SimulationApp(config)

import cv2
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction

# --- Configuration ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/Collected_franka-umi-scene/panda.usd"
FRANKA_PANDA_PRIM_PATH = "/World/panda"
CAD_ASSETS_DIR = Path("/workspace/voilab/assets/CADs")

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
    "panda_leftfinger": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
    "panda_rightfinger": {"stiffness": 2e3, "damping": 1e2, "pos": 0.04},
}

# Mapping from object names to USD file names
OBJECT_NAME_TO_USD = {
    # Kitchen objects
    "pink_cup": "cup_finalized.usd",
    "blue_cup": "cup_finalized.usd",
    # Dining room objects
    "fork": "fork_final.usd",
    "knife": "knife_final.usd",
    "plate": "plate.usd",
    # Living room objects
    "blue_block": "storage_box.usd",
    "green_block": "storage_box.usd",
    "red_block": "storage_box.usd",
}

def update_simulation(num_steps=5):
    """Update simulation for num_steps"""
    for _ in range(num_steps):
        simulation_app.update()

def load_object_poses(session_dir: Path, episode: int):
    """
    Load object poses from object_poses.json for a specific episode.
    
    Args:
        session_dir: Path to the session directory (e.g., datasets/1124_gopro3_kitchen)
        episode: Episode number to load poses for
    
    Returns:
        List of object pose dictionaries, or empty list if not found
    """
    object_poses_path = session_dir / "demos" / "mapping" / "object_poses.json"
    
    if not object_poses_path.exists():
        print(f"[ObjectPoses] WARNING: object_poses.json not found at {object_poses_path}")
        return []
    
    try:
        with open(object_poses_path, "r") as f:
            all_episodes_data = json.load(f)
        
        # object_poses.json contains a list of episodes
        if episode >= len(all_episodes_data):
            print(f"[ObjectPoses] WARNING: Episode {episode} not found in object_poses.json (only {len(all_episodes_data)} episodes)")
            return []
        
        episode_data = all_episodes_data[episode]
        objects = episode_data.get("objects", [])
        status = episode_data.get("status", "unknown")
        
        print(f"[ObjectPoses] Loaded episode {episode} with status: {status}")
        print(f"[ObjectPoses] Found {len(objects)} objects: {[obj['object_name'] for obj in objects]}")
        
        return objects
    
    except Exception as e:
        print(f"[ObjectPoses] ERROR loading object_poses.json: {e}")
        return []

def rvec_to_quaternion(rvec):
    """
    Convert rotation vector (OpenCV format) to quaternion (w, x, y, z).
    
    Args:
        rvec: Rotation vector [x, y, z] from OpenCV/cv2.Rodrigues
    
    Returns:
        Quaternion as numpy array [w, x, y, z]
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    
    # Convert rotation matrix to quaternion
    # Using the formula from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z], dtype=np.float64)

def spawn_objects_from_poses(world: World, object_poses: list):
    """
    Spawn objects in the Isaac Sim scene based on loaded poses.
    
    Args:
        world: Isaac Sim World instance
        object_poses: List of object pose dictionaries from object_poses.json
    """
    if not object_poses:
        print("[ObjectSpawn] No objects to spawn")
        return
    
    print(f"[ObjectSpawn] Spawning {len(object_poses)} objects...")
    
    for idx, obj_pose in enumerate(object_poses):
        object_name = obj_pose.get("object_name")
        rvec = obj_pose.get("rvec")
        tvec = obj_pose.get("tvec")
        
        if not object_name or rvec is None or tvec is None:
            print(f"[ObjectSpawn] WARNING: Incomplete data for object {idx}, skipping")
            continue
        
        # Get USD file path
        usd_filename = OBJECT_NAME_TO_USD.get(object_name)
        if not usd_filename:
            print(f"[ObjectSpawn] WARNING: No USD mapping for object '{object_name}', skipping")
            continue
        
        usd_path = CAD_ASSETS_DIR / usd_filename
        if not usd_path.exists():
            print(f"[ObjectSpawn] WARNING: USD file not found: {usd_path}, skipping")
            continue
        
        # Create unique prim path for this object
        prim_path = f"/World/objects/{object_name}_{idx}"
        
        # Convert rotation vector to quaternion
        quaternion = rvec_to_quaternion(rvec)
        
        # Convert tvec to numpy array
        position = np.array(tvec, dtype=np.float64)
        
        print(f"[ObjectSpawn] Spawning '{object_name}' at position {position} with rotation {quaternion}")
        
        try:
            # Add object to stage
            stage_utils.add_reference_to_stage(
                usd_path=str(usd_path),
                prim_path=prim_path
            )
            
            # Create XFormPrim to set pose
            obj_prim = XFormPrim(prim_path=prim_path, name=f"{object_name}_{idx}")
            world.scene.add(obj_prim)
            
            # Set world pose (position and orientation)
            obj_prim.set_world_pose(
                position=position,
                orientation=quaternion  # (w, x, y, z)
            )
            
            print(f"[ObjectSpawn] Successfully spawned '{object_name}' at {prim_path}")
            
        except Exception as e:
            print(f"[ObjectSpawn] ERROR spawning '{object_name}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim workspace with ROS2 bridge and object poses")
    parser.add_argument("--task", type=str, 
                        choices=["kitchen", "dining-table", "living-room"],
                        required=True,
                        help="Task registry to use")
    parser.add_argument("--session-dir", type=str,
                        help="Path to session directory containing object_poses.json (e.g., datasets/1124_gopro3_kitchen)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode number to load object poses from (default: 0)")
    args = parser.parse_args()
    print(f"[Main] Starting with task: {args.task}")
    
    # Load object poses if session_dir is provided
    object_poses = []
    if args.session_dir:
        session_dir = Path(args.session_dir)
        if not session_dir.is_absolute():
            # Make it absolute relative to /workspace/voilab
            session_dir = Path("/workspace/voilab") / session_dir
        
        print(f"[Main] Loading object poses from session: {session_dir}, episode: {args.episode}")
        object_poses = load_object_poses(session_dir, args.episode)
    else:
        print("[Main] No session-dir provided, skipping object pose loading")

    enable_extension("isaacsim.ros2.bridge")
    update_simulation()
    
    # Load registry
    registry_class = registry.get_task_registry(args.task)
    print(f"[Main] Loaded registry: {args.task}")
    if not registry_class.validate_environment():
        print(f"[Main] WARNING: Registry validation failed for {args.task}")
    
    cfg = registry_class.get_config()
    franka_pose = cfg.get("franka_pose", {})
    franka_translation = franka_pose.get("translation")
    
    # Open the stage from the provided USD file
    stage_utils.open_stage(BASE_SCENE_FP)
    world = World()
    stage_utils.add_reference_to_stage(usd_path=FRANKA_PANDA_FP, prim_path=FRANKA_PANDA_PRIM_PATH)
    panda = Articulation(
        prim_paths_expr=FRANKA_PANDA_PRIM_PATH, name="panda_robot",
    )
    print(f"[Main] Robot moved to: {franka_translation}")
    world.scene.add(panda)

    world.reset()

    panda.set_world_poses(
        positions=np.array([franka_translation])
    )

    joint_indices = []
    kps = [] # Stiffness
    kds = [] # Damping
    default_positions = [] # Joint positions

    for joint_name in PANDA_ARM_JOINTS:
        dof_index = panda.get_dof_index(joint_name)
        if dof_index is not None:
            config = DRIVE_CONFIGS.get(joint_name)

            if not config: continue

            joint_indices.append(dof_index)
            kps.append(config["stiffness"])
            kds.append(config["damping"])
            default_positions.append(config["pos"]) 

    kps_array = np.array([kps], dtype=np.float32)
    kds_array = np.array([kds], dtype=np.float32)
    indices_array = np.array(joint_indices, dtype=np.int32)
    positions_array = np.array([default_positions], dtype=np.float32)

    panda.set_gains(kps=kps_array, kds=kds_array, joint_indices=indices_array)
    panda.set_joint_positions(positions=positions_array, joint_indices=indices_array)

    # Spawn objects from loaded poses
    spawn_objects_from_poses(world, object_poses)
    
    # Reset world after spawning objects
    world.reset()

    print("Starting simulation loop...")
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
