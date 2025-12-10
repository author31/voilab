"""
Object loading utilities for Isaac Sim.
Loads objects from object_poses.json and spawns them in the scene.
"""

import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim

# Registry mapping object names to USD asset filenames
# This should be customized based on the actual asset library
OBJECT_NAME_TO_ASSET = {
    'blue_cup': 'cup_finalized.usd',
    'cup': 'cup_finalized.usd',
    'white_plate': 'plate.usd',
    'plate': 'plate.usd',
    'fork': 'fork_final.usd',
    'knife': 'knife_final.usd',
    'spoon': 'spoon.usd',
    'storage_box': 'storage_box.usd',
    'table': 'table.usd',
}


def map_object_name_to_asset(object_name: str) -> str:
    """
    Map object name from reconstruction to USD asset filename.

    Args:
        object_name: Name from object_poses.json (e.g., "blue_cup", "white_plate")

    Returns:
        str: USD filename (e.g., "cup_finalized.usd")
    """
    # Try exact match first
    if object_name in OBJECT_NAME_TO_ASSET:
        return OBJECT_NAME_TO_ASSET[object_name]

    # Try partial match (e.g., "blue_cup" contains "cup")
    for key, asset in OBJECT_NAME_TO_ASSET.items():
        if key in object_name.lower():
            return asset

    # Default: try to construct filename from object_name
    # e.g., "blue_cup" -> "cup.usd"
    base_name = object_name.lower().split('_')[-1]  # Get last word
    return f"{base_name}.usd"


def load_objects_from_json(json_path: str, assets_dir: str, world: World, episode_index: int = 0):
    """
    Load objects from a specific episode in object_poses.json and spawn them in the scene.

    The JSON format is an array of episodes, each containing objects with rvec (rotation vector)
    and tvec (translation vector) from the reconstruction system.

    Args:
        json_path: Path to object_poses.json (contains array of episodes)
        assets_dir: Base directory for CAD assets (e.g., /workspace/voilab/assets/CADs)
        world: Isaac Sim World instance
        episode_index: Index of the episode to load (default 0)
    """
    # Load JSON with error handling
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ObjectLoader] ERROR: File not found: {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"[ObjectLoader] ERROR: Invalid JSON format in {json_path}: {str(e)}")
        return
    except Exception as e:
        print(f"[ObjectLoader] ERROR: Unexpected error loading JSON: {str(e)}")
        return

    # data is an array of episodes
    if isinstance(data, list):
        episodes = data
    else:
        episodes = [data]

    # Validate episode index
    if episode_index < 0 or episode_index >= len(episodes):
        print(f"[ObjectLoader] ERROR: Episode index {episode_index} out of range (0-{len(episodes)-1}), skipping object loading")
        return

    episode = episodes[episode_index]
    objects = episode.get('objects', [])
    print(f"[ObjectLoader] Found {len(objects)} objects in episode {episode_index}")

    total_objects = 0
    for idx, obj in enumerate(objects):
        object_name = obj.get('object_name', f'object_{idx}')
        rvec = np.array(obj.get('rvec', [0, 0, 0]))
        tvec = np.array(obj.get('tvec', [0, 0, 0]))

        # Convert rotation vector (axis-angle) to quaternion WXYZ
        # rvec is a rotation vector where:
        # - Direction is the axis of rotation
        # - Magnitude is the angle in radians
        rot = R.from_rotvec(rvec)
        quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Map object_name to USD asset file
        asset_filename = map_object_name_to_asset(object_name)
        full_asset_path = os.path.join(assets_dir, asset_filename)
        # Check if asset exists
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
            continue

        # Create unique prim path
        prim_path = f"/World/objects/{object_name}_{total_objects}"

        # Add reference to stage
        try:
            stage_utils.add_reference_to_stage(
                usd_path=full_asset_path,
                prim_path=prim_path
            )
        except Exception as e:
            print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
            continue

        # Create XFormPrim and set pose
        obj_prim = XFormPrim(prim_path=prim_path, name=f"{object_name}_{total_objects}")
        obj_prim.set_world_pose(position=tvec, orientation=quat_wxyz)

        # Add to world scene
        world.scene.add(obj_prim)

        print(f"[ObjectLoader] Spawned {object_name} at position {tvec}")
        total_objects += 1

    print(f"[ObjectLoader] Total objects spawned: {total_objects}")
