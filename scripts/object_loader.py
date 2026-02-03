"""
Object loading utilities for Isaac Sim.
Loads objects from object_poses.json and spawns them in the scene.
"""

import json
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import omni.usd
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from pxr import Sdf, UsdGeom, UsdPhysics
from utils import set_prim_scale, pose_to_transform_matrix


# Registry mapping object names to USD asset filenames
OBJECT_NAME_TO_ASSET = {
    'blue_cup': 'cup_blue.usd',
    'pink_cup': 'cup_pink.usd',
    'white_plate': 'plate.usd',
    'plate': 'plate.usd',
    'fork': 'fork_final.usd',
    'knife': 'knife_final.usd',
    'spoon': 'spoon.usd',
    'storage_box': 'storage_box.usd',
    'table': 'table.usd',
}

MAXIMUM_Z_HEIGHT = 0.9


def _ensure_xform_prim(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[ObjectLoader] USD stage is not available")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        return
    UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))


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


def load_object_transforms_from_json(
    json_path: str,
    episode_index: int = 0,
    aruco_tag_pose: dict = None,
    cfg: dict = None,
):
    """
    Simplified loader: compute object transforms from a specific episode.

    Returns:
        list[dict]: Each entry has keys:
            object_name (str)
            transform (np.ndarray, shape (4, 4))
            pose_rt (np.ndarray, shape (2, 3)) [rotvec, translation]
            position (np.ndarray, shape (3,))
            quat_wxyz (np.ndarray, shape (4,))
    """
    # Load JSON with error handling
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ObjectLoader] ERROR: File not found: {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[ObjectLoader] ERROR: Invalid JSON format in {json_path}: {str(e)}")
        return []
    except Exception as e:
        print(f"[ObjectLoader] ERROR: Unexpected error loading JSON: {str(e)}")
        return []

    episodes = data if isinstance(data, list) else [data]
    if episode_index < 0 or episode_index >= len(episodes):
        print(f"[ObjectLoader] ERROR: Episode index {episode_index} out of range (0-{len(episodes)-1}), skipping object loading")
        return []

    episode = episodes[episode_index]
    if episode.get('status') != 'full':
        print(f"[ObjectLoader] Episode {episode_index} is not complete, skipping object loading")
        return []

    object_maximum_z_height = (cfg or {}).get("environment_vars", {}).get("OBJECT_MAXIMUM_Z_HEIGHT", 0.0)
    objects = episode.get('objects', [])

    if aruco_tag_pose is not None:
        aruco_tag_position = np.array(aruco_tag_pose.get('translation', [0, 0, 0]), dtype=np.float64)
        aruco_tag_quat_wxyz = np.array(aruco_tag_pose.get('rotation_quat', [1, 0, 0, 0]), dtype=np.float64)
        T_aruco_tag = pose_to_transform_matrix(aruco_tag_position, aruco_tag_quat_wxyz)
    else:
        T_aruco_tag = np.eye(4, dtype=np.float64)

    results = []
    for obj in objects:
        object_name = obj.get('object_name')
        rvec = obj.get('rvec')
        tvec = obj.get('tvec')
        if not object_name or rvec is None or tvec is None:
            continue
        if len(rvec) != 3 or len(tvec) != 3:
            continue

        rvec = np.array(rvec, dtype=np.float64)
        tvec = np.array(tvec, dtype=np.float64)

        rot = R.from_rotvec(rvec)
        T_object = np.eye(4, dtype=np.float64)
        T_object[:3, :3] = rot.as_matrix()
        T_object[:3, 3] = tvec

        T_world = T_aruco_tag @ T_object
        world_position = T_world[:3, 3].copy()
        world_position[2] = min(world_position[2], object_maximum_z_height)
        T_world[:3, 3] = world_position

        world_rot = R.from_matrix(T_world[:3, :3])
        quat_xyzw = world_rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        world_rvec = world_rot.as_rotvec()
        pose_rt = np.stack([world_rvec, world_position], axis=0)

        results.append(
            {
                "object_name": object_name,
                "transform": T_world,
                "pose_rt": pose_rt,
                "position": world_position,
                "quat_wxyz": quat_wxyz,
            }
        )

    return results


def load_objects_from_json(
    json_path: str,
    assets_dir: str,
    world: World,
    episode_index: int = 0,
    aruco_tag_pose: dict = None,
    cfg: dict = None,
    parent_prim_path: str = "/World",
):
    """
    Load objects from a specific episode in object_poses.json and spawn them in the scene.

    The JSON format is an array of episodes, each containing objects with rvec (rotation vector)
    and tvec (translation vector) from the reconstruction system.

    Args:
        json_path: Path to object_poses.json (contains array of episodes)
        assets_dir: Base directory for CAD assets (e.g., /workspace/voilab/assets/CADs)
        world: Isaac Sim World instance
        episode_index: Index of the episode to load (default 0)
        aruco_tag_pose: Aruco tag pose dict with 'translation' (3D position) and 
                    'rotation_quat' (quaternion WXYZ). Object poses will be
                    transformed from aruco tag frame to world frame using
                    homogeneous transformation: T_world = T_aruco_tag @ T_object
        cfg: Configuration dictionary containing object maximum z height
        parent_prim_path: Parent prim under which to spawn all objects (default: /World).
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
    if episode.get('status') != 'full':
        print(f"[ObjectLoader] Episode {episode_index} is not complete, skipping object loading")
        return False

    parent_prim_path = parent_prim_path.rstrip("/") if parent_prim_path != "/" else parent_prim_path
    _ensure_xform_prim(parent_prim_path)

    objects = episode.get('objects', [])
    print(f"[ObjectLoader] Found {len(objects)} objects in episode {episode_index}")

    object_maximum_z_height = cfg.get("environment_vars", {}).get("OBJECT_MAXIMUM_Z_HEIGHT", 0.0)

    total_objects = 0
    for idx, obj in enumerate(objects):
        assert obj.get('object_name') is not None, f"object_name is None"
        object_name = obj.get('object_name')

        assert obj.get('rvec') is not None, f"rvec is None for object {object_name}"
        assert obj.get('tvec') is not None, f"tvec is None for object {object_name}"
        assert len(obj.get('rvec')) == 3, f"rvec must be a 3D vector for object {object_name}"
        assert len(obj.get('tvec')) == 3, f"tvec must be a 3D vector for object {object_name}"

        rvec = np.array(obj.get('rvec'))
        tvec = np.array(obj.get('tvec'))

        # Build object transformation matrix from rvec/tvec
        rot = R.from_rotvec(rvec)
        T_object = np.eye(4)
        T_object[:3, :3] = rot.as_matrix()
        T_object[:3, 3] = tvec

        # Transform from robot base frame to world frame using homogeneous transformation
        if aruco_tag_pose is not None:
            aruco_tag_position = np.array(aruco_tag_pose.get('translation', [0, 0, 0]))
            aruco_tag_quat_wxyz = np.array(aruco_tag_pose.get('rotation_quat', [1, 0, 0, 0]))
            T_aruco_tag = pose_to_transform_matrix(aruco_tag_position, aruco_tag_quat_wxyz)
            
            # Matrix multiplication: T_world = T_aruco_tag @ T_object
            T_world = T_aruco_tag @ T_object
            
            # Extract world position and orientation
            world_position = T_world[:3, 3]
            world_rot = R.from_matrix(T_world[:3, :3])
            quat_xyzw = world_rot.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        else:
            world_position = tvec
            quat_xyzw = rot.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Map object_name to USD asset file
        asset_filename = map_object_name_to_asset(object_name)
        print(f"[ObjectLoader] Asset filename: {asset_filename}")
        full_asset_path = os.path.join(assets_dir, asset_filename)
        # Check if asset exists
        if not os.path.exists(full_asset_path):
            print(f"[ObjectLoader] WARNING: Asset not found: {full_asset_path}, skipping {object_name}")
            continue

        # Create unique prim path (spawn under a dedicated parent for easy cleanup)
        prim_path = f"{parent_prim_path}/{object_name}"

        try:
            stage_utils.add_reference_to_stage(
                usd_path=full_asset_path,
                prim_path=prim_path
            )
        except Exception as e:
            print(f"[ObjectLoader] ERROR: Failed to load asset {full_asset_path}: {str(e)}")
            continue
        
        # Create SingleXFormPrim and set pose
        obj_prim = SingleXFormPrim(prim_path=prim_path, name=f"{object_name}_{total_objects}")
        world.scene.add(obj_prim)
        world_position[2] = min(world_position[2], object_maximum_z_height)
        obj_prim.set_world_pose(position=world_position)

        print(f"[ObjectLoader] Spawned {object_name} at world position {world_position}")
        total_objects += 1

    print(f"[ObjectLoader] Total objects spawned: {total_objects}")
