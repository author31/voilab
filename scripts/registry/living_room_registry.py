import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation
from utils import get_object_pose, get_object_world_boundary


class LivingRoomTaskRegistry:
    """Registry for living room task configuration"""

    TASK_NAME = "living-room"
    # ArUco tag pose
    ARUCO_TAG_TRANSLATION = np.array([1.18, 11.31, 0.83])
    ARUCO_TAG_ROTATION_EULER = np.array([0.0, 0.0, 90])
    ARUCO_TAG_ROTATION_QUAT = Rotation.from_euler('xyz', ARUCO_TAG_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w
    BLUE_BLOCK = "/World/cylinder"
    GREEN_BLOCK = "/World/bridge"
    RED_BLOCK = "/World/triangle"
    STORAGE_BOX = "/World/storage_box"

    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([1.045, 11.31, 0.50])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, -30.0])
    FRANKA_ROTATION_QUAT = Rotation.from_euler('xyz', FRANKA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    # Camera poses
    CAMERA_TRANSLATION = np.array([2.66, 11.41, 1.96])
    CAMERA_ROTATION_EULER = np.array([75.80000305175781, 0.0, -91.0])
    CAMERA_ROTATION_QUAT = Rotation.from_euler('xyz', CAMERA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "aruco_tag_pose": {
                "translation": cls.ARUCO_TAG_TRANSLATION,
                "rotation_euler_deg": cls.ARUCO_TAG_ROTATION_EULER,
                "rotation_quat": cls.xyzw_to_wxyz(cls.ARUCO_TAG_ROTATION_QUAT),
            },
            "camera_pose": {
                "translation": cls.CAMERA_TRANSLATION,
                "rotation_euler_deg": cls.CAMERA_ROTATION_EULER,
                "rotation_quat": cls.xyzw_to_wxyz(cls.CAMERA_ROTATION_QUAT),
            },
            "franka_pose": {
                "translation": cls.FRANKA_TRANSLATION,
                "rotation_euler_deg": cls.FRANKA_ROTATION_EULER,
                "rotation_quat": cls.xyzw_to_wxyz(cls.FRANKA_ROTATION_QUAT),
            },
            "environment_vars": {
                "TASK_NAME": cls.TASK_NAME,
                "SCENE_CONFIG": "living_scene",
                "OBJECT_MAXIMUM_Z_HEIGHT": 1.1,
                "BLUE_BLOCK_PATH": cls.BLUE_BLOCK,
                "GREEN_BLOCK_PATH": cls.GREEN_BLOCK,
                "RED_BLOCK_PATH": cls.RED_BLOCK,
                "STORAGE_BOX_PATH": cls.STORAGE_BOX,
                "PRELOAD_OBJECTS": [
                    {
                        "name": "blue_block",
                        "assets": "cylinder.usd",
                        "prim_path": "/World/cylinder",
                        "quat_wxyz": np.array([0.707107, 0.707107, 0, 0]),
                    },
                    {
                        "name": "green_block",
                        "assets": "bridge.usd",
                        "prim_path": "/World/bridge",
                        "quat_wxyz": np.array([0.707107, 0.707107, 0, 0]),
                    },
                    {
                        "name": "red_block",
                        "assets": "triangle.usd",
                        "prim_path": "/World/triangle",
                        "quat_wxyz": np.array([0.707107, 0.707107, 0, 0]),
                    },
                ],
                "FIXED_OBJECTS": [
                    {
                        "name": "storage_box",
                        "position": [2.03664, 11.37101, 0.89187],
                        "rotation_quat_wxyz": [0.525322, 0, 0, -0.8509035],
                    }
                ],
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate living room environment setup"""

        if np.any(np.isnan(cls.FRANKA_TRANSLATION)) or np.any(np.isnan(cls.FRANKA_ROTATION_EULER)):
            return False

        if np.any(np.isnan(cls.CAMERA_TRANSLATION)) or np.any(np.isnan(cls.CAMERA_ROTATION_EULER)):
            return False

        return True

    @classmethod
    def is_episode_completed(cls, episode_record: Dict[str, Any]) -> bool:
        blue_block_pos, _ = get_object_pose(cls.BLUE_BLOCK)
        green_block_pos, _ = get_object_pose(cls.GREEN_BLOCK)
        red_block_pos, _ = get_object_pose(cls.RED_BLOCK)

        box_min, box_max = get_object_world_boundary(cls.STORAGE_BOX)
        margin = 0.01

        def in_box(p, mn, mx, margin=0.0):
            return (
                mn[0] + margin <= p[0] <= mx[0] - margin and
                mn[1] + margin <= p[1] <= mx[1] - margin and
                mn[2] + margin <= p[2] <= mx[2] - margin
            )

        blue_inside = in_box(blue_block_pos, box_min, box_max, margin)
        green_inside = in_box(green_block_pos, box_min, box_max, margin)
        red_inside = in_box(red_block_pos, box_min, box_max, margin)

        success = blue_inside and green_inside and red_inside

        return success

    @staticmethod
    def xyzw_to_wxyz(q_xyzw):
        assert q_xyzw.shape[0] == 4
        x, y, z, w = q_xyzw
        return (w, x, y, z)
