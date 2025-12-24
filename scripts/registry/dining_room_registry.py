import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation


class DiningRoomTaskRegistry:
    """Registry for dining table task configuration"""

    TASK_NAME = "dining-table"
    # ArUco tag pose
    ARUCO_TAG_TRANSLATION = np.array([1.65, 4.75, 0.8])
    ARUCO_TAG_ROTATION_EULER = np.array([0.0, 0.0, 90.0])
    ARUCO_TAG_ROTATION_QUAT = Rotation.from_euler('xyz', ARUCO_TAG_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w
    FORK_PATH = "/World/fork"
    KNIFE_PATH = "/World/knife"
    PLATE_PATH = "/World/plate"

    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([1.4471314866267897, 4.953638444125494, 0.7547650876392805])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, -18.0])
    FRANKA_ROTATION_QUAT = Rotation.from_euler('xyz', FRANKA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    # Camera poses
    CAMERA_TRANSLATION = np.array([5.300000078976154, 4.90000007301569, 1.600000023841858])
    CAMERA_ROTATION_EULER = np.array([78.5, 0.0, -270.0])
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
                "SCENE_CONFIG": "dining_scene",
                "OBJECT_MAXIMUM_Z_HEIGHT": 1.1,
                "KNIFE_PATH": cls.KNIFE_PATH,
                "FORK_PATH": cls.FORK_PATH,
                "PLATE_PATH": cls.PLATE_PATH,
                "PRELOAD_OBJECTS": [
                    {
                        "name": "knife",
                        "assets": "knife.usd",
                        "prim_path": "/World/knife",
                        "quat_wxyz": np.array([1.,0.,0.,0.]),
                    },
                    {
                        "name": "fork",
                        "assets": "fork.usd",
                        "prim_path": "/World/fork",
                        "quat_wxyz": np.array([0.707, 0.0, 0.0, -0.707]),
                    },
                    {
                        "name": "plate",
                        "assets": "plate.usd",
                        "prim_path": "/World/plate",
                        "quat_wxyz": np.array([0.707, 0.0, 0.0, -0.707]),
                    },
                ],
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate dining table environment setup"""

        if np.any(np.isnan(cls.FRANKA_TRANSLATION)) or np.any(np.isnan(cls.FRANKA_ROTATION_EULER)):
            return False

        if np.any(np.isnan(cls.CAMERA_TRANSLATION)) or np.any(np.isnan(cls.CAMERA_ROTATION_EULER)):
            return False

        return True

    @classmethod
    def is_episode_completed(cls, episode_record: Dict[str, Any]) -> bool:
        return True

    @staticmethod
    def xyzw_to_wxyz(q_xyzw):
        assert q_xyzw.shape[0] == 4
        x, y, z, w = q_xyzw
        return (w, x, y, z)
