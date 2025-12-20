import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation



class KitchenTaskRegistry:
    """Registry for kitchen task configuration"""

    ENVIRONMENT_NAME = "kitchen"
    TASK_NAME = "stacking"
    # ArUco tag pose
    ARUCO_TAG_TRANSLATION = np.array([4.9652, 2.45, 0.9])
    ARUCO_TAG_ROTATION_EULER = np.array([0.0, 0.0, 180.0])
    ARUCO_TAG_ROTATION_QUAT = Rotation.from_euler('xyz', ARUCO_TAG_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w
    TARGET_OBJECT_PATH = "/World/blue_cup"
    SUPPORT_OBJECT = "/World/pink_cup"
    
    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([4.5, 2.7, 0.9000000134110451])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, 0.0])
    FRANKA_ROTATION_QUAT = Rotation.from_euler('xyz', FRANKA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    # Camera poses (enhanced from current)
    CAMERA_TRANSLATION = np.array([7.5, 2.68664950400609, 2.2])
    CAMERA_ROTATION_EULER = np.array([71.30, 0.0, 89.])
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
                "ENVIRONMENT_NAME": cls.ENVIRONMENT_NAME,
                "SCENE_CONFIG": "kitchen_scene",
                "OBJECT_MAXIMUM_Z_HEIGHT": 1.1,
                "TARGET_OBJECT_PATH": cls.TARGET_OBJECT_PATH,
                "SUPPORT_OBJECT": cls.SUPPORT_OBJECT,
                "PRELOAD_OBJECTS": [
                    {"name": "pink cup", "assets": "cup_pink.usd"},
                    {"name": "blue cup", "assets": "cup_blue.usd"},
                ],
            }
        }

    @classmethod
    def validate_environment(cls) -> bool:
        """Validate kitchen environment setup"""

        if np.any(np.isnan(cls.FRANKA_TRANSLATION)) or np.any(np.isnan(cls.FRANKA_ROTATION_EULER)):
            return False

        if np.any(np.isnan(cls.CAMERA_TRANSLATION)) or np.any(np.isnan(cls.CAMERA_ROTATION_EULER)):
            return False

        return True

    @staticmethod
    def xyzw_to_wxyz(q_xyzw):
        assert q_xyzw.shape[0] == 4
        x, y, z, w = q_xyzw
        return (w, x, y, z)
