import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.spatial.transform import Rotation


class LivingRoomTaskRegistry:
    """Registry for living room task configuration"""

    TASK_NAME = "living-room"
    # ArUco tag pose
    ARUCO_TAG_TRANSLATION = np.array([1.62, 4.96, 0.75])
    ARUCO_TAG_ROTATION_EULER = np.array([0.0, 0.0, 0.0])
    ARUCO_TAG_ROTATION_QUAT = Rotation.from_euler('xyz', ARUCO_TAG_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([1.475993595877246, 11.351613434539319, 0.8570439802062628])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, -18.0])
    FRANKA_ROTATION_QUAT = Rotation.from_euler('xyz', FRANKA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    # Camera poses
    CAMERA_TRANSLATION = np.array([-1.3000000193715096, 11.60000017285347, 2.0000000298023224])
    CAMERA_ROTATION_EULER = np.array([75.80000305175781, 0.0, -91.0])
    CAMERA_ROTATION_QUAT = Rotation.from_euler('xyz', CAMERA_ROTATION_EULER, degrees=True).as_quat() # x,y,z,w

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
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
        return True

    @staticmethod
    def xyzw_to_wxyz(q_xyzw):
        assert q_xyzw.shape[0] == 4
        x, y, z, w = q_xyzw
        return (w, x, y, z)
