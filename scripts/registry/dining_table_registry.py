import numpy as np
from pathlib import Path
from typing import Dict, Any


class DiningTableTaskRegistry:
    """Registry for dining table task configuration"""

    TASK_NAME = "dining-table"

    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([1.4471314866267897, 4.953638444125494, 0.7547650876392805])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, -18.0])

    # Camera poses
    CAMERA_TRANSLATION = np.array([5.300000078976154, 4.90000007301569, 1.600000023841858])
    CAMERA_ROTATION_EULER = np.array([78.5, 0.0, -270.0])

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "camera_pose": {
                "translation": cls.CAMERA_TRANSLATION,
                "rotation_euler_deg": cls.CAMERA_ROTATION_EULER,
            },
            "franka_pose": {
                "translation": cls.FRANKA_TRANSLATION,
                "rotation_euler_deg": cls.FRANKA_ROTATION_EULER,
            },
            "environment_vars": {
                "TASK_NAME": cls.TASK_NAME,
                "SCENE_CONFIG": "dining_scene",
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
