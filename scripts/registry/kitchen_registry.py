import numpy as np
from pathlib import Path
from typing import Dict, Any


class KitchenTaskRegistry:
    """Registry for kitchen task configuration"""

    TASK_NAME = "kitchen"
    # Robot poses (Franka)
    FRANKA_TRANSLATION = np.array([4.90000007301569, 3.2254199483665964, 0.9000000134110451])
    FRANKA_ROTATION_EULER = np.array([0.0, 0.0, 0.0])

    # Camera poses (enhanced from current)
    CAMERA_TRANSLATION = np.array([2.974631262928433, 2.68664950400609, 1.966344261667377])
    CAMERA_ROTATION_EULER = np.array([75.43, 0.0, -91.64])

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
                "SCENE_CONFIG": "kitchen_scene",
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
