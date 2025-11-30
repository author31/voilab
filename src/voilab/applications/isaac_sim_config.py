import numpy as np

# Task-specific camera configurations
TASK_CONFIGURATIONS = {
    "kitchen": {
        "usd_path": "assets/franka_panda/franka_panda_umi-isaacsim/franka_panda_umi-isaacsim.usd",
        "camera_position": np.array([2.5, 1.8, 2.0]),
        "camera_target": np.array([0.0, 0.0, 0.8]),
        "environment_vars": {
            "TASK_NAME": "kitchen",
            "SCENE_CONFIG": "kitchen_scene"
        }
    },
    "dining-table": {
        "usd_path": "assets/Collected_franka-umi-scene/franka-umi-scene.usd",
        "camera_position": np.array([1.8, 2.2, 1.8]),
        "camera_target": np.array([0.0, 0.0, 0.7]),
        "environment_vars": {
            "TASK_NAME": "dining-table",
            "SCENE_CONFIG": "dining_scene"
        }
    },
    "living-room": {
        "usd_path": "assets/Collected_franka-umi-scene/franka-umi-scene.usd",
        "camera_position": np.array([3.2, 1.2, 1.9]),
        "camera_target": np.array([0.0, 0.5, 0.8]),
        "environment_vars": {
            "TASK_NAME": "living-room",
            "SCENE_CONFIG": "living_scene"
        }
    }
}

# Isaac Sim runtime configuration
ISAAC_SIM_CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "physics_engine": "PhysX"
}