import registry
import argparse
import numpy as np
from isaacsim import SimulationApp

config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.ros2.bridge"]
}
simulation_app = SimulationApp(config)

import numpy as np
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.prims import Articulation
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction

# --- Configuration ---
BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/Collected_franka-umi-scene/panda.usd"
FRANKA_PANDA_PRIM_PATH = "/World/panda"

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
def update_simulation(num_steps=5):
    """Update simulation for num_steps"""
    for _ in range(num_steps):
        simulation_app.update()

def main():
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim workspace with ROS2 bridge")
    parser.add_argument("--task", type=str, 
                        choices=["kitchen", "dining-table", "living-room"],
                        required=True,
                        help="Task registry to use")
    args = parser.parse_args()
    print(f"[Main] Starting with task: {args.task}")

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

    print("Starting simulation loop...")
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
