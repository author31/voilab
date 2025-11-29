# launch.py - FIXED VERSION
import numpy as np
import registry
import argparse
import os
import sys

# Critical: Set ROS2 environment BEFORE importing omniverse modules
os.environ.setdefault("ROS_DISTRO", "humble")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

from isaacsim import SimulationApp
from scipy.spatial.transform import Rotation as R

# CRITICAL: Configuration must disable rendering for headless ROS2 mode
config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": ["isaacsim.ros2.bridge"]
}

print("[Isaac Sim] Initializing SimulationApp with config:", config)
simulation_app = SimulationApp(config)

# NOW import Omniverse modules (must happen after SimulationApp)
print("[Isaac Sim] Importing Omniverse modules...")
import omni.usd
from pxr import Sdf
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.extensions import enable_extension
import omni.kit.commands

# Enable ROS2 bridge explicitly
print("[Isaac Sim] Enabling ROS2 bridge extension...")
try:
    enable_extension("isaacsim.ros2.bridge")
    print("[Isaac Sim] ✓ ROS2 bridge enabled successfully")
except Exception as e:
    print(f"[Isaac Sim] WARNING: ROS2 bridge enable failed: {e}")

# Try importing ROS2 bridge
try:
    from isaacsim.ros2.bridge import Ros2Bridge
    print("[Isaac Sim] ✓ ROS2 bridge imported successfully")
    ros2_available = True
except ImportError as e:
    print(f"[Isaac Sim] WARNING: ROS2 bridge import failed: {e}")
    ros2_available = False

BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/Collected_franka-umi-scene/panda.usd"
PANDA_PRIM_PATH = "/World/panda"

DEFAULT_CAMERA_TRANSLATION = np.array([0.0, 0.0, 0.0])

def update_simulation(num_steps=5):
    """Update simulation for num_steps"""
    for _ in range(num_steps):
        simulation_app.update()

def setup_scene_from_registry(registry_class):
    """Set up scene using registry configuration"""
    config_dict = registry_class.get_config()
    
    franka_pose = config_dict.get("franka_pose", {})
    franka_translation = franka_pose.get("translation")
    franka_rotation = franka_pose.get("rotation_euler_deg")
    
    if franka_translation is not None and franka_rotation is not None:
        print(f"[Scene Setup] Setting Franka pose: position={franka_translation}, rotation={franka_rotation}")
        
        rotation_quat = R.from_euler('xyz', franka_rotation, degrees=True).as_quat()
        
        omni.kit.commands.execute(
            "IsaacSimTeleportPrim",
            prim_path=PANDA_PRIM_PATH,
            translation=tuple(franka_translation),
            rotation=tuple(rotation_quat)
        )

def main():
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim workspace with ROS2 bridge")
    parser.add_argument("--task", type=str, 
                        choices=["kitchen", "dining-table", "living-room"],
                        required=True,
                        help="Task registry to use")
    args = parser.parse_args()
    
    print(f"[Main] Starting with task: {args.task}")
    print(f"[Main] ROS2 Available: {ros2_available}")
    
    # Load registry
    registry_class = registry.get_task_registry(args.task)
    print(f"[Main] Loaded registry: {args.task}")
    
    if not registry_class.validate_environment():
        print(f"[Main] WARNING: Registry validation failed for {args.task}")
    
    # Load scene
    print(f"[Main] Opening stage: {BASE_SCENE_FP}")
    ctx = omni.usd.get_context()
    ctx.open_stage(BASE_SCENE_FP)
    update_simulation()
    
    # Add robot
    print(f"[Main] Adding Franka Panda...")
    omni.kit.commands.execute(
        "CreateReference",
        usd_context=ctx,
        path_to=Sdf.Path(PANDA_PRIM_PATH),
        asset_path=FRANKA_PANDA_FP,
        prim_path=Sdf.Path()
    )
    update_simulation()
    
    # Setup scene
    setup_scene_from_registry(registry_class)
    update_simulation()
    
    # Simulation loop
    print("[Main] Entering simulation loop...")
    step_count = 0
    while simulation_app.is_running():
        simulation_app.update()
        
        # Periodic ROS2 status check
        if step_count % 1000 == 0:
            print(f"[Main] Simulation step: {step_count}")
        step_count += 1
    
    print("[Main] Shutting down...")
    simulation_app.close()

if __name__ == "__main__":
    main()

