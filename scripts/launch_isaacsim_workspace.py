import numpy as np
import registry
import argparse

from omni.isaac.kit import SimulationApp
from scipy.spatial.transform import Rotation as R


BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/Collected_franka-umi-scene/panda.usd"
PANDA_PRIM_PATH = "/World/panda"

config = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(config)

# Import necessary modules (must happen after SimulationApp starts)
import omni.usd
from pxr import Sdf
from omni.isaac.core.utils.viewports import set_camera_view
from action_graph import IsaacSimActionGraph
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2_bridge")
enable_extension("omni.graph.action")

# Camera Settings (fallback defaults)
DEFAULT_CAMERA_TRANSLATION = np.array([0.0, 0.0, 0.0])
DEFAULT_CAMERA_TARGET = np.array([0.0, 0.0, 0.0])

def update_simulation():
    for _ in range(5):
        simulation_app.update()

def setup_scene_from_registry(registry_class):
    """Set up scene using registry configuration"""
    config_dict = registry_class.get_config()

    # Set Franka robot pose
    franka_pose = config_dict.get("franka_pose", {})
    franka_translation = franka_pose.get("translation")
    franka_rotation = franka_pose.get("rotation_euler_deg")

    if franka_translation is not None and franka_rotation is not None:
        print(f"[Isaac Sim] Setting Franka pose to: position={franka_translation}, rotation={franka_rotation}")

        # Convert Euler angles (degrees) to quaternion for Isaac Sim
        rotation_quat = R.from_euler('xyz', franka_rotation, degrees=True).as_quat()

        omni.kit.commands.execute(
            "IsaacSimTeleportPrim",
            prim_path=PANDA_PRIM_PATH,
            translation=tuple(franka_translation),
            rotation=tuple(rotation_quat)
        )

    # Set camera pose and apply with set_camera_view
    camera_pose = config_dict.get("camera_pose", {})
    camera_translation = camera_pose.get("translation", DEFAULT_CAMERA_TRANSLATION)
    camera_rotation = camera_pose.get("rotation_euler_deg", DEFAULT_CAMERA_TRANSLATION)

    set_camera_view(eye=camera_translation, target=tuple(franka_translation))
    print(f"[Isaac Sim] Applied camera view: position={camera_translation}, target={franka_translation}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Launch Isaac Sim workspace with registry configuration")
    parser.add_argument("--task", type=str, choices=["kitchen", "dining-table", "living-room"],
                        help="Task registry to use for scene configuration")
    args = parser.parse_args()

    if not args.task:
        raise RuntimeError("Missing --task argument")

    # Use registry configuration
    registry_class = registry.get_task_registry(args.task)
    print(f"[Isaac Sim] Using task registry: {args.task}")

    # Validate environment
    if not registry_class.validate_environment():
        print(f"[Isaac Sim] Warning: Registry validation failed for {args.task}")

    # Open the USD stage - This is the correct API
    print(f"[Isaac Sim] Opening stage: {BASE_SCENE_FP}")
    ctx = omni.usd.get_context()
    ctx.open_stage(BASE_SCENE_FP)

    update_simulation()

    # Add Franka Panda robot
    print(f"[Isaac Sim] Adding Franka Panda at {PANDA_PRIM_PATH}")
    omni.kit.commands.execute(
        "CreateReference",
        usd_context=ctx,
        path_to=Sdf.Path(PANDA_PRIM_PATH),
        asset_path=FRANKA_PANDA_FP,
        prim_path=Sdf.Path()
    )
    update_simulation()

    setup_scene_from_registry(registry_class)
    update_simulation()

    action_graph = IsaacSimActionGraph(task_name=args.task, usd_path=BASE_SCENE_FP)
    action_graph.create_action_graph()
    action_graph.set_pipeline_stage()
    action_graph.start()


    # Main Simulation Loop
    while simulation_app.is_running():
        simulation_app.update()

    # Cleanup on exit
    simulation_app.close()


if __name__ == "__main__":
    main()
