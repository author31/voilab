from omni.isaac.kit import SimulationApp

USD_FILEPATH = "/workspace/305_v3.usdz" 
config = {"headless": False, "width": 1280, "height": 720}  # Remove "open_usd"
simulation_app = SimulationApp(config)

# Import necessary modules (must happen after SimulationApp starts)
import omni.usd
import numpy as np
from omni.isaac.core.utils.viewports import set_camera_view

# Camera Settings
CAMERA_POSITION = np.array([2.0, 2.0, 2.0])  # X, Y, Z coordinates
CAMERA_TARGET = np.array([0.0, 0.0, 0.0])    # The point the camera looks at

def main():
    # Open the USD stage - This is the correct API
    print(f"[Isaac Sim] Opening stage: {USD_FILEPATH}")
    omni.usd.get_context().open_stage(USD_FILEPATH)

    # Perform update steps to ensure the stage is fully loaded
    for _ in range(10):  # Multiple updates recommended for stability
        simulation_app.update()

    # Set the camera view
    print(f"[Isaac Sim] Setting camera to: {CAMERA_POSITION}")
    set_camera_view(eye=CAMERA_POSITION, target=CAMERA_TARGET)

    # Main Simulation Loop
    while simulation_app.is_running():
        simulation_app.update()

    # Cleanup on exit
    simulation_app.close()

if __name__ == "__main__":
    main()

