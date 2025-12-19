# Data Collection in Isaac Sim v5.1.0
**Software:** NVIDIA Isaac Sim v5.1.0

**Pipeline:** UMI-Processing to Simulation-Policy Training

---

## 1. Prerequisites

Before proceeding with data collection in the simulator, ensure the following conditions are met:

* **Trajectory Dataset:** You must have a successfully constructed trajectory dataset processed via the **UMI-processing pipeline**.
* **Documentation Check:** If you have not generated these trajectories yet, please refer to the [UMI_README.md](docs/UMI_README.md) for calibration and extraction steps.
* **Environment Setup:** Ensure your `voilab` environment is active and the Isaac Sim v5.1.0 dependencies are installed.

---

## 2. Observation Replacement (Replay Workflow)

To train a policy model for simulation, we must bridge the gap between real-world observations and simulated environments. This is achieved by replaying the collected real-world trajectories and replacing the observation frames with simulation-rendered frames.

Run the following command to launch the simulator and begin the replay session:

```sh
uv run voilab launch-simulator --task kitchen --session_dir datasets/1124_gopro3_kitchen

```

### How it works:

1. **Trajectory Loading:** The simulator loads the poses and actions from the specified `session_dir`.
2. **Frame-by-Frame Replay:** The robot state is updated at every timestep to match the recorded data.
3. **Sim-Observation Capture:** The Isaac Sim cameras (e.g., RGB, Depth, Segmentation) capture the scene, replacing the original GoPro/RealSense footage.

---

## 3. Data Augmentation & Synthetic Tricks

To improve the robustness of the policy we could utilize synthetic data generation tricks.

Modify the `run_intervention` script logic to introduce noise into the replay at `scripts/launch_isaacsim_workspace.py`

* **Task:** Locate the intervention logic in your script and adjust the perturbation parameters.
* **Goal:** Force the robot to recover from slightly "off-path" states.

* **Example Modification:**
* Inject Gaussian noise into the gripper pose during the approach phase.
* Change the object's physical properties (mass/friction) to simulate diverse handling conditions.
