# Data Collection in Isaac Sim v5.1.0
**Software:** NVIDIA Isaac Sim v5.1.0

**Pipeline:** UMI-Processing to Simulation-Policy Training

---
1. Environment Setup (Docker & Dependencies)

Before running any data processing or simulation, the execution environment must be prepared.

Docker is required. Please install Docker by following the official Ubuntu installation guide:
https://docs.docker.com/engine/install/ubuntu/

After installation, verify that Docker is available by running:

```bash
docker --version
or
docker version
```
Which supposed to show the version information of docker if succesfully installed.
In addition, ensure that the voilab environment is correctly configured and that all dependencies required for NVIDIA Isaac Sim v5.1.0 are installed and accessible.

---

2. UMI Trajectory Processing (Real-World Data)

This step converts your raw sensor recordings (e.g., GoPro or RealSense data).

Command for running the UMI SLAM and trajectory processing pipeline:

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml --session-dir {data_path}
```

The output of this step is a trajectory dataset stored under the specified session directory. This dataset will be used as input for simulation replay.

For detailed calibration and extraction procedures, please refer to docs/UMI_README.md.

---

3. Isaac Sim Replay and Observation Replacement

Command for data replay and collection in Isacc Sim.

--task -> Your Task Scene

--session -> Your Data Address
```bash
uv run voilab launch-simulator --task kitchen --session_dir {data_path}
```

The resulting dataset preserves real-world motion while providing fully simulated visual observations.

Every time collecting dataset, remember to purge the previous failed data:

```bash
voilab/dataset/{dataset_name}/.previous_progress.json
```

---



4. Diffusion Policy Training

After simulation replay, the generated dataset can be used to train a diffusion-based policy.

Command for training the diffusion policy:

```bash
uv run packages/diffusion_policy/train.py --config-path=src/diffusion_policy/config --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=/path/to/your/dataset.zarr.zip
```







