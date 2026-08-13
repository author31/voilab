# Voilab

Turn handheld GoPro demonstrations into robot-learning datasets, then inspect, train on, and simulate them.

Voilab is a fork of [UMI](https://github.com/real-stanford/universal_manipulation_interface) (Universal Manipulation Interface). Instead of teleoperating a robot to collect training data, a person holds a 3D-printed gripper with a GoPro on it and performs the task by hand; the recorded video is reverse-engineered into end-effector trajectories using visual-inertial SLAM. This fork rebuilds UMI's pipeline as a YAML-configured service pipeline, swaps conda for [uv](https://docs.astral.sh/uv/), adds GoPro 13 telemetry support, and layers on Jupyter/Voila viewers, diffusion-policy training, and an Isaac Sim simulation path.

```text
  GoPro MP4s  ->  umi run-slam-pipeline  ->  session artifacts  ->  dataset.zarr.zip
                                                                          |
                                                        +-----------------+-----------------+
                                                        |                                   |
                                                 diffusion policy                     Isaac Sim
                                                    training                        rollout / SDG
```

New here? Start with **[Getting started](./docs/getting-started.md)**.

---

## Documentation

### Start here

| Doc | What it covers |
|---|---|
| [Getting started](./docs/getting-started.md) | Install the toolchain and run your first GoPro-to-dataset pipeline end to end |
| [Repository map](./docs/repository-map.md) | What every directory is for, and which parts are live, dead, or missing from git |
| [Project status](./docs/project-status.md) | What is finished, what is under development, and why rollout is moving to ROS 2 |
| [Known issues](./docs/known-issues.md) | Ranked register of broken configs, silent data corruption, dead code and footguns |

### Environment and tooling

| Doc | What it covers |
|---|---|
| [uv workspace](./docs/uv-workspace.md) | The uv workspace, dependency declaration, daily commands, and Makefile targets |
| [CLI reference](./docs/cli-reference.md) | Every `umi` and `voilab` command, with flags, defaults and failure modes |

### The SLAM pipeline

| Doc | What it covers |
|---|---|
| [Pipeline overview](./docs/pipeline-overview.md) | How the YAML-driven SLAM pipeline is assembled and what each stage reads, writes and skips |
| [Pipeline configuration](./docs/pipeline-config.md) | Pipeline YAML reference: stage keys, merge and propagation rules, and per-stage settings |
| [Data formats](./docs/data-formats.md) | Session directory tree and the schema of every artifact the pipeline writes |
| [Versus upstream UMI](./docs/vs-upstream-umi.md) | Why this fork's pipeline is structured differently from real-stanford UMI, and what that changes |

### Camera, SLAM and calibration

| Doc | What it covers |
|---|---|
| [GoPro telemetry](./docs/gopro-telemetry.md) | GPMF telemetry extraction with py-gpmf-parser, and the `imu_data.json` ORB-SLAM3 consumes |
| [GoPro 9 to GoPro 13](./docs/gopro9-to-gopro13.md) | Camera-specific constants inventory and the checklist for migrating off the HERO9 defaults |
| [ORB-SLAM3](./docs/orb-slam3.md) | How ORB-SLAM3 runs as a Docker subprocess, and how to read and write its settings file |
| [Calibration with OpenICC](./docs/calibration-openicc.md) | Measure GoPro intrinsics, IMU noise and camera-IMU extrinsics for ORB-SLAM3 |

### Downstream

| Doc | What it covers |
|---|---|
| [Visualization](./docs/visualization.md) | The Jupyter/Voila viewers for inspecting sessions, replay buffers and ArUco detections |
| [Training and evaluation](./docs/training-and-eval.md) | Train a diffusion policy from a `dataset.zarr.zip`, locally or on SLURM, and evaluate it |
| [Simulation and Docker](./docs/simulation-and-docker.md) | Isaac Sim data generation and rollout, the object-pose handoff, and the container images |

### Reference notes

| Doc | What it covers |
|---|---|
| [Diffusion policy layers](./docs/diffusion_policy_layers.md) | Layer-by-layer tour of the training package: what each layer owns, and which upstream modules this fork removed |
| [ROS 2 integration design](./docs/ros2_integration_design.md) | The three-layer ROS 2 stack in the training package, its real class signatures, and its known defects |

---

## Quickstart

**Before you run this:** Linux with Docker (stage 2 runs ORB-SLAM3 in a container and pulls ~1 GB on first use), plus `make install-ffmpeg install-exiftool` for the video and metadata tools. Full prerequisites: [Getting started](./docs/getting-started.md).

```bash
# 1. Install uv and all dependencies (dev extra needed for the viewers)
make install-dev

# 2. Check the two CLIs are on the path
uv run umi --help
uv run voilab --help

# 3. Run the pipeline on a session directory of GoPro clips
# my_session/ = a folder of GoPro .MP4 clips (one slow room sweep + gripper-calibration clips + demos).
# No footage yet? Getting started links a downloadable example session — it is GoPro 9
# footage, so run that one with official_gopro9_pipeline_config.yaml instead.
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
    --session-dir /abs/path/to/my_session

# 4. Open dataset.zarr.zip
uv run voilab launch-viewer
uv run voilab launch-dataset-visualizer   # or: review the session (SLAM, ArUco, per-demo quality)
```

Run every command from the repository root — relative paths inside the shipped configs resolve against the current working directory. Full walkthrough: [Getting started](./docs/getting-started.md).

---

## Contributing

- **Pipeline stage**: subclass `BaseService` in `packages/umi/src/umi/services/`, then reference it by dotted path from a config in `umi_pipeline_configs/`. See [Pipeline overview](./docs/pipeline-overview.md).
- **Viewer**: put the logic in `src/voilab/applications/`, data loading in `src/voilab/utils/`, and a notebook entry point in `nbs/`. See [Visualization](./docs/visualization.md).
- **Dependencies**: add them with `uv add --package <member>` so `uv.lock` stays in sync; the lockfile is committed. See [uv workspace](./docs/uv-workspace.md).
- **Docs**: one topic per file under `docs/`, and add the row to the index above.
