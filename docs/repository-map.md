# Repository map

A guided tour of the directory layout, so you know which few thousand lines to read and which ten thousand to skip.

**Read this if:** you have cloned the repo and want to know what each directory is for before you touch anything.

**Before you start:** [Getting started](./getting-started.md), if you want a working install while you read.

---

## 1. The mental model: four halves and some ballast

| Half | Lives in | What it does |
|---|---|---|
| SLAM pipeline | `packages/umi/` | Turns handheld GoPro videos into a robot-learning dataset. SLAM ("simultaneous localization and mapping") is the technique that recovers where a camera was while it filmed. |
| Visualization | `src/voilab/`, `nbs/` | Notebook viewers that show what each pipeline stage produced. Voila serves a notebook as a read-only web page. |
| Training | `packages/diffusion_policy/` | Trains a diffusion policy — a neural network that generates short sequences of robot actions — on that dataset. |
| Simulation | `scripts/v1/`, `assets/` | Rebuilds the real captured scene inside NVIDIA Isaac Sim to generate extra data or evaluate a trained policy. |
| Ballast | `real_world/`, `traj_eval/`, `scripts/generate_data.py`, `scripts/run_isaacsim_eval.py`, `dependencies/` | Roughly 9,000 lines carried over from upstream UMI or abandoned mid-refactor. None of it runs. Section 3 and [Known issues](./known-issues.md) name every piece. |

The two live halves meet in the session directory. A session is one capture run; see [Data formats](./data-formats.md) for its full layout.

```text
GoPro .MP4  ──►  umi run-slam-pipeline  ──►  <session>/
                 (packages/umi)              ├── dataset.zarr.zip ──►  train.py
                                             │                        (packages/diffusion_policy)
                                             └── demos/mapping/
                                                 object_poses.json ──►  scripts/v1/sdg.py
                                                                       scripts/v1/rollout.py
                 src/voilab + nbs/  ──►  read <session>/ at any stage
```

`dataset.zarr.zip` is a zarr store (a chunked array format, like HDF5 but directory-shaped) packed into a single zip file. `object_poses.json` records where each real object sat in the capture, written at `packages/umi/src/umi/services/frame_to_pose.py:327` and read back by the simulator at `scripts/v1/rollout.py:257`. That file is the only link between the real half and the simulated half.

## 2. Top-level directories

Status vocabulary: **Source** (hand-written, tracked), **Config**, **Generated** (gitignored output), **Simulation**, **Vendored** (third-party copy), **Dead/Stale**, **Untracked** (on this machine but not in git).

| Directory | What it is | Status |
|---|---|---|
| `src/` | The root `voilab` package: the `voilab` CLI, three viewer modules and two Isaac Sim helpers (`isaac_sim_config.py`, `isaac_sim_launcher.py`) under `applications/`, and the viewers' loaders under `utils/`. 11 Python files. | Source |
| `packages/` | The two workspace member packages, `umi` (pipeline, 14,989 lines) and `diffusion_policy` (training, 28,264 lines). See [uv workspace](./uv-workspace.md). | Source |
| `umi_pipeline_configs/` | Five pipeline YAML files plus `CONFIG_PROPAGATION_GUIDE.md`. One config per camera and lens. See [Pipeline config](./pipeline-config.md). | Config |
| `nbs/` | Four notebooks: `replay_buffer_viewer`, `dataset_visualizer`, `aruco_detection_viewer`, `slam_viewer`. These are the Voila entry points. See [Visualization](./visualization.md). | Source |
| `docs/` | This documentation set. | Source |
| `assets/` | 110 MB of USD scene files and URDF robot descriptions for Isaac Sim (`franka_panda/`, `ED305_scene/`, `CADs/`, `lula/`, …). USD is Pixar's Universal Scene Description, the format Isaac Sim loads. | Simulation |
| `deps/` | Exactly one vendored wheel, `jupyterlab_urdf-0.6.0-py3-none-any.whl`, referenced at `pyproject.toml:59`. | Vendored |
| `scripts/` | Isaac Sim entry points. `v1/` is the working rewrite. Of the four `.py` files beside it, `generate_data.py` and `run_isaacsim_eval.py` are the superseded entry points and no longer import; `umi_replay.py` and `motion_plan.py` are still imported by `v1/engine.py`. See section 5. | Simulation |
| `submodules/` | A gitlink to `IsaacSim-ros_workspaces`, never initialized and referenced by no code in the repo. `Dockerfile.isaacsim:118` is its only consumer, and that Dockerfile is itself orphaned. | Dead/Stale |
| `experiments/` | Camera-calibration scratch work: `calibration.py`, plus ChArUco notebooks. A ChArUco board is a chessboard with ArUco fiducial markers in the white squares, used to measure a lens. See [Calibration](./calibration-openicc.md). | Source |
| `media/` | 30 MB of screenshots and GIFs embedded by [Visualization](./visualization.md). | Source |
| `video/` | Skeleton for capture data. `video/raw_videos/*` is gitignored (`.gitignore:218`); the three `example_*` session directories are tracked and contain only `object_poses.json`. | Generated |
| `data/` | 23 MB of `eval_outputs/session_*`, written by containers running as root. Not in git and not gitignored. | Untracked |
| `cache/` | Empty Omniverse cache stubs created by the Isaac Sim container, root-owned. | Untracked |
| `dependencies/` | A 58 MB clone of NVIDIA IsaacLab that nothing in the repo imports — the single reference is a code comment at `scripts/v1/configs/simple_room.py:46`. Safe to delete. | Untracked, Dead |
| `experiment_wpnoise0629/` | 2.2 GB experiment dump: one training dataset, two SLURM job scripts and their checkpoints. The SLURM scripts are the only working end-to-end training example in the tree. See [Training and eval](./training-and-eval.md). | Untracked |
| `Dockerfile` | The Isaac Sim image (compose service `isaac-sim`). The current one. | Config |
| `Dockerfile.client` | ROS 2 Humble + cuRobo + nvblox image (compose service `voilab-workspace`). | Config |
| `Dockerfile.isaacsim` | Builds ROS 2 from source with pip instead of uv. Referenced by nothing. | Dead/Stale |
| `docker-compose.yaml` | Defines the two services above. See [Simulation and Docker](./simulation-and-docker.md). | Config |
| `Makefile` | Ten targets: uv bootstrap, `uv sync`, JupyterLab, and apt installs for exiftool/cmake/ffmpeg. See [uv workspace](./uv-workspace.md). | Config |
| `*.sh` (root) | `launch_workspace.sh` starts the dev container (`make launch-workspace`, `Makefile:66`) and `entrypoint.sh` is that image's entrypoint (`Dockerfile:95`). `launch_isaacsim.sh`, `entrypoint.client.sh` and `entrypoint.isaacsim.sh` are referenced by nothing. See [Simulation and Docker](./simulation-and-docker.md). | Config |

## 3. Inside `packages/umi/src/umi/`

This is where a newcomer wastes the most time. Over half of it does not run.

| Subpackage | Purpose | Size | Verdict |
|---|---|---|---|
| `services/` | The pipeline stages. 14 modules defining 12 `BaseService` subclasses plus `BaseService` itself (`services/base_service.py:7`) and `ConfigService` (`services/config.py:5`). Start here. | ~3,500 lines | live |
| `infrastructure/` | `replay_buffer.py` (the zarr dataset writer/reader) and `imagecodecs_numcodecs.py` (image compression codecs, including JPEG-XL). | ~2,000 lines | live |
| `common/` | Pose maths, video and timecode helpers used by `services/`. Four modules have zero importers repo-wide: `exiftool_util.py`, `k3d_util.py`, `mocap_util.py`, `latency_util.py`. | ~1,500 lines | mostly live |
| `real_world/` | Drivers for real robots and cameras: UR5, Franka, a Weiss WSG-50 gripper, capture cards, a SpaceMouse. | ~4,500 lines | dead |
| `shared_memory/` | Lock-free ring buffers that let each `real_world/` driver run in its own process. Imported only by `real_world/`. | ~600 lines | dead |
| `traj_eval/` | A verbatim copy of `uzh-rpg/rpg_trajectory_evaluation` plus Christoph Gohlke's `transformations.py`. Five files still carry `#!/usr/bin/env python2` shebangs. Zero importers anywhere. | ~2,500 lines | vendored, dead |
| `pipeline/` | One file, `aruco_detection.py`, which is 0 bytes. | 0 lines | dead |
| `asset/` | Not code: `mask.json` and `mask_4k.json`, polygon outlines used to mask the mirror out of a fisheye frame. | 2 files | see below |
| `defaults/` | Not under `src/` — the real path is `packages/umi/defaults/`. Ships the ORB-SLAM3 settings and camera calibration files described in section 4, plus `eval_robots_config.yaml` (UR5e and gripper IP addresses) which nothing in the repo reads. | 13 files | mostly live |

**Known issue:** `real_world/` cannot be imported after a normal `uv sync`. `packages/umi/src/umi/real_world/franka_interpolation_controller.py:14` imports `zerorpc`, which is declared in no `pyproject.toml` and appears nowhere in `uv.lock`; both env classes import that controller unconditionally. There is no entry point for any of it either. See [Known issues](./known-issues.md).

**Known issue:** `mask_pts_json_path` is set by three pipeline configs but is only assigned, never read — `packages/umi/src/umi/services/slam_mapping.py:36` is its sole occurrence in the package. The `asset/` masks therefore have no effect. See [Known issues](./known-issues.md).

## 4. Where configuration lives

Three separate places, read by three different layers.

| Location | Contents | Read by |
|---|---|---|
| `umi_pipeline_configs/` | Five pipeline YAMLs. Each top-level key is a stage; each stage names a service class and its settings. | `PipelineExecutor._load_config` at `packages/umi/src/umi/pipeline_executor.py:101`, via `uv run umi run-slam-pipeline <config>`. |
| `packages/umi/defaults/orb_slam3_settings/` | Seven ORB-SLAM3 settings files (camera model, intrinsics — the lens's focal length, optical centre and distortion coefficients — ORB feature counts, IMU noise). Named per camera and resolution. | ORB-SLAM3 inside its Docker container, handed the path by the SLAM stages. See [ORB-SLAM3](./orb-slam3.md). |
| `packages/umi/defaults/calibration/` | Four camera-intrinsics JSON files (`gopro13_intrinsics_2_7k.json`, `gopro13_intrinsics_4k.json`, `gopro9_intrinsics_2_7k.json`, `gopro9_intrinsics_normal_lens.json`) and `aruco_config.yaml`. | The ArUco detection and calibration stages, plus the ArUco viewer. See [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md). |

The pipeline YAMLs reach the other two by **repo-root-relative** string paths, for example `umi_pipeline_configs/official_gopro9_pipeline_config.yaml:20`:

```yaml
slam_settings_file: "packages/umi/defaults/orb_slam3_settings/gopro10_maxlens_fisheye_setting_v1_720.yaml"
```

Consequence: run `umi` with the repo root as your working directory, always.

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir <session>
```

**Known issue:** `umi_pipeline_configs/gopro13_wide_angle_pipeline_config.yaml:20` and `:32` point at `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_60fps_2-7k.yaml`, which does not exist in that directory. See [Known issues](./known-issues.md).

## 5. What a fresh clone does not have

Several directories exist on the machine this repo was developed on but are absent from git. `git check-ignore` reports that of these, only `.venv` is actually gitignored — the rest are plain untracked files that were never committed.

| Path | Size | Effect on a fresh clone |
|---|---|---|
| `scripts/v1/` | ~2,850 lines | **The entire working simulation side is missing.** |
| `dependencies/IsaacLab/` | 58 MB | No effect; nothing imports it. |
| `experiment_wpnoise0629/` | 2.2 GB | The worked training example is missing. |
| `data/`, `cache/` | 23 MB | No effect; both are container output. |

The simulation gap is the dangerous one. `git status` on this working tree also shows nine **deleted but uncommitted** files under `scripts/` — `registry/`, `utils.py` and `object_loader.py` among them. A fresh clone gets those files back but not their replacement; this tree has the opposite problem:

- `scripts/generate_data.py:15` does `import registry`, a package that was deleted here and will be present but stale in a clone. `scripts/run_isaacsim_eval.py:10,17` is in the same state.
- `scripts/v1/` is not self-contained. `scripts/v1/engine.py:48-50` imports `object_loader`, `umi_replay` and `motion_plan` from the parent `scripts/` directory, and every v1 entry point imports `engine`. `object_loader.py` is one of the deleted files, so v1 also fails at import here until you run `git checkout -- scripts/object_loader.py`.
- `umi_replay.py` and `motion_plan.py` are still tracked and still used. `motion_plan.py:4`'s `from utils import ...` resolves to `scripts/v1/utils.py`, which shadows the deleted `scripts/utils.py` during a v1 run.
- `uv run voilab launch-simulator` invokes `scripts/generate_data.py` (`src/voilab/cli.py:62`), so that CLI command fails either way.

**Known issue:** the simulation half of this repo is not reproducible from git alone. Treat `scripts/v1/` as the real implementation, and `generate_data.py` and `run_isaacsim_eval.py` beside it as stale. See [Known issues](./known-issues.md) and [Simulation and Docker](./simulation-and-docker.md).

Also root-owned, because containers bind-mount the repo and run as root: `data/`, `cache/`, `debug.log`, `entrypoint.sh`. You will hit `Permission denied` editing them.

## 6. Conventions

- **Services subclass `BaseService`.** The executor imports each stage's class by dotted path from the YAML `instance:` field (`pipeline_executor.py:180`) and rejects anything that is not a `BaseService` instance (`pipeline_executor.py:204`). Adding a stage means adding a class and a YAML block, nothing else.
- **Stage keys are numbered by convention only.** `list_stages()` returns `list(self.config.keys())` (`pipeline_executor.py:232`) and `execute_all` walks that list in order (`pipeline_executor.py:288`). Names like `00_process_video` are ordinary YAML keys; the digits are for humans. Reorder the file and you reorder the pipeline.
- **`nbs/*.ipynb` are Voila entry points, not scratch notebooks.** Two of them have CLI launchers (`src/voilab/cli.py:16`, `:23`) that pass a path relative to the repo root, so they only work from there. The other two are opened by hand.
- **Two spellings of the session flag exist.** The `umi` CLI uses `--session-dir`; `voilab launch-simulator` and every `scripts/v1/` entry point use `--session_dir`. See [CLI reference](./cli-reference.md).

---

**Next:** [Pipeline overview](./pipeline-overview.md) · [Known issues](./known-issues.md) · [uv workspace](./uv-workspace.md)
