# What changed versus upstream UMI

This fork keeps upstream UMI's algorithms but replaces its eight-script command-line pipeline with one YAML-configured, class-based pipeline so that a different camera is a config change rather than a source edit.

**Read this if:** you have used or read [real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) and want to know why the code here does not look like it.

**Before you start:** [Pipeline overview](./pipeline-overview.md), [Repository map](./repository-map.md)

---

Upstream paths and line numbers anywhere on this page — `scripts_slam_pipeline/`, `run_slam_pipeline.py`, `conda_environment.yaml`, `tests/`, `scripts_real/`, and the whole "Upstream" column below — are to an upstream checkout and cannot be checked from this repository, which vendors only upstream's README as `packages/umi/README.md`. Paths under `packages/`, `src/`, `umi_pipeline_configs/` or the repo root are fork-side and checkable here.

## Upstream in one paragraph

Upstream UMI turns handheld GoPro video into a robot-learning dataset using SLAM (Simultaneous Localization and Mapping — recovering the camera's 3D trajectory from its own video). It ships that as a directory of eight numbered standalone scripts in `scripts_slam_pipeline/`, driven by a top-level `run_slam_pipeline.py` that runs each one as a separate `python <script> ...` subprocess. All configuration is command-line flags, with per-script defaults and several values baked into the source. The environment is conda (a package manager that ships non-Python binaries alongside Python) pinned to Python 3.9 in `conda_environment.yaml:7`. The eight scripts total 1806 lines, of which `06_generate_dataset_plan.py` alone is 787.

```text
upstream:   run_slam_pipeline.py  --subprocess-->  00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06
                                                                          (07 run by hand)
this fork:  umi run-slam-pipeline <config.yaml>  -->  PipelineExecutor  -->  00..07 in-process
```

## The developer-experience problems

| Problem | What it looks like upstream | Why it hurts |
|---|---|---|
| Eight separate flag vocabularies | Each script defines its own `click` options; short flags collide. `-i` is `--input_dir` in `02_create_map.py:26` but `--input` in `06_generate_dataset_plan.py:83`. `-ml` is `--max_lost_frames` in `03_batch_slam.py:44` but `--min_episode_length` in `06_generate_dataset_plan.py:88`. `-nm` is `--no_mask` in `02_create_map.py:30` but `--no_mirror` in `07_generate_replay_buffer.py:42`. | Nothing transfers between stages. Re-running one stage means re-reading its source to recall which spelling it wants. |
| `sys.path` and `chdir` in every file | Every script opens with `sys.path.append(ROOT_DIR)` then `os.chdir(ROOT_DIR)` (`00_process_videos.py:9-10`, repeated in 01-07 and the driver). | The repo is not an installable package. `os.chdir` also silently reinterprets any relative path you typed as relative to the repo root. |
| Hardcoded device constants | The ORB-SLAM3 settings file is a literal path inside the Docker image, `/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml`, at `02_create_map.py:80` and again at `03_batch_slam.py:117`. The SLAM mask resolution `(2028, 2704)` is literal at `02_create_map.py:62` and `03_batch_slam.py:99`. | No flag exists for either. Supporting a different camera means editing source, or rebuilding the container image. |
| No shared configuration | The session path is retyped on every subprocess line, and the map atlas location is recomputed in the driver (`run_slam_pipeline.py:59`) and defaulted again inside `02_create_map.py:36` and `03_batch_slam.py:53`. | Three places encode one fact. They drift. |
| The driver stops before the last stage | The last block in `run_slam_pipeline.py` is stage 06 (`:108-116`). `07_generate_replay_buffer.py` is never invoked; the upstream README, vendored here as [`packages/umi/README.md:98`](../packages/umi/README.md), tells you to run it by hand with a different argument shape (`-o <output>` plus a positional input). | "One command" is really two, and the second is the one that produces the dataset you actually train on. |
| Conda environment drift | `mamba env create -f conda_environment.yaml` plus manual `apt` installs (`packages/umi/README.md:55-60`). Some pins are conda-only, e.g. `exiftool=12`, with `PyExifTool` annotated as implicitly depending on it. | No lockfile for the mixed conda/pip set, so two machines resolve differently. |
| No pipeline tests | Upstream's `tests/` holds three files: `test_multi_uvc_camera.py`, `test_pose_util.py`, `test_uvc_camera.py`. Two need physical cameras. This repo has no top-level `tests/` — its suite is `packages/umi/tests/`. | Nothing covers stages 00-07. Refactoring is unguarded. |

## Stage mapping

Every upstream script became a class implementing one method, `execute()`. Stage keys are the top-level keys of a pipeline YAML file; see [Pipeline config](./pipeline-config.md).

| Upstream script | Stage key in this repo | Service class (file:line) |
|---|---|---|
| `00_process_videos.py` | `00_process_video` | `VideoOrganizationService` — packages/umi/src/umi/services/video_organization.py:11 |
| `01_extract_gopro_imu.py` | `01_extract_gopro_imu` | `IMUExtractionService` — packages/umi/src/umi/services/imu_extraction.py:11 |
| `02_create_map.py` | `02_create_map` | `SLAMMappingService` (`slam_process_mode: create_map`) — packages/umi/src/umi/services/slam_mapping.py:20 |
| `03_batch_slam.py` | `03_batch_slam` | `SLAMMappingService` (`slam_process_mode: batch_slam`) — packages/umi/src/umi/services/slam_mapping.py:20 |
| `04_detect_aruco.py` | `04_detect_aruco` | `ArucoDetectionService` — packages/umi/src/umi/services/aruco_detection.py:22 |
| `05_run_calibrations.py` | `05_run_calibrations` | `CalibrationService` — packages/umi/src/umi/services/calibration.py:17 |
| `06_generate_dataset_plan.py` | `06_generate_dataset_plan` | `DatasetPlanningService` — packages/umi/src/umi/services/dataset_planning.py:26 |
| `07_generate_replay_buffer.py` | `07_generate_replay_buffer` | `ReplayBufferService` — packages/umi/src/umi/services/replay_buffer.py:26 |
| *(none)* | `07_frame_to_pose` | `FrameToPoseService` — packages/umi/src/umi/services/frame_to_pose.py:342 |

Upstream's stages 02 and 03 are ~250 lines of near-duplicate Docker-invocation code; here they are one class selected by a `slam_process_mode` key (packages/umi/src/umi/services/slam_mapping.py:44-47). Upstream stages 04 and 05 shell out to a *second* layer of helper scripts in `scripts/`; here they are ordinary in-process methods, so exceptions carry real tracebacks instead of a return code.

## Design changes

### One YAML file instead of scattered flags

A pipeline is a YAML document; each top-level key is a stage, and its `instance:` value is a dotted class path imported at runtime. Adding or reordering a stage is a YAML edit with no Python change — `umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml:58` inserts `07_frame_to_pose` and renumbers the replay buffer to `08_generate_replay_buffer` (`:67`).

### Classes instead of scripts

Every stage subclasses `BaseService` (packages/umi/src/umi/services/base_service.py:7), whose one abstract method is `execute(*args, **kwargs)` (`:20`). By convention stages take no arguments and return a dict, but nothing enforces that: `PipelineExecutor` only rejects classes that do not inherit from `BaseService` (packages/umi/src/umi/pipeline_executor.py:204-205), and it forwards `*args, **kwargs` straight through (`:302`). There is no `if stage == ...` ladder anywhere.

### One CLI

```bash
# from the repo root
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml
uv run umi run-slam-pipeline umi_pipeline_configs/official_gopro9_pipeline_config.yaml --session-dir /path/to/session --task kitchen
```

One positional argument and two options, `--session-dir` and `--task` (packages/umi/src/umi/cli.py:11-14), registered as a console script at `packages/umi/pyproject.toml:93`. The first form relies on the config's own `session_dir`; `official_gopro9_pipeline_config.yaml:4` names a directory that is not in the repo, so it needs the override. Full flag list in [CLI reference](./cli-reference.md).

### uv workspace instead of conda

`uv` is a Python package manager with a lockfile. The root `pyproject.toml:52-58` declares a workspace whose members are `packages/*`, so `voilab`, `umi` and `diffusion_policy` install together from one resolve. `requires-python = ">=3.10, <3.13"` (root `pyproject.toml:13`) — the services use 3.10+ syntax such as `Path | None`, so upstream's 3.9 cannot run this code. System dependencies that upstream leaves to prose are Makefile targets: `make install` (runs `uv sync`), `make install-exiftool`, `make install-ffmpeg`, `make install-cmake`. See [uv workspace](./uv-workspace.md).

### Config propagation between stages

Each stage's effective config is merged forward into the next (packages/umi/src/umi/pipeline_executor.py:309), so `session_dir` is declared once — in stage `00_process_video` — and inherited by all later stages. Resolution order is: inherited config, minus `config_exclude`, merged with the stage's own `config`, merged with `config_override` (packages/umi/src/umi/pipeline_executor.py:183-195). Dicts deep-merge; lists are replaced. Rules and worked examples in [Pipeline config](./pipeline-config.md).

### A pytest suite

`packages/umi/tests/` holds 12 files and 155 tests, against upstream's zero pipeline tests. Read the honest assessment below before trusting it.

## Behavioural differences that change your dataset

Do not treat a dataset built here as comparable to an upstream-trained policy's data without checking these.

| Parameter | Upstream | This fork | Notes |
|---|---|---|---|
| `min_episode_length` | `24` (`06_generate_dataset_plan.py:88`) | `10` (packages/umi/src/umi/services/dataset_planning.py:34) | No shipped config sets it, so the fork default always applies. More than twice as many short episodes survive. |
| `nominal_z` (dataset planning) | `0.072` (`06_generate_dataset_plan.py:87`) | default `0.0` (packages/umi/src/umi/services/dataset_planning.py:33) | `official_gopro9_pipeline_config.yaml:52` sets `0.072`; `gopro13_fisheye_2-7k_pipeline_config.yaml` sets `0.01` on stage 05 (`:50`) and nothing on stage 06, which then inherits `0.01`. See the collision note below. |
| Output image resolution | `--out_res 224,224` (`07_generate_replay_buffer.py:39`) | default `[256, 256]` (packages/umi/src/umi/services/replay_buffer.py:33) | `official_gopro9_pipeline_config.yaml:59` sets `[256, 256]`, so the "official gopro9" config does **not** reproduce upstream's output size. All four gopro13 configs set `[224, 224]`. |
| ArUco worker count | `cpu_count()` (`04_detect_aruco.py:37-38`) | `cpu_count() // 2` (packages/umi/src/umi/services/aruco_detection.py:28) | Halved. Slower, not different output. |
| IMU extraction backend | `chicheng/openicc` Docker image running a Node.js script (`01_extract_gopro_imu.py:63-73`) | `py-gpmf-parser` Python library (packages/umi/src/umi/services/imu_extraction.py:4) | Different parser reading the same GPMF stream (GoPro's per-frame telemetry track). See [GoPro telemetry](./gopro-telemetry.md). |
| Python | `3.9` (`conda_environment.yaml:7`) | `>=3.10, <3.13` (root `pyproject.toml:13`) | |

Unchanged: `tcp_offset` 0.205, `max_lost_frames` 60, `timeout_multiple` 16, `compression_level` 99. The hardcoded gripper geometry (`cam_to_center_height = 0.086`, `cam_to_mount_offset = 0.01465`) survives verbatim at packages/umi/src/umi/services/dataset_planning.py:48-50.

**Known issue:** config propagation is a flat namespace, so a key set by one stage silently becomes the default for every later stage. `nominal_z` means different things to `CalibrationService` (packages/umi/src/umi/services/calibration.py:33) and `DatasetPlanningService` (packages/umi/src/umi/services/dataset_planning.py:33), and `umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml:50` leaks `0.01` from stage 05 into stage 06. See [Known issues](./known-issues.md).

## GoPro 13 support

Upstream targets GoPro 9/10 only. Three changes here make GoPro 13 footage work.

- **Mountable SLAM settings.** The host settings file is bind-mounted over a fixed container path (`f"{settings_file_abs_path}:/slam_settings.yaml"`, packages/umi/src/umi/services/slam_mapping.py:106) and passed as `--setting /slam_settings.yaml` (`:158`). This turns upstream's hardcoded in-image path into a config key, `slam_settings_file`. Seven ready-made profiles ship in `packages/umi/defaults/orb_slam3_settings/` and four intrinsics files in `packages/umi/defaults/calibration/`. Details in [ORB-SLAM3](./orb-slam3.md) and [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md).
- **Telemetry parsing without Docker.** `py-gpmf-parser` (declared at `packages/umi/pyproject.toml:85`) replaces the OpenICC container, and the extracted stream list is a config key with 11 defaults (packages/umi/src/umi/services/imu_extraction.py:18-33). See [GoPro telemetry](./gopro-telemetry.md).
- **Automatic 120 fps transcode.** GoPro 13 records at 120 fps; ORB-SLAM3 expects 60. `_detect_and_convert_fps` (packages/umi/src/umi/services/slam_mapping.py:401) detects `abs(fps - 120.0) < 1.0` (`:420`) and re-encodes with ffmpeg to `converted_60fps_<name>.mp4` (`:424`), caching the result. Downstream stages prefer the converted file when present (packages/umi/src/umi/services/aruco_detection.py:46-47).

## Where the fork is not better yet

- **Dead services.** `LoggingService` (packages/umi/src/umi/services/logging.py:8), `ProgressService` (packages/umi/src/umi/services/progress.py:8), `ConfigService` (packages/umi/src/umi/services/config.py:5) and `BatchSLAMService` (packages/umi/src/umi/services/batch_slam.py:9) are fully written and never instantiated. So there is no `pipeline.log`, no `progress.json`, and no stage-level resume — resumability is still upstream's "does the output file already exist" check, per stage.
- **Undocumented working-directory dependence.** Shipped configs use repo-root-relative paths such as `packages/umi/defaults/...`, and the resolver anchors relative paths to `Path.cwd()` (packages/umi/src/umi/services/slam_mapping.py:484-486). Upstream's `os.chdir(ROOT_DIR)` was ugly but made this deterministic. **Known issue:** `umi run-slam-pipeline` only works from the repo root, and `FrameToPoseService` uses a *different* base — `Path(__file__).resolve().parents[3]`, i.e. `packages/umi/` (packages/umi/src/umi/services/frame_to_pose.py:15,367) — so two path keys in one YAML file resolve against two different directories. See [Known issues](./known-issues.md).
- **No single-stage command.** `PipelineExecutor.execute_stage()` (packages/umi/src/umi/pipeline_executor.py:234) and `validate_stages()` (`:327`) exist and work, but `cli.py` exposes only `run-slam-pipeline` and `visualize-slam-gui`. Re-running one stage or dry-running a config requires writing Python.
- **Test suite health.** 155 tests collect, but `uv run pytest tests/` never terminates: `tests/services/test_aruco_detection.py` mocks `ThreadPoolExecutor` so `submit()` returns a plain `Mock`, and production code then calls the real `concurrent.futures.wait()` on it (packages/umi/src/umi/services/aruco_detection.py:61). The runnable subset is `cd packages/umi && uv run pytest tests/ --ignore=tests/services/test_aruco_detection.py -q`, which reports **41 failed, 100 passed** on a clean checkout. Many failures assert methods that were never implemented, e.g. `DatasetPlanningService.validate_plan`. Treat those tests as aspirational specs, not documentation.
- **The real-robot half of upstream does not run here.** Upstream ships `eval_real.py` and 14 scripts under `scripts_real/` for driving UR5/Franka hardware. Neither exists in this repo. The supporting modules were vendored — `packages/umi/src/umi/real_world/` still contains `umi_env.py`, `franka_interpolation_controller.py` and friends — but nothing invokes them, and they import `diffusion_policy`, which `packages/umi/pyproject.toml` does not declare as a dependency. Policy evaluation here means [training and eval](./training-and-eval.md) plus [simulation](./simulation-and-docker.md), not a physical arm.

---

**Next:** [Pipeline overview](./pipeline-overview.md) · [Project status](./project-status.md) · [Known issues](./known-issues.md) · [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md)
