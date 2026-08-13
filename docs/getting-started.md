# Getting started

Install the toolchain, then turn a folder of handheld GoPro clips into a robot-learning dataset with one command.

**Read this if:** you have just cloned voilab and want a working first run.

**Before you start:** Nothing — this is the entry point.

---

## What voilab is

voilab is a fork of **UMI** (Universal Manipulation Interface, from real-stanford). UMI's idea: instead of teleoperating a robot to collect training data, a person holds a 3D-printed gripper with a GoPro bolted to it and does the task by hand. The recorded video is then reverse-engineered into robot end-effector trajectories. voilab keeps that pipeline and builds tooling around it.

The repository has four halves. Only the first two are needed for your first run.

| Half | What it does | Lives in |
|---|---|---|
| **SLAM pipeline** | GoPro MP4s to a training dataset. *SLAM* = simultaneous localisation and mapping: recovering the camera's 3D path plus a map of the room, from video alone. | `packages/umi/` |
| **Visualization** | Jupyter notebooks served as single-page web apps (via Voila) for eyeballing a session or a finished dataset. | `src/voilab/`, `nbs/` |
| **Diffusion-policy training** | Trains a visuomotor policy on that dataset. | `packages/diffusion_policy/` |
| **Isaac Sim simulation** | Replays the real object layout in NVIDIA Isaac Sim to generate more data or evaluate a policy. | `scripts/v1/` — untracked, so a fresh clone does not have it ([Simulation and Docker](./simulation-and-docker.md)) |

You do **not** need a physical robot, and you cannot currently drive one from this fork — the real-robot control code under `packages/umi/src/umi/real_world/` has an undeclared `zerorpc` dependency and no entry point. See [Known issues](./known-issues.md).

## The data flow

```text
  GoPro MP4s                 (one session = one recording day)
      |
      |  uv run umi run-slam-pipeline <config.yaml> --session-dir <SESSION>
      v
  <SESSION>/demos/*/         imu_data.json, camera_trajectory.csv,
      |                      tag_detection.pkl, map_atlas.osa
      |                      (+ object_poses.json, reconstruct config only)
      v
  <SESSION>/dataset.zarr.zip     <- the deliverable
      |
      +--> diffusion policy training  (packages/diffusion_policy/train.py)
      +--> Isaac Sim rollout / more synthetic data  (scripts/v1/)
```

`imu_data.json` — the IMU (inertial measurement unit: accelerometer + gyroscope) stream the GoPro records alongside the video; ORB-SLAM3 fuses it with the images. `map_atlas.osa` — the ORB-SLAM3 *atlas*: the saved 3-D map of the room, which stage 3 localises every demo clip against. Deleting it is how you force stage 2 to re-run. *zarr* is an array storage format, like HDF5 but directory- or zip-based; `dataset.zarr.zip` is one zip file holding all episode images and poses. Its schema is in [Data formats](./data-formats.md).

## Prerequisites

The repo is a **uv workspace**: uv is the package manager, and one virtualenv at `.venv` serves all three packages. Python **3.10** is what you want — it is pinned in `.python-version`, and although `pyproject.toml:13` says `>=3.10, <3.13`, the pinned `torch==2.1.0` only ships cp310/cp311 wheels.

| Tool | Minimum | Needed for | Install |
|---|---|---|---|
| uv | any recent | everything | `make install-uv` (`Makefile:2`) |
| Python 3.10 | 3.10 | the venv | downloaded automatically by `uv sync` |
| exiftool | **12.5** | reading GoPro camera serial numbers out of MP4 metadata (`Makefile:32`) | `make install-exiftool` |
| cmake | any | compiling `py-gpmf-parser`, a C extension built from git during install | `make install-cmake` |
| C/C++ toolchain | any | compiling `py-gpmf-parser` and `spnav`, which build from source during `uv sync` | `sudo apt install -y build-essential` |
| ffmpeg | any | transcoding 120 fps clips down to 60 fps inside the SLAM stage (`packages/umi/src/umi/services/slam_mapping.py:401`) | `make install-ffmpeg` |
| Docker + `docker compose` | Engine with the compose plugin | ORB-SLAM3 runs as a container, not as a library | your distro's docker install, then [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/) so `docker run hello-world` works **without `sudo`** — the pipeline invokes `docker` directly and never escalates |
| `chicheng/orb_slam3:latest` | — | the SLAM engine itself | pulled automatically on first run (`slam_mapping.py:29`, `pull_docker` defaults to `True`) — budget a multi-GB download |
| X11 display (`$DISPLAY`) | — | **only** for `umi visualize-slam-gui` and `enable_gui: true`; the normal pipeline is headless | already present on a desktop; use `ssh -X` otherwise (`visualize_slam_gui.py:126`) |
| NVIDIA GPU + driver | — | diffusion-policy training and Isaac Sim; not needed for the SLAM pipeline | vendor driver + `nvidia-container-toolkit` |

## Install

```bash
cd /path/to/voilab
make install-uv                       # bootstraps uv if it is missing
make install-exiftool                 # required by the pipeline
sudo apt install -y build-essential   # C/C++ compiler for the source builds
make install-cmake                    # required so py-gpmf-parser can compile
make install-ffmpeg
make install-dev   # uv sync --extra dev: runtime deps plus the plotly the viewers need
```

`make install-dev` (`Makefile:19`) runs `uv sync --extra dev`: it resolves against `uv.lock`, downloads CPython 3.10 if absent, creates `.venv`, installs `voilab`, `umi` and `diffusion_policy` in editable mode, and adds the `dev` extra from `pyproject.toml:34-45` (pytest, ruff, jupyter notebook, `jupyterlab-urdf`, plotly). Editable means **edits to `packages/umi/src/umi/*.py` take effect immediately** — there is no reinstall step.

`make install` (`Makefile:13`) is the same thing without the dev extra, which leaves the viewers broken: they import plotly at module load (`src/voilab/applications/dataset_visualizer.py:16`).

Details on the workspace, extras and lockfile live in [uv workspace](./uv-workspace.md).

## Verify your install

```bash
cd /path/to/voilab
uv run umi --help
uv run voilab --help
```

You should see two commands under `umi` (`run-slam-pipeline`, `visualize-slam-gui`) and four under `voilab` (`launch-dataset-visualizer`, `launch-simulator`, `launch-viewer`, `replay-trajectory`). Neither group has a `--version` flag.

Do not use the test suite as an install check. **Known issue:** 41 of the 155 tests in `packages/umi/tests/` fail against the current code (they call service methods that were removed), and `packages/umi/tests/services/test_aruco_detection.py` **hangs forever** — a plain `uv run pytest packages/umi/tests` never terminates; add `--ignore=packages/umi/tests/services/test_aruco_detection.py`. There is no `make test` target. See [Known issues](./known-issues.md).

## The golden rule: run every command from the repository root

Path strings inside the shipped pipeline YAMLs are resolved against the **current working directory**, not against the config file and not against the package. `_resolve_settings_file_path` does `settings_path = Path.cwd() / settings_path` for any relative path (`packages/umi/src/umi/services/slam_mapping.py:484-486`), and the shipped configs use repo-root-relative strings such as `packages/umi/defaults/orb_slam3_settings/...` (`umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml:21`).

The Voila viewers have the same constraint for a different reason: `voilab launch-viewer` opens the relative path `nbs/replay_buffer_viewer.ipynb` (`src/voilab/cli.py:18`).

So: **always `cd` to the repository root first.** Running `umi run-slam-pipeline` from anywhere else fails with `FileNotFoundError: SLAM settings file not found`.

## Your first pipeline run

### 1. Lay out a session directory

A *session* is one recording day: one mapping sweep (a slow walk around the room so SLAM can build a map), one or more gripper-calibration clips (a few seconds of the gripper opening and closing fully with its ArUco tag — a square black-and-white fiducial marker — in frame, so the pipeline can measure it to learn the gripper's min and max width), and the demonstration clips.

```bash
export SESSION=/path/to/voilab/data/2026_01_20_kitchen
mkdir -p "$SESSION/raw_videos"
cp /media/gopro/DCIM/100GOPRO/*.MP4 "$SESSION/raw_videos/"
```

**No footage yet?** Download upstream UMI's demo session and use it as your `$SESSION`: `wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative --reject="index.html*" https://real.stanford.edu/umi/data/example_demo_session/` (see [packages/umi/README.md](../packages/umi/README.md)). It is GoPro 9 footage, so run it with `official_gopro9_pipeline_config.yaml`, not the GoPro 13 config.

The first stage sorts this out for you (`packages/umi/src/umi/services/video_organization.py:19-111`):

- the **largest** clip is promoted to `raw_videos/mapping.mp4` — this must be your slow sweep of the room (`:45-52`);
- the **earliest** clip per camera serial is moved into `raw_videos/gripper_calibration/` (`:54-75`);
- everything else becomes a demo, fanned out into `<SESSION>/demos/demo_<serial>_<timestamp>/raw_video.mp4`.

If the automatic choice is wrong, rename the file by hand before running: `mv "$SESSION/raw_videos/GX010042.MP4" "$SESSION/raw_videos/mapping.mp4"`.

### 2. Pick a config

Five YAMLs ship in `umi_pipeline_configs/`. Use **`gopro13_fisheye_2-7k_pipeline_config.yaml`** for a GoPro 13 in 2.7K 4:3 fisheye, or `official_gopro9_pipeline_config.yaml` for the original UMI GoPro 9 hardware. Always pass `--session-dir`, because two of the configs hardcode someone else's absolute path (`official_gopro9_pipeline_config.yaml:4`).

**Known issue:** `gopro13_wide_angle_pipeline_config.yaml:20` and `:32` point at `gopro13_fisheye_60fps_2-7k.yaml`, which does not exist in `packages/umi/defaults/orb_slam3_settings/`, so that config dies at stage 2. `gopro13_normal_pipeline_config.yaml` is broken too: it names no `slam_settings_file` at all, so it falls back to an in-container path that does not exist on the host. See [Known issues](./known-issues.md).

### 3. Run it

```bash
cd /path/to/voilab
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
  --session-dir "$SESSION" 2>&1 | tee "$SESSION/pipeline.log"
```

`--session-dir` must already exist; Click rejects it otherwise. There is no `--dry-run` and no way to run a single stage — the CLI always calls `execute_all()` (`packages/umi/src/umi/cli.py:17`). Reruns are cheap, though: most stages skip work whose output file is already on disk.

Rough timing: the first two stages take seconds; SLAM (stages 2 and 3) runs at roughly video rate, so budget about the total length of your footage, plus the one-time Docker image pull. Stage 3 parallelises across `cpu_count() // 2` threads.

Expect this afterwards:

```text
$SESSION/
├── raw_videos/
├── demos/
│   ├── mapping/          map_atlas.osa, mapping_camera_trajectory.csv,
│   │                     camera_trajectory.csv, tag_detection.pkl,
│   │                     tx_slam_tag.json, slam_stdout.txt
│   ├── gripper_calibration_<serial>_<ts>/    gripper_range.json
│   └── demo_<serial>_<ts>/                   camera_trajectory.csv, tag_detection.pkl
├── dataset_plan.pkl
└── dataset.zarr.zip      <- the deliverable
```

**Known issue:** if IMU extraction throws for a clip, the service catches the exception, prints one line to stdout and writes a **fabricated** IMU file — a single accelerometer sample of `[0.0, 0.0, 9.8]` at 30 fps (`packages/umi/src/umi/services/imu_extraction.py:127-144`). SLAM then runs on garbage motion data instead of failing. Grep your log for `Error processing` before trusting a result. See [Known issues](./known-issues.md).

### 4. Look at the result

```bash
cd /path/to/voilab
uv run voilab launch-dataset-visualizer
```

This serves `nbs/dataset_visualizer.ipynb` with Voila and prints a URL. Voila listens on `localhost:8866` and the command takes no flags to change that, so if you are working over SSH, forward the port from your laptop first: `ssh -L 8866:localhost:8866 <lab-host>`, then open the printed URL locally. Paste your **session directory** (not the zarr file) into the "Session Dir" box and press Load; the widget defaults to `../video`. You get per-stage completion status, per-demo quality metrics, the 3D SLAM trajectory, and the ArUco tag overlay — the markers stuck to the gripper and the table, which the pipeline uses to measure gripper width and to anchor the world frame.

To inspect the finished `dataset.zarr.zip` itself, use `uv run voilab launch-viewer` instead. Both are covered in [Visualization](./visualization.md).

If stage 2 failed, read `$SESSION/demos/mapping/slam_stdout.txt` first, then see [ORB-SLAM3](./orb-slam3.md) for the interactive GUI debugger.

## Where to go next

| You want to... | Go to |
|---|---|
| Understand what each of the eight stages actually does | [Pipeline overview](./pipeline-overview.md) |
| Change a config key, or write your own YAML | [Pipeline config](./pipeline-config.md) |
| Use a different camera, or migrate from GoPro 9 to 13 | [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md), [Calibration](./calibration-openicc.md) |
| Know what every file in a session means | [Data formats](./data-formats.md) |
| Find your way around the repo | [Repository map](./repository-map.md) |
| Train a policy on your dataset | [Training and eval](./training-and-eval.md) |
| Run the simulator | [Simulation and Docker](./simulation-and-docker.md) |
| Fix something that broke | [Known issues](./known-issues.md) |

---

**Next:** [Pipeline overview](./pipeline-overview.md) · [Repository map](./repository-map.md) · [Project status](./project-status.md) · [Known issues](./known-issues.md)
