# CLI reference

Exact reference for the two console scripts this repo installs: `umi` (the data pipeline) and `voilab` (viewers, simulator, trajectory replay).

**Read this if:** you want the real flags, defaults, and failure modes of a command before you run it.

**Before you start:** [Getting started](./getting-started.md), [uv workspace](./uv-workspace.md)

---

## The two entry points

| Script | Package | Declared in | Source | Purpose |
|---|---|---|---|---|
| `umi` | `umi` (workspace member) | `packages/umi/pyproject.toml:92-93` | `packages/umi/src/umi/cli.py` | Turn GoPro videos into a training dataset |
| `voilab` | `voilab` (root) | `pyproject.toml:48-49` | `src/voilab/cli.py` | Launch viewers, the simulator, trajectory replay |

Both are `[project.scripts]` entries. `uv sync` installs them into `.venv/bin/`, so there are two equivalent ways to call them:

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run umi --help            # uv picks the workspace venv for you
```

```bash
cd /home/hcis-s17/author_workdir/voilab
source .venv/bin/activate
umi --help                   # the same binary, .venv/bin/umi
```

**Run from the repo root.** Almost every path these commands touch is resolved against the current working directory: SLAM settings files (`packages/umi/src/umi/services/slam_mapping.py:484-486`), and the notebook paths handed to Voila (`src/voilab/cli.py:18`, `:25`) are bare relative strings.

Neither group has global options: no `--verbose`, no `--config`, no `--version`. Command names are the Python function names with underscores turned into dashes, which is why `def run_slam_pipeline` is called as `run-slam-pipeline`.

## `umi run-slam-pipeline`

Runs an entire pipeline described by a YAML file: sorting raw videos, extracting IMU (inertial measurement unit) samples, SLAM (Simultaneous Localization And Mapping — recovering the camera's 3D path from the video), fiducial-marker detection, calibration, and packing everything into a zarr archive (zarr = a chunked array format; here the final `dataset.zarr.zip`). What each stage does is in [Pipeline overview](./pipeline-overview.md).

```text
umi run-slam-pipeline [OPTIONS] CONFIG_PATH
```

| Option | Type | Default | Required | Effect |
|---|---|---|---|---|
| `CONFIG_PATH` | positional string | — | yes | Path to the pipeline YAML. Click does **not** check it exists; a bad path raises `FileNotFoundError` later, at `packages/umi/src/umi/pipeline_executor.py:104` |
| `--session-dir` | existing path | value in the YAML | no | Overrides the capture-session directory. The directory must already exist or Click rejects the call before anything runs (`packages/umi/src/umi/cli.py:13`) |
| `--task` | choice: `kitchen`, `living_room`, `dining_room` | value in the YAML | no | Overrides the scene/task name (`packages/umi/src/umi/cli.py:14`) |

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
  --session-dir /home/hcis-s17/author_workdir/voilab/data/2026_01_20_kitchen
```

### How `--session-dir` and `--task` actually behave

Neither flag is a global setting. Each rewrites **one** key in **one** stage of the YAML and stops; later stages only see the new value because config is inherited stage to stage ([Pipeline config](./pipeline-config.md)).

- `--session-dir` walks the stages in file order and replaces `session_dir` in the first stage that declares that key, then returns (`packages/umi/src/umi/pipeline_executor.py:126-132`). All five shipped configs declare it only in their first stage, so it works. If no stage declares it you get a log warning and nothing else (`:136`), and the run fails later on an assertion.
- `--task` behaves the same way but only rewrites a stage that **already has** a `task` key (`packages/umi/src/umi/pipeline_executor.py:115-120`). Only `umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml:62` has one.

**Known issue:** on the other four shipped configs `--task` is silently a no-op, with no warning (`packages/umi/src/umi/pipeline_executor.py:115-120`). See [Known issues](./known-issues.md).

The task names here use underscores (`living_room`); `voilab launch-simulator` uses dashes (`living-room`). They are not interchangeable.

## `umi visualize-slam-gui`

Runs ORB-SLAM3 on a single video inside its Docker container with the graphical debugger open, so you can watch tracking hold or drop. Reach for it when a mapping stage fails. Details in [ORB-SLAM3](./orb-slam3.md).

```text
umi visualize-slam-gui [OPTIONS] VIDEO_PATH
```

| Option | Type | Default | Required | Effect |
|---|---|---|---|---|
| `VIDEO_PATH` | existing path | — | yes | Bind-mounted into the container as `/input/video.mp4` |
| `--session-dir` | existing path | — | yes | Bind-mounted as `/data`; the container reads `/data/imu_data.json` and writes `camera_trajectory.csv` and `map_atlas.osa` there |
| `--docker-image` | text | `chicheng/orb_slam3:latest` | no | Image to run (`packages/umi/src/umi/cli.py:24`) |
| `--settings-file` | text | none — see below | optional on paper, required in practice | Camera settings YAML, bind-mounted to `/slam_settings.yaml`; resolved relative to the working directory |
| `--force` | flag | off | no | Declared at `packages/umi/src/umi/cli.py:28` but the service never reads it — no effect |

This command needs X11: the service constructor raises `RuntimeError` when `DISPLAY` is unset (`packages/umi/src/umi/services/visualize_slam_gui.py:124-131`). Over SSH, connect with `ssh -X`. It also always runs `docker pull` on the image first and raises `RuntimeError` if the pull fails, so it needs network access; no flag skips it, because the CLI never sets the `pull_docker` key the service reads (`packages/umi/src/umi/services/visualize_slam_gui.py:24`, used `:216-222`).

**Known issue:** leaving out `--settings-file` fails instead of falling back to a default. The CLI always inserts the key with value `None`, so the `.get()` fallback at `packages/umi/src/umi/services/visualize_slam_gui.py:19-22` never fires and you get `ValueError: slam_settings_file is not configured` (`:198`). See [Known issues](./known-issues.md).

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run umi visualize-slam-gui \
  /home/hcis-s17/author_workdir/voilab/data/2026_01_20_kitchen/demos/mapping/raw_video.mp4 \
  --session-dir /home/hcis-s17/author_workdir/voilab/data/2026_01_20_kitchen/demos/mapping \
  --settings-file packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml
```

## `voilab` commands

### `launch-viewer` and `launch-dataset-visualizer`

Neither takes any options. Both start Voila (a server that renders a Jupyter notebook as a standalone web page) with `--no-browser`, so they print a URL instead of opening one. Voila listens on `localhost:8866` and neither command takes a flag to change that, so if you are working over SSH, forward the port from your laptop first: `ssh -L 8866:localhost:8866 <lab-host>`, then open the printed URL locally.

| Command | Notebook served | Point it at |
|---|---|---|
| `launch-viewer` | `nbs/replay_buffer_viewer.ipynb` (`src/voilab/cli.py:18`) | a `dataset.zarr.zip`; the widget defaults to `../video/dataset.zarr.zip` (`nbs/replay_buffer_viewer.ipynb:32`) |
| `launch-dataset-visualizer` | `nbs/dataset_visualizer.ipynb` (`src/voilab/cli.py:25`) | a session directory; the widget defaults to `../video` (`nbs/dataset_visualizer.ipynb:38`) |

Prerequisites: both notebook paths are relative, so these only work from the repo root. Both viewer modules import plotly at module load (`src/voilab/applications/replay_buffer_viewer.py:5`, `src/voilab/applications/dataset_visualizer.py:16`), but plotly ships only in the `dev` extra (`pyproject.toml:39`), so a plain `uv sync` leaves you with an `ImportError`. Install with `make install-dev` (equivalently `uv sync --extra dev`). More in [Visualization](./visualization.md).

### `launch-simulator`

Builds the `isaac-sim` Docker service and runs the Isaac Sim scene generator inside it.

| Option | Type | Default | Required | Effect |
|---|---|---|---|---|
| `--task` | choice: `kitchen`, `dining-room`, `living-room` | — | yes | Sets the `TASK_NAME` env var and is passed to `scripts/generate_data.py --task` |
| `--session_dir` | text (underscore, not dash) | none | no | Passed to `scripts/generate_data.py --session_dir` |
| `--width` | integer | `1280` | no | Sets `WINDOW_WIDTH` |
| `--height` | integer | `720` | no | Sets `WINDOW_HEIGHT` |

It always runs `docker compose build isaac-sim` first (`src/voilab/cli.py:59`); there is no skip flag. It needs an NVIDIA GPU, the nvidia container runtime, and X11 (`docker-compose.yaml:2-33`). See [Simulation and Docker](./simulation-and-docker.md).

**Known issue:** this command does not currently work. It runs `scripts/generate_data.py`, whose line 15 is `import registry`, and `scripts/registry/` has been removed in favour of the rewrite under `scripts/v1/`, so the container exits with `ModuleNotFoundError: No module named 'registry'`. See [Known issues](./known-issues.md).

**Known issue:** `--session_dir` is interpolated into the container command unconditionally (`src/voilab/cli.py:62`), so omitting it passes the literal four-character string `None`. See [Known issues](./known-issues.md).

### `replay-trajectory`

Publishes a recorded session's poses over ROS 2 from inside the `voilab-workspace` container.

| Option | Type | Default | Required | Effect |
|---|---|---|---|---|
| `--session-dir` | text (dash, not underscore) | — | yes | Session directory to replay (`src/voilab/cli.py:82`) |

It runs `docker compose run voilab-workspace python packages/diffusion_policy/examples/run_dataset_pose_publisher.py --session_dir <dir>` (`src/voilab/cli.py:97-101`). The underlying script also accepts `--episode` (default `0`, `packages/diffusion_policy/examples/run_dataset_pose_publisher.py:405`), but the CLI cannot pass it, so you always replay episode 0 — one episode is one continuous demonstration inside `dataset.zarr.zip`, delimited by `meta/episode_ends`. The `docker compose run` has no `--rm` (`src/voilab/cli.py:97-101`), so each invocation leaves a stopped container behind — clean up with `docker container prune`.

## Common recipes

All of these assume `cd /home/hcis-s17/author_workdir/voilab` and `export SESSION=/home/hcis-s17/author_workdir/voilab/data/2026_01_20_kitchen`.

Run the full pipeline on a new capture:

```bash
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
  --session-dir "$SESSION" 2>&1 | tee "$SESSION/pipeline.log"
```

Re-run the map-building stage. There is no CLI flag for this; most stages skip work whose output already exists, so you either delete the output or set `force: true` under the `02_create_map` stage's `config:` block (`packages/umi/src/umi/services/slam_mapping.py:33`, skip check at `:79-80`). Deleting is simpler:

```bash
rm "$SESSION/demos/mapping/map_atlas.osa"
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir "$SESSION"
```

Re-run fiducial-marker (ArUco tag) detection for one demo. The stage skips any demo that already has a detection file (`packages/umi/src/umi/services/aruco_detection.py:49-52`), so delete just that one and re-run the config:

```bash
rm "$SESSION/demos/demo_C3541324566267_2026.01.20_10.11.12.000000/tag_detection.pkl"
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir "$SESSION"
```

Point the same config at a different session — no YAML editing needed, because every shipped config declares `session_dir` only in its first stage:

```bash
uv run umi run-slam-pipeline \
  umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
  --session-dir /home/hcis-s17/author_workdir/voilab/data/2026_01_21_dining
```

Debug one video's SLAM interactively after a mapping failure:

```bash
less "$SESSION/demos/mapping/slam_stdout.txt"
uv run umi visualize-slam-gui "$SESSION/demos/mapping/raw_video.mp4" \
  --session-dir "$SESSION/demos/mapping" \
  --settings-file packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml
```

## What the CLI does not have yet

- No `--version` on either script. Versions live in `pyproject.toml` and `packages/umi/pyproject.toml`.
- No global options at all — every flag belongs to a specific subcommand.
- No way to run a single stage. `PipelineExecutor` implements `list_stages()` (`packages/umi/src/umi/pipeline_executor.py:226`), `execute_stage()` (`:234`) and `validate_stages()` (`:327`), but no command exposes them; the CLI only calls `execute_all()` (`packages/umi/src/umi/cli.py:17`). To run a subset, copy the YAML and delete the stages you do not want.
- No `--dry-run`, `--from`, or `--only`.
- No launcher for `nbs/aruco_detection_viewer.ipynb` or `nbs/slam_viewer.ipynb`; open those through JupyterLab.

---

**Next:** [Pipeline config](./pipeline-config.md) · [Known issues](./known-issues.md) · [Visualization](./visualization.md)
