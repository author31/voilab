# Pipeline overview

The SLAM pipeline — SLAM = Simultaneous Localization and Mapping, recovering where the camera was frame by frame — turns a folder of handheld GoPro videos into one training dataset (`dataset.zarr.zip`, a single chunked-array archive) by running an ordered list of Python classes named in a YAML file.

**Read this if:** you want to understand what happens between "I plugged in the GoPro" and "I have a `dataset.zarr.zip`", or you want to add or change a stage.

**Before you start:** [Getting started](./getting-started.md) for install and a first run, [Repository map](./repository-map.md) for where the code lives.

---

## The idea in one paragraph

There is no stage registry in Python. **The YAML file is the registry.** Each top-level key in a pipeline config is a stage; its `instance:` value is a dotted import path to a class. `PipelineExecutor` reads the YAML, walks the keys in file order, imports each class with `importlib`, builds one config dictionary for that stage, constructs the class with it, calls `execute()`, and then merges that stage's config into a running dictionary that the next stage inherits. Stage names like `00_process_video` are labels only — the numeric prefix is never parsed (`packages/umi/src/umi/pipeline_executor.py:226-232`). Ordering comes from YAML key order.

One command runs the whole chain, from the repo root. Every shipped config's `session_dir` points at a path that will not exist on your machine, so always pass `--session-dir`:

```bash
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir /abs/path/to/session
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml --session-dir /abs/path/to/session --task living_room
```

`--session-dir` and `--task` are the only flags this command takes (`packages/umi/src/umi/cli.py:13-14`); everything else is config. The `umi` CLI has one other command, `visualize-slam-gui` — see [CLI reference](./cli-reference.md).

## The stage chain

Everything downstream depends on the camera trajectory SLAM recovers, and on the map of the room it builds along the way.

```text
<session_dir>/  raw GoPro .MP4 files
  │
  ├─ 00 VideoOrganizationService  ──▶ demos/<demo>/raw_video.mp4
  ├─ 01 IMUExtractionService      ──▶ demos/<demo>/imu_data.json
  ├─ 02 SLAMMappingService(map)   ──▶ demos/mapping/map_atlas.osa
  ├─ 03 SLAMMappingService(batch) ──▶ demos/<demo>/camera_trajectory.csv
  ├─ 04 ArucoDetectionService     ──▶ demos/<demo>/tag_detection.pkl
  ├─ 05 CalibrationService        ──▶ demos/mapping/tx_slam_tag.json
  │                                   demos/gripper_calibration_*/gripper_range.json
  ├─ 06 DatasetPlanningService    ──▶ dataset_plan.pkl
  ├─ 07 FrameToPoseService        ──▶ demos/mapping/object_poses.json   (optional)
  └─ 08 ReplayBufferService       ──▶ dataset.zarr.zip
        │
        └── config flows left-to-right the whole way down
```

Every artifact's schema lives in [Data formats](./data-formats.md); this page does not repeat them. *Replay buffer* is UMI's name for the finished training dataset: every episode's images and robot state packed into one array store (`dataset.zarr.zip`).

## The two building blocks

### `BaseService` — the whole contract

`packages/umi/src/umi/services/base_service.py` is 68 lines, and that is the entire interface a stage must satisfy:

- Constructor takes one plain `dict` and stores it on `self.config` (`base_service.py:10-17`). No schema, no validation, no dataclass. Services pull their own keys with `self.config.get(...)`.
- `execute(*args, **kwargs)` is the single `@abstractmethod` (`base_service.py:19-30`). Its return type is `Any`; in practice every service returns a differently shaped dict.
- Two helpers you may use: `_ensure_output_dir()` (`:32`) and `_get_num_workers()` (`:45`).

What the base does **not** give you:

| Not provided | Consequence |
|---|---|
| Status enum | Only some services put a `status` key in their return dict, with per-service values. Nothing reads them. |
| Error handling | Services raise; the executor logs and re-raises, aborting the run (`pipeline_executor.py:314-316`). No retries. |
| Skip / resume logic | Each stage re-implements its own "does the output file exist?" check, or has none. |
| Output validation | `validate_output()` (`base_service.py:59`) returns `True` and is never called by anything. |

### `PipelineExecutor` — the driver

`packages/umi/src/umi/pipeline_executor.py`, five things in order:

1. **Load YAML.** `yaml.safe_load` into `self.config`; missing file raises `FileNotFoundError`, bad YAML raises `ValueError` (`:101-124`). `--session-dir` and `--task` overrides are applied here, each patching only the **first** stage that declares the key (`:126-136`, `:115-121`).
2. **Resolve per-stage config.** Inherited config, minus `config_exclude`, merged with the stage's own `config:`, merged with `config_override:` (`:182-195`). Dicts deep-merge; lists are replaced. Full rules in [Pipeline config](./pipeline-config.md).
3. **Import the dotted path.** `class_path.rsplit(".", 1)` then `importlib.import_module` + `getattr` (`:150-154`). A typo raises `ImportError`.
4. **Type-check and instantiate.** `if not isinstance(service_instance, BaseService): raise TypeError` (`:204-205`). This is what makes the plugin model safe.
5. **Execute and propagate.** `service_instance.execute()` runs, then the stage's *effective* config is merged into the running dictionary for later stages (`:299`, `:308-310`).

`execute_all()` forwards the same `*args/**kwargs` to every service (`:302`), and the CLI passes none — so every pipeline stage must have a zero-argument `execute()`.

## Stage table

| Stage key | Service class | Reads | Writes | Skips when |
|---|---|---|---|---|
| `00_process_video` | `VideoOrganizationService` | loose `*.MP4` under the session dir | `demos/<demo>/raw_video.mp4`, symlinks in `raw_videos/` | `raw_videos/` or `gripper_calibration/` already exists; the source file is already a symlink (`video_organization.py:35,44,56,82`) |
| `01_extract_gopro_imu` | `IMUExtractionService` | each `demos/*/raw_video.mp4` | `demos/<demo>/imu_data.json` | never — rewrites every run |
| `02_create_map` | `SLAMMappingService` (`slam_process_mode: create_map`) | `demos/mapping/raw_video.mp4` + `imu_data.json` | `map_atlas.osa`, `mapping_camera_trajectory.csv`, `slam_mask.png`, SLAM logs | `map_atlas.osa` exists and `force` is falsy (`slam_mapping.py:79-82`) |
| `03_batch_slam` | `SLAMMappingService` (`slam_process_mode: batch_slam`) | `demos/demo*/` and `demos/map*/` videos + IMU + `map_atlas.osa` | per-demo `camera_trajectory.csv`, `slam_mask.png`, SLAM logs | per demo, `camera_trajectory.csv` exists (`slam_mapping.py:242-244`) |
| `04_detect_aruco` | `ArucoDetectionService` | every `demos/*/raw_video.mp4`, camera intrinsics JSON, ArUco config YAML | `demos/<demo>/tag_detection.pkl` | per demo, `tag_detection.pkl` exists (`aruco_detection.py:49-52`) |
| `05_run_calibrations` | `CalibrationService` | `demos/mapping/tag_detection.pkl` + trajectory CSV, `demos/gripper_calibration*/tag_detection.pkl` | `demos/mapping/tx_slam_tag.json`, `gripper_calibration*/gripper_range.json` | never |
| `06_generate_dataset_plan` | `DatasetPlanningService` | trajectories, tag detections, both calibrations, optional `check_result.txt` | `<session>/dataset_plan.pkl` | never |
| `07_frame_to_pose` *(reconstruct config only)* | `FrameToPoseService` | `dataset_plan.pkl`, `tx_slam_tag.json`, episode video frames | `demos/mapping/object_poses.json` | never |
| `07`/`08_generate_replay_buffer` | `ReplayBufferService` | `dataset_plan.pkl`, demo videos, tag detections, optional `object_poses.json` | `<session>/dataset.zarr.zip` | never |

The replay-buffer stage is key `07_generate_replay_buffer` in the 8-stage configs and `08_generate_replay_buffer` in the 9-stage reconstruct config, because `07_frame_to_pose` was inserted ahead of it with no Python change.

Despite the stage key, `03_batch_slam` is not backed by `services/batch_slam.py`: `BatchSLAMService` is an unused stub whose SLAM step only touches placeholder files, and it is imported by tests alone. `services/config.py` (`ConfigService`) is unused as well.

## What each stage does

**00_process_video.** Moves every loose MP4 into `raw_videos/`, renames the largest file to `mapping.mp4`, sets aside the earliest clip per camera as gripper calibration, then explodes each video into its own `demos/<name>/raw_video.mp4` directory keyed by camera serial and start time, leaving a symlink behind. Most common failure: ExifTool is missing or too old, so `QuickTime:CameraSerialNumber` lookup dies (`video_organization.py:66-67`).

**01_extract_gopro_imu.** Reads GPMF (the metadata track GoPro muxes into the MP4, carrying accelerometer and gyroscope samples) and writes it as JSON in the shape ORB-SLAM3 expects. See [GoPro telemetry](./gopro-telemetry.md).
**Known issue:** any extraction error is swallowed and a fake single-sample IMU file is written instead, reported as success (`packages/umi/src/umi/services/imu_extraction.py:127-144`). SLAM then runs on garbage with nothing in the logs. See [Known issues](./known-issues.md).

**02_create_map.** Runs ORB-SLAM3 in a Docker container over the mapping clip to build a map atlas — a reusable saved map of the room. This is the slow one. Most common failure: `slam_settings_file` does not resolve, or Docker cannot reach the display. See [ORB-SLAM3](./orb-slam3.md).

**03_batch_slam.** Same class as 02, different `slam_process_mode` — it dispatches at `slam_mapping.py:43-48`. It localizes every demo clip against the atlas from 02, in parallel threads, with a per-video timeout of `duration × timeout_multiple`. Gripper-calibration clips are deliberately excluded from the glob (`slam_mapping.py:222-223`). Most common failure: a clip times out or loses tracking, leaving no `camera_trajectory.csv`, and stage 06 silently drops that demo.

**04_detect_aruco.** Decodes every frame of every clip and finds ArUco markers — printed square black-and-white fiducial patterns, each with an ID, whose 3D pose can be solved from its four corners. Output is one pickle per demo. Most common failure: `camera_intrinsics_path` is resolved relative to the current working directory, so running from anywhere but the repo root gives `FileNotFoundError` (`aruco_detection.py:93-96`).

**05_run_calibrations.** Two independent jobs: fit the transform from the SLAM coordinate frame to the fixed table tag (`tx_slam_tag.json`), and measure each gripper's open/closed finger width from its finger tags (`gripper_range.json`). Most common failure: the config sets `resolution` under a different key, tripping `assert self.resolution` (`calibration.py:73`).

**06_generate_dataset_plan.** The real brain of the pipeline: aligns cameras by wall-clock timestamp, converts trajectories into tool-center-point (TCP, the gripper tip) poses, attaches gripper widths, drops bad frames and short episodes, and writes the plan. Most common failure: `RuntimeError("Gripper calibration not found.")` because stage 05 quietly returned nothing (`dataset_planning.py:476`).

**07_frame_to_pose.** Reconstruct configs only. Scans the planned episode frames for task-object markers and writes their poses in the SLAM-tag frame so Isaac Sim can rebuild the scene. Requires `task:` in its config. Most common failure: `intrinsics_path` here is resolved relative to `packages/umi/`, not the repo root, unlike every other path key (`frame_to_pose.py:15,367`).

**08_generate_replay_buffer.** Materializes the plan into a single zarr archive — zarr is a chunked array format, here a `.zip` store holding per-frame images and robot state — with the gripper region masked out and images JPEG-XL compressed. The mirror region is masked only when `no_mirror: true`, which reads backwards; every shipped config sets it `false` ([Pipeline config](./pipeline-config.md)). Most common failure: running out of disk or memory on long sessions.
**Known issue:** without `demos/mapping/object_poses.json` the stage does not complete at all — it raises `ValueError: missing object_codec for object array` (`replay_buffer.py:239`) while writing the `reconstruct/` group, after every video has already been decoded and compressed. Only the reconstruct config runs `07_frame_to_pose`, so the other four shipped configs hit it. See [Known issues](./known-issues.md).

## Resume and re-run semantics

There is no pipeline-level resume. `execute_all()` always walks every stage; each stage decides for itself whether there is work to do, always by asking "does the output file exist?".

| Behaviour | Stages |
|---|---|
| Skips whole stage on existing output | `02_create_map` (`map_atlas.osa`) |
| Skips per demo on existing output | `03_batch_slam` (`camera_trajectory.csv`), `04_detect_aruco` (`tag_detection.pkl`) |
| Partially guarded by directory existence | `00_process_video` |
| Redoes everything, every run | `01`, `05`, `06`, `07_frame_to_pose`, replay buffer |

To force a redo, delete the artifact. Stage 02 is the only one with a config flag:

```yaml
02_create_map:
  instance: "umi.services.slam_mapping.SLAMMappingService"
  config:
    slam_process_mode: create_map
    force: true          # re-run SLAM even though map_atlas.osa exists
```

`force` is honoured only inside `execute_create_map_slam` (`slam_mapping.py:79`); setting it on stage 03 does nothing. There is also a manual per-demo kill switch: writing a `check_result.txt` whose content does not begin with `true` into a demo directory makes stage 06 drop that camera (`dataset_planning.py:407-410`).

## What you can observe while it runs

- **loguru to stderr.** Every service logs through `loguru` with no sink configured, so all output goes to stderr. Per stage you get the config diff (`pipeline_executor.py:81-99`), `Stage i/N: <name>`, `... completed`, and `Updated propagated configuration with N keys` (`:292-310`). Stage 02/03 also log the exact `docker run` command before issuing it (`slam_mapping.py:171,337`).
- **tqdm bars** for per-item progress in batch SLAM, ArUco detection and replay-buffer construction (`slam_mapping.py:236`, `aruco_detection.py:40`, `replay_buffer.py:204`). Nothing is persisted.
- **Stage 06 yield line.** `dataset_planning.py:580-581` logs `<N>% of raw data are used.` and then a per-camera-serial drop count. This is the only signal that demos were silently discarded for lost tracking or a missing `camera_trajectory.csv`.
- **`slam_stdout.txt` / `slam_stderr.txt`** written into each video directory by both SLAM modes (`slam_mapping.py:172-173`, `:338-339`). These are where you look when a clip fails.

What is **not** produced: no pipeline log file, no progress file, no machine-readable stage status. `LoggingService` (`services/logging.py:8`) would have written `<output_dir>/pipeline.log`, and `ProgressService` (`services/progress.py:8`) would have written `progress.json` and even exposes `is_stage_completed()` (`progress.py:77`) — the resume primitive the pipeline lacks. Neither class is instantiated anywhere.

**Known issue:** `04_detect_aruco`'s returned counts are inverted — after a clean run every video has a pickle and is therefore labelled *skipped*, so `videos_processed` counts only the failures (`packages/umi/src/umi/services/aruco_detection.py:75-78`). Do not read those numbers as progress. See [Known issues](./known-issues.md).

## Adding your own stage

1. Subclass `BaseService` and implement a zero-argument `execute()`. Read everything you need from `self.config`.
2. Put the class anywhere importable inside the `umi` package.
3. Add a key to the YAML wherever you want it to run:

```yaml
06b_my_filter:
  instance: "umi.services.my_filter.MyFilterService"
  inherit_config: true
  config:
    my_threshold: 0.5
```

That is the entire procedure — `07_frame_to_pose` was added to the reconstruct pipeline exactly this way, with no change to `PipelineExecutor`.

**The gotcha:** constructor defaults do not propagate. The executor forwards `service_instance.config` (`pipeline_executor.py:299`), which is the dictionary it passed in — services read `self.config.get("k", default)` into an attribute but never write the default back. So a value you rely on being "defaulted" is invisible to later stages, and a key you spell wrong fails several stages later instead of at load time. `umi_pipeline_configs/gopro13_normal_pipeline_config.yaml:43` demonstrates both: it sets `video_resolution`, a key no code reads, and `CalibrationService` then dies on `assert self.resolution` (`packages/umi/src/umi/services/calibration.py:73`). Declare every key you depend on, explicitly, in the stage that needs it.

You can dry-run a config without executing anything, but only from Python — `PipelineExecutor.validate_stages()` (`pipeline_executor.py:327`) instantiates every stage and returns `{stage: True/False}`. No CLI command exposes it.

---

**Next:** [Pipeline config](./pipeline-config.md) · [Data formats](./data-formats.md) · [Known issues](./known-issues.md)
