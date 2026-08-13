# Pipeline configuration reference

What every key in `umi_pipeline_configs/*.yaml` means, which service reads it, and how values flow from one stage to the next.

**Read this if:** you are editing a pipeline YAML, writing one for a new camera, or debugging a "Missing X in configuration" crash.

**Before you start:** [Pipeline overview](./pipeline-overview.md), [CLI reference](./cli-reference.md)

---

## 1. Anatomy: the file *is* the stage list

- No `stages:` wrapper, no schema version, no `defaults:` block. **Every top-level key is one stage.**
- Stages run in **file order**: `list_stages()` returns `list(self.config.keys())` (`packages/umi/src/umi/pipeline_executor.py:232`) and Python preserves insertion order.
- The `00_` / `01_` prefixes are pure convention. Renaming `03_batch_slam` to `zz_batch_slam` does not move it; moving its YAML block does.
- `instance` is a dotted Python path imported at runtime with `importlib` (`pipeline_executor.py:151-153`). There is no name-to-class registry — the YAML *is* the registry, and the class must subclass `BaseService`, else `TypeError` (`:204-205`). One class can appear twice: `SLAMMappingService` runs at both `02_create_map` and `03_batch_slam`, told apart only by `slam_process_mode`.

## 2. Stage block keys

These five are the **only** keys the executor understands. Anything else at stage level is silently ignored; there is no schema validation anywhere.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `instance` | dotted class path | **required** | Class to import and instantiate. Missing → `KeyError` (`pipeline_executor.py:177-178`) |
| `config` | mapping | `{}` | This stage's own settings (`pipeline_executor.py:184`) |
| `inherit_config` | bool | `true` | Start from the accumulated upstream config (`pipeline_executor.py:183`) |
| `config_override` | mapping | `{}` | Applied last; beats `config` (`pipeline_executor.py:185`) |
| `config_exclude` | list[str] | `[]` | Keys dropped from the **inherited** config only (`pipeline_executor.py:186`) |

Expected but absent: no `enabled:` / `skip:` per stage (comment the block out instead), no `depends_on:` (order is positional), no global `defaults:` block (sharing happens by propagation, §4), no key-name or type checking (a typo does nothing). `config_override` and `config_exclude` work but are used by **zero** shipped configs.

## 3. Resolution order

Per stage (`packages/umi/src/umi/pipeline_executor.py:188-195`):

1. If `inherit_config` is true **and** something has propagated, start from the propagated config. Stage 1 always starts empty (`:283`).
2. Delete every key named in `config_exclude`.
3. Deep-merge this stage's `config` on top.
4. Deep-merge this stage's `config_override` on top of that.

With `inherit_config: false`, only steps 3-4 run and `config_exclude` is ignored entirely. **Merge rule** (`:42-46`): mappings merge key by key; everything else, lists included, is **replaced wholesale**. (`opts` below is a stand-in: no shipped service reads a nested mapping, but the rule applies if you add one.)

```text
inherited   nominal_z: 0.072   input_patterns: ["*.MP4","*.mp4"]   opts: {a: 1, b: 2}
stage cfg   nominal_z: 0.01    input_patterns: ["*.mp4"]           opts: {b: 9}
effective   nominal_z: 0.01    input_patterns: ["*.mp4"]           opts: {a: 1, b: 9}
                                ^ list replaced whole               ^ mapping merged key-by-key
```

## 4. Propagation between stages

After a stage finishes, its **effective** config — inherited plus its own, not just its own block — is merged into the running propagated config (`pipeline_executor.py:299,309`), so the dict grows down the file.

```text
00_process_video          {session_dir, input_patterns}
01_extract_gopro_imu        + {stream_types}
02_create_map               + {docker_image, slam_process_mode, resolution, ...}
05_run_calibrations         sees session_dir AND resolution AND all of the above
07_generate_replay_buffer   sees 16 accumulated keys plus its own 7 (23 effective)
```

That is why `session_dir` is declared once, in `00_process_video`, and every later stage still finds it — and why `--session-dir` only rewrites the first stage that declares it (`:126-133`).

**The trap: constructor defaults never propagate.** Each service reads settings with `self.config.get("key", default)` into an instance attribute and never writes the default back. `BaseService` stores the dict verbatim (`services/base_service.py:16`) and propagation forwards that dict, so a default a service *used* stays invisible to later stages. Concretely, `SLAMMappingService` falls back to `resolution = (2028, 2704)` (`services/slam_mapping.py:35`); if `02_create_map` does not write `resolution` in YAML, `05_run_calibrations` inherits nothing and dies on `assert self.resolution` (`services/calibration.py:73`).

Two more consequences: deleting `06_generate_dataset_plan` breaks the replay-buffer stage, because `dataset_plan_filename` is declared only there (`services/replay_buffer.py:52`); and a value set once becomes the default for every later stage reading that key name — `nominal_z: 0.01` at stage 05 is inherited by stage 06, whose own default is `0.0` (`services/dataset_planning.py:33`).

## 5. Annotated reference config

`umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml` in full — the healthiest shipped config, and the one to copy from. SLAM = simultaneous localisation and mapping, the step that recovers the camera's 3D trajectory; ORB-SLAM3 is the implementation this repo drives inside Docker.

```yaml
00_process_video:
  instance: "umi.services.video_organization.VideoOrganizationService"
  config:
    session_dir: "video"                 # root of all input/output, CWD-relative (video_organization.py:22)
    input_patterns: ["*.MP4", "*.mp4"]   # globs for finding raw clips (video_organization.py:17)
01_extract_gopro_imu:
  instance: "umi.services.imu_extraction.IMUExtractionService"
  config:
    # GPMF = GoPro's telemetry format, muxed inside the MP4. These streams are pulled out of
    # each clip into demos/*/imu_data.json for ORB-SLAM3 (imu_extraction.py:18)
    stream_types: ["ACCL", "GYRO", "GPS5", "GPSP", "GPSU", "GPSF", "GRAV", "MAGN", "CORI", "IORI", "TMPC"]
02_create_map:
  instance: "umi.services.slam_mapping.SLAMMappingService"
  inherit_config: true                   # explicit but redundant: true is the default
  config:
    docker_image: "chicheng/orb_slam3:latest"   # image holding the SLAM binary (slam_mapping.py:26)
    timeout_multiple: 16                 # slam_mapping.py:27 - UNUSED in create_map mode
    max_lost_frames: 60                  # slam_mapping.py:28 - UNUSED in create_map mode
    slam_process_mode: create_map        # picks execute_create_map_slam (slam_mapping.py:31,44)
    slam_settings_file: "packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml"
    resolution: [2028,2704]              # (h, w) of your footage, numpy order; sizes the SLAM mask (:35,:264,:396)
    mask_pts_json_path: "packages/umi/src/umi/asset/mask_4k.json"   # DEAD KEY, see §10
03_batch_slam:
  instance: "umi.services.slam_mapping.SLAMMappingService"
  inherit_config: true
  config:
    docker_image: "chicheng/orb_slam3:latest"
    timeout_multiple: 16                 # per-clip timeout = clip seconds x 16 (slam_mapping.py:263)
    max_lost_frames: 60                  # --max_lost_frames given to ORB-SLAM3 (slam_mapping.py:331)
    slam_process_mode: batch_slam        # overrides the inherited "create_map"
    slam_settings_file: "packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml"
    resolution: [2028,2704]
04_detect_aruco:
  instance: "umi.services.aruco_detection.ArucoDetectionService"
  inherit_config: true
  config:
    camera_intrinsics_path: "packages/umi/defaults/calibration/gopro13_intrinsics_2_7k.json"  # :29, open :96
    aruco_config_path: "packages/umi/defaults/calibration/aruco_config.yaml"                  # :30, open :93
05_run_calibrations:
  instance: "umi.services.calibration.CalibrationService"
  inherit_config: true
  config:
    resolution: [2028,2704]              # calibration.py:34; asserted at :73, used at :145
    tag_id: 13                           # fiducial fixed to the table (calibration.py:28)
    dist_to_center_threshold: 0.6        # reject tags far from image centre (calibration.py:149)
    nominal_z: 0.01                      # expected camera-to-finger-tag depth, m (calibration.py:33)
06_generate_dataset_plan:
  instance: "umi.services.dataset_planning.DatasetPlanningService"
  inherit_config: true
  config:
    dataset_plan_filename: dataset_plan.pkl   # written to <session>/ (dataset_planning.py:36)
    # nominal_z is NOT re-declared: stage 06 inherits 0.01 from 05, not its own 0.0 default
07_generate_replay_buffer:
  instance: "umi.services.replay_buffer.ReplayBufferService"
  inherit_config: true
  config:
    output_filename: dataset.zarr.zip    # zarr = a chunked array archive (replay_buffer.py:32)
    output_resolution: [224, 224]        # stored RGB size (replay_buffer.py:33)
    output_fov: null                     # null keeps the fisheye image as-is (replay_buffer.py:40)
    output_fov_intrinsic_path: null      # both must be non-null to rectify (replay_buffer.py:99)
    compression_level: 99                # JPEG-XL quality level (replay_buffer.py:193)
    no_mirror: false                     # replay_buffer.py:38 - reads backwards, see §10
    mirror_swap: false                   # replay_buffer.py:39, applied at :331
    # dataset_plan_filename is inherited from stage 06 and is required (replay_buffer.py:52)
```

The reconstruct variant is identical through stage 06, then inserts `07_frame_to_pose` and renumbers the replay buffer to `08_`. (Blank lines between stages have been removed here for length; the file on disk has them.)

## 6. Per-stage config keys

Every key any shipped config sets, plus every key the services read; dead keys are in §10 instead. "Read by" paths are relative to `packages/umi/src/umi/services/`. `session_dir` is asserted by every stage and inherited from stage 00 in all shipped configs.

| Stage | Key | Type | Default | Read by | Meaning |
|---|---|---|---|---|---|
| 00 | `session_dir` | str | none, asserted | `video_organization.py:16`, assert `:20` | Capture-session root directory |
| 00 | `input_patterns` | list[str] | `["*.MP4","*.mp4"]` | `video_organization.py:17` | Globs for finding raw clips |
| 01 | `stream_types` | list[str] | the 11 names at `imu_extraction.py:19-33` | `imu_extraction.py:18` | GPMF telemetry streams to extract |
| 02/03 | `slam_process_mode` | str | `"slam_mapping"` | `slam_mapping.py:31`, dispatch `:44-48` | `create_map` or `batch_slam`; the default matches neither and raises |
| 02/03 | `slam_settings_file` | str | in-container GoPro 10 path | `slam_mapping.py:34`, resolved `:468-495` | ORB-SLAM3 camera + IMU tuning YAML |
| 02/03 | `resolution` | list[int] (H, W) | `(2028, 2704)` | `slam_mapping.py:35`, used `:264,:396` | Canvas size of the generated SLAM mask |
| 02/03 | `docker_image` | str | `chicheng/orb_slam3:latest` | `slam_mapping.py:26` | Image holding the ORB-SLAM3 binary |
| 02/03 | `timeout_multiple` | int | `16` | `slam_mapping.py:27`, used `:263` | Batch timeout = clip seconds x this |
| 02/03 | `max_lost_frames` | int | `60` | `slam_mapping.py:28`, used `:331` | Batch-mode tracking-loss budget |
| 02/03 | `pull_docker` | bool | `true` | `slam_mapping.py:29`, used `:386` | Run `docker pull` before running |
| 02/03 | `generate_mask` | bool | `true` | `slam_mapping.py:30`, used `:90` | Write `slam_mask.png` |
| 02/03 | `num_workers` | int | `cpu_count()//2` | `slam_mapping.py:32`, used `:237` | Batch-mode thread pool size |
| 02 | `force` | bool | `false` | `slam_mapping.py:33`, used `:79` | Rebuild even if `map_atlas.osa` exists |
| 02/03 | `enable_gui` | bool | `false` | `slam_mapping.py:37`, used `:40` | X11 GUI; raises if `DISPLAY` is unset |
| 04 | `camera_intrinsics_path` | str | none, asserted | `aruco_detection.py:29`, assert `:90`, open `:96` | Lens model JSON, CWD-relative |
| 04 | `aruco_config_path` | str | none, asserted | `aruco_detection.py:30`, assert `:91`, open `:93` | ArUco (printed square fiducial marker) dictionary and tag sizes |
| 04 | `num_workers` | int | `cpu_count()//2` | `aruco_detection.py:28`, used `:41` | Decode/detect thread pool |
| 05 | `resolution` | list[int] | `None`, asserted | `calibration.py:34`, assert `:73`, used `:145` | Image size for the centre-distance filter |
| 05 | `tag_id` | int | `13` | `calibration.py:28`, assert `:45` | Table fiducial id; `0` fails the assert |
| 05 | `dist_to_center_threshold` | float | `0` | `calibration.py:29-30`, used `:149` | Reject tags this far off-centre |
| 05 | `nominal_z` | float | `0.072` | `calibration.py:33`, used `:279` | Expected finger-tag depth, metres |
| 05 | `keyframe_only` | bool | `true` | `calibration.py:27`, used `:97` | Use SLAM keyframes only |
| 05 | `tag_detection_threshold` | float | `0.1` | `calibration.py:31-32`, used `:263` | Minimum gripper-tag detection rate |
| 06 | `dataset_plan_filename` | str | `"dataset.pkl"` | `dataset_planning.py:36` | Plan file; all configs set `dataset_plan.pkl` |
| 06 | `tcp_offset` | float | `0.205` | `dataset_planning.py:32`, used `:50` | Gripper tool-centre offset, metres |
| 06 | `nominal_z` | float | `0.0` | `dataset_planning.py:33`, used `:484` | Finger-tag depth; usually inherited from 05 |
| 06 | `min_episode_length` | int | `10` | `dataset_planning.py:34`, used `:533` | Drop episodes shorter than this |
| 06 | `ignore_cameras` | str | `None` | `dataset_planning.py:35`, used `:77-79` | Comma-separated camera serials to exclude |
| 07 | `session_dir` | str | **required** | `frame_to_pose.py:348` | `KeyError` if absent — no `.get()` here |
| 07 | `task` | str | **required** | `frame_to_pose.py:350-354` | `kitchen`, `dining_room` or `living_room` |
| 07 | `marker_size_m` | float | `0.018` | `frame_to_pose.py:357` | Printed object-tag edge length, metres |
| 07 | `intrinsics_path` | str | `defaults/calibration/gopro13_intrinsics_2_7k.json` | `frame_to_pose.py:363-367` | Lens model, **rooted at `packages/umi/`** |
| 07 | `dataset_plan_filename` | str | `"dataset_plan.pkl"` | `frame_to_pose.py:359` | Plan file to read episodes from |
| last | `output_filename` | str | none, asserted | `replay_buffer.py:32`, assert `:51` | Zarr archive name |
| last | `dataset_plan_filename` | str | none, asserted | `replay_buffer.py:41`, assert `:52` | Usually inherited from stage 06 |
| last | `output_resolution` | list[int] (W, H) | `[256, 256]` | `replay_buffer.py:33`, used `:58` | Stored image size |
| last | `compression_level` | int | `99` | `replay_buffer.py:36`, used `:193` | JPEG-XL level |
| last | `num_workers` | int | `cpu_count()//2` | `replay_buffer.py:37`, used `:206` | Encode thread pool |
| last | `no_mirror` | bool | `false` | `replay_buffer.py:38`, used `:322` | Passed as `mirror=`, so `true` *enables* the mask |
| last | `mirror_swap` | bool | `false` | `replay_buffer.py:39`, used `:331` | Horizontally flip the mirror region |
| last | `task` | str | `"unknown"` | `replay_buffer.py:94` | Stored as `reconstruct/location` in the zarr |
| last | `output_fov` | float | `None` | `replay_buffer.py:40`, used `:99` | Fisheye-to-pinhole rectification, see §10 |
| last | `output_fov_intrinsic_path` | str | `None` | `replay_buffer.py:35`, used `:99` | Target lens model for that rectification |

## 7. Choosing a config for your camera

Configs live in `umi_pipeline_configs/`, SLAM settings in `packages/umi/defaults/orb_slam3_settings/`, intrinsics in `packages/umi/defaults/calibration/`.

| Camera | Lens / mode | Res | Pipeline config | ORB-SLAM3 settings | Intrinsics JSON |
|---|---|---|---|---|---|
| GoPro 9 | MAX Lens fisheye | 2.7K 4:3 | `official_gopro9_pipeline_config.yaml` | `gopro10_maxlens_fisheye_setting_v1_720.yaml` | `gopro9_intrinsics_2_7k.json` |
| GoPro 13 | MAX Lens fisheye | 2.7K 4:3 | `gopro13_fisheye_2-7k_pipeline_config.yaml` | `gopro13_fisheye_ratio_4-3_2-7k.yaml` | `gopro13_intrinsics_2_7k.json` |
| GoPro 13 | fisheye + object-pose reconstruction | 2.7K 4:3 | `gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml` | `gopro13_fisheye_ratio_4-3_2-7k.yaml` | same, declared twice (see §8) |
| GoPro 13 | wide | 2.7K | `gopro13_wide_angle_pipeline_config.yaml` — broken | file missing | `gopro13_intrinsics_2_7k.json` |
| GoPro 13 | linear / "normal" | 2.7K | `gopro13_normal_pipeline_config.yaml` — broken | not set | `gopro13_intrinsics_2_7k.json` |

**Known issue:** `gopro13_wide_angle_pipeline_config.yaml:20` and `:32` point at `gopro13_fisheye_60fps_2-7k.yaml`, which does not exist, so `02_create_map` raises `FileNotFoundError` (`packages/umi/src/umi/services/slam_mapping.py:490`). Use `gopro13_fisheye_2-7k_pipeline_config.yaml` instead. See [Known issues](./known-issues.md).

**Known issue:** `gopro13_normal_pipeline_config.yaml:43` sets `video_resolution`, but `CalibrationService` reads `resolution` (`packages/umi/src/umi/services/calibration.py:34`, asserted `:73`). It also omits `slam_settings_file`, falling back to an in-container absolute path absent on the host (`packages/umi/src/umi/services/slam_mapping.py:34`). Use `gopro13_fisheye_2-7k_pipeline_config.yaml` instead. See [Known issues](./known-issues.md).

No shipped config's `session_dir` exists on a fresh clone: two hardcode absolute paths from the author's machine (`official_gopro9_pipeline_config.yaml:4`, `gopro13_normal_pipeline_config.yaml:4`) and the other three set the relative `video`. Always pass `--session-dir`:

```bash
cd /path/to/voilab            # the repository root
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir /path/to/your/session
```

## 8. Path resolution rules

| Key | Resolved against |
|---|---|
| `session_dir` | Current working directory (`services/video_organization.py:22`) |
| `slam_settings_file` | Current working directory when relative (`services/slam_mapping.py:484-486`) |
| `camera_intrinsics_path`, `aruco_config_path` | Current working directory — passed to bare `open()` (`services/aruco_detection.py:93,96`) |
| `intrinsics_path` (stage 07 only) | **`packages/umi/`** — `ROOT / value` (`services/frame_to_pose.py:15,367`) |

That last row is why the reconstruct config names one file two ways: `packages/umi/defaults/calibration/gopro13_intrinsics_2_7k.json` at line 40 for stage 04, and `defaults/calibration/gopro13_intrinsics_2_7k.json` at line 64 for stage 07. Both are correct.

**Practical rule: always run the pipeline from the repository root.** Every relative path in every shipped config assumes it.

## 9. Errata for `umi_pipeline_configs/CONFIG_PROPAGATION_GUIDE.md`

That guide predates the current executor. Where the two disagree, the code and this page win.

| Guide claim | Reality |
|---|---|
| `:164` Set `inherit_config: false` on the first stage | Pointless. The executor starts with an empty propagated config (`pipeline_executor.py:283`) and short-circuits on it (`:188`), so stage 1 inherits nothing either way. No shipped config does this. |
| `:168` Use `validate_stages()` before execution | It exists (`pipeline_executor.py:327-353`) but no CLI command exposes it, and it constructs every service, which has side effects. |
| `:62-76` Scenario 1 shares an `output_dir` key | No pipeline service reads `output_dir`; only the unused `services/logging.py:13` and `services/progress.py:13` do. The real shared key is `session_dir`. |
| `:111` Sample log shows `Stage 1/8: video_organization (inherit_config: false)` | Real runs log `inherit_config: True` for stage 1, and the guide's stage names match no shipped config (which uses `00_process_video`, `02_create_map`, …). |
| Undocumented | `config_exclude` is silently ignored when `inherit_config: false` (`pipeline_executor.py:193-195`). |
| Undocumented | The "Overridden keys / New keys" log lines are computed from `config` only, so `config_override` keys never appear (`pipeline_executor.py:66-99`, called at `:198`). |

## 10. Dead config keys

These parse fine, appear in shipped configs or guides, and do nothing. The last row is a CLI flag rather than a key.

| Key | Where it appears | Why it does nothing |
|---|---|---|
| `mask_pts_json_path` | three gopro13 configs, e.g. `gopro13_fisheye_2-7k_pipeline_config.yaml:23` | Read into a field at `services/slam_mapping.py:36`, never used. Masks come from hardcoded polygons in `packages/umi/src/umi/common/cv_util.py`. |
| `video_resolution` | `gopro13_normal_pipeline_config.yaml:43` | No Python reads it. The real key is `resolution`. |
| `slam_tag_calibration_timeout`, `gripper_range_timeout` | no shipped config | Read at `services/calibration.py:23-26`, never used. |
| `num_workers` at stage 01 | no shipped config | Read at `services/imu_extraction.py:17`; that extraction loop is serial. |
| `timeout_multiple`, `max_lost_frames` at `02_create_map` | all configs | Consulted only in `batch_slam` mode (`services/slam_mapping.py:263,331`). Harmless: they propagate to stage 03, where they matter. |
| `output_fov` alone | all configs set `null` | Rectification needs both `output_fov` and `output_fov_intrinsic_path` (`services/replay_buffer.py:99`), so setting `output_fov` by itself is a no-op. **Known issue:** setting both crashes with `AttributeError: 'str' object has no attribute 'open'` at `:101`, because YAML supplies the path as a plain string. See [Known issues](./known-issues.md). |
| `--task` (flag, not key) | — | `--task` only rewrites a stage that already declares `task` (`pipeline_executor.py:115-121`), so on the four non-reconstruct configs the flag is a silent no-op and the zarr records `location: "unknown"`. Declaring `task:` in the replay-buffer stage's `config:` does work (`services/replay_buffer.py:94`). |

**Known issue:** `no_mirror` is passed straight into the `mirror=` argument at `packages/umi/src/umi/services/replay_buffer.py:322`, so `no_mirror: true` *enables* mirror masking instead of disabling it. See [Known issues](./known-issues.md).

---

**Next:** [Pipeline overview](./pipeline-overview.md) · [Data formats](./data-formats.md) · [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md)
