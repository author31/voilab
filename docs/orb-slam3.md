# ORB-SLAM3 in this pipeline

How the pipeline runs visual-inertial SLAM as a Docker subprocess, and how to read and write the settings file it depends on.

**Read this if:** you are debugging a SLAM stage, swapping cameras, or wondering why some demonstrations vanish from the dataset.

**Before you start:** [Pipeline overview](./pipeline-overview.md), [Getting started](./getting-started.md)

---

## 1. Why there is a SLAM stage at all

SLAM ("simultaneous localization and mapping") estimates where a camera was, frame by frame, while building a map of the scene from what it sees. This pipeline needs that camera trajectory because the GoPro is rigidly bolted to the handheld UMI gripper: once you know the camera pose, a fixed transform gives you the gripper pose, which is what the robot policy is trained to reproduce.

The variant used here is *visual-inertial*: images plus the camera's IMU (inertial measurement unit — the accelerometer and gyroscope inside the GoPro). The IMU is what makes the trajectory metric, i.e. expressed in real metres rather than an arbitrary scale.

**This repo does not build ORB-SLAM3.** No C++ source, no submodule, no compile step. The Python side shells out to the `docker` CLI and runs a prebuilt image. Everything below describes that subprocess boundary.

```text
demos/mapping/raw_video.mp4  ─┐
demos/mapping/imu_data.json  ─┼─> [docker run gopro_slam --save_map] ─> map_atlas.osa
<settings>.yaml              ─┘                                          + mapping_camera_trajectory.csv
                                                                              │
demos/demo*/raw_video.mp4    ─┐                                               │ (atlas reused)
demos/demo*/imu_data.json    ─┼─> [docker run gopro_slam --load_map] <────────┘
<settings>.yaml              ─┘        (N in parallel)      └─> demo*/camera_trajectory.csv
```

`demos/mapping/` is created by stage 00 from the clip whose filename starts with `mapping`; every other clip becomes a `demo_<camera_serial>_<timestamp>` directory (`video_organization.py:90-95`) — see [Data formats](./data-formats.md).

## 2. The fork and the image

- Upstream C++ is <https://github.com/cheng-chi/ORB_SLAM3>, a fork of `urbste/ORB_SLAM3`, itself a fork of the original `UZ-SLAMLab/ORB_SLAM3`. Linked from `packages/umi/README.md:29`, with the fork lineage at `:253`.
- The image is `chicheng/orb_slam3:latest` on Docker Hub (`packages/umi/README.md:30`), the default in `packages/umi/src/umi/services/slam_mapping.py:26` and in every shipped pipeline config.
- The fork adds an executable the upstream project does not have: `/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam`. It reads an MP4 directly plus a JSON of IMU samples, instead of the EuRoC/TUM image-folder formats vanilla expects, and it takes the map load/save paths as arguments and uses them verbatim — which is why Python passes a full `/map/map_atlas.osa` path rather than a bare name.
- It adds a `--mask_img` flag: a PNG whose white pixels are blacked out before tracking, used to hide the UMI mirrors and the gripper fingers from the feature detector.
- It adds a `--max_lost_frames` early exit, an ArUco-tag-based initialiser (a printed square fiducial marker used to fix scale at startup), a `--enable_gui` switch, and a SIGINT handler so Ctrl-C works when the binary is PID 1 in a container.
- It writes a CSV trajectory with an `is_lost` column, which is the signal the rest of this pipeline keys off.
- The fork's `Dockerfile` declares no `ENTRYPOINT`, `WORKDIR` or `USER`, and no `CMD` of its own — `docker inspect chicheng/orb_slam3:latest` reports only the `["/bin/bash"]` inherited from `ubuntu:22.04`. So the binary path must be given explicitly after the image name, and without `--user` the container runs as root.
- The fork's README is the unmodified upstream README — none of the above is documented there.

## 3. How the pipeline invokes it

`SLAMMappingService` builds an argv list and runs it with `subprocess`. There is no Docker SDK anywhere in the repo — mapping uses `subprocess.Popen` (`slam_mapping.py:178`), batch localization uses `subprocess.run` (`slam_mapping.py:210`). Before either, `docker pull <image>` runs unless `pull_docker: false` is set (`slam_mapping.py:384-390`), so a run normally needs network access.

Mapping, fully expanded (this is what `slam_mapping.py:123-169` assembles):

```bash
SESSION=/path/to/session
REPO=/home/hcis-s17/author_workdir/voilab
docker run \
  --volume "$SESSION/demos/mapping:/data" \
  --volume "$SESSION/demos/mapping:/map" \
  --volume "$REPO/packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:/slam_settings.yaml" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --env "DISPLAY=$DISPLAY" \
  chicheng/orb_slam3:latest \
  /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
  --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --setting /slam_settings.yaml \
  --input_video /data/raw_video.mp4 \
  --input_imu_json /data/imu_data.json \
  --output_trajectory_csv /data/mapping_camera_trajectory.csv \
  --save_map /map/map_atlas.osa \
  --mask_img /data/slam_mask.png
```

Batch localization, one container per demo directory (`slam_mapping.py:291-335`):

```bash
SESSION=/path/to/session
REPO=/home/hcis-s17/author_workdir/voilab
DEMO="$SESSION/demos/demo_<serial>_<timestamp>"   # any directory matching demos/demo* or demos/map*
docker run --rm \
  --volume "$DEMO:/data" \
  --volume "$SESSION/demos/mapping:/map" \
  --volume "$REPO/packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:/slam_settings.yaml" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --env "DISPLAY=$DISPLAY" \
  chicheng/orb_slam3:latest \
  /ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam \
  --vocabulary /ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --setting /slam_settings.yaml \
  --input_video /data/raw_video.mp4 \
  --input_imu_json /data/imu_data.json \
  --output_trajectory_csv /data/camera_trajectory.csv \
  --load_map /map/map_atlas.osa \
  --mask_img /data/slam_mask.png \
  --max_lost_frames 60
```

| Argument | What it is | Notes |
|---|---|---|
| `--volume <dir>:/data` | bind mount (host path exposed inside the container) of the directory being processed | mapping: `demos/mapping`; localization: the demo dir. All inputs and outputs live here |
| `--volume <mapping dir>:/map` | the directory holding the atlas file | In mapping mode this is the *same* host dir as `/data`, mounted twice |
| `--volume <yaml>:/slam_settings.yaml` | the settings file, mounted read-write as a single file | Host path resolved by `slam_mapping.py:468-495`; relative paths resolve against your shell's current directory, so run `uv run umi ...` from the repo root |
| `--volume /tmp/.X11-unix:/tmp/.X11-unix` | X11 socket for the optional viewer window | Mounted unconditionally, even with `enable_gui: false` (`slam_mapping.py:107`) |
| `--env DISPLAY=...` | forwarded whenever the host has `DISPLAY` set | Also unconditional; harmless without the GUI |
| `--rm` | delete the container when it exits | **Localization only.** Mapping omits it (`slam_mapping.py:123`) and leaves a stopped container behind on every run |
| `--user <uid>:<gid>`, `--ipc host` | run as you, share IPC namespace | Added only when `enable_gui: true` (`slam_mapping.py:128-130`) |
| `--vocabulary` | the ORB bag-of-words dictionary used for place recognition | Always the file baked into the image |
| `--save_map` / `--load_map` | write a new atlas / reuse an existing one | The only difference between the two modes |
| `--mask_img` | PNG of pixels to blank out | Mapping passes it only if `generate_mask` is true; localization always passes it |
| `--max_lost_frames` | abort after N untracked frames | Localization only; default `60` (`slam_mapping.py:28`) |

Additional Python-side behaviour worth knowing: a 120 fps input video is transcoded to 60 fps with ffmpeg first and the container is fed `converted_60fps_raw_video.mp4` instead (`slam_mapping.py:401-466`). Localization runs `num_workers` containers concurrently and gives each a timeout of `video_duration_sec * timeout_multiple` (default multiplier 16, `slam_mapping.py:27, 263`).

## 4. Mapping versus localization

Both modes run the same binary. `slam_process_mode` picks between them (`slam_mapping.py:44-48`).

`map_atlas.osa` is ORB-SLAM3's *atlas*: a binary dump of everything ORB-SLAM3 learned from one video — the keyframes it kept, the 3-D map points it triangulated, and which keyframes saw the same points (the covisibility graph). It is written to `<session>/demos/mapping/map_atlas.osa` on shutdown.

The mapping video is processed first and alone because every demonstration must be expressed in **one common coordinate frame**. If each demo built its own map, each trajectory would sit in its own arbitrary frame and the tag-based calibration in stage 05 could not relate them. Mapping produces the shared frame; localization mode then loads that atlas read-only and solves only for the camera pose within it.

Localization globs both `demos/demo*/` and `demos/map*/` (`slam_mapping.py:222-223`). The `map*` glob deliberately re-runs the mapping video in localization mode so that `demos/mapping/camera_trajectory.csv` exists; `calibration.py:83-88` prefers that file and only falls back to `mapping_camera_trajectory.csv`.

Two skip rules: mapping is skipped if `map_atlas.osa` already exists unless `force: true` (`slam_mapping.py:79`); a demo is skipped if its `camera_trajectory.csv` already exists, with no override at all (`slam_mapping.py:242-244`). Delete the artifact to force a re-run.

**Known issue:** the atlas embeds the vocabulary filename and an MD5 checksum; loading it with a different `ORBvoc.txt` makes the binary exit immediately. Atlases are only portable between identical images. See [Known issues](./known-issues.md).

## 5. Settings file reference

One YAML per camera/lens/resolution combination, in `packages/umi/defaults/orb_slam3_settings/`. Seven files ship; only two are referenced by any config. The two live ones are compared below:

- **G9 column** = `gopro10_maxlens_fisheye_setting_v1_720.yaml`, used by `official_gopro9_pipeline_config.yaml:20,30`.
- **G13 column** = `gopro13_fisheye_ratio_4-3_2-7k.yaml`, used by the two `gopro13_fisheye_2-7k*` configs.

The first line of every file must be the literal `%YAML:1.0` directive — OpenCV's parser refuses the file without it.

**Dialect.** `File.version: "1.0"` selects the modern parser, which expects `Camera1.fx` and `IMU.T_b_c1`. Without that key the legacy parser runs and expects `Camera.fx` and `Tbc` instead. Both live files are v1.0. Three shipped files are legacy and unreferenced: `gopro9_maxlens_fisheye_setting.yaml`, `gopro9_wide_setting.yaml` and `gopro10_maxlens_fisheye_setting.yaml` — they have no `File.version` line. Do not mix key styles in one file.

| Group / key | Meaning | Comes from | G9 | G13 |
|---|---|---|---|---|
| `Camera.type` | lens projection model; `KannalaBrandt8` is the equidistant fisheye model | choice of lens | `KannalaBrandt8` | `KannalaBrandt8` |
| `Camera1.fx`, `Camera1.fy` | focal length in pixels, at the *processing* resolution | fisheye intrinsic calibration, rescaled | `282.906909765` | `282.906909765` |
| `Camera1.cx`, `Camera1.cy` | principal point (where the optical axis meets the sensor), in pixels | same calibration | `480.0`, `360.0` | `480.0`, `360.0` |
| `Camera1.k1`..`k4` | fisheye radial distortion coefficients | same calibration | `-0.01746`, `-0.02808`, `0.01283`, `-0.00452` | `0.08133`, `-0.08527`, `0.07741`, `-0.02968` |
| `Camera.width`, `Camera.height` | resolution every frame is resized to before tracking; *not* the video resolution | whatever the fx/cx above were rescaled to | `960`, `720` | `960`, `720` |
| `Camera.fps` | nominal frame rate | the recording mode | `60` | `60` |
| `Camera.RGB` | channel order, 0 = BGR, 1 = RGB | fixed | `1` | `1` |
| `IMU.T_b_c1` | 4x4 rigid transform mapping a point in the camera frame into the IMU body frame (body-from-camera), row-major, translation in metres. Ignore the `# Transformation from body-frame (imu) to camera` comment at `gopro13_fisheye_ratio_4-3_2-7k.yaml:29` — it states the direction backwards, and the identical matrix in the GoPro 10 file is commented the other way round | camera-IMU extrinsic calibration; OpenICC's `T_imu_cam` goes in as-is, see [Calibration](./calibration-openicc.md) §8, then a permutation documented inline at `gopro13_fisheye_ratio_4-3_2-7k.yaml:34-41` | identical matrix | identical matrix |
| `IMU.NoiseGyro` | gyroscope white-noise density | Allan-variance run on a static recording | `0.0015` | `0.033863333333333336` |
| `IMU.NoiseAcc` | accelerometer white-noise density | same | `0.017` | `0.15026666666666666` |
| `IMU.GyroWalk` | gyroscope bias random walk (slow drift of the zero point) | same | `5.0e-5` | `0.25133333333333335` |
| `IMU.AccWalk` | accelerometer bias random walk | same | `0.0055` | `0.0016666666666666668` |
| `IMU.Frequency` | IMU sample rate in Hz | the camera's GPMF telemetry rate, see [GoPro telemetry](./gopro-telemetry.md) | `200.0` | `197.577` |
| `ORBextractor.nFeatures` | keypoints extracted per frame | tuning; higher is slower and more robust | `1250` | `1250` |
| `ORBextractor.scaleFactor` | scale ratio between image-pyramid levels | tuning | `1.2` | `1.2` |
| `ORBextractor.nLevels` | pyramid levels | tuning; scales with resolution | `8` | `8` |
| `ORBextractor.iniThFAST` / `minThFAST` | corner-detector thresholds, tried in that order | lower them for low-contrast footage | `20` / `7` | `20` / `7` |
| `System.thFarPoints` | discard map points beyond this many metres | tuning | `20.0` (line 79) | *absent* |
| `Viewer.*` | Pangolin viewer cosmetics (keyframe size, point size, viewpoint) | taste; only matter with `--enable_gui` | present | present |

All IMU keys are mandatory: `gopro_slam` always constructs the system in IMU-monocular mode, so a missing `IMU.T_b_c1` or noise value is a hard parse failure, not a fallback.

**Known issue:** the GoPro 13 file is not a GoPro 13 calibration. Its `fx/fy/cx/cy` at `gopro13_fisheye_ratio_4-3_2-7k.yaml:9-12` are byte-identical to the GoPro 9-derived values in the GoPro 10 file, and its `IMU.T_b_c1` at `:42-45` is byte-identical to the GoPro 10 extrinsic. Only `k1..k4` and the IMU noise numbers are camera-13-specific. See [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) for the corrected numbers and [Known issues](./known-issues.md).

## 6. Outputs and how to read them

| Artifact | Where | What it is |
|---|---|---|
| `mapping_camera_trajectory.csv` | `demos/mapping/` | trajectory from the mapping run |
| `camera_trajectory.csv` | each `demos/demo*/`, and `demos/mapping/` | trajectory from a localization run — the file everything downstream consumes |
| `map_atlas.osa` | `demos/mapping/` | the shared map, written on clean shutdown |
| `slam_stdout.txt`, `slam_stderr.txt` | next to the video in each processed dir | raw container output; nothing parses them, they exist for you |
| `slam_mask.png` | next to the video | generated by Python, blanks out mirrors and fingers |

The CSV columns and their exact semantics are specified in [Data formats](./data-formats.md); they come from the fork's C++ writer, which is not vendored here, and `chicheng/orb_slam3:latest` is an unpinned tag, so a rebuilt image could reorder the columns or add tracking states without anything in this repository noticing — only `timestamp`, `is_lost`, `is_keyframe` and the pose columns are read by any code here. Two things matter here: the `is_lost` column marks frames the tracker could not place, and rows with `is_lost=true` carry an all-zero quaternion which is invalid — `orb_slam_util.load_csv_trajectory` (`packages/umi/src/umi/common/orb_slam_util.py:28-46`) drops those rows, and `dataset_planning.py:431` repairs them with `q_w = 1` before converting to rotations.

Useful strings to grep for in the logs: `n_lost_frames=`, `Lost tracking on`, `Fail to track local map!`, `CSV camera trajectory saved!`, `Atlas loaded!`.

## 7. Diagnosing a failed run

| Symptom | Likely cause | What to do |
|---|---|---|
| `FileNotFoundError: SLAM settings file not found: /ORB_SLAM3/Examples/...` | the default `slam_settings_file` is a path *inside* the container but is validated on the host (`slam_mapping.py:34` vs `:489`) | set `slam_settings_file` to a host path in your config |
| `FileNotFoundError` naming a settings file under `packages/` that you can see exists | relative settings paths resolve against your shell's working directory (`slam_mapping.py:484-486`) | run the CLI from the repo root |
| `ValueError: Unknown mode, only accepts: create_map, batch_slam` | `slam_process_mode` was omitted; its default is the string `"slam_mapping"`, which matches neither mode (`slam_mapping.py:31`) | set the key explicitly in both SLAM stages |
| Demo has `slam_stdout.txt` ending in `Lost tracking on N >= 60 frames. Terminating!` and **no** `camera_trajectory.csv` | the `--max_lost_frames` early exit fires before the CSV is written; the process exits non-zero and the batch marks the demo `failed` | re-shoot the demo, or raise `max_lost_frames` to inspect where it degrades |
| Demo status `timeout` in the stage result | the run exceeded `duration_sec * timeout_multiple` (default 16x realtime) | raise `timeout_multiple`, or lower `num_workers` if the machine is oversubscribed |
| Mapping stage hangs forever with no new log lines | mapping drains the container's stdout to EOF *before* reading stderr (`slam_mapping.py:185-192`); a chatty stderr fills the ~64 KiB pipe buffer and both sides block. There is no timeout in mapping mode | kill the container, re-run; watch `slam_stdout.txt` for progress |
| `Permission denied` writing into the session dir, or root-owned CSV/OSA/log files | `--user` is only passed when `enable_gui: true`, so the container writes as root (`slam_mapping.py:128-129`) | `sudo chown -R "$USER" <session>` after the stage |
| Every demo runs but the dataset is nearly empty | demos are dropped later, in `dataset_planning.py:411-430`: no `camera_trajectory.csv`, more than 10 lost frames, or fewer than 60 valid frames in the plan window. A demo dies if *any* of its gripper cameras is dropped | read the `Skipping <demo>, ...` lines in the stage log |

**Known issue:** the count of dropped demos is swallowed by a logging bug at `dataset_planning.py:582` and never printed. Count the `Skipping` lines instead. See [Known issues](./known-issues.md).

## 8. Interactive debugging with the SLAM GUI

`--enable_gui` opens ORB-SLAM3's Pangolin viewer: the current frame with tracked keypoints drawn on it, and a 3D view of the map points, keyframes and the live camera pose. It is the fastest way to see *where* tracking degrades — featureless walls, motion blur, the mask covering too much.

It needs an X11 display: `DISPLAY` set, `/tmp/.X11-unix` present, and a readable `~/.Xauthority`. Over SSH use `ssh -X`. The service refuses to construct without `DISPLAY` (`visualize_slam_gui.py:126-131`). Software rendering is forced via `LIBGL_ALWAYS_SOFTWARE=1`, so expect it to be slow.

```bash
cd /home/hcis-s17/author_workdir/voilab
DEMO="/path/to/session/demos/demo_<serial>_<timestamp>"   # a real demo dir, see: ls <session>/demos
uv run umi visualize-slam-gui $DEMO/raw_video.mp4 \
  --session-dir $DEMO \
  --settings-file packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml
```

**Known issue:** `--settings-file` looks optional but is mandatory. `cli.py:37` always writes the key into the config dict, so omitting the flag stores `None`, the documented default never applies, and the service raises `ValueError: slam_settings_file is not configured`. See [Known issues](./known-issues.md).

Two more traps: it always runs in mapping mode (hardcoded `--save_map /data/map_atlas.osa`) and cannot load an existing atlas, and the `--session-dir` above is the demo directory on purpose — the container mounts it at `/data` (`visualize_slam_gui.py:150`) and reads `<session-dir>/imu_data.json` (`:187`), which the pipeline writes per demo directory, not at the session root.

## 9. Rough edges

- `gopro13_wide_angle_pipeline_config.yaml:20,32` points at `gopro13_fisheye_60fps_2-7k.yaml`, which does not exist. That config cannot run.
- `gopro13_normal_pipeline_config.yaml` sets no `slam_settings_file` at all, so it hits the unusable in-container default.
- `mask_pts_json_path` is read into a field (`slam_mapping.py:36`) and never used. Three configs set it; it does nothing.
- `packages/umi/src/umi/services/batch_slam.py` is a placeholder that touches no SLAM code. The real batch stage is `SLAMMappingService` with `slam_process_mode: batch_slam`.
- `SLAMMappingService.create_map()` (`slam_mapping.py:497-548`) fabricates empty outputs when `pull_docker` is false. It exists for tests; the pipeline never calls it.
- Five of the seven shipped settings files are referenced by nothing: the three legacy ones named in §5, plus `gopro10_maxlens_fisheye_setting_v1.yaml` (working size 2704 x 2028) and `gopro10_maxlens_fisheye_setting_v1_480.yaml` (640 x 480). Those two are v1.0-schema copies of the same GoPro 10 optics at other working resolutions — their `k1..k4` are identical to the live G9 file — so either can serve as a template.

All of these are ranked with severity in [Known issues](./known-issues.md).

---

**Next:** [Data formats](./data-formats.md) · [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) · [Known issues](./known-issues.md)
