# Session layout and data formats

Every file the SLAM pipeline reads or writes inside a session directory, with its exact schema and a snippet that loads it.

**Read this if:** you need to know what a pipeline artifact contains, or you are writing code that consumes one.

**Before you start:** [Pipeline overview](./pipeline-overview.md) — which stage produces what, and in what order. Run every snippet on this page with `uv run python` from the repository root; the `umi` package is only importable inside the uv environment.

---

## The session directory

A *session* is one recording day: a folder you point `session_dir` at. A *demo* is one continuous clip from one camera, exploded into its own `demos/demo_<serial>_<ts>/` directory; an *episode* is a contiguous stretch of frames inside a demo where every camera tracked successfully, and is what becomes one training trajectory. Everything below is created by the pipeline; you only supply the loose `.MP4` files. The one exception is `check_result.txt` — if you write that file into a *gripper* camera's demo directory and its text does not start with `true`, stage 06 skips that camera (`dataset_planning.py:403-410`), the camera count then falls short and the entire demo is dropped, every episode it would have produced (`dataset_planning.py:511-514`).

```text
<session_dir>/
├── raw_videos/                              # 00_process_video (video_organization.py:36)
│   ├── mapping.mp4                          #   largest input clip, renamed (video_organization.py:45-51)
│   ├── GX010007.MP4                         #   symlink -> ../demos/demo_.../raw_video.mp4
│   └── gripper_calibration/                 # 00_process_video (video_organization.py:57)
│       └── GX010002.MP4                     #   symlink, one clip per camera serial
├── demos/                                   # 00_process_video (video_organization.py:24)
│   ├── mapping/
│   │   ├── raw_video.mp4                    # 00_process_video (video_organization.py:101)
│   │   ├── converted_60fps_raw_video.mp4    # 02, only if the source is ~120 fps (slam_mapping.py:88)
│   │   ├── imu_data.json                    # 01_extract_gopro_imu (imu_extraction.py:100)
│   │   ├── slam_mask.png                    # 02_create_map (slam_mapping.py:395)
│   │   ├── map_atlas.osa                    # 02_create_map (slam_mapping.py:78)
│   │   ├── mapping_camera_trajectory.csv    # 02_create_map (slam_mapping.py:93)
│   │   ├── camera_trajectory.csv            # 03_batch_slam re-runs mapping/ too (slam_mapping.py:223)
│   │   ├── slam_stdout.txt / slam_stderr.txt  # 02/03, raw container logs
│   │   ├── tag_detection.pkl                # 04_detect_aruco (aruco_detection.py:123)
│   │   ├── tx_slam_tag.json                 # 05_run_calibrations (calibration.py:183)
│   │   └── object_poses.json                # 07_frame_to_pose (frame_to_pose.py:327)
│   ├── gripper_calibration_<serial>_<timestamp>/
│   │   ├── raw_video.mp4, imu_data.json, tag_detection.pkl   # stages 00 / 01 / 04
│   │   └── gripper_range.json               # 05_run_calibrations (calibration.py:300)
│   │                                        # no camera_trajectory.csv: 03 skips these dirs
│   └── demo_<serial>_<timestamp>/
│       ├── raw_video.mp4                    # 00_process_video
│       ├── converted_60fps_raw_video.mp4    # 02/03, only if the source is ~120 fps
│       ├── imu_data.json                    # 01_extract_gopro_imu
│       ├── slam_mask.png                    # 03_batch_slam (slam_mapping.py:255)
│       ├── camera_trajectory.csv            # 03_batch_slam (slam_mapping.py:251)
│       ├── slam_stdout.txt / slam_stderr.txt  # 03_batch_slam
│       ├── tag_detection.pkl                # 04_detect_aruco
│       └── check_result.txt                 # OPTIONAL manual kill switch, you write it
├── dataset_plan.pkl                         # 06_generate_dataset_plan (dataset_planning.py:583)
└── dataset.zarr.zip                         # 0{7,8}_generate_replay_buffer (replay_buffer.py:246)
```

## Demo directory names and symlinks

Stage 00 moves each video out of `raw_videos/` into its own directory and leaves a symlink behind, so the original filename still resolves.

| Source clip | Destination directory | Rule |
|---|---|---|
| filename starts with `mapping` | `demos/mapping/` | `video_organization.py:90-91` |
| filename or parent dir starts with `gripper_cal` | `demos/gripper_calibration_<serial>_<ts>/` | `video_organization.py:92-93` |
| anything else | `demos/demo_<serial>_<ts>/` | `video_organization.py:94-95` |

- `<serial>` is the GoPro's `QuickTime:CameraSerialNumber`, read with ExifTool (`video_organization.py:87`) — this is how stage 06 pairs a demo with the right gripper calibration. `<ts>` is the clip start time as `%Y.%m.%d_%H.%M.%S.%f`, giving names like `demo_C3441328164125_2025.09.11_14.02.07.123456`.
- The video is renamed to `raw_video.mp4` (`video_organization.py:101`) and a **relative** symlink replaces its old path in `raw_videos/` (`video_organization.py:104-107`). Copying a session with `cp -r` without `-L`, or moving `demos/` on its own, breaks those links. Re-running stage 00 is safe: entries that are already symlinks are skipped (`video_organization.py:82`).

## Artifact reference

### imu_data.json

**Written by** `01_extract_gopro_imu` into every demo directory (`imu_extraction.py:100`). It holds GPMF telemetry — the sensor metadata track GoPro embeds in the MP4 alongside video. The schema imitates the older `gopro-telemetry` JavaScript output because the ORB-SLAM3 container parses exactly that shape.

```json
{"1": {"streams": {"ACCL": {"samples": [{"value": [8.19, -0.33, 5.18], "cts": 1.201}]},
                   "GYRO": {"samples": [{"value": [-0.07, -0.01, -0.03], "cts": 1.201}]}}}, "frames/second": 0.0}
```

`"1"` is a fixed device key. Stream keys are GPMF four-character codes, **omitted entirely when empty**. `cts` is milliseconds from video start (`imu_extraction.py:118`) and can be slightly negative. `frames/second` is hard-coded to `0.0` (`imu_extraction.py:110`); the SLAM binary reads frame rate from the video instead. Stream-by-stream units: [GoPro telemetry](./gopro-telemetry.md).

```python
import json
d = json.load(open("<session>/demos/demo_XXXX/imu_data.json"))
accl = d["1"]["streams"]["ACCL"]["samples"]
print(len(accl), accl[0]["cts"], accl[0]["value"])  # count, ms, [x, y, z] m/s^2
```

**Known issue:** if GPMF parsing raises, the service writes a fabricated one-sample file and reports success (`imu_extraction.py:127-144`); if the MP4 is unreadable it writes `{"1": {"streams": {}}, ...}` with no error at all. An `imu_data.json` that exists is not proof of good IMU data — see [Known issues](./known-issues.md).

### slam_mask.png

**Written by** `02_create_map` (`slam_mapping.py:393-399`) and `03_batch_slam` (`slam_mapping.py:264-266`). Single-channel PNG sized by the `resolution` config key (default `(2028, 2704)`, height first). White (255) pixels cover the mirror and gripper fingers; the SLAM binary blanks them so the hand hardware is not tracked as scene geometry.

```python
import cv2, numpy as np
m = cv2.imread("<session>/demos/mapping/slam_mask.png", cv2.IMREAD_GRAYSCALE)
print(m.shape, np.unique(m))  # (2028, 2704) [0 255]
```

### camera_trajectory.csv

**Written by** the `gopro_slam` binary inside the `chicheng/orb_slam3` container, one row per video frame. Stage 02 names it `mapping_camera_trajectory.csv`; stage 03 names it `camera_trajectory.csv`. Stage 05 reads `camera_trajectory.csv` from `demos/mapping/` and falls back to `mapping_camera_trajectory.csv` with a `logger.warning` if stage 03 never re-ran the mapping clip (`calibration.py:83-88`). Header, in this exact order:

| Column | Meaning |
|---|---|
| `frame_idx`, `timestamp` | 0-based frame index in the video passed to SLAM, and seconds from its start |
| `state` | tracker state, see below |
| `is_lost`, `is_keyframe` | `True`/`False`; the row's pose is meaningless when `is_lost` |
| `x`, `y`, `z`, `q_x`, `q_y`, `q_z`, `q_w` | camera pose in the SLAM map frame: position in metres, orientation as a quaternion in scipy's xyzw order |

`state` is ORB-SLAM3's tracking enum: `-1` system-not-ready, `0` no-images-yet, `1` not-initialized, `2` OK, `3` recently-lost, `4` lost, `5` OK-KLT, `6` init-relocalize (a fork addition, used while a loaded map has not relocalized). `is_lost` is the negation of "has tracking": it is **false** for the tracked states (`2` OK, `3` recently-lost) and **true** for every other state, so filter with `df.loc[~df.is_lost]` — see [ORB-SLAM3](./orb-slam3.md). A demo whose tracking failed early may have **no CSV at all**: the binary exits non-zero once `--max_lost_frames` is hit and never writes the file.

**The `q_w = 0` trap.** Lost rows carry an all-zero position *and* an all-zero quaternion. `scipy.spatial.transform.Rotation.from_quat` rejects a zero-norm quaternion, so a naive whole-file conversion raises. The two in-repo fixes: drop lost rows before converting (`common/orb_slam_util.py:33`), or overwrite `q_w` with 1 on lost rows to keep the array length (`dataset_planning.py:431`).

```python
import pandas as pd
df = pd.read_csv("<session>/demos/demo_XXXX/camera_trajectory.csv")
print(df.columns.tolist(), len(df), int(df.is_lost.sum()), int(df.is_keyframe.sum()))
print(df.loc[~df.is_lost, ["timestamp", "x", "y", "z", "q_w"]].head())
```

### map_atlas.osa

**Written by** `02_create_map` via `--save_map` (`slam_mapping.py:78,162`), read back by `03_batch_slam` via `--load_map` (`slam_mapping.py:224,329`). An opaque Boost binary serialization of the ORB-SLAM3 map, prefixed with the vocabulary checksum: only the same container image can load it, and there is no Python reader. Deleting it is how you force stage 02 to rebuild the map.

### tag_detection.pkl

**Written by** `04_detect_aruco` into every directory under `demos/` that has a `raw_video.mp4` (`aruco_detection.py:123`). ArUco tags are printed square fiducial markers; the stage detects them and solves each one's 6-DoF pose relative to the camera. The pickle is a `list[dict]`, one entry per decoded frame. `rvec` is an axis-angle rotation and `tvec` a translation in metres, both camera-to-tag; `corners` are pixel coordinates. `tag_dict` is empty on frames with no detections. A tag is kept only if its ID has a size in `marker_size_map` (`common/cv_util.py:167-168`); because the shipped `aruco_config.yaml` sets `default: 0.16`, that is every ID in `DICT_4X4_50`, not just the 20 listed explicitly.

```python
{"frame_idx": int,          # decode order
 "time": float,             # seconds from video start
 "tag_dict": {tag_id: {"rvec": (3,), "tvec": (3,), "corners": (4, 2)}}}
```

```python
import pickle
res = pickle.load(open("<session>/demos/demo_XXXX/tag_detection.pkl", "rb"))
print(len(res), res[0]["frame_idx"], res[0]["time"])
for tag_id, t in res[0]["tag_dict"].items():
    print(tag_id, t["rvec"].round(3), t["tvec"].round(3), t["corners"].shape)
```

### tx_slam_tag.json

**Written by** `05_run_calibrations` into `demos/mapping/` (`calibration.py:182-183`). One 4x4 homogeneous transform that maps the calibration tag's frame into the SLAM map frame. Stage 06 inverts it to express every camera pose relative to the table tag instead of the arbitrary SLAM origin.

```python
import json, numpy as np
tx = np.array(json.load(open("<session>/demos/mapping/tx_slam_tag.json"))["tx_slam_tag"])
print(tx.shape, tx[:3, 3])  # (4, 4); tag origin in SLAM coordinates, metres
```

### gripper_range.json

**Written by** `05_run_calibrations` into each `gripper_calibration_*` directory (`calibration.py:293-300`). It records how far that physical gripper opens, measured from the two finger tags during the calibration clip. Tag IDs follow `left = gripper_id * 6`, `right = left + 1` (`calibration.py:240-241`).

```python
import json
g = json.load(open("<session>/demos/gripper_calibration_XXXX/gripper_range.json"))
print(g["gripper_id"], g["left_finger_tag_id"], g["right_finger_tag_id"])
print(round(g["min_width"], 4), round(g["max_width"], 4))  # metres
```

### object_poses.json

**Written by** `07_frame_to_pose`, reconstruct config only, into `demos/mapping/` (`frame_to_pose.py:327-329`). A JSON list with one entry per planned episode, giving where the task objects sat at the start of that episode so a simulator can rebuild the scene. `status` is `full` (every expected object found), `partial`, or `none` (`frame_to_pose.py:307-313`); object names come from the hard-coded task registry at `frame_to_pose.py:20-35`. Three real files are committed here — `video/example_kitchen/`, `video/example_living_room/` and `video/example_dining_room/`, each under `demos/mapping/` (50, 56 and 55 entries). They came from an older revision and carry a fifth key, `sample_id`, that the current writer no longer emits (`frame_to_pose.py:316-321`).

```json
[{"video_name": "converted_60fps_raw_video.mp4", "episode_range": [0, 362],
  "objects": [{"object_name": "blue_cup", "rvec": [...3], "tvec": [...3]}],
  "status": "full"}]
```

```python
import json
eps = json.load(open("<session>/demos/mapping/object_poses.json"))
print(len(eps), eps[0]["status"], eps[0]["episode_range"])
for o in eps[0]["objects"]:
    print(o["object_name"], [round(v, 3) for v in o["tvec"]])
```

### dataset_plan.pkl

**Written by** `06_generate_dataset_plan` at the session root (`dataset_planning.py:583`). A pickled `list[dict]`, one entry per episode: which video frames form the episode, and the robot state on each of those frames. No pixels are copied yet — this is a manifest. TCP means tool centre point, the gripper tip, offset from the camera by the hard-coded rig geometry at `dataset_planning.py:48-52`. `video_path` names `converted_60fps_raw_video.mp4` instead whenever that file exists (`dataset_planning.py:558-560`), which is every demo on 120 fps GoPro 13 footage; downstream stages resolve it verbatim.

```python
{"episode_timestamps": np.ndarray,                    # (T,) seconds
 "grippers": [{"tcp_pose": np.ndarray,                # (T, 6) pos + axis-angle, tag frame
               "gripper_width": np.ndarray,           # (T,) metres
               "demo_start_pose": np.ndarray,         # (6,) first valid pose of the whole demo
               "demo_end_pose": np.ndarray}],         # (6,)
 "cameras": [{"video_path": "demo_XXXX/raw_video.mp4",  # relative to <session>/demos/
              "video_start_end": (start, end)}]}        # frame index half-open range
```

```python
import pickle
plan = pickle.load(open("<session>/dataset_plan.pkl", "rb"))
ep = plan[0]
print(len(plan), ep["episode_timestamps"].shape, len(ep["grippers"]), len(ep["cameras"]))
print(ep["grippers"][0]["tcp_pose"].shape, ep["grippers"][0]["gripper_width"].shape)
print(ep["cameras"][0]["video_path"], ep["cameras"][0]["video_start_end"])
```

**Known issue:** the filename comes from the `dataset_plan_filename` config key, whose in-code fallback is `dataset.pkl` while every shipped config sets `dataset_plan.pkl` (`dataset_planning.py:36`). Tools that guess the name check both — see [Known issues](./known-issues.md).

### dataset.zarr.zip

**Written by** the replay-buffer stage at the session root (`replay_buffer.py:246-249`), and the only artifact training consumes. The name is the `output_filename` config key (`replay_buffer.py:32,57`); unlike `dataset_plan_filename` it has no in-code default, and the stage asserts it is set (`replay_buffer.py:51`). All five shipped configs set `dataset.zarr.zip`. Zarr is a chunked array format; here the whole store is packed into one ZIP file. Frames are decoded from the videos, tag pixels inpainted, masked, resized, and stored as one compressed chunk per frame.

```text
/
├── meta/
│   └── episode_ends            int64   (n_episodes,)   # exclusive end index of each episode
├── data/                                               # N = episode_ends[-1] frames
│   ├── camera{c}_rgb           uint8   (N, o0, o1, 3)  # o = output_resolution, JPEG XL
│   ├── robot{g}_eef_pos        float32 (N, 3)          # metres, tag frame
│   ├── robot{g}_eef_rot_axis_angle  float32 (N, 3)
│   ├── robot{g}_gripper_width  float32 (N, 1)          # metres
│   ├── robot{g}_demo_start_pose     float64 (N, 6)     # constant within an episode
│   └── robot{g}_demo_end_pose       float64 (N, 6)
└── reconstruct/                                        # 07_frame_to_pose only; see known issue below
    ├── location                <U32    (1,)            # the `task` config value, else "unknown"
    ├── object_poses            float64 (M, 2, 3)       # [rvec, tvec] per detection
    ├── object_name             <U32    (M,)
    └── object_video_name       <U64    (M,)
```

`c` and `g` start at 0; camera index 0 is the right gripper, 1 the left (`dataset_planning.py:360`). Episode `i` spans `episode_ends[i-1]:episode_ends[i]`, with `0` as the implicit first start — the ends are **exclusive** (`infrastructure/replay_buffer.py:549-553`). Poses are float32, except `demo_start_pose`/`demo_end_pose`, which inherit float64 from the plan (`replay_buffer.py:154-158`). `camera{c}_rgb` is shaped `(N,) + tuple(output_resolution) + (3,)` (`replay_buffer.py:198`), while the resizer reads the same pair as (W, H) (`replay_buffer.py:269-272`) — only square values are unambiguous, and every shipped config is square. It uses the JPEG XL codec `imagecodecs_jpegxl` (`replay_buffer.py:193`), which is not a zarr built-in — register it before opening, or the read fails with an unknown-codec error:

```python
import zarr
from umi.infrastructure.imagecodecs_numcodecs import register_codecs
register_codecs(verbose=False)
root = zarr.group(zarr.ZipStore("<session>/dataset.zarr.zip", mode="r"))
print(root["meta/episode_ends"][:5], root["data/camera0_rgb"].shape)
```

Pass `mode=` by keyword: `zarr.ZipStore(path, "r")` binds `"r"` to the `compression` argument and raises `NotImplementedError`. Training code registers the same codec itself at import (`packages/diffusion_policy/src/diffusion_policy/dataset/umi_dataset.py:27`); see [Training and eval](./training-and-eval.md).

**Name trap:** `robot{g}_eef_rot_axis_angle` is a genuine axis-angle triple **on disk only** (training-package paths here and below are relative to `packages/diffusion_policy/src/diffusion_policy/`). `shape_meta` declares it `raw_shape: [3]` / `shape: [6]` with `rotation_rep: rotation_6d` (`config/task/umi.yaml:45-51`), `UmiDataset` overwrites it with the 6-D form before the policy sees it (`dataset/umi_dataset.py:364`), and `ROS2Environment.get_obs` emits six values under the same key (`environments/ros2_environment.py:406,420`). Its companion `robot{g}_eef_rot_axis_angle_wrt_start` is 6-D in every producer — the dataset derives it (`dataset/umi_dataset.py:312-325`), the simulator computes it directly (`scripts/v1/engine.py:572`) — and is stored in no zarr.

**Known issue:** with no `demos/mapping/object_poses.json`, the stage raises `ValueError: missing object_codec for object array` at `replay_buffer.py:239` — the fallback at `replay_buffer.py:87-90` builds `dtype=object` arrays where the has-file path uses `'U32'`/`'U64'` (`:84-85`). Only the reconstruct config runs `07_frame_to_pose`, so the other four shipped configs fail here, after every video has already been decoded and compressed. See [Known issues](./known-issues.md).

**Forcing a redo.** No stage resumes: each decides whether its work is already done purely by checking whether its own output exists, so deleting an artifact is how you re-run the stage that writes it. The per-stage rules are in [Pipeline overview](./pipeline-overview.md).

## The action vector

No artifact stores an action: the dataset assembles one and every runner consumes it, so the 10-element layout is the contract between them. One row, per robot:

| Index | Meaning | Built from |
|---|---|---|
| `0:3` | end-effector position, metres | `robot{g}_eef_pos` |
| `3:9` | rotation as `rotation_6d` — the first two rows of the rotation matrix | `robot{g}_eef_rot_axis_angle` |
| `9` | gripper width, metres | `robot{g}_gripper_width` |

`SequenceSampler` concatenates those three arrays into seven columns per robot (`common/sampler.py:93-103`), then `UmiDataset.__getitem__` widens the rotation to 6-D (`dataset/umi_dataset.py:341-360`), matching the `shape: [10]` and `rotation_rep: rotation_6d` that `shape_meta.action` declares (`config/task/umi.yaml:78-83`). Both consumers agree with that layout: `ROS2Environment.step` rejects any array that is not `(Ta, 10)` (`environments/ros2_environment.py:177-178`), and the Isaac Sim rollout slices the same three ranges out of `predict_action` (`scripts/v1/rollout.py:207-218`). Positions and rotations are expressed relative to the newest observed pose under the shipped `pose_repr: relative` (`config/task/umi.yaml:87-89`), which is why the rollout multiplies each predicted pose back onto its reference frame.

---

**Next:** [Pipeline overview](./pipeline-overview.md) · [Training and eval](./training-and-eval.md) · [Known issues](./known-issues.md)
