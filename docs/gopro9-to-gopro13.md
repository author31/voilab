# GoPro 9 constants and the GoPro 13 migration checklist

Every camera-specific number in the pipeline, where it lives, and what to change when you swap cameras.

**Read this if:** you are recording with a camera that is not the GoPro HERO9 the original UMI pipeline was tuned for, or you want to know why your camera-trajectory tracking drifts.

**Before you start:** [Pipeline overview](./pipeline-overview.md), [Pipeline config](./pipeline-config.md)

---

## 1. Why this page exists

Upstream UMI was built around one hardware setup: a GoPro HERO9 in a fixed lens mode, recording 2704x2028 at 60 fps, bolted to a 3D-printed gripper. Dozens of numbers in this repo encode that setup — the lens model, the pixel size of the gripper in frame, the distance from the camera body to the gripper tip. Some are configurable from YAML; a significant number are literals in Python. This fork targets GoPro 13; in practice that support is partial, and Section 4 is the part to read before trusting any GoPro 13 trajectory.

Four terms used throughout:

- **SLAM** — simultaneous localisation and mapping: recovering the camera's 3D trajectory from the video itself. This repo runs ORB-SLAM3 in monocular-inertial mode, meaning it fuses one camera with the camera's own IMU (inertial measurement unit: accelerometer + gyroscope).
- **Intrinsics** — how a camera projects the world onto pixels: focal length `fx`/`fy`, principal point `cx`/`cy` (the pixel where the optical axis hits the sensor, near but not exactly the image centre), and distortion coefficients `k1..k4`.
- **Extrinsic** — the fixed rigid transform between two sensors; here, between the camera and its IMU.
- **ArUco tag** — a printed square black-and-white fiducial marker whose pose a camera can recover; UMI sticks them on the gripper fingers and on the table.

The pipeline reads camera constants from **two independent places** that are never cross-checked:

```text
raw video (2704x2028, 60 or 120 fps)
  |
  +-- 02/03 SLAM      <- slam_settings_file: ORB-SLAM3 YAML
  |                      fx/cx/cy/k1..k4 expressed at 960x720, plus the whole IMU block
  |                      + slam_mask.png, drawn by Python at config key `resolution`
  |
  +-- 04 ArUco        <- camera_intrinsics_path: OpenICC JSON at native resolution
  |                      auto-rescaled to the real frame size at aruco_detection.py:102-104
  |
  +-- 05/06 planning  <- nominal_z, tcp_offset, gripper geometry (partly hardcoded)
  |
  +-- last stage      <- output_resolution -> dataset.zarr.zip
```

Nothing validates that the two intrinsics sources agree. They currently do not. (`dataset.zarr.zip` is the final chunked-array dataset — see [Data formats](./data-formats.md).)

## 2. The constants inventory

"SLAM YAML" below means the file named by the `slam_settings_file` config key — for GoPro 13 that is `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml`, for GoPro 9 it is `.../gopro10_maxlens_fisheye_setting_v1_720.yaml` (yes, a GoPro 10 file). Those files are canonical for every SLAM-YAML row below; [ORB-SLAM3](./orb-slam3.md) §5 tabulates the same values, so after a recalibration read the YAML rather than either table.

| Constant | Where it lives (file:line) | GoPro 9 value | GoPro 13 value in this repo | Overridable from YAML? |
|---|---|---|---|---|
| ArUco-stage intrinsics JSON | `aruco_detection.py:29` selects it | `gopro9_intrinsics_2_7k.json` | `gopro13_intrinsics_2_7k.json` | yes, `camera_intrinsics_path` |
| SLAM camera model | SLAM YAML `:7` | `KannalaBrandt8` | `KannalaBrandt8` | file edit only |
| SLAM `fx` / `fy` | SLAM YAML `:9-10` | 282.906909765 | **282.906909765 (identical)** | file edit only |
| SLAM `cx` / `cy` | SLAM YAML `:11-12` | 480.0 / 360.0 | **480.0 / 360.0 (identical)** | file edit only |
| SLAM `k1..k4` | SLAM YAML `:14-17` | -0.0174584, -0.0280825, 0.0128327, -0.0045198 | 0.0813337, -0.0852681, 0.07741, -0.0296763 | file edit only |
| SLAM image size | SLAM YAML `:20-21` | 960 x 720 | 960 x 720 | file edit only |
| SLAM `Camera.fps` | SLAM YAML `:24` | 60 | 60 | file edit only |
| Camera-to-IMU extrinsic | SLAM YAML `:30-45` (`IMU.T_b_c1`) | 4x4, tz = -0.05175258 | **byte-identical to GoPro 9/10** | file edit only |
| IMU noise densities | GoPro 9 YAML `:49-50`; GoPro 13 YAML `:48-49` | NoiseGyro 0.0015, NoiseAcc 0.017 | 0.033863333, 0.150266667 | file edit only |
| IMU random walks | GoPro 9 YAML `:51-52`; GoPro 13 YAML `:50-51` | GyroWalk 5.0e-5, AccWalk 0.0055 | 0.251333333, 0.001666667 | file edit only |
| IMU sample rate | GoPro 9 YAML `:53`; GoPro 13 YAML `:52` | 200.0 Hz | 197.577 Hz | file edit only |
| Video resolution `[h, w]` | `slam_mapping.py:35`, `calibration.py:34` | `[2028, 2704]` | `[2028, 2704]` | yes, `resolution` |
| Image mask polygons | `cv_util.py:256-337` | pixel points at `[2028, 2704]` | same | **no** |
| Mask JSON files | `slam_mapping.py:36` | unused | unused | key exists, does nothing |
| Dataset image size | `replay_buffer.py:33` | `[256, 256]` | `[224, 224]` | yes, `output_resolution` |
| ArUco tag sizes | `packages/umi/defaults/calibration/aruco_config.yaml` | same file | same file | yes, `aruco_config_path` |
| SLAM reference tag id | `calibration.py:28` | 13 | 13 | yes, `tag_id` |
| Finger-tag depth gate | `calibration.py:33`, `dataset_planning.py:33` | `nominal_z` 0.072 m | `nominal_z` **0.01 m** | yes, `nominal_z` |
| Depth-gate tolerance | `cv_util.py:210` | 0.008 m | 0.008 m | **no** |
| Gripper TCP offset | `dataset_planning.py:32` | 0.205 m (default) | 0.205 m (default) | yes, `tcp_offset` |
| Camera-to-gripper geometry | `dataset_planning.py:48-49` | 0.086 m, 0.01465 m | same | **no** |
| 120 -> 60 fps transcode | `slam_mapping.py:420` | triggers on any 120 fps clip | same | **no** |

## 3. The four shipped intrinsics files

All four live in `packages/umi/defaults/calibration/` and use the OpenImuCameraCalibrator (OpenICC) JSON schema, parsed at `packages/umi/src/umi/common/cv_util.py:11-61`. Every one declares `"intrinsic_type": "FISHEYE"`, meaning the Kannala-Brandt fisheye model that ORB-SLAM3 calls `KannalaBrandt8`.

| | gopro9_intrinsics_2_7k | gopro9_intrinsics_normal_lens | gopro13_intrinsics_2_7k | gopro13_intrinsics_4k |
|---|---|---|---|---|
| declared resolution (w x h) | 2704 x 2028 | 2704 x 2028 | 2704 x 2028 | **3000 x 4000** |
| `focal_length` (fx) | 796.8545 | 2477.1334 | 856.6956 | 413.33 |
| `principal_pt_x` | 1354.4265 | 2531.8859 | 1365.8744 | 1669.4582 |
| `principal_pt_y` | 1011.4847 | 2573.6044 | 987.0008 | 1989.4062 |
| `radial_distortion_1..4` | -0.02196, -0.01896, 0.00169, -0.00017 | -0.27923, 0.09938, -0.00089, 0.00006 | -0.28880, 0.08287, 0.00221, -0.00103 | 0.02349, -0.02209, 0.02300, -0.01390 |
| reprojection error (px) | 0.2916 | 3.4844 | 1.1632 | 0.2916 |
| calibration images | 59 | 11 | 216 | 59 |
| used by | `official_gopro9_pipeline_config.yaml:36` | nothing | all four gopro13 configs | nothing |

Note that `focal_length` alone is `fx`; `aspect_ratio` is meant to give `fy`, but `cv_util.py:58` builds `K` with `fx == fy` and ignores it, while `frame_to_pose.py:174` does use it. See [Known issues](./known-issues.md).

**Known issue:** the two unreferenced files are both unusable. `gopro13_intrinsics_4k.json:2,4-5,18` is a hand-edited copy of the GoPro 9 file (identical reprojection error and image count) with width and height swapped relative to a real 4:3 4K frame; `gopro9_intrinsics_normal_lens.json:10-11` puts the principal point at y = 2573.6 in a frame 2028 px tall. Do not copy either as a template. See [Known issues](./known-issues.md).

## 4. The headline warning: the GoPro 13 SLAM file carries GoPro 9 optics

**Known issue:** `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:9-12` holds a GoPro 9 focal length and a placeholder principal point. See [Known issues](./known-issues.md).

The shipped file says, at its 960x720 working resolution:

```yaml
Camera1.fx: 282.906909765
Camera1.fy: 282.906909765
Camera1.cx: 480.0
Camera1.cy: 360.0
```

Where those numbers come from. Rescaling the GoPro **9** JSON from 2028 px tall to 720 px tall:

```text
796.8544625226342 * 720 / 2028 = 282.90690977   (gopro9_intrinsics_2_7k.json:9)
shipped Camera1.fx             = 282.906909765
```

Ten significant figures of agreement. And `cx: 480.0`, `cy: 360.0` are exactly `Camera.width / 2` and `Camera.height / 2` — an image centre, not a measured principal point.

What the numbers should be, derived from the repo's own GoPro 13 calibration with the helper at `cv_util.py:64-89`:

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run python -c "
import json
from umi.common.cv_util import parse_fisheye_intrinsics, convert_fisheye_intrinsics_resolution
d = parse_fisheye_intrinsics(json.load(open('packages/umi/defaults/calibration/gopro13_intrinsics_2_7k.json')))
o = convert_fisheye_intrinsics_resolution(d, (960, 720))
print(o['K'])
"
```

| | shipped | from `gopro13_intrinsics_2_7k.json` | error |
|---|---|---|---|
| fx = fy | 282.9069 | 304.1523 | **-6.99 %** |
| cx | 480.0 | 484.9258 | -4.93 px (-13.9 px at native 2704 x 2028) |
| cy | 360.0 | 350.4145 | +9.59 px (+27.0 px at native 2704 x 2028) |

A focal length 7 % too small tells ORB-SLAM3 the camera sees a wider field of view than it does, so the bearing direction it computes for every tracked feature is systematically wrong, worst at the frame edges. In a monocular-inertial system this raises reprojection error, degrades IMU initialisation, and biases the metric scale of the trajectory. It is not a crash — it is silent, consistent inaccuracy.

The distortion terms `k1..k4` in that same file *are* genuine GoPro 13 measurements, but they do not match the `radial_distortion_*` values in `gopro13_intrinsics_2_7k.json:12-15` either. The file mixes a GoPro 9 focal length with GoPro 13 distortion and a GoPro 9/10 camera-to-IMU extrinsic (`:42-45`, byte-identical to `gopro10_maxlens_fisheye_setting_v1_720.yaml` and `gopro10_maxlens_fisheye_setting_v1.yaml`).

What to do, in order of increasing effort:

1. **Minimum:** patch `Camera1.fx/fy/cx/cy` in your own copy of the settings file with the values printed by the command above, so SLAM and the ArUco stage at least agree on one camera.
2. **Correct:** recalibrate your own GoPro 13 and regenerate both the JSON and the settings file. Follow [Calibrating a GoPro with OpenICC](./calibration-openicc.md).
3. **Do not** assume `IMU.T_b_c1` or the IMU noise block is GoPro 13 data. It is not.

## 5. Resolution and frame rate

- **Capture resolution.** Both live settings files assume a 4:3 frame and resize it to **960 x 720** inside the ORB-SLAM3 container. Python never resizes the video: `slam_mapping.py:155-159` hands the raw MP4 to the container binary. So `Camera.width`/`Camera.height` in the settings YAML is the *SLAM working size*, and every intrinsic except `k1..k4` must be expressed at that size. The config key `resolution: [2028, 2704]` is a different thing: it is `[height, width]` of your *source video*, used to size the mask canvas (`slam_mapping.py:264, 396`) and the image centre in tag calibration (`calibration.py:146`).
- **Lens mode is not recorded anywhere.** No Python code mentions "Wide", "Linear", "SuperView" or "MAX SuperView". The lens mode is implied only by which calibration JSON you point at and by file names. Calibrate in the exact lens mode you record in.
- **Aspect ratio is baked into the mask.** The polygons that black out the gripper and the two side mirrors (`cv_util.py:239-246`) are normalised by image *height* and centred horizontally. That makes them correct for any 4:3 frame — 2704x2028, 1920x1440, 960x720 — and **wrong for 16:9**, where they land in the wrong horizontal position.
- **120 fps is silently transcoded.** `slam_mapping.py:420` tests `abs(fps - 120.0) < 1.0` and, if true, re-encodes to 60 fps with `ffmpeg -vf fps=60 -c:v libx264 -preset fast -crf 23` into `converted_60fps_<name>.mp4` (`:424, :430-442`). 240 fps is not handled. If you record at 120 fps, keep `Camera.fps: 60` in the settings file.
- **Output size differs between the two camera families:** `output_resolution: [256, 256]` for GoPro 9 (`official_gopro9_pipeline_config.yaml:59`) versus `[224, 224]` for all GoPro 13 configs. Anything you train on the resulting dataset must match — see [Training and eval](./training-and-eval.md).

**Known issue:** `umi_pipeline_configs/gopro13_wide_angle_pipeline_config.yaml:20,32` names `gopro13_fisheye_60fps_2-7k.yaml`, which does not exist; the run dies at `slam_mapping.py:489-490`. That config is one a newcomer is likely to pick by name, so check `slam_settings_file` before running it. See [Known issues](./known-issues.md).

**Known issue:** `umi_pipeline_configs/gopro13_normal_pipeline_config.yaml:43` sets `video_resolution`, a key no Python reads; the correct key is `resolution`, and its absence trips `assert self.resolution` at `calibration.py:73`. See [Known issues](./known-issues.md).

## 6. Gripper and ArUco geometry

The pipeline uses ArUco tags to measure gripper opening width and to anchor the SLAM map to the world. These constants describe **the physical gripper**, not the camera, so a camera swap alone does not change them — a new gripper build or a differently-scaled printout does.

| Constant | file:line | Meaning | Change it when |
|---|---|---|---|
| `aruco_dict: DICT_4X4_50` | `aruco_config.yaml:2` | which ArUco dictionary the printed tags come from | you print tags from a different dictionary |
| `marker_size_map` 0.016 m (ids 0,1,6,7) | `aruco_config.yaml:5-6,11-12` | edge length of the small tags on each gripper finger | you reprint the finger tags at another size |
| `marker_size_map` 0.06 m (ids 2-5, 8-11) | `aruco_config.yaml:7-10,13-16` | edge length of the mid-size gripper-body tags | same |
| `marker_size_map` 0.16 m (ids 12-19, `default`) | `aruco_config.yaml:4,17-24` | edge length of the large table-mounted tag | you print a different world-anchor tag |
| `tag_id: 13` | `calibration.py:28` | which large tag defines the world frame | your world-anchor tag has another id |
| `tag_per_gripper = 6` | `calibration.py:232`, `dataset_planning.py:217` | tags per gripper; gripper *g* owns ids `6g` (left finger) and `6g+1` (right finger) | never, without a code edit |
| `nominal_z` | `calibration.py:33`, `dataset_planning.py:33` | expected camera-to-finger-tag depth, in metres; detections shallower than `nominal_z - 0.008` are discarded | the camera sits at a different distance from the fingers |
| `z_tolerance = 0.008` | `cv_util.py:210` | half-width of that depth gate | never, without a code edit |
| `tcp_offset = 0.205` | `dataset_planning.py:32` | camera mount to gripper tip along the tool axis | you change the finger length |
| `cam_to_center_height = 0.086` | `dataset_planning.py:48` | camera optical centre to gripper centre line | you change the camera mount |
| `cam_to_mount_offset = 0.01465` | `dataset_planning.py:49` | camera optical centre to the mount face | you change the camera body (a GoPro 13 mount is not a GoPro 9 mount) |
| `marker_size_m: 0.018` | `frame_to_pose.py:357` | printed size of the *object* tags used by the reconstruct pipeline only | you reprint object tags |

Watch `nominal_z`: GoPro 9 ships 0.072 m, every GoPro 13 config ships **0.01 m** (`gopro13_fisheye_2-7k_pipeline_config.yaml:50`). With the 0.008 m tolerance that lowers the depth gate from 0.064 m to 0.002 m — effectively no filtering. Measure it on your rig rather than inheriting either value.

## 7. The migration checklist

Run everything from the repo root; relative paths in the configs resolve against the current working directory (`slam_mapping.py:484-486`).

1. **Record a calibration set** with the same camera, lens mode, resolution and frame rate you will use for data. Stabilisation off. How: [Calibrating a GoPro with OpenICC](./calibration-openicc.md).
2. **Produce fisheye intrinsics** (fx, cx, cy, k1..k4) with OpenICC in `FISHEYE` mode. Save as `packages/umi/defaults/calibration/<camera>_intrinsics_<mode>.json` in the same JSON schema as the shipped files: copy `packages/umi/defaults/calibration/gopro9_intrinsics_2_7k.json` as a template and replace the numbers (the parser and the full key list are at `packages/umi/src/umi/common/cv_util.py:11-61`). Point stage `04_detect_aruco` at it via **`camera_intrinsics_path`**.
3. **Rescale those intrinsics** to the SLAM working size with the command in Section 4 (swap in your JSON path and target size).
4. **Copy a settings file** and edit it. Only files that begin with a `File.version: "1.0"` line (`Camera1.*`, `IMU.T_b_c1`) are safe to copy. The three files with no `File.version` line — `gopro9_maxlens_fisheye_setting.yaml`, `gopro9_wide_setting.yaml` and `gopro10_maxlens_fisheye_setting.yaml` — use the incompatible legacy schema (`Camera.fx`, `Tbc`).
   ```bash
   cd /home/hcis-s17/author_workdir/voilab
   cp packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml \
      packages/umi/defaults/orb_slam3_settings/mycam_setting.yaml
   ```
   Edit `Camera1.fx`, `Camera1.fy`, `Camera1.cx`, `Camera1.cy` (step 3 values), `Camera1.k1..k4` (unitless, never rescaled), `Camera.width`, `Camera.height`, `Camera.fps`.
5. **Measure the camera-to-IMU extrinsic** and write it into `IMU.T_b_c1` (`data:` list of 16 row-major floats). The GoPro axis permutation this needs is documented in the YAML comment at `gopro13_fisheye_ratio_4-3_2-7k.yaml:34-41` and explained in [Calibrating a GoPro with OpenICC](./calibration-openicc.md). Do not inherit the shipped matrix.
6. **Measure IMU noise** — noise densities and random walks, from a long static recording (Allan variance, a standard way to separate an IMU's white noise from its slow bias drift). Write `IMU.NoiseGyro`, `IMU.NoiseAcc`, `IMU.GyroWalk`, `IMU.AccWalk` and the real `IMU.Frequency` (a GoPro samples near 197-200 Hz, not exactly 200). How: [Calibrating a GoPro with OpenICC](./calibration-openicc.md).
7. **Copy a pipeline config** and repoint it.
   ```bash
   cd /home/hcis-s17/author_workdir/voilab
   cp umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml \
      umi_pipeline_configs/mycam_pipeline_config.yaml
   ```
   Edit `slam_settings_file` (both SLAM stages), `resolution: [h, w]` of your source video, `camera_intrinsics_path`, `nominal_z`, `dist_to_center_threshold`, `output_resolution`. Key-by-key semantics: [Pipeline config](./pipeline-config.md).
8. **Measure the gripper geometry** (Section 6) and set `nominal_z` and `tcp_offset`. `cam_to_center_height` and `cam_to_mount_offset` need a source edit at `dataset_planning.py:48-49`.
9. **Check the mask.** Run one video through and open `demos/*/slam_mask.png` in the session directory. If the mirrors or gripper are not covered, you are in Section 8 territory.
10. **Run it.**
    ```bash
    cd /home/hcis-s17/author_workdir/voilab
    uv run umi run-slam-pipeline umi_pipeline_configs/mycam_pipeline_config.yaml --session-dir /path/to/session
    ```

## 8. Hardcoded in Python: the footgun list

In addition to the **no** rows in Section 2 and the fixed tag layout in Section 6, no YAML key reaches any of these. Patching means editing source.

| Value | file:line | Why a camera swap cares |
|---|---|---|
| Finger polygon ratios `height=0.37, top_width=0.25, bottom_width=1.4` | `cv_util.py:305`, `:357` | tuned to the GoPro 9 field of view; a wider or narrower lens shifts the fingers |
| `fx == fy`, `aspect_ratio` ignored | `cv_util.py:58` | non-square pixels are silently mismodelled |
| Max depth gate disabled | `cv_util.py:211` (commented out) | no upper bound on accepted tag depth |
| Tag distance gate `< 0.3 m` or `> 4 m` | `calibration.py:138` | a wider lens sees the anchor tag from further away |
| Image centre uses `[h, w]` against an `(x, y)` point | `calibration.py:145-147` | the `dist_to_center_threshold` filter measures the wrong quantity, so the tuned value does not transfer |
| Lost-frame gates `> 10` lost, `< 60` valid | `dataset_planning.py:294, 298, 420, 425` | frame-rate dependent: 60 frames is 1 s at 60 fps, 0.5 s at 120 fps |
| Default `slam_settings_file`, an in-container path (the key itself is settable) | `slam_mapping.py:34`, `visualize_slam_gui.py:21` | omit the key and the stage raises `FileNotFoundError: SLAM settings file not found: /ORB_SLAM3/...`; the default is validated on the host (`slam_mapping.py:489-490`), so GoPro 10 tuning never silently applies |
| `"frames/second": 0.0` in the IMU handoff JSON | `imu_extraction.py:110` (`# TODO: update`) | see [GoPro telemetry](./gopro-telemetry.md) |

**Known issue:** `mask_pts_json_path` is read into a field at `slam_mapping.py:36` and never used again. `packages/umi/src/umi/asset/mask.json` and `mask_4k.json` are dead files; three GoPro 13 configs set the key anyway. All masking comes from the hardcoded polygons in `cv_util.py` (Section 2). See [Known issues](./known-issues.md).

---

**Next:** [Calibrating a GoPro with OpenICC](./calibration-openicc.md) · [ORB-SLAM3 as a Docker subprocess](./orb-slam3.md) · [Known issues](./known-issues.md)
