# Calibrating a GoPro with OpenImuCameraCalibrator

How to measure the camera and IMU numbers that ORB-SLAM3 needs, using the external OpenICC tool, and how to transfer them into a settings file in this repo.

**Read this if:** you have a camera model or lens mode that no shipped settings file covers, and SLAM tracking is drifting or failing to initialise.

**Before you start:** [Pipeline overview](./pipeline-overview.md), [ORB-SLAM3](./orb-slam3.md). You also need Docker, a GoPro running GoPro Labs firmware, a ChArUco board printed at 100 % and glued flat, calipers, and (for IMU noise) a 2-hour undisturbed recording window.

---

## 1. What you need to produce, and why

ORB-SLAM3 runs *visual-inertial* SLAM (simultaneous localisation and mapping — it estimates the camera trajectory and a 3-D map at the same time), fusing images with the camera's inertial measurement unit (IMU: 3-axis accelerometer plus 3-axis gyroscope). Fusing two sensors only works if you tell it exactly how they relate. There are four unknowns:

| Unknown | What it is | Settings-file fields it fills |
|---|---|---|
| Camera intrinsics + fisheye distortion | Focal length, principal point, and the 4 Kannala–Brandt radial terms that model a wide fisheye lens | `Camera1.fx/fy/cx/cy`, `Camera1.k1..k4` |
| IMU noise densities and random walks | How noisy the gyro/accel are, and how fast their biases drift | `IMU.NoiseGyro`, `IMU.NoiseAcc`, `IMU.GyroWalk`, `IMU.AccWalk` |
| Camera-to-IMU rigid transform | The fixed 4x4 rotation + translation between the lens and the IMU chip inside the camera body | `IMU.T_b_c1` |
| Camera-IMU time offset | Constant delay between an image timestamp and an IMU timestamp | *No field exists* — see the trap in §8 |

The settings file is a YAML consumed by the ORB-SLAM3 Docker container. See [ORB-SLAM3](./orb-slam3.md) for every field and how the container reads it; this doc only covers where the numbers come from. Example: `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:9-52`.

## 2. The tool: OpenICC

**OpenImuCameraCalibrator** (OpenICC), by Steffen Urban — `https://github.com/urbste/OpenImuCameraCalibrator`, licensed **AGPL-3.0**. It is the tool upstream UMI used (`packages/umi/README.md:254`) and the only one that solves all four unknowns for an action camera.

It is a **C++ cmake project with a Python driver. There is no pip package and no conda environment.** You either build it natively or use its Docker image. Its readme file is `Readme.md` — capital R only, `README.md` does not exist.

Native build dependencies, in order: OpenCV >= 4.5 **with contrib** (needed for ArUco marker detection), Ceres 2.1.0, `urbste/pyTheiaSfM` at commit `69c3d37`, then OpenICC itself, then `pip install -r requirements.txt` (`natsort`, `scipy`, `opencv-python==4.7.0.72`, `matplotlib`, `py_gpmf_parser==0.0.3`).

Use Docker instead — it pins all of that:

```bash
git clone https://github.com/urbste/OpenImuCameraCalibrator
cd OpenImuCameraCalibrator
docker build -t openicc .
docker run -it --rm --user $(id -u):$(id -g) -v `pwd`:/home -v /path/to/parent_of_MyDataset:/dataset openicc
```

Mount the directory that *contains* `MyDataset/` (see §5), so that `/dataset/MyDataset` exists inside the container. Without `--user` the image runs as root and every output lands root-owned on the host — `sudo chown -R "$USER"` afterwards.

The image builds OpenICC at `/OpenImuCameraCalibrator` (`Dockerfile:48-52`), so the compiled binaries live at `/OpenImuCameraCalibrator/build/applications` inside the container.

**Known issue:** the C++ board generator uses `cv::aruco::CharucoBoard::create` and `PREDEFINED_DICTIONARY_NAME`, both removed in OpenCV 4.7. The container works because Ubuntu 22.04 ships OpenCV 4.5.4. Do not upgrade the system OpenCV inside the image.

## 3. Prepare the calibration target

Print `resource/board.png` from the OpenICC clone. It is a **ChArUco board** — a chessboard whose white squares each contain a unique ArUco marker, so the detector still knows which corner is which when part of the board is out of frame. The default geometry the driver script assumes is **10 x 8 squares at 21 mm** (`python/run_gopro_calibration.py:32-40`).

1. Print at "Actual size" / 100 %. **Turn off "Fit to page"** — it silently rescales.
2. Glue the print flat onto something rigid (foam board, a wall). A wavy print becomes a wavy calibration.
3. **Measure one black square with calipers** and pass that value to `--checker_size_m`.

Why step 3 matters: nothing else in the pipeline knows how big the world is. `--checker_size_m` is the only metric input, so it sets the scale of every distance the calibration produces — including the translation part of the camera-to-IMU transform, and therefore the scale of the SLAM trajectories that feed the dataset.

## 4. GoPro capture settings

| Setting | Requirement |
|---|---|
| Firmware | GoPro Labs firmware, as UMI requires (`packages/umi/README.md:147`) |
| Stabilisation | **HyperSmooth, Horizon Lock and Horizon Levelling all OFF** |
| Lens mode | Exactly the mode you will record data in (Wide / Linear / SuperView / HyperView) |
| Resolution and frame rate | Exactly what you will record data in |
| Shutter | As fast as lighting allows, e.g. 1/480 |

Stabilisation warps each frame independently, so no single lens model or fixed camera-to-IMU transform exists — OpenICC even stamps `"stabelized": false` into its output JSON (`src/io/write_camera_calibration.cc:49`) to record the assumption. Lens mode, resolution and frame rate must match because each combination is a genuinely different optical model and sensor readout: the digital lens modes are different projections rather than crops, focal length and principal point scale with resolution, and the IMU sample rate and rolling-shutter line delay change with readout mode.

## 5. The three recordings

Three separate videos in three subdirectories:

```text
MyDataset/
  cam/       GX010001.MP4    camera intrinsics
  imu_bias/  GX010002.MP4    static gyro/accel bias
  cam_imu/   GX010003.MP4    camera-to-IMU transform and time offset
```

| Folder | Duration | Motion |
|---|---|---|
| `cam` | 20-30 s | Move **slowly** around the board, covering the whole frame including corners. Slow motion avoids blur and rolling-shutter skew. |
| `imu_bias` | 10-20 s | Camera flat on a table, untouched. Do not bump the table. |
| `cam_imu` | 30-60 s | Board filling most of the frame, moving **fast but not blurry**. Excite all six degrees of freedom: three translations and three rotations. |

Filenames must keep the GoPro `G...` prefix and an uppercase `.MP4` extension — the driver globs `*.MP4` and `G*.MP4` and exits if a folder is empty.

## 6. Run the calibration

Inside the container (`/home` is your mounted OpenICC clone, `/dataset` your footage):

```bash
cd /home
python3 python/run_gopro_calibration.py \
  --path_calib_dataset=/dataset/MyDataset \
  --path_to_build=/OpenImuCameraCalibrator/build/applications \
  --camera_model=FISHEYE \
  --checker_size_m=0.021 \
  --num_squares_x=10 --num_squares_y=8 \
  --image_downsample_factor=2
```

`--path_to_build` defaults to the author's own machine path (`python/run_gopro_calibration.py:21`), so you must always pass it. Use `--camera_model=FISHEYE` for any wide or fisheye lens mode — that is the Kannala–Brandt model ORB-SLAM3's `KannalaBrandt8` expects. Only use `PINHOLE` for GoPro Linear mode.

The script chains nine stages and prints each one's progress: board-corner extraction, camera intrinsics, GPMF telemetry extraction (GPMF = the metadata track GoPro embeds in the MP4, carrying IMU samples), telemetry normalisation, static bias, per-frame camera poses, spline knot spacing, an initial gyro-to-camera rotation, and finally a continuous-time spline optimisation for the full transform. The last stage should converge in 8-15 iterations; if it does not, the recordings are not good enough.

### IMU noise: Allan variance

Allan variance is a statistical fit over a long *static* recording that separates white noise from slow bias drift.

1. Record **more than 2 hours** completely still, lowest resolution and frame rate, lens covered. GoPro chapters this into several `GX01…`, `GX02…` files.
2. Extract one telemetry JSON per chapter. The `{"1": {"streams": {...}}}` schema stage 01 writes is the one OpenICC's reader expects (`python/telemetry_converter.py:87-113`), so the four `py_gpmf_parser` calls in [GoPro telemetry](./gopro-telemetry.md) produce a usable file — with two caveats. The reader indexes `ACCL`, `GYRO`, `CORI` and `GRAV` unconditionally (`:89, :92, :95, :102`), so check all four landed; and when `GPS5` is present it also wants a per-sample `precision` key that stage 01 never writes (`:111`), so leave `GPS5` out of the stream list. Then merge, from the root of the OpenICC clone:

   ```bash
   python3 -c "from python.telemetry_converter import TelemetryConverter; TelemetryConverter().convert_gopro_telemetry_file(['a.json', 'b.json'], '/dataset/static/merged_telemetry.json')"
   ```
3. Fit:

```bash
/OpenImuCameraCalibrator/build/applications/fit_allan_variance \
  --telemetry_json=/dataset/static/merged_telemetry.json | tee allan.log
```

**Known issue:** `fit_allan_variance` writes **no file at all** — `RunFit()` only prints to stdout (`applications/fit_allan_variance.cc:41`). Pipe it through `tee` or the numbers are gone.

**Known issue:** OpenICC's own merge helper is broken, which is why step 2 does it by hand. `python/merge_gopro_telemetry_from_folder.py:27,35` still shells out to `node javascript/extract_metadata.js`, a directory deleted when OpenICC moved to `py_gpmf_parser`. The same bug breaks `python/static_multipose_imu_calibration.py:53`, whose documented `--path_to_src` flag does not exist either; IMU intrinsics are optional here, so skip that script.

The gyro fit prints **four** numbers per axis and the accel fit two. Take the *second* "Bias Instability" and the *second* "White Noise" line for the gyro — those are the getter-based values (`src/allanvariance/fitallan_gyr.cc:56-63`). Then average x, y and z.

## 7. Outputs and the mapping table

| OpenICC output file | What you take from it |
|---|---|
| `cam/cam_calib_<VID>_fi_<factor>.json` | intrinsics: `focal_length`, `aspect_ratio`, `principal_pt_x/y`, `radial_distortion_1..4`, `final_reproj_error`, `image_width/height` |
| `imu_bias/imu_bias_<VID>.json` | static `gyro_bias` / `accl_bias` (diagnostic only) |
| `cam_imu/cam_imu_calib_result_<VID>.json` | `q_i_c`, `t_i_c`, `time_offset_imu_to_cam_s`, `final_reproj_error` |
| stdout of `fit_allan_variance` | the four IMU noise numbers |

Let `s` be the resize scale from the calibration resolution to the SLAM `Camera.width` x `Camera.height`.

| Settings field | Expression |
|---|---|
| `Camera.type` | `"KannalaBrandt8"` when `intrinsic_type` is `FISHEYE` |
| `Camera1.fx` | `focal_length * s` |
| `Camera1.fy` | `focal_length * aspect_ratio * s` |
| `Camera1.cx` / `cy` | `principal_pt_x * s` / `principal_pt_y * s` (see formula below) |
| `Camera1.k1..k4` | `radial_distortion_1..4` — unitless, **never rescaled** |
| `IMU.T_b_c1` | 4x4 built from `q_i_c` (w,x,y,z) and `t_i_c`, then permuted (§8) |
| `IMU.NoiseGyro` / `NoiseAcc` | mean White Noise over x,y,z, **divided by sqrt(f)** |
| `IMU.GyroWalk` / `AccWalk` | mean Bias Instability over x,y,z, **multiplied by sqrt(f)** |
| `IMU.Frequency` | the measured GPMF sample rate (a GoPro is near, not exactly, 200 Hz) |

For the rescale, use the same formula the repo already uses in `packages/umi/src/umi/common/cv_util.py:64-89`, which assumes no vertical crop and symmetric horizontal crop:

```python
s   = out_h / in_h
fx2 = fx * s
cx2 = (cx - in_w / 2) * s + out_w / 2
cy2 = cy * s
```

## 8. Traps that cost days

**Transform direction — no inversion needed.** OpenICC's `q_i_c` / `t_i_c` is `T_imu_cam`: it maps a point from the camera frame into the IMU body frame. ORB-SLAM3's `IMU.T_b_c1` is body-from-camera, the same direction. **Copy it straight through.** (Kalibr, a different calibration tool, publishes the opposite direction and *does* need inverting — do not carry that habit over.) OpenICC's own readme calls the quantity "T_camera_to_imu"; that is the same matrix, worded differently.

**GoPro IMU axis ordering — a permutation is needed.** OpenICC reorders the raw GPMF triplets as `[raw1, raw2, raw0]` when it loads them (`python/telemetry_converter.py:297,299`), while the ORB-SLAM3 GoPro entry point consumes them in raw order. So `T_i_c` must be re-permuted before it becomes `IMU.T_b_c1`:

```python
perm_mat = np.array([[0, 0, 1, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])
T_b_c1 = perm_mat @ T_i_c
```

This is recorded as a comment in `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:34-41`. Skipping it silently ruins IMU initialisation rather than erroring.

**Continuous vs discrete noise units.** ORB-SLAM3 multiplies the noise densities by `sqrt(IMU.Frequency)` and divides the walks by it, in `Tracking.cc:211-212` of the ORB-SLAM3 fork. The YAML must therefore hold *continuous-time* values. OpenICC's `getWhiteNoise()` already includes a `sqrt(freq)` factor, so pasting the printout in directly double-counts it and makes the value roughly 14x too large at 200 Hz.

**Two incompatible settings schemas.** A file with `File.version: "1.0"` is read by the newer parser and uses `Camera1.fx`, `Camera1.k1`, `IMU.T_b_c1`. A file without that line uses the legacy parser: `Camera.fx`, `Camera.k1`, `Tbc`. Both families exist in `packages/umi/defaults/orb_slam3_settings/`. They are not interchangeable — pasting a `Tbc` block into a `File.version: "1.0"` file aborts on a missing required parameter. Write new files in the v1.0 schema.

**Time offset has nowhere to go.** ORB-SLAM3 has no field for `time_offset_imu_to_cam_s`; this pipeline assumes it is zero. Use the number as a *quality check* only (§9).

## 9. Quality gates

Check these before pasting anything into a settings file. Reference values are from OpenICC's own GoPro 9 results.

| Metric | Good | Where |
|---|---|---|
| Camera reprojection error | < 0.5 px | `cam_calib_*.json` `final_reproj_error` |
| Camera-IMU reprojection error | < 1 px | `cam_imu_calib_result_*.json` `final_reproj_error` |
| Time offset magnitude | tens of milliseconds, < 0.1 s | `time_offset_imu_to_cam_s` |
| Rotation shape | near a signed axis permutation — the quaternion has two components near +/-0.707 and two near 0 | `q_i_c` |
| Translation magnitude | 1-3 cm | `t_i_c` |

A failure here is a **recording** problem, not a processing problem. Re-record: brighter lighting and faster shutter for blur, more aggressive six-axis motion for a weak camera-IMU solve, slower motion and fuller board coverage for bad intrinsics. Do not paste failing numbers in and hope.

## 10. State of calibration in this repo

Everything below concerns GoPro 13, the camera this fork targets. See [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) for the full migration checklist.

None of the OpenICC inputs or outputs of that run are checked into this repository — there is no `cam_calib_*.json`, `cam_imu_calib_result_*.json` or `imu_bias_*.json` anywhere in the tree. The only surviving in-tree artifact is `Camera1.k1..k4` in the settings file. On the machine this fork was developed on, the run itself survives one level above the repo, under `/home/hcis-s17/author_workdir/OpenImuCameraCalibrator/`: `gopro13_calib_2-7k_120fps/` (the three input videos), `gopro13_2-7k_result/{cam,cam_imu,imu_bias}/` (every output) and `imu_0829/` (the static recording). Copy what you need before assuming it is lost.

Done: a three-video OpenICC dataset was recorded at 2.7K/120 fps and run end to end, producing two good fisheye intrinsics solutions (0.37 and 0.38 px reprojection error) and a static IMU bias. The genuine GoPro 13 distortion terms did reach the shipped file — `Camera1.k1..k4` at `gopro13_fisheye_ratio_4-3_2-7k.yaml:14-17` match that run exactly.

**Known issue:** the shipped GoPro 13 settings file mixes three cameras — a GoPro 9 focal length, a placeholder image-centre principal point, and the GoPro 10 `IMU.T_b_c1`. Only `k1..k4` are GoPro 13. Derivation and corrected values: [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) §4. See [Known issues](./known-issues.md).

**Known issue:** the IMU noise block at `gopro13_fisheye_ratio_4-3_2-7k.yaml:48-52` is not trustworthy. `IMU.GyroWalk` (0.2513) is 7x *larger* than `IMU.NoiseGyro` (0.0339); for any MEMS IMU the random walk is orders of magnitude smaller. Dividing the two noise densities by `sqrt(197.577)` lands them in the expected range, which points at the missing continuous-time conversion from §8. No script, notebook or log in the repo produces these numbers.

**Known issue:** `packages/umi/defaults/calibration/gopro13_intrinsics_2_7k.json` — used by the ArUco stage — did not come from OpenICC. It came from `experiments/calibration.py:114`, which calls `cv2.aruco.calibrateCameraCharuco` (a Brown-Conrady pinhole model — the standard narrow-lens distortion model, whose coefficients mean something different from the fisheye Kannala-Brandt terms) and then writes its `k1,k2,p1,p2` coefficients into `radial_distortion_1..4` while labelling the file `"intrinsic_type": "FISHEYE"` (`experiments/calibration.py:126-136`). Every consumer reads it as Kannala-Brandt. Its `final_reproj_error` is 1.16 px. That script is the only calibration code in the tree, and its video path and board geometry are hardcoded (`experiments/calibration.py:14-15, 29-32, 39`) — it records how that one file was made rather than offering a workflow to repeat.

Must be re-recorded, not just re-processed:

- **Camera-to-IMU transform.** The GoPro 13 run failed its own quality gates on every count: 0.494 s time offset, 1.95 px reprojection error, a rotation 43 degrees from the nearest signed axis permutation, and a 3.7 cm translation against the 1-3 cm gate in §9.

Re-processable from footage that still exists (paths above):

- **IMU noise.** The static recording survives as twelve 1920x1080 chapters totalling 9.12 h at `imu_0829/GX*.MP4`, but no telemetry JSON was ever produced from it — consistent with the broken merge script in §6 — and the Allan variance was never run. Step 2 of §6 is the recipe.
- **Intrinsics for the ArUco stage.** A real fisheye solve of the same camera survives at `gopro13_2-7k_result/cam/cam_calib_GX010555_fi_*.json`, in the schema `cv_util.parse_fisheye_intrinsics` already reads. It was shot at 2.7K/120 fps, so §4 decides whether it transfers to your recording mode, and the two solutions differ by 6 % in `focal_length` — only the `fi_0.0` one matches the shipped `k1..k4`.

---

**Next:** [ORB-SLAM3](./orb-slam3.md) · [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) · [Known issues](./known-issues.md)
