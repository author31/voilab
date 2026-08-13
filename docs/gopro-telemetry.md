# GoPro telemetry and IMU extraction

Pipeline stage `01_extract_gopro_imu` reads the sensor track hidden inside a GoPro MP4 and writes it next to the video as `imu_data.json`, which is the only IMU input ORB-SLAM3 ever sees.

**Read this if:** you need to know where IMU samples come from, what units they are in, or why a session produced an empty `imu_data.json`.

**Before you start:** [Pipeline overview](./pipeline-overview.md) for how stages are wired, and [Data formats](./data-formats.md) for the session directory layout.

---

## Why this stage exists

A GoPro writes more than video and audio. Every MP4 also carries a **GPMF** track (GoPro Metadata Format — a binary key-value stream keyed by four-character codes) holding accelerometer, gyroscope, orientation and GPS samples recorded while the camera was rolling.

The pipeline runs **visual-inertial SLAM** (Simultaneous Localisation and Mapping — recovering camera trajectory from video, here helped by inertial measurements). ORB-SLAM3 fuses image features with accelerometer and gyroscope samples to fix the scale and survive fast motion, so it needs those samples on the same clock as the video frames. GPMF timestamps are already relative to the start of the video track, which is exactly the alignment SLAM wants — the pipeline does no resampling and applies no time offset of its own.

```text
raw_video.mp4 ── GPMF track ──> py_gpmf_parser (C extension)
                                     │ numpy arrays + timestamps
                                     v
                            imu_data.json  (stage 01)
                                     v
                 gopro_slam inside chicheng/orb_slam3:latest  (stages 02/03)
```

## Stream reference

The eleven streams requested are listed in `packages/umi/src/umi/services/imu_extraction.py:18-33` and repeated verbatim in the `stream_types` key of every shipped config (for example `umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml:11`). Rates below were measured on a 28 s HERO13 clip and a 55 s HERO9 clip; where a cell gives two figures they are labelled by camera.

| FourCC | Meaning | Units | Typical rate | HERO9 | HERO13 |
|---|---|---|---|---|---|
| `ACCL` | 3-axis accelerometer | m/s² | HERO9 199.9 Hz / HERO13 197.5 Hz | yes | yes |
| `GYRO` | 3-axis gyroscope | rad/s | HERO9 199.9 Hz / HERO13 197.5 Hz | yes | yes |
| `GRAV` | gravity direction in camera frame | unit vector, no units | video frame rate (59.94 Hz here) | yes | yes |
| `CORI` | camera orientation since capture start | quaternion `w,x,y,z` | video frame rate (59.94 Hz here) | yes | yes |
| `IORI` | image orientation relative to camera body | quaternion `w,x,y,z` | video frame rate (59.94 Hz here) | yes | yes |
| `TMPC` | device temperature | °C per GoPro spec | ~2 Hz effective | yes | yes |
| `GPS5` | lat, lon, altitude, 2D speed, 3D speed | deg, deg, m, m/s, m/s | 18.2 Hz | yes | **no** |
| `GPSF` | GPS fix quality (0 none, 2 = 2D, 3 = 3D) | enum | 18 Hz | yes | **no** |
| `GPSP` | GPS dilution of precision, x100 | unitless | 18 Hz | yes | **no** |
| `GPSU` | GPS UTC timestamp | date string | — | **never** | **never** |
| `MAGN` | magnetometer | µT | — | **never** | **never** |

`GRAV`, `CORI` and `IORI` are frame-locked — one sample per video frame. On GoPro 13 clips they measured 119.88 Hz at 120 fps and 29.97 Hz at 30 fps, while `ACCL` and `GYRO` stayed near 197.5 Hz in both. If you change frame rate, only those three rates move.

Notes on the four streams that never arrive:

- **GPS on a HERO13.** GoPro retired `GPS5` after HERO11 and newer bodies emit `GPS9` instead. No file in this repo mentions `GPS9` (`grep -rn GPS9` returns nothing outside `dependencies/`), so a HERO13 session yields no GPS at all. Nothing downstream reads GPS, so this is latent rather than breaking — but do not expect location data from a HERO13. See [Known issues](./known-issues.md).
- **`GPSU`** is a UTC date string. The extractor asks the parser for `DOUBLE` samples, and the string cannot be converted, so it comes back empty even on a HERO9 that has a GPS lock.
- **`MAGN`** only exists on Fusion and MAX bodies. It is dead weight in every config's `stream_types`.
- **Axis convention.** GoPro's HERO6+ ordering for `ACCL` and `GYRO` is `Y, -X, Z`. Nothing in `imu_extraction.py` remaps axes or converts units — the permutation is baked into the camera-to-IMU extrinsic in the SLAM settings file instead, documented in the comment at `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:34-41`. See [ORB-SLAM3](./orb-slam3.md).

**Known issue:** `TMPC` values measured 0.036–0.096 on a HERO13 and 0.124 on a HERO9 — not plausible degrees Celsius. Something in the scaling path is wrong. Nothing reads it, so treat `TMPC` as unusable rather than as temperature. See [Known issues](./known-issues.md).

## The library: py-gpmf-parser

Declared at `packages/umi/pyproject.toml:85` as a git dependency:

```toml
"py-gpmf-parser @ git+https://github.com/urbste/py-gpmf-parser.git",
```

`uv.lock:3835-3840` pins it to version `0.1.1` at commit `8af151545c188f55d3d46d78eefe4e5881a9057c`, and its only runtime dependency is numpy.

| Aspect | Detail |
|---|---|
| What it is | pybind11 bindings around GoPro's own C reference parser, `github.com/gopro/gpmf-parser` (pulled in as a git submodule) |
| Installed artifacts | `.venv/lib/python3.10/site-packages/py_gpmf_parser/` (Python wrapper) and `gpmf_parser.cpython-310-x86_64-linux-gnu.so` (compiled extension) |
| Build requirements | a C/C++17 toolchain and pybind11; both handled by the PEP 517 build |
| CMake | **not** required — the package's `setup.py` compiles the C sources through a plain setuptools `Extension` and never invokes the submodule's `CMakeLists.txt` |

There is no wheel on PyPI, so `uv` clones the repo (recursively, initialising the submodule) and compiles it. That happens automatically:

```bash
cd /home/hcis-s17/author_workdir/voilab
make install     # = install-uv, then `uv sync`
```

`make install-cmake` exists in the `Makefile:46-49` but nothing wires it to this package. See [uv workspace](./uv-workspace.md) for the full target list.

## The four calls the pipeline makes

The whole stage uses four methods of `GoProTelemetryExtractor`, and nothing else:

| Call | Site |
|---|---|
| `GoProTelemetryExtractor(str(video_path))` | `packages/umi/src/umi/services/imu_extraction.py:102` |
| `.open_source()` | `packages/umi/src/umi/services/imu_extraction.py:104` |
| `.extract_data("<FOURCC>")` | `packages/umi/src/umi/services/imu_extraction.py:114` |
| `.close_source()` | `packages/umi/src/umi/services/imu_extraction.py:147` |

The same four calls, standalone, so you can inspect a video of your own:

```python
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

extractor = GoProTelemetryExtractor("/path/to/raw_video.mp4")
extractor.open_source()
for stream in ["ACCL", "GYRO", "GRAV", "CORI", "IORI", "TMPC"]:
    data, timestamps = extractor.extract_data(stream)
    print(stream, data.shape, timestamps[:1])
extractor.close_source()
```

Return types of `extract_data`:

- `data` — `np.ndarray` of shape `(n_samples, n_elements)`, dtype float64, already scaled to physical units by the GPMF parser. An absent stream returns shape `(0,)`, which is what the `len(payload[0]) > 0` test at `imu_extraction.py:115` keys on.
- `timestamps` — `np.ndarray` of shape `(n_samples,)` in **seconds**. Each GPMF payload carries a start and end time; samples inside a payload are spread linearly between them, then the whole array is shifted by a per-stream `start` offset derived from the camera's shutter stream. The origin is the start of the video track.

That offset can be negative: on the HERO9 clip `ACCL` started at `-0.002237 s`, so the first sample lands at `cts = -2.237`. Do not assume the first timestamp is zero or positive.

`get_image_timestamps_s()` and `extract_data_to_json()` exist in the library but are unused here — which is why the frame rate written to the JSON is a placeholder (below).

## The output file: imu_data.json

Written inside each demo directory as `imu_data.json` (`imu_extraction.py:100`). Passing an `output_dir` to `execute()` instead writes `<demo_name>_imu.json` into that directory (`imu_extraction.py:98`), but the pipeline never does — `PipelineExecutor` calls `execute()` with no arguments.

```json
{
  "1": {
    "streams": {
      "ACCL": {
        "samples": [
          { "value": [8.189448441247002, -0.3333333333333333, 5.175059952038369], "cts": 1.201 }
        ]
      },
      "GYRO": { "samples": [] },
      "GRAV": { "samples": [] }
    }
  },
  "frames/second": 0.0
}
```

| Key | Meaning |
|---|---|
| `"1"` | literal string key, a device id inherited from the Node `gopro-telemetry` output format |
| `streams.<FOURCC>` | present **only** if the stream produced at least one sample; empty streams are omitted entirely |
| `value` | always a list, even for one-element streams — `TMPC` values look like `[0.0803]` |
| `cts` | timestamp in **milliseconds** — seconds x `SECS_TO_MS = 1e3` (`imu_extraction.py:8`, applied at `:118`). May be negative. |
| `frames/second` | hardcoded `0.0` with a `# TODO: update` comment (`imu_extraction.py:110`) |

The odd shape is deliberate. The consumer is the unmodified C++ binary `gopro_slam` inside the `chicheng/orb_slam3:latest` image, which was written against the Node tool's schema and reads `j["1"]["streams"]["ACCL"]["samples"]`, `["GYRO"]`, `["CORI"]` and multiplies `cts` by `1e-3`. Keeping the schema byte-compatible meant the container never had to change. It is handed over at `packages/umi/src/umi/services/slam_mapping.py:160` as `--input_imu_json`, and `slam_mapping.py:75` asserts both `raw_video.mp4` and `imu_data.json` exist before launching.

**Known issue:** `frames/second` is never filled in. It is harmless today only because `gopro_slam` takes the frame rate from OpenCV rather than from the JSON, and the error path writes a different lie (`30.0`). See [Known issues](./known-issues.md).

## Why this replaced upstream's approach

| | upstream UMI | this fork |
|---|---|---|
| Mechanism | `docker run chicheng/openicc:latest node .../extract_metadata_single.js` | in-process C extension call |
| Prerequisites | Docker daemon plus a multi-GB image pull | one `uv sync` |
| Data path | subprocess to container to Node to JSON on disk | direct call returning numpy arrays |
| Speed | dominated by container startup | **0.19 s** measured for a 302 MB / 28 s HERO13 clip, all eleven streams, 3.4 MB of JSON |
| Stream selection | fixed inside the JS script | configurable via the `stream_types` YAML key |
| Failure mode | non-zero exit code, visible | silent — see below |

See [Versus upstream UMI](./vs-upstream-umi.md) for the rest of the divergences.

## What exiftool is still for

exiftool is still a hard prerequisite (`make install-exiftool`, minimum version 12.5, `Makefile:30-44`; the Python wrapper `PyExifTool==0.5` at `packages/umi/pyproject.toml:76`), but it has nothing to do with IMU.

- `packages/umi/src/umi/services/video_organization.py:67,87` — reads `QuickTime:CameraSerialNumber` to build demo directory names.
- `packages/umi/src/umi/services/dataset_planning.py:63,86` — same key, to map a gripper calibration to the camera that recorded it.
- `packages/umi/src/umi/common/exiftool_util.py:4` — a `get_videos_metadata` helper that nothing imports. Dead code.

Video creation time and timecode come from PyAV, not exiftool (`packages/umi/src/umi/common/timecode_util.py`).

## Troubleshooting

Run the stage as part of the pipeline, from the repository root:

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run umi run-slam-pipeline umi_pipeline_configs/gopro13_fisheye_2-7k_pipeline_config.yaml --session-dir /path/to/session
```

There is no CLI subcommand for stage 01 alone — `umi` exposes only `run-slam-pipeline` and `visualize-slam-gui`. To drive the service directly:

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run python -c "
from umi.services.imu_extraction import IMUExtractionService
print(IMUExtractionService({'session_dir': '/path/to/session'}).execute())"
```

Inspect a result by hand. Demo directories are named `demo_<camera_serial>_<timestamp>` (`video_organization.py:95`), so glob for one:

```bash
uv run python -c "
import glob, json
d = json.load(open(glob.glob('/path/to/session/demos/demo_*/imu_data.json')[0]))
print(sorted(d['1']['streams']), d['frames/second'])
for k, v in d['1']['streams'].items():
    print(k, len(v['samples']), v['samples'][0])"
```

A healthy HERO13 run prints six streams (`ACCL GYRO GRAV CORI IORI TMPC`) with thousands of samples each. Two failure shapes to recognise:

**Known issue: silent empty output.** If the file is not a parseable GoPro MP4, the underlying `OpenMP4Source` returns a falsy-but-valid handle instead of raising. Every stream then comes back empty, the `len(payload[0]) > 0` guard at `imu_extraction.py:115` skips all of them, and the service writes `{"1": {"streams": {}}, "frames/second": 0.0}` while reporting `{"extracted": [...], "failed": []}`. Reproduced end to end. Stage 02's assertion at `slam_mapping.py:75` passes because the file does exist, and the failure only surfaces inside the SLAM container much later. **Detect it by checking that `d['1']['streams']` is non-empty.** See [Known issues](./known-issues.md).

**Known issue: fabricated fallback samples.** Any exception during extraction is caught at `packages/umi/src/umi/services/imu_extraction.py:127-144`, which writes an invented single-sample file — `ACCL` = `[0.0, 0.0, 9.8]` at `cts: 0`, `frames/second: 30.0` — and still returns it as a successful extraction. The comment says "For testing purposes"; it is on the production path. **Detect it by looking for exactly one `ACCL` sample, a `cts` of `0`, and `frames/second == 30.0`.** See [Known issues](./known-issues.md).

To recover from either: fix or replace the MP4 and re-run stage 01 — it always overwrites `imu_data.json` and has no skip logic (`imu_extraction.py:100, 122`). If a SLAM stage already ran on the bad file, also delete `demos/mapping/map_atlas.osa` and the affected `camera_trajectory.csv`, or SLAM will skip and keep the bad result (`slam_mapping.py:79, 242-244`).

Two more things that will not do what you expect: `num_workers` is read at `imu_extraction.py:17` and never used (extraction is a serial loop, `imu_extraction.py:65`), and the unit tests write the literal text `mock video data` into a `.MP4` and only assert that a JSON file appeared, so they cover none of the parsing above:

```bash
cd /home/hcis-s17/author_workdir/voilab
uv run --extra dev python -m pytest packages/umi/tests/services/test_imu_extraction.py -v
```

`pytest` is only in the root project's `dev` extra (`pyproject.toml:34-36`), which plain `make install` does not install; `make install-dev` does.

---

**Next:** [ORB-SLAM3](./orb-slam3.md) · [GoPro 9 to GoPro 13](./gopro9-to-gopro13.md) · [Known issues](./known-issues.md)
