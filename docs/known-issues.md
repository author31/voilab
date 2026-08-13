# Known issues

One ranked register of everything in this repository that is currently broken, dead, or surprising.

**Read this if:** a command failed, an output looks wrong, or you want to know what not to trust before you start.

**Before you start:** Nothing — but [Getting started](./getting-started.md) explains the commands most of these issues break.

---

Entries are ranked by how likely a new user is to hit them, not by how hard they are to fix. Every entry carries a `file:line` or a config path so you can check it yourself. Severity means: **Blocker** stops a documented command dead; **High** produces wrong results or wastes hours; **Medium** is a real defect you can work around; **Low** is cosmetic or narrow. All citations were verified against commit `ef712e0` on 2026-08-12; line numbers drift after refactors, so search the quoted code rather than trusting the number.

Terms used throughout: **SLAM** (Simultaneous Localization and Mapping) is the step that recovers the camera's 3D path from video; **ArUco** tags are the printed square black-and-white fiducial markers glued to the gripper and to task objects; a **replay buffer** here is the final training dataset the pipeline writes — one zarr archive, `dataset.zarr.zip`, holding every episode's images and poses.

## The register

| # | Severity | Area | Symptom you will see | Root cause | Workaround |
|---|---|---|---|---|---|
| 1 | Blocker | Pipeline config | Stage `02_create_map` aborts with `FileNotFoundError: SLAM settings file not found` | `umi_pipeline_configs/gopro13_wide_angle_pipeline_config.yaml:20,32` name `gopro13_fisheye_60fps_2-7k.yaml`, which is not on disk | Run `gopro13_fisheye_2-7k_pipeline_config.yaml` instead |
| 2 | Blocker | Isaac Sim | `voilab launch-simulator` dies with `ModuleNotFoundError: No module named 'registry'` | `scripts/generate_data.py:15` imports `scripts/registry/`, deleted in this working tree (a fresh clone still has it, but the v0 scripts are unmaintained) | `scripts/v1/sdg.py` is the replacement, but restore `git checkout -- scripts/object_loader.py` first — `scripts/v1/engine.py:48` imports it — and `scripts/v1/` is untracked in git |
| 3 | Blocker | Pipeline config | Stage `05_run_calibrations` aborts with `AssertionError: Missing resolution in configuration` | `umi_pipeline_configs/gopro13_normal_pipeline_config.yaml:43` sets `video_resolution`; the code reads `resolution` (`packages/umi/src/umi/services/calibration.py:34`, asserted `:73`) | Rename the key, or use a different config |
| 4 | Blocker | Pipeline config | Stage `00_process_video` finds no videos, or writes into a stranger's folder | `official_gopro9_pipeline_config.yaml:4` and `gopro13_normal_pipeline_config.yaml:4` hardcode one developer's absolute paths | Always pass `--session-dir /abs/path` |
| 5 | Blocker | Training | `train.py` exits with a Hydra (the YAML config framework `train.py` uses) "config directory not found" error | `packages/diffusion_policy/train.py:20-23` defaults to `packages/diffusion_policy/diffusion_policy/config`, which does not exist | Always pass `--config-path=src/diffusion_policy/config` |
| 6 | Blocker | Real robot | `import umi.real_world.umi_env` raises `ModuleNotFoundError: No module named 'zerorpc'` | `packages/umi/src/umi/real_world/franka_interpolation_controller.py:14` imports `zerorpc`, declared in no pyproject and absent from `uv.lock` | Treat `packages/umi/src/umi/real_world/` as out of scope |
| 7 | Blocker | Replay buffer | Stage `08_generate_replay_buffer` dies with `ValueError: missing object_codec for object array`, after every video has been decoded and recompressed | With `demos/mapping/object_poses.json` absent, `replay_buffer.py:89-90` builds the empty name arrays as `dtype=object` instead of the `'U32'`/`'U64'` used at `:84-85`; `:239` then writes them into zarr | Only the `_reconstruct` config writes that file (stage `07_frame_to_pose`); otherwise create it containing `[]` before stage 08 |
| 8 | High | IMU | SLAM tracks badly or not at all, and no stage reported an error | `packages/umi/src/umi/services/imu_extraction.py:127-144` catches every exception and writes a fabricated one-sample `imu_data.json` | Check each `demos/*/imu_data.json` is megabytes, not bytes |
| 9 | High | IMU | `imu_data.json` contains `{"1": {"streams": {}}}` and the stage still reports success | An unreadable MP4 makes the GPMF reader (GPMF is the metadata track GoPro cameras write IMU samples into) return empty arrays instead of raising; `imu_extraction.py:65-73` treats that as success | Same as #8 |
| 10 | High | SLAM GUI | `umi visualize-slam-gui` always exits `ValueError: slam_settings_file is not configured` | `packages/umi/src/umi/cli.py:33-39` always stores the key, so the default at `visualize_slam_gui.py:19-22` never applies | Always pass `--settings-file` |
| 11 | High | Tests | `pytest packages/umi/tests --ignore=packages/umi/tests/services/test_aruco_detection.py` reports 41 failures out of 141 run (155 collect) | Large parts of the suite call service methods that do not exist, e.g. `DatasetPlanningService.validate_plan` | Expected on a clean checkout — see below; the `--ignore` is needed because of #12 |
| 12 | High | Tests | `pytest` appears to hang forever partway through | `packages/umi/tests/services/test_aruco_detection.py:147-149` mocks `ThreadPoolExecutor`, then real `concurrent.futures.wait` blocks on the mock (`aruco_detection.py:61`) | Add `--ignore=packages/umi/tests/services/test_aruco_detection.py` |
| 13 | High | Calibration | GoPro 13 footage is fed to ORB-SLAM3 with GoPro 9 optics | `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_ratio_4-3_2-7k.yaml:9-12` is `fx=fy=282.906909765, cx=480, cy=360` — byte-identical to the GoPro 10 file | Re-derive; see [GoPro 13 migration](./gopro9-to-gopro13.md) |
| 14 | Medium | CLI | `--task kitchen` changes nothing and prints no warning | `packages/umi/src/umi/pipeline_executor.py:115-121` rewrites only the first stage that *already* declares `task`, then breaks | Only the reconstruct config declares `task` |
| 15 | Medium | Replay buffer | `output_fov` never takes effect; setting it raises `AttributeError` | Assigned twice, `replay_buffer.py:34` then `:40`; `:101` calls `.open()` on a plain string | Leave `output_fov` and `output_fov_intrinsic_path` null |
| 16 | Medium | ArUco | Stage 04 reports `videos_processed: 0` after a fully successful run | Labels are swapped at `packages/umi/src/umi/services/aruco_detection.py:76,78` | Count `demos/*/tag_detection.pkl` files instead |
| 17 | Medium | Calibration | `dist_to_center_threshold` rejects the wrong detections | `calibration.py:145` builds the image centre from `resolution` (height, width) and compares it against a pixel (x, y) | Tune the value empirically per camera |
| 18 | Medium | Config propagation | Stage 06 silently uses stage 05's `nominal_z` | Config propagates forward flat (`pipeline_executor.py:299,309`); `dataset_planning.py:33`'s own default of `0.0` never applies | Re-declare `nominal_z` in stage 06 |
| 19 | Medium | Docker | Stopped `chicheng/orb_slam3` containers accumulate after every mapping run | The `create_map` command is built without `--rm` (`packages/umi/src/umi/services/slam_mapping.py:123-169`) | `docker container prune` |
| 20 | Medium | Docker | Stage 02 sits forever with the SLAM container still running | stdout is drained to completion before stderr is read (`slam_mapping.py:185-192`); a full stderr pipe blocks the child | Kill the container and rerun |
| 21 | Medium | Docker | SLAM outputs in your session dir are owned by `root` | The container runs as root unless `enable_gui: true` is set | `sudo chown -R "$USER" <session_dir>` |
| 22 | Medium | Config keys | Editing `mask_pts_json_path` has no effect at all | Read into a field at `slam_mapping.py:36` and never referenced again | Masks come from hardcoded polygons in `packages/umi/src/umi/common/cv_util.py` |
| 23 | Low | Replay buffer | `no_mirror: true` *enables* the mirror mask | The value is passed straight through as `mirror=` at `replay_buffer.py:322` | Leave it `false` |
| 24 | Low | Logging | The dropped-demo count never appears in the log | loguru treats the second argument as a format value: `logger.info("n_dropped_demos", n_dropped_demos)` (`dataset_planning.py:582`) | Count episodes in `dataset_plan.pkl` |
| 25 | Low | CLI | The simulator receives the literal string `None` as its session dir | f-string interpolation at `src/voilab/cli.py:62` | Always pass `--session_dir` |
| 26 | Low | Make | `make make-init-submodule` succeeds and does nothing | `Makefile:73` accidentally declares that name as a real target | Use `make init-submodule` |
| 27 | Low | Training package | Importing `diffusion_policy.env_runner.pusht_image_runner` (or eight sibling modules) raises `ModuleNotFoundError: No module named 'diffusion_policy.env'` | This fork removed upstream's vendored gym environments but kept the runners that import them (`env_runner/pusht_image_runner.py:10`) | Only `ROS2Runner` and `RealPushTImageRunner` are reachable; treat the rest as dead |
| 28 | Low | ROS 2 | In the container, importing `diffusion_policy.environments.ros2.ros2_env` or `diffusion_policy.examples.ros2_integration_example` raises `ImportError: cannot import name 'ROS2EnvironmentFactory'` (outside it, `rclpy` fails first) | Both import it (`environments/ros2/ros2_env.py:16`, `examples/ros2_integration_example.py:20`) from a module that defines only `ROS2Environment` | Construct `ROS2Environment` directly |
| 29 | Low | ROS 2 | `env.step(action)` returns a normal-looking observation with `info['error']` set and the arm never moves | `_publish_joint_states` calls `.tolist()` on the Python list built at `environments/ros2_environment.py:209` (`:281`), inside the blanket `except` at `:231-234` | Check `info` on every step; see [the ROS 2 stack](#the-ros-2-stack-in-the-training-package) |

## Broken configuration files

Five pipeline YAMLs ship in `umi_pipeline_configs/`. Two of them cannot run. Full key-by-key reference: [Pipeline config](./pipeline-config.md).

**`gopro13_wide_angle_pipeline_config.yaml` — dead on arrival.** Lines 20 and 32 both point at `packages/umi/defaults/orb_slam3_settings/gopro13_fisheye_60fps_2-7k.yaml`. That directory holds seven files and none has that name. `_resolve_settings_file_path` raises `FileNotFoundError` before SLAM starts.

**`gopro13_normal_pipeline_config.yaml` — broken twice.** Stages 02 and 03 declare no `slam_settings_file`, so the default at `slam_mapping.py:34` is used — an absolute path *inside* the ORB-SLAM3 container that fails host-side validation. Separately, line 43 sets `video_resolution`, a key no Python in this repo reads.

**Two configs bake in one developer's home directory.** `official_gopro9_pipeline_config.yaml:4` points at `example_demo_session/`, which is not in the repo; `gopro13_normal_pipeline_config.yaml:4` points at `0911_basement_test_normal_lens`. The `--session-dir` flag overrides both.

**The healthy pick** is `gopro13_fisheye_2-7k_pipeline_config.yaml` (or its `_reconstruct` variant, which adds the `07_frame_to_pose` stage). Their settings file exists and their `session_dir` is the relative `"video"`. Only the `_reconstruct` variant reaches the end, though: stage `07_frame_to_pose` writes the `object_poses.json` whose absence crashes stage 08 (#7).

## Fabricated or silently-wrong data

**Fake IMU on any failure.** `imu_extraction.py:127-144` swallows every exception and writes `{"ACCL": [{"value": [0.0, 0.0, 9.8], "cts": 0}]}` with `"frames/second": 30.0`, then reports the demo as extracted. The comment says "For testing purposes"; it runs in the pipeline path. ORB-SLAM3 then consumes garbage without complaint. Details: [GoPro telemetry](./gopro-telemetry.md).

**Empty IMU without any failure.** For a file the parser cannot open, the reader returns empty arrays rather than raising, so the service writes `{"1": {"streams": {}}, "frames/second": 0.0}` and reports success. The next stage asserts only that the file exists.

**`frames/second` is hardcoded.** `imu_extraction.py:110` writes `0.0` with a `# TODO: update` comment.

**Inverted ArUco accounting.** `aruco_detection.py:70-87` computes its summary *after* detection by testing whether `tag_detection.pkl` exists, then labels every file that exists as *skipped*. A clean run reports `videos_processed: 0`.

**Test-only fabrication in production code.** `SLAMMappingService.create_map()` (`slam_mapping.py:497`) contains a branch that writes fake trajectory outputs when `pull_docker` is false. `BatchSLAMService._run_batch_slam` (`packages/umi/src/umi/services/batch_slam.py:82-95`) only touches an empty `optimized_trajectory.txt` and writes `{"keyframes": []}`; no config references it.

## Camera calibration correctness

**Intrinsics** are the lens numbers (focal length `fx/fy`, image centre or *principal point* `cx/cy`, distortion `k1..k4`); the **extrinsic** is the fixed rotation+translation between the camera and the IMU inside the same GoPro body.

**The "GoPro 13" SLAM settings are not GoPro 13.** `gopro13_fisheye_ratio_4-3_2-7k.yaml:9-12` carries `fx=fy=282.906909765`, `cx=480.0`, `cy=360.0`. Those are the GoPro 9 focal length scaled to 720p and an image centre that is exactly width/2, height/2 — i.e. assumed, not measured. The camera-to-IMU extrinsic at `:42-45` appears verbatim in the GoPro 10 settings files. Only `k1..k4` are demonstrably GoPro 13 numbers. The IMU noise block at `:48-52` is not copied from the GoPro 9/10 files, but nothing in the repository reproduces it either, and `IMU.GyroWalk` (0.2513) is 7x *larger* than `IMU.NoiseGyro` (0.0339) — the reverse of what a MEMS IMU gives. See [Calibration](./calibration-openicc.md).

**Two intrinsics files should not be used.** `packages/umi/defaults/calibration/gopro13_intrinsics_4k.json` shares its `final_reproj_error` and `nr_calib_images` with the GoPro 9 file and swaps width and height (3000×4000). `gopro9_intrinsics_normal_lens.json` puts its principal point outside the declared frame. Neither is referenced by any pipeline config.

**The 2.7K GoPro 13 intrinsics claim a lens model they were not fitted with.** `experiments/calibration.py` fits a pinhole (Brown–Conrady) model and writes its coefficients into the `radial_distortion_1..4` fields, which every consumer feeds to `cv2.fisheye.*`. Use OpenImuCameraCalibrator instead — see [Calibration](./calibration-openicc.md).

**The centre-distance filter measures the wrong thing.** `calibration.py:145-147` divides `np.array(self.resolution)/2` — a (height, width) pair — against a pixel (x, y) coordinate, then normalises by half the *height*. That is why `dist_to_center_threshold` is `0.6` in some configs and `0.1` in others.

**`nominal_z` changed by 7x between camera generations.** GoPro 9 configs use `0.072`, all GoPro 13 configs use `0.01` (`gopro13_fisheye_2-7k_pipeline_config.yaml:50`). With the hardcoded `z_tolerance = 0.008` in `cv_util.py:210`, `0.01` effectively disables depth filtering of finger-tag detections.

## Docker and SLAM execution

The SLAM stages shell out to `docker run` with the image `chicheng/orb_slam3:latest`, built outside this repository. See [ORB-SLAM3](./orb-slam3.md).

- **No `--rm` in mapping mode.** The command built at `slam_mapping.py:123-169` leaves a stopped container behind each run; the batch command at `:291` does pass `--rm`.
- **No timeout in mapping mode.** `timeout_multiple` (default 16) is only applied to batch localization at `slam_mapping.py:263`. A stuck mapping run blocks the pipeline indefinitely.
- **Pipe deadlock.** `slam_mapping.py:185-192` reads all of stdout before touching stderr. If ORB-SLAM3 fills the stderr pipe buffer first, the child blocks and the parent never returns.
- **Root-owned outputs.** The container runs as root unless `enable_gui: true`, so `camera_trajectory.csv`, `map_atlas.osa` and the logs land in your session directory owned by root.
- **`docker compose up isaac-sim` exits immediately.** The service declares no `command:` and `Dockerfile:95` sets an `ENTRYPOINT`, so nothing runs. Use `docker compose run --rm isaac-sim bash`. See [Simulation and Docker](./simulation-and-docker.md).
- **`.dockerignore` excludes almost nothing.** Untracked directories such as `experiment_wpnoise0629/` and `dependencies/` are shipped to the daemon on every build.

## Dead code and dead config keys

Config keys that are read once and never used again:

| Key | Read at | Set in |
|---|---|---|
| `mask_pts_json_path` | `slam_mapping.py:36` | three GoPro 13 configs, line 22 or 23 |
| `video_resolution` | nowhere | `gopro13_normal_pipeline_config.yaml:43` |
| `slam_tag_calibration_timeout`, `gripper_range_timeout` | `calibration.py:23-26` | nowhere |
| `num_workers` (stage 01) | `imu_extraction.py:17` | extraction is a serial loop |
| `--force` (`visualize-slam-gui`) | stored in the config dict | never read by the service |

Whole modules with no caller in the pipeline: `services/logging.py` (so no `pipeline.log` is ever written), `services/progress.py` (so no `progress.json`), `services/config.py`, `services/batch_slam.py`, and `packages/umi/src/umi/traj_eval/` (a vendored, Python-2-era copy of a third-party trajectory-evaluation tool). Only the tests reach `config.py` and `batch_slam.py`, so `tests/services/test_config.py` and `tests/services/test_batch_slam.py` (20 passing tests) would have to go with them. `packages/umi/src/umi/pipeline/aruco_detection.py` is a 0-byte file.

`packages/umi/src/umi/real_world/` (4,500 lines of robot and camera drivers) has no entry point in this fork and cannot be imported (#6). `packages/umi/src/umi/shared_memory/` exists only to serve it.

In `packages/diffusion_policy/src/diffusion_policy/`, nine modules under `env_runner/` and `dataset/` plus `scripts/generate_bet_blockpush.py:19-21` import the removed `diffusion_policy.env` package at module level, so none of them can be imported at all (#27) — `pusht_image_runner.py:10`, `pusht_keypoints_runner.py:10`, `blockpush_lowdim_runner.py:10`, `robomimic_runner.py:21`, `robomimic_image_runner.py:21`, `robomimic_lowdim_runner.py:21`, `franka_assembly_image_runner.py:11`, `pick_and_place_cup_image_runner.py:10` and `dataset/kitchen_mjl_lowdim_dataset.py:12`. `env_runner/kitchen_lowdim_runner.py` imports it inside its methods (`:56-57`) and fails at call time instead. Full tour: [Diffusion policy layers](./diffusion_policy_layers.md).

Both halves of the Isaac Sim side are broken in this working tree. The tracked `scripts/*.py` import `scripts/registry/`, `scripts/utils.py` and four other tracked v0 modules that are deleted but not committed as deleted. The replacement `scripts/v1/` is **untracked in git** *and* depends on one of the same deletions: `scripts/v1/engine.py:48` imports `object_loader`, and `sdg.py:87`, `rollout.py:76` and `preview.py:61` all pull `engine` in. `git checkout -- scripts/object_loader.py` restores the v1 half; a fresh clone gets the unmaintained v0 half and none of the v1 half. [Repository map](./repository-map.md) lists what is live.

## The ROS 2 stack in the training package

Three layers under `packages/diffusion_policy/src/diffusion_policy/` — `infrastructure/ros2_infrastructure.py`, `environments/ros2_environment.py`, `env_runner/ros2_runner.py` — are the intended route to policy rollout on a real arm. No pipeline stage, simulation script or shipped pipeline config drives them today, so these defects only reach you if you pick that work up. Design and signatures: [ROS 2 integration design](./ros2_integration_design.md).

- **`ROS2EnvironmentFactory` is imported but never defined** (#28). `environments/ros2/ros2_env.py:16` and `examples/ros2_integration_example.py:20` (used at `:70`) both import it; `environments/ros2_environment.py` defines only `ROS2Environment`. The legacy shim and the example script are broken outright, not merely deprecated.
- **`step()` publishes nothing** (#29). `_publish_joint_states` calls `.tolist()` (`ros2_environment.py:281`) on the plain Python list built at `:209`, raising `AttributeError` before either the joint command or the gripper command goes out. The blanket `except` at `:231-234` turns that into an ordinary return value.
- **`reset()` and `close()` send the wrong shape.** Both call `step(np.zeros(6))` (`:146`, `:469`), which the `(N, 10)` check at `:177-178` rejects; the same blanket `except` swallows it, so no stop command is ever sent. Do not rely on either to halt an arm.
- **`create_ros2_runner` drops `real_world`.** The factory accepts the argument at `env_runner/ros2_runner.py:216` and never forwards it — `ROS2Runner` has no such parameter — and passes neither of the required `shape_meta` and `urdf_path` (`:231-235`), so they have to arrive through `**kwargs` or the call raises `TypeError`. Construct `ROS2Runner` directly.
- **`get_observation_shapes()` disagrees with `get_obs()`.** `ros2_environment.py:512-524` advertises `robot0_eef_quat` and omits `robot0_eef_rot_axis_angle` and `robot0_eef_rot_axis_angle_wrt_start`, both of which `get_obs()` returns (`:417-423`). Read `get_obs()`, not the advertised shapes.
- **`obs_history` grows without bound.** It is a plain list (`:73`) appended on every `get_obs()` call (`:262`) and never trimmed, while only the last `n_obs_steps` entries are read. `collections.deque(maxlen=n_obs_steps)` is the fix; until then a long rollout keeps every frame in memory.
- **The environment is Franka Panda only.** Hard-coded in four places that no argument reaches: the neutral joint pose (`:59-67`), `base_elements=["panda_link0"]` (`:122`), the nine-element `active_joints_mask` (`:118`) and the joint names `panda_joint1..panda_joint7` (`:276-279`). Passing a different `urdf_path` is not enough for another arm.
- **Infrastructure makes a robot-control decision on shutdown.** `ROS2Infrastructure.shutdown` publishes a zero `Twist` to every publisher whose attribute name contains `cmd_vel` (`ros2_infrastructure.py:329-331`) — a message type this stack never publishes, in the layer meant to hold no robot knowledge.

## Dependency and environment problems

- **`zerorpc` is never installed** (#6). `spnav` is declared but also needs the system library `libspnav.so`, which no Makefile target installs.
- **ROS 2 Python is not in the venv.** `packages/diffusion_policy/examples/run_dataset_pose_publisher.py` imports `rclpy`; neither `rclpy` nor `cv_bridge` is declared anywhere. They come only from the system ROS 2 install baked into `Dockerfile.client`, which is why `voilab replay-trajectory` runs through `docker compose`.
- **`requires-python = ">=3.10, <3.13"` is too generous.** `torch==2.1.0` and `ur-rtde==1.5.6` publish only cp310/cp311 wheels. Use the 3.10 in `.python-version`. See [uv workspace](./uv-workspace.md).
- **Many dependencies serve dead code.** `ur-rtde`, `v4l2py`, `atomics`, `pynput`, `spnav`, `k3d`, `robosuite` are all pulled in for modules nothing reaches, which is a large part of why `uv sync` is slow.
- **The Isaac Sim image does not install exiftool**, although stages 00 and 06 read GoPro serial numbers through it. The image also runs `uv pip install hydra-core --upgrade` after `uv sync --frozen`, so the container does not match `uv.lock`.
- **The Makefile's `.PHONY` lines are malformed** — seven of ten use `.PHONY install:` (space) instead of `.PHONY: install`, so those declarations do nothing.

## Test suite state

155 tests collect, but the full suite never finishes — `tests/services/test_aruco_detection.py` hangs after five passing tests (#12). Excluding that file, **41 fail and 100 pass** in about a second (from the repository root):

```bash
uv run --no-sync pytest packages/umi/tests -q --ignore=packages/umi/tests/services/test_aruco_detection.py
# 41 failed, 100 passed in ~1.2s
```

- `tests/test_pipeline_executor.py` and `tests/test_services.py` are near-identical duplicates (6 failures each), and neither actually tests `PipelineExecutor`. Config merging, inheritance and the `--session-dir` / `--task` overrides have no coverage at all.
- `tests/services/test_calibration.py` (11 failures) and `tests/services/test_dataset_planning.py` (13 failures) describe an API that no longer exists, including artifact names (`slam_tag_calibration.json`) the pipeline never writes. Do not read them as documentation.
- `tests/services/test_slam_mapping.py` (3 failures of 11) and `tests/services/test_visualize_slam_gui.py` (2 of 19) are partly stale. Those five plus the twelve above account for all 41.
- The files that do reflect current behaviour: `test_config.py`, `test_video_organization.py`, `test_imu_extraction.py`, `test_batch_slam.py`, `test_replay_buffer.py`.
- `tests/services/test_aruco_detection.py` stalls after five passing tests (#12). Everything else runs in about a second.
- `packages/umi/tests/` is the only test directory in the repository. `src/voilab/`, `packages/diffusion_policy/` and `scripts/v1/` have no tests at all.

## Documentation drift

Pre-existing documents in `docs/` were written against earlier versions of the code.

- **`packages/umi/README.md`** is largely inherited upstream prose: `make deps`, `mamba env create`, `python run_slam_pipeline.py` and `eval_real.py` are all commands or files that do not exist here. See [vs. upstream UMI](./vs-upstream-umi.md).
- **Task names are spelled five ways.** `kitchen|living_room|dining_room` (`packages/umi/src/umi/cli.py:15`), `kitchen|dining-room|living-room` (`src/voilab/cli.py:30`), `kitchen|dining-table|living-room` (the dict keys in `src/voilab/applications/isaac_sim_config.py`), `kitchen|dining_table|living_room` (the `TASK_NAME` values the same file emits, `:20,29`), and `ed305_kitchen|ed305_dining_room|ed305_living_room|simple_room_kitchen` (`scripts/v1/configs/__init__.py:16-21`).
- **Flag names disagree.** `umi run-slam-pipeline` and `voilab replay-trajectory` take `--session-dir`; `voilab launch-simulator` and every `scripts/v1/` entry point take `--session_dir`. Full list: [CLI reference](./cli-reference.md).

## Working as intended, commonly mistaken for bugs

- **The `00_`, `01_`, `02_` prefixes on stage names do nothing.** Stages run in YAML key order (`pipeline_executor.py:226-232` returns `list(self.config.keys())`). Renaming `03_batch_slam` to `zz_batch_slam` keeps its position; moving its YAML block moves it.
- **You must run from the repository root.** `slam_settings_file` is resolved against `Path.cwd()` (`slam_mapping.py:484-486`), and `camera_intrinsics_path` / `aruco_config_path` are opened as given. The configs write these as `packages/umi/defaults/...`, so any other working directory fails.
- **One path key breaks that rule on purpose.** `intrinsics_path` in `07_frame_to_pose` is resolved against `packages/umi/` (`frame_to_pose.py:15,367`), which is why the reconstruct config writes it without the `packages/umi/` prefix.
- **`--session-dir` rewrites only the first stage that declares it** (`pipeline_executor.py:126-136`), and that is enough: every shipped config declares `session_dir` only in stage 00, and later stages inherit it.
- **Stage 02 skipping is normal.** If `demos/mapping/map_atlas.osa` exists, the stage returns immediately (`slam_mapping.py:78-82`). Delete the file or set `force: true` to redo it. Stages 03 and 04 skip per demo on `camera_trajectory.csv` and `tag_detection.pkl`.
- **`reconstruct/location` says `"unknown"` in most datasets.** Only the reconstruct config sets `task`; without it `replay_buffer.py:94` falls back to that string.
- **`converted_60fps_raw_video.mp4` appearing next to your footage is intended.** Clips at roughly 120 fps are transcoded once to 60 fps (`slam_mapping.py:420-442`) and the result is reused.
- **The first pipeline run downloads about a gigabyte.** `pull_docker` defaults to true, so the ORB-SLAM3 image is pulled before stage 02.
- **`submodules/isaac-ros-ws` is empty and nothing needs it.** No code in this repo references the submodule.

---

**Next:** [Getting started](./getting-started.md) · [Pipeline config](./pipeline-config.md) · [Repository map](./repository-map.md)
