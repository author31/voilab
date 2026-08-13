# Simulation and containers

The Isaac Sim half of the repo: what it generates, which scripts still work, and the Docker images they run inside.

**Read this if:** you want to make synthetic training data or score a trained policy in simulation, or you need to know what each Dockerfile actually contains.

**Before you start:** [Getting started](./getting-started.md), [Pipeline overview](./pipeline-overview.md), [Repository map](./repository-map.md)

---

## 1. Why there is a simulation half

The real pipeline turns handheld GoPro videos into a training dataset. It is slow (one demonstration per recording) and it cannot re-run a scene. The simulation half exists to do two things the real pipeline cannot:

- **Generate more data.** Replay a scripted pick-and-place in Isaac Sim (NVIDIA's GPU robotics simulator) and record camera + end-effector + gripper observations into a dataset with the same schema as the real one.
- **Evaluate a policy.** Load a trained diffusion-policy checkpoint, run it closed-loop against the simulated robot, and count how often it succeeds. There is no working real-robot path in this fork, so this is the only evaluation available.

The bridge between the two halves is a single file: **`object_poses.json`**. The pipeline's `07_frame_to_pose` stage detects ArUco markers (printed square fiducial tags whose 6-DoF pose a camera can recover) stuck to the real objects, converts each detection into a pose relative to a reference tag, and writes one entry per episode (`packages/umi/src/umi/services/frame_to_pose.py:327`). The simulation reads that file and spawns the same objects at the same places, episode by episode. That is why a separate `*_reconstruct_pipeline_config.yaml` exists — it is the only pipeline config that includes the `07_frame_to_pose` stage (`umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml:58-65`).

```text
GoPro videos
     |
     |  uv run umi run-slam-pipeline \
     |     umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml \
     |     --session-dir S
     v
 +--------------------------------------------------------------+
 |  00..06  video split, IMU, SLAM, ArUco, calibration, plan     |
 |  07_frame_to_pose   -> S/demos/mapping/object_poses.json      |   real object layout
 |  08_generate_replay_buffer -> S/dataset.zarr.zip              |   real demonstrations
 +--------------------------------------------------------------+
          |                                     |
          | object_poses.json                   | dataset.zarr.zip
          v                                     v
  scripts/v1/preview.py   (look at it)     packages/diffusion_policy/train.py
  scripts/v1/sdg.py       -> S/simulation_dataset.zarr.zip ---^
  scripts/v1/rollout.py   (score a checkpoint)  ------------->  checkpoint
```

`.zarr.zip` is a chunked, compressed array store packed into a single zip file — the one format both halves of the repo exchange.

The object-name contract is worth knowing because it silently drops objects when it breaks: the pipeline writes keys like `pink_cup` from a hard-coded task registry (`packages/umi/src/umi/services/frame_to_pose.py:20-35`), and the simulation lower-cases and underscores its own object names to match (`scripts/v1/engine.py:72-73`). A `PreloadObject("pink cup", ...)` in a sim config therefore binds to the `pink_cup` entry in the JSON.

## 2. Read this before you clone

**Known issue:** the working simulation code lives in `scripts/v1/`, which is **untracked in git** — a fresh `git clone` does not contain it. At the same time nine helper modules are deleted in the working tree, with the deletion uncommitted, so a fresh clone has them and this tree does not. Seven are needed only by the older `scripts/*.py`; the other two are `scripts/object_loader.py` and `scripts/utils.py`, both reached by the current path. See [Known issues](./known-issues.md).

`scripts/v1/` is not self-contained. Each entry point puts `scripts/v1` ahead of `scripts` on `sys.path` (`preview.py:22-24`, `sdg.py:27-29`, `rollout.py:23-25`), and `engine.py:48-50` then imports three modules from the parent directory:

- `object_loader` — **deleted in the working tree**, so all three entry points fail at import today. Restore it before running anything: `git checkout -- scripts/object_loader.py`.
- `umi_replay`, `motion_plan` — committed and present, nothing to do.

`scripts/utils.py` is deleted too, but it does not need restoring: `scripts/v1/utils.py` shadows it in a v1 run and defines the symbols the parent modules import.

| Path | State | Works? |
|---|---|---|
| `scripts/v1/` — `preview.py`, `sdg.py`, `rollout.py`, `engine.py`, `config.py`, `geometry.py`, `utils.py`, `configs/` | untracked, present locally | The current path, but not as-is — `engine.py:48` imports the deleted `object_loader` |
| `scripts/generate_data.py` | committed | No — `scripts/generate_data.py:15` does `import registry`, deleted |
| `scripts/run_isaacsim_eval.py` | committed | No — imports `registry` and `utils` (`:10`, `:17`), both deleted |
| `scripts/motion_plan.py` | committed | Standalone: no — `utils` (`:4`) is deleted. In a v1 run: yes, and `scripts/v1/engine.py:50` needs it |
| `scripts/umi_replay.py` | committed | Yes — imported by the current path at `scripts/v1/engine.py:49` |

**Known issue:** `uv run voilab launch-simulator --task kitchen --session_dir <dir>` builds a container command around `scripts/generate_data.py` (`src/voilab/cli.py:62`), which fails on `import registry` (`scripts/generate_data.py:15`). Its `--task kitchen` spelling is also gone. Use `scripts/v1/sdg.py` instead. See [Known issues](./known-issues.md).

If you are picking this repo up fresh and `scripts/v1/` is missing, ask whoever ran the simulation last for the directory; there is no other copy. Copying `scripts/v1/` alone is not enough — `scripts/object_loader.py`, `scripts/umi_replay.py` and `scripts/motion_plan.py` must sit in the parent directory beside it.

## 3. The three entry points — `preview.py` (look), `sdg.py` (synthetic data generation), `rollout.py` (score)

All three take `--config`, and the legal values are exactly the keys of the config registry (`scripts/v1/configs/__init__.py:16-21`):

| `--config` | Scene | Robot | Object spawning |
|---|---|---|---|
| `ed305_kitchen` | lab scene `assets/ED305_scene/ED305.usd` | Franka Panda + UMI gripper | from `object_poses.json` |
| `ed305_dining_room` | same | same | from `object_poses.json` |
| `ed305_living_room` | same | same | from `object_poses.json` |
| `simple_room_kitchen` | Isaac's stock `Simple_Room` | Franka FR3 | on a table, with random jitter |

Note the naming: scene and task are fused into one token, replacing the older `--task kitchen` spelling used everywhere else in the repo.

All three must run inside the `isaac-sim` container, because they construct an Isaac Sim `SimulationApp` at import time (`scripts/v1/preview.py:43-51`, `sdg.py:71-79`, `rollout.py:56-65`). Get a shell first.

First time only, build the image — this is long (tens of minutes) and large (tens of GB of layers): `docker compose build isaac-sim`. `xhost +local:` needs an X session on the *host*; over plain SSH there is none, and because all three entry points run with `"headless": False` there is no headless fallback today (see §5).

```bash
# on the host, from the repo root
xhost +local:
docker compose run --rm isaac-sim /bin/bash
```

Then, inside the container (the repo is bind-mounted at `/workspace/voilab`):

```bash
cd /workspace/voilab

# 1. Look at a scene before spending GPU hours on it
.venv/bin/python scripts/v1/preview.py --config ed305_kitchen \
    --session_dir /workspace/voilab/video/example_kitchen \
    --episode_index 0 --move_to_start

# 2. Generate a synthetic dataset (one episode per object_poses.json entry)
.venv/bin/python scripts/v1/sdg.py --config ed305_kitchen \
    --session_dir /workspace/voilab/video/example_kitchen \
    --output simulation_dataset.zarr.zip

# 3. Score a trained checkpoint
.venv/bin/python scripts/v1/rollout.py --config ed305_kitchen \
    --checkpoint /workspace/voilab/data/outputs/<date>/<time>_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt \
    --session_dir /workspace/voilab/video/example_kitchen \
    --num_eval_rounds 10 --round_timeout 30.0 --device cuda:0
```

`video/example_kitchen/` is a session shipped with the repo (50 episodes in `demos/mapping/object_poses.json`), so commands 1 and 2 run as written once `scripts/object_loader.py` is restored (§2). Command 3 also needs a trained checkpoint; [Training and evaluation](./training-and-eval.md) §5 says where training writes them.

Flags that matter, verified against the argument parsers:

| Flag | Scripts | Default | Notes |
|---|---|---|---|
| `--session_dir` | all three | none | Root of a pipeline session; `demos/mapping/object_poses.json` is read from under it |
| `--object_poses_path` | `preview`, `rollout` | none | Overrides `--session_dir`. **Not available on `sdg.py`** — it always derives the path (`sdg.py:332`) |
| `--episode_index` | `preview`, `rollout` | `0` | Which JSON entry to spawn |
| `--move_to_start` | `preview` | off | Also drives the arm to the task's initial pose |
| `--num_episodes` | `sdg` | JSON length | Required for `simple_room_kitchen`; otherwise `sdg.py:298-301` raises |
| `--output` | `sdg` | `simulation_dataset.zarr.zip` | Filename only; a path is rejected (`sdg.py:462-464`) |
| `--num_eval_rounds`, `--round_timeout`, `--device` | `rollout` | `1`, `30.0`, `cuda:0` | Rounds run back to back |

`sdg.py` can also inject action-proportional noise into the scripted demonstrations, which is how the `experiment_wpnoise0629/` dataset was made. Switch it on with `--inject_wp_action_noise` (Cartesian waypoints) and `--inject_gripper_noise` (gripper width); `--noise_seed` (default `0`) seeds each episode as `noise_seed + episode_idx`. The magnitudes are `min(alpha * commanded_change, cap)` and are tunable with `--wp_pos_noise_alpha 0.15 --wp_pos_noise_cap 0.01 --wp_rot_noise_alpha 0.10 --wp_rot_noise_cap 0.05 --gripper_noise_alpha 0.10 --gripper_noise_cap 0.005` (`sdg.py:51-68`).

`sdg.py` resumes: completed episode indices are written to `<session_dir>/.previous_progress.json` (`sdg.py:202-220`). Delete that file to restart a session from scratch.

## 4. Which Python runs what

Three interpreters exist and they are not interchangeable. Running a `scripts/v1/` file under the host venv fails at `from isaacsim import SimulationApp`.

| Interpreter | Path | Version | Contains | Runs |
|---|---|---|---|---|
| Host uv venv | `<repo>/.venv/bin/python` | 3.10 (`.python-version`) | `umi`, `voilab`, `diffusion_policy`, torch — **no `isaacsim`** | `uv run umi ...`, `uv run voilab ...`, `train.py`, the notebook viewers |
| Container venv, `isaac-sim` service | `/workspace/voilab/.venv/bin/python` | 3.11 (`Dockerfile:56`) | the same lockfile plus `isaacsim[all,extscache]==5.1.0` (`Dockerfile:78-82`) | everything in `scripts/` and `scripts/v1/` |
| ROS 2 Humble, `voilab-workspace` service | `/opt/venv/bin/python` with `/opt/ros/humble` on `PYTHONPATH` (`Dockerfile.client:189-191`) | 3.10 | `rclpy`, `cv_bridge`, cuRobo, nvblox | `packages/diffusion_policy/examples/*ros2*`, `voilab replay-trajectory` |

The two container venvs are separate images, not the host's. The compose file deliberately shadows the bind-mounted `.venv` with an anonymous volume (`docker-compose.yaml:7-8`) so the container keeps its own 3.11 environment even though the repo is mounted over the top.

## 5. Container topology

| Dockerfile | Compose service | Contains | Status |
|---|---|---|---|
| `Dockerfile` | `isaac-sim` (`docker-compose.yaml:2-33`) | CUDA 12.2 + Python 3.11 venv + `uv sync --frozen` + `isaacsim==5.1.0`. **No ROS 2** | Current — the only image used for simulation |
| `Dockerfile.client` | `voilab-workspace` (`docker-compose.yaml:35-65`) | CUDA 12.2 + ROS 2 Humble desktop (`:60-68`) + cuRobo + nvblox built from source | Older, still referenced by `voilab replay-trajectory` |
| `Dockerfile.isaacsim` | *(none)* | ROS 2 Humble built from source with colcon + `isaacsim==5.1.0` via pip | Orphaned — nothing builds it |

**The `isaac-sim` image has no ROS 2.** `grep -i ros Dockerfile` returns nothing. ROS 2 lives only in `Dockerfile.client`, which builds the *other* service (`voilab-workspace`) — so do not expect a `ros2` command inside the sim container.

**Known issue:** `docker compose up isaac-sim` looks like it should launch the Isaac Sim GUI, but it will not: the service declares no `command:`, and `Dockerfile:95` sets `ENTRYPOINT ["/entrypoint.sh"]`, whose whole body is `exec "$@"` (`entrypoint.sh:9`). With no arguments the container exits immediately. Use `docker compose run --rm isaac-sim /bin/bash`. See [Known issues](./known-issues.md).

Two standalone launch scripts predate the compose file and disagree with it:

- `launch_workspace.sh` builds only under `--force-rebuild` (`launch_workspace.sh:31-37`), i.e. via `make launch-workspace-force`. That build is `docker build -t voilab-workspace:latest .` (`:33`), which builds the **Isaac Sim** `Dockerfile` and tags it with the *client* service's name. Two different images now answer to "voilab-workspace".
- `launch_isaacsim.sh` runs the stock NGC image `nvcr.io/nvidia/isaac-sim:5.0.0` (`:15`), a different Isaac version from the 5.1.0 this repo installs, and contains none of this repo's code.

Prefer `docker compose` over both.

Host requirements for either service: an NVIDIA driver plus the `nvidia-docker2` runtime (both services set `runtime: nvidia` and reserve all GPUs), and X11 access for the viewport — run `xhost +local:` once per boot. `DISPLAY` defaults to `:1` if unset (`docker-compose.yaml:16`). All three sim entry points open a window (`"headless": False`), so there is no headless mode today.

## 6. Assets

`assets/` is about 110 MB of USD (Universal Scene Description, Pixar's scene format) and URDF (the ROS robot-description XML) files, tracked in git. The sim configs reference them by **absolute container path**, so they only resolve under the `/workspace/voilab` bind mount:

| Directory | Used by | Referenced at |
|---|---|---|
| `assets/ED305_scene/ED305.usd` | all three `ed305_*` configs | `scripts/v1/configs/ed305.py:25` |
| `assets/franka_panda/franka_panda_arm_v2.usd` | ED305 robot | `scripts/v1/configs/ed305.py:26` |
| `assets/franka_panda/franka_panda_umi-isaacsim.urdf` | Lula — Isaac Sim's built-in inverse-kinematics library; the URDF gives it the link geometry, the `*_descriptor.yaml` (next row) gives it the joint set and limits | `scripts/v1/configs/ed305.py:33` |
| `assets/lula/frank_umi_descriptor.yaml` | Lula IK descriptor (note the typo in the filename) | `scripts/v1/configs/ed305.py:32` |
| `assets/CADs/*.usd` | spawnable objects: cups, fork, knife, plate, blocks | `scripts/v1/config.py:36` |

`simple_room_kitchen` is the exception: its scene and FR3 robot come from NVIDIA's asset server at runtime, not from `assets/` (`scripts/v1/configs/simple_room.py:27-28`, resolved through `get_assets_root_path()` at `:70-71`). Only the cup CADs come from the repo. Those paths are Isaac Sim 5.1 paths — 5.1 moved robots under a manufacturer directory, so `FrankaRobotics/FrankaFR3/fr3.usd` will not resolve on 5.0.

## 7. What the simulation writes

`sdg.py` writes that archive at `<session_dir>/<--output>` (`sdg.py:462-464`). The arrays are (`sdg.py:238-269`):

| Array | dtype | Meaning |
|---|---|---|
| `data/camera0_rgb` | `uint8` | wrist-camera frames, 224x224 (`scripts/v1/config.py:33`) |
| `data/robot0_eef_pos` | `float32` | end-effector position, 3 values |
| `data/robot0_eef_rot_axis_angle` | `float32` | end-effector rotation as an axis-angle vector, 3 values |
| `data/robot0_gripper_width` | `float32` | gripper opening in metres |
| `data/robot0_demo_start_pose` | `float64` | episode start pose, repeated per frame |
| `data/robot0_demo_end_pose` | `float64` | episode end pose, repeated per frame |
| `meta/episode_ends` | int | cumulative frame counts, one per episode |

Those are exactly the keys the real pipeline's replay-buffer stage writes (`packages/umi/src/umi/services/replay_buffer.py:154-158`, `:195`) and exactly the keys the training `shape_meta` asks for (`packages/diffusion_policy/src/diffusion_policy/config/task/umi.yaml:14-83`). The one entry in `shape_meta` that neither writer produces, `robot0_eef_rot_axis_angle_wrt_start`, is derived at load time from `robot0_demo_start_pose` (`packages/diffusion_policy/src/diffusion_policy/dataset/umi_dataset.py:312-325`). A simulated dataset is therefore a drop-in for `task.dataset_path=` — see [Training and evaluation](./training-and-eval.md).

One difference: the real pipeline compresses `camera0_rgb` with JPEG-XL and needs the `imagecodecs` codec registered before reading, while `sdg.py` uses Blosc/zstd (`sdg.py:224`), which zarr handles without registration.

`rollout.py` writes no dataset. It prints per-round grasp and task success and, if you omit `--session_dir`, defaults its working directory to `./data/eval_outputs/session_<timestamp>` (`rollout.py:491`). Because the container runs as root over a bind mount, files it creates there are root-owned on the host.

## 8. ROS 2

Nothing in `scripts/v1/` uses ROS 2 — the simulation talks to Isaac Sim through its Python API directly. ROS 2 code exists elsewhere in the repo (all three scripts in `packages/diffusion_policy/examples/` — `run_dataset_pose_publisher.py`, `run_ros2_rgb.py`, `run_go_to_home.py`, the last also needing cuRobo — plus a `ros2_runner` env-runner), but `rclpy` and `cv_bridge` are declared in no `pyproject.toml`, so those files only import inside the `voilab-workspace` container. `docs/ros2_integration_design.md` documents the three-layer ROS 2 architecture against the real class signatures and ends with the defects that stop it working — the layer split is the intended design, several pieces of the code are broken, and no pipeline stage or sim script depends on any of it.

---

**Next:** [Project status](./project-status.md) · [Known issues](./known-issues.md) · [Training and evaluation](./training-and-eval.md) · [Data formats](./data-formats.md)
