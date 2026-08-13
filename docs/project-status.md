# Project status

Which parts of this repository are finished, which are still being built, and which are deliberately abandoned.

**Read this if:** you are deciding what to build on, or you hit code that looks half-finished and want to know whether that is a bug or a work in progress.

**Before you start:** [Getting started](./getting-started.md), [Repository map](./repository-map.md)

*Last reviewed: 2026-08-13.*

---

## Status at a glance

| Area | Status | Lives in |
|---|---|---|
| Data collection — handheld GoPro capture, session organisation | **Done** | `packages/umi/src/umi/services/video_organization.py` |
| Transformation to training data — SLAM, calibration, ArUco, replay buffer | **Done** | `packages/umi/src/umi/services/`, [Pipeline overview](./pipeline-overview.md) |
| Policy training — diffusion policy, local and SLURM | **Done** | `packages/diffusion_policy/`, [Training and evaluation](./training-and-eval.md). The SLURM job scripts are the **untracked** `experiment_wpnoise0629/*/train.slurm`. |
| Visualization and dataset QA | **Done** | `src/voilab/`, [Visualization](./visualization.md) |
| **Real-world rollout — running a trained policy on hardware** | **Under development** | targeted at ROS 2; see below |
| **Isaac Sim simulator integration** | **Under development** | `scripts/`, [Simulation and Docker](./simulation-and-docker.md) |
| Upstream's vendored real-robot stack | **Not used, not being revived** | `packages/umi/src/umi/real_world/` |

"Done" means it has been run end to end on real data in this lab. It does not mean bug-free — see [Known issues](./known-issues.md).

## What is finished: capture through training

The path from a memory card to a trained checkpoint works today:

```text
GoPro clips -> umi run-slam-pipeline -> session artifacts -> dataset.zarr.zip -> diffusion policy checkpoint
```

Each arrow has a doc: [Getting started](./getting-started.md) for the run, [Pipeline overview](./pipeline-overview.md) for the stages, [Data formats](./data-formats.md) for what lands on disk, and [Training and evaluation](./training-and-eval.md) for training and offline evaluation.

If your work stops at "I need a dataset" or "I need a trained policy", nothing on this page should slow you down.

## Under development: real-world rollout

Everything up to and including training is done. **Deploying that checkpoint onto a physical robot is not.** There is no working real-robot entry point in this fork — no equivalent of upstream's `eval_real.py` or `demo_real_robot.py`.

### The decision: build rollout on ROS 2

Rollout is being rebuilt on the ROS 2 ecosystem rather than continued from upstream UMI's control stack.

The reason is upstream's dependency chain. UMI's Franka deployment path is built on **Polymetis** — its `franka_instruction.md` tells you to "install *Polymetis* on a computer with realtime kernel" and to run `python scripts_real/launch_franka_interface_server.py` on the control PC. [`facebookresearch/polymetis`](https://github.com/facebookresearch/polymetis) **was archived on 6 January 2023 and is read-only**; its source was folded into the `droidlet` monorepo. Both of those are facts about upstream, checked against the upstream repository and not against this one: neither `franka_instruction.md` nor `scripts_real/` is vendored into this fork. Building new lab infrastructure on an archived controller with a realtime-kernel requirement is not a foundation worth investing in, whereas ROS 2 gives us maintained drivers, a standard message layer, and hardware we can swap.

The vendored upstream code shows exactly this dependency. `packages/umi/src/umi/real_world/franka_interpolation_controller.py:233` is commented `# start polymetis interface`, and the `FrankaInterface` class at `:36-38` is a `zerorpc` client pointed at that Polymetis-backed server. That module does not even import in this workspace: `zerorpc` is declared in no `pyproject.toml` and is absent from `uv.lock`. Treat all of `packages/umi/src/umi/real_world/` as dead reference material.

### What exists toward ROS 2 today

| Piece | File | State |
|---|---|---|
| Policy runner | `packages/diffusion_policy/src/diffusion_policy/env_runner/ros2_runner.py` | Early, not validated end to end |
| Environment layer | `packages/diffusion_policy/src/diffusion_policy/environments/ros2_environment.py` | Early |
| Example scripts | `packages/diffusion_policy/examples/run_ros2_rgb.py`, `run_dataset_pose_publisher.py`, `run_go_to_home.py` | Sketches |
| Layered example | `packages/diffusion_policy/src/diffusion_policy/examples/ros2_integration_example.py` | Broken: it imports `ROS2EnvironmentFactory` (`:20`), which is defined nowhere. The legacy shim `environments/ros2/ros2_env.py:16` imports the same missing name |
| Architecture write-up | [ROS 2 integration design](./ros2_integration_design.md) | Current: documents the real signatures, and lists the defects it found in them |

None of this runs from the `uv` environment alone — ROS 2 comes from a system installation, not from `uv.lock`. Expect these interfaces to change.

## Under development: Isaac Sim integration

Everything under `scripts/` is the simulation half, and it is mid-rewrite. There are two generations in the tree at once:

| Generation | Files | State |
|---|---|---|
| v0 | `scripts/generate_data.py`, `run_isaacsim_eval.py`, `motion_plan.py`, `umi_replay.py`, plus `registry/`, `utils.py`, `action_graph.py`, `object_loader.py`, `teleop_controller.py`, `launch_isaacsim_workspace.py` | Committed, superseded. `scripts/registry/`, `utils.py` and four other tracked v0 modules are deleted in the working tree but not committed as deleted (`git status`), so `generate_data.py:15`'s `import registry` raises `ModuleNotFoundError`. |
| v1 | `scripts/v1/` — `sdg.py`, `rollout.py`, `preview.py`, `engine.py`, `config.py`, `geometry.py`, `utils.py`, `configs/` | The active config-based rewrite. **Untracked in git**, so a fresh clone does not get it. |

Consequences worth knowing before you plan work around it:

- `uv run voilab launch-simulator` shells out to the v0 script (`src/voilab/cli.py:62`) and therefore fails.
- v1 is not independent of v0: `scripts/v1/engine.py:48` imports `object_loader`, one of the deleted-but-uncommitted modules, so every v1 entry point fails at import in this tree until `git checkout -- scripts/object_loader.py` restores it.
- Do not treat `scripts/v1/` APIs as stable; config names, flags and the scene registry are still moving.
- The simulation side reads `object_poses.json`, produced by the reconstruct pipeline variant. That handoff is the load-bearing link between the two halves — see [Simulation and Docker](./simulation-and-docker.md).

Getting `scripts/v1/` committed is the prerequisite for anyone else reproducing simulation results.

## What this means for you

| If you want to... | Today's answer |
|---|---|
| Collect demonstrations and build a dataset | Supported. Follow [Getting started](./getting-started.md). |
| Train and evaluate a policy offline | Supported. See [Training and evaluation](./training-and-eval.md). |
| Train on the HCISLAB SLURM cluster | Supported, but `train.slurm` is not in git — copy it from a lab machine, or from §6 of [Training and evaluation](./training-and-eval.md). |
| Run a policy on a real robot | Not yet. Rollout is being built on ROS 2; nothing runnable exists. |
| Run a policy in simulation | In progress, and only from `scripts/v1/` inside the Isaac Sim container. Expect breakage. |
| Reuse upstream UMI's Franka or UR5 controllers | No. `packages/umi/src/umi/real_world/` does not import, and its Polymetis-based design is not the direction. |
| Reproduce simulation results from a fresh clone | Not possible until `scripts/v1/` is committed. |

---

**Next:** [Simulation and Docker](./simulation-and-docker.md) · [Known issues](./known-issues.md) · [Versus upstream UMI](./vs-upstream-umi.md)
