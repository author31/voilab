# Training and evaluating a diffusion policy

How to turn a `dataset.zarr.zip` into a trained checkpoint, where that checkpoint lands, and how to run it back.

**Read this if:** you have a dataset and want a policy, locally or on the lab cluster.

**Before you start:** [Getting started](./getting-started.md), [Data formats](./data-formats.md)

---

## 1. What the training package is

`packages/diffusion_policy` is a fork of the Columbia *Diffusion Policy* codebase, trimmed to the
UMI setup: it keeps the training loop, the models and the dataset adapters, and drops the simulated
gym benchmarks. A diffusion policy generates a short sequence of future robot actions by
iteratively denoising random noise, conditioned on recent camera images and gripper state.

It consumes exactly one thing: a **replay buffer** — a `.zarr.zip` file (zarr is an array-on-disk
format; the `.zip` is a single-file zarr store) carrying the UMI arrays under `/data` plus
`/meta/episode_ends`. That is stage `07_generate_replay_buffer` of the SLAM pipeline, and also what
the Isaac Sim generator writes. Full schema: [Data formats](./data-formats.md). There is **no
console script** — `pyproject.toml` declares none, so `train.py` and `eval.py` are invoked by path.

```text
dataset.zarr.zip ──> UmiDataset ──> TrainDiffusionUnetImageWorkspace ──> checkpoints/*.ckpt
                     (whole zip        (accelerate + wandb loop)          (torch+dill payload,
                      into RAM)                                            config embedded)
```

## 2. Prerequisites

| Requirement | Detail |
|---|---|
| GPU | One NVIDIA GPU. The shipped batch size of 64 with a ViT-B/16 encoder wants roughly 24 GB of VRAM; drop `dataloader.batch_size` on smaller cards. |
| Environment | The uv workspace at the repo root. `make install` (which runs `uv sync`) is enough — training needs no dev extras. See [uv workspace](./uv-workspace.md). |
| Dataset | A `.zarr.zip` path. Hydra (the config framework this codebase uses; command-line `key.subkey=value` pairs override the YAML) does *not* chdir into the run directory here — `train.py:21` passes `version_base=None`, which with the pinned hydra-core 1.2.0 leaves `hydra.job.chdir` false (`hydra/core/utils.py:148-152`), so relative paths resolve against your shell's cwd. Use an **absolute** path anyway, so the command is copy-pasteable from any directory and identical in the sbatch script. |
| RAM | With the shipped `cache_dir: null` (`config/task/umi.yaml:98`) the entire zip is decompressed into a `zarr.MemoryStore`. Budget several times the file size. |
| wandb | Weights & Biases — a hosted experiment tracker; you need a free account at wandb.ai first. Logging defaults to `mode: online` under project `umi` (`config/train_diffusion_unet_timm_umi_workspace.yaml:181`). Run `uv run wandb login` once with your API key, or add `logging.mode=offline` to every command. |

## 3. Launching training

From the repo root:

```bash
uv run packages/diffusion_policy/train.py \
  --config-path=src/diffusion_policy/config \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/absolute/path/to/dataset.zarr.zip
```

`--config-path` is not optional here. Hydra computes its config directory at
`packages/diffusion_policy/train.py:22-23` as `packages/diffusion_policy/diffusion_policy/config`,
which does not exist — the package lives under `src/`. The flag is resolved relative to `train.py`'s
own directory, hence `src/diffusion_policy/config`. Everything after `--config-name` is a Hydra
override, `key.subkey=value`, no dashes. `train.py` cannot run without the flag — see
[Known issues](./known-issues.md).

**Multi-GPU.** The workspace is built on HuggingFace `accelerate`
(`workspace/train_diffusion_unet_image_workspace.py:93`, `:188`), so the same script runs
data-parallel under `accelerate launch`; checkpointing, the normalizer and sampling are guarded by
`accelerator.is_main_process` (`:117`, `:310`, `:330`). Every recorded run here is single-GPU.

```bash
uv run accelerate launch --num_processes 2 \
  packages/diffusion_policy/train.py \
  --config-path=src/diffusion_policy/config \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/absolute/path/to/dataset.zarr.zip
```

## 4. The config tree

All `config/` paths here are relative to `packages/diffusion_policy/src/diffusion_policy/`.
Workspace configs sit at the top of `config/`; task configs (dataset, observation shapes,
horizons) sit in `config/task/`, pulled in by each workspace config's `defaults:` block.

| Workspace config | Task | Use it for |
|---|---|---|
| `train_diffusion_unet_timm_umi_workspace.yaml` | `umi` | The default. Single-arm UMI, U-Net denoiser, timm ViT encoder. |
| `train_diffusion_transformer_umi_workspace.yaml` | `umi` | Same data, transformer denoiser. |
| `train_diffusion_unet_umi_timm_workspace.yaml` | `umi` | Older variant, **not** interchangeable with the default: DDPM (100 timesteps) and a from-scratch ResNet-18 encoder instead of DDIM (50) + pretrained ViT-B/16. It also has no top-level `n_action_steps`, so that override fails against it. |
| `train_diffusion_unet_timm_umi_ros2_workspace.yaml` | `umi_ros2` | Differs only in `env_runner`, which training never uses. |
| `train_diffusion_unet_umi_bimanual_workspace.yaml`, `train_diffusion_transformer_umi_bimanual_workspace.yaml` | `umi_bimanual` | Two-arm datasets. |

The overrides people actually type:

| Override | Effect | Default (file:line) |
|---|---|---|
| `task.dataset_path=<abs path>` | Which `.zarr.zip` to train on | `example_demo_session/dataset.zarr.zip` (`config/task/umi.yaml:86`) |
| `training.num_epochs=300` | Length of the run | `120` (`..._umi_workspace.yaml:160`) |
| `dataloader.batch_size=32 val_dataloader.batch_size=32` | VRAM pressure; set both together | `64` / `64` (`:118`, `:125`) |
| `task.img_obs_horizon=3` | How many past camera frames the policy sees | `2` (`config/task/umi.yaml:10`) |
| `task.action_horizon=16` | How many future actions the network predicts | `16` (`config/task/umi.yaml:11`) |
| `n_action_steps=8` | How many of those actions are executed per inference | `8` (`..._umi_workspace.yaml:12`) |
| `optimizer.lr=1e-4` | Learning rate (the vision encoder gets 0.1x this when pretrained) | `3.0e-4` (`:147`) |
| `training.checkpoint_every=10` | Epochs between checkpoint writes | `10` (`:169`) |
| `logging.mode=offline` / `logging.name=<run>` | wandb behaviour and run name | `online` (`:181`) |
| `policy.obs_encoder.model_name=resnet34.a1_in1k` | Cheaper vision backbone | `vit_base_patch16_clip_224.openai` (`:47`) |
| `hydra.run.dir=<dir>` | Where the whole run is written | `data/outputs/<date>/<time>_<name>_<task>` (`:204`) |

`training.device` appears in the config (`:153`) and in the shipped job scripts, but the U-Net
workspace has that line commented out (`workspace/train_diffusion_unet_image_workspace.py:181`)
and takes its device from `accelerate`. Mask GPUs with `CUDA_VISIBLE_DEVICES`, not this key.

## 5. Run directory, checkpoints and resuming

With the default `hydra.run.dir`, one run produces:

```text
data/outputs/2026.08.12/14.30.05_train_diffusion_unet_timm_umi/
├── .hydra/config.yaml          # the fully-resolved config for this run
├── normalizer.pkl              # action/observation statistics
├── logs.json.txt               # one JSON line per logged step
└── checkpoints/
    ├── latest.ckpt                                  # overwritten every checkpoint_every epochs
    └── epoch=0010-train_loss=0.123.ckpt             # top-k, k=20, ranked by train_loss
```

- `normalizer.pkl` and `logs.json.txt` come from `workspace/train_diffusion_unet_image_workspace.py:116`
  and `:208`; checkpoints from `workspace/base_workspace.py:35-40`, with `k: 20` set at `..._umi_workspace.yaml:187-192`.
- A checkpoint is a `torch.save` payload pickled with `dill` embedding the full config, so `eval.py`
  needs no config flags. Each is about 860 MB for the default ViT encoder; `k: 20` fills tens of GB.

**Resuming** is `training.resume=True`, which reloads `<output_dir>/checkpoints/latest.ckpt`
(`workspace/train_diffusion_unet_image_workspace.py:103-107`). Point `hydra.run.dir` at the *same*
directory as the original run, or there is no `latest.ckpt` to find and training silently restarts.

**Known issue:** optimizer state is excluded from checkpoints unless `training.resume=True` was
set on the *original* run (`workspace/train_diffusion_unet_image_workspace.py:87-88`). Resuming a
run that started with the default `resume: False` restores weights but restarts the optimizer.
See [Known issues](./known-issues.md).

## 6. Training on the HCISLAB SLURM cluster

Taken from the job scripts that produced the checkpoints in `experiment_wpnoise0629/` — like
`scripts/v1/` and `.claude/`, that directory is untracked in git, so a fresh clone has neither the
sbatch scripts nor the checkpoints, and the script below is reproduced in full. SLURM is the
cluster's batch scheduler: you submit a shell script with `sbatch`, it runs when a GPU frees up.

Everything below uses `chunchuan@hpc.hcislab.org` and `/mnt/bapve/thome/chunchuan/voilab` as an
example — substitute your own cluster account and your own checkout everywhere they appear,
including inside the sbatch script. You need a HCISLAB cluster account and a `uv`-synced clone of
this repo under `/mnt/bapve/thome/<you>/` first; ask whoever administers the cluster.

1. Copy the dataset up first — it is the slow part, and `.zarr.zip` files run to gigabytes:
   `rsync -avP dataset.zarr.zip chunchuan@hpc.hcislab.org:/mnt/bapve/thome/chunchuan/voilab/<experiment>/data/`
2. Check what the cluster offers before picking resources: `sinfo -o "%P %.6D %.10G %.15l %.20N"`.
3. Create the `logs/` directory referenced by `--output`/`--error`; SLURM will not create it.
4. Give each run its own `hydra.run.dir` so concurrent jobs do not overwrite each other.
5. `sbatch train.slurm`, then watch `squeue -u $USER` and `tail -f logs/<job>-<id>.out`.

This is `experiment_wpnoise0629/h3/train.slurm` with one line removed: the real file also passes
`training.device=cuda:0`, which the U-Net workspace ignores (see §4).

```bash
#!/bin/bash
#SBATCH --job-name=voilab-wpnoise-h3
#SBATCH --partition=devq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/bapve/thome/chunchuan/voilab/logs/%x-%j.out
#SBATCH --error=/mnt/bapve/thome/chunchuan/voilab/logs/%x-%j.err

set -euo pipefail
# uv is not on PATH for non-interactive (sbatch) shells; add it.
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/bapve/thome/chunchuan/voilab

# wandb credentials/config
source ~/.wandb_env

uv run --no-sync packages/diffusion_policy/train.py \
  --config-path=src/diffusion_policy/config \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=/mnt/bapve/thome/chunchuan/voilab/experiment_wpnoise0629/data/wp_noise_dataset.zarr.zip \
  task.img_obs_horizon=3 \
  training.num_epochs=300 \
  logging.mode=online \
  logging.name=wpnoise0629_h3 \
  hydra.run.dir=/mnt/bapve/thome/chunchuan/voilab/experiment_wpnoise0629/h3/output
```

The four lines that matter and are easy to omit:

- `export PATH="$HOME/.local/bin:$PATH"` — `uv` is per-user and is not on a non-interactive PATH.
- `uv run --no-sync` — without it, uv re-resolves and reinstalls the lockfile on every job start,
  on a shared filesystem.
- `source ~/.wandb_env` — not in the repo. Create it on the cluster so wandb authenticates without
  a prompt, or use `logging.mode=offline`.
- `--cpus-per-task` should track `dataloader.num_workers` (default `8`,
  `..._umi_workspace.yaml:119`); the runs above used 16 CPUs and 48 GB for a single GPU.

If you are driving this from Claude Code, the repo-local skill `launch-slurm-training` automates
the same flow — but `.claude/` is untracked in git, so a fresh clone will not have it, and the copy
in this tree hard-codes one person's cluster account. Ask whoever set up the cluster for a copy and
change the account.

## 7. Evaluating a checkpoint

```bash
uv run packages/diffusion_policy/eval.py \
  --checkpoint /path/to/checkpoints/latest.ckpt \
  --output_dir /path/to/eval_out \
  --device cuda:0 \
  --env_runner_path diffusion_policy.env_runner.ros2_runner.ROS2Runner \
  --urdf_path assets/franka_panda/franka_panda.urdf
```

| Flag | Short | Default | Meaning |
|---|---|---|---|
| `--checkpoint` | `-c` | required | The `.ckpt` to load; its embedded config rebuilds the workspace |
| `--output_dir` | `-o` | required | Written to; you are prompted to confirm if it exists (`eval.py:32`) |
| `--device` | `-d` | `cuda:0` | Torch device for the policy |
| `--env_runner_path` | `-e` | `None` | Dotted class path overriding `cfg.task.env_runner` (`eval.py:52-56`) |
| `--urdf_path` | `-urdf` | `None` | Robot description handed to the runner |

It loads the checkpoint, picks `workspace.ema_model` when `cfg.training.use_ema` is true
(`eval.py:43-45`), instantiates an *env runner* (the object that steps a robot or simulator and
scores the policy), runs it, and writes only `<output_dir>/eval_log.json` (`eval.py:73-74`), which
carries the per-episode video paths.

**Known issue:** `eval.py:58-62` passes `urdf_path` and `shape_meta` to every runner, but the
runner selected by the default task config — `RealPushTImageRunner`
(`config/task/umi.yaml:92`) — takes only `output_dir`
(`src/diffusion_policy/env_runner/real_pusht_image_runner.py:5-6`) and its `run()` returns an empty
dict. Running `eval.py` without `-e` raises `TypeError`, and the only signature-compatible runner is
`ROS2Runner`. See [Known issues](./known-issues.md).

In practice policies here are evaluated in Isaac Sim with `scripts/v1/rollout.py`, which loads the
same checkpoint and writes into `--session_dir`, or into `./data/eval_outputs/session_<timestamp>/`
when `--session_dir` is omitted (`scripts/v1/rollout.py:491`).
Both evaluation routes have prerequisites you may not have: `ROS2Runner` needs a system ROS 2
Humble install (the `voilab-workspace` container), and `scripts/v1/` is untracked in git, so a
fresh clone contains no `rollout.py`. Read [Simulation and Docker](./simulation-and-docker.md) §2
before you count on either.

## 8. Known issues specific to training

**Known issue:** `config/task/umi.yaml:6` spells the key `dataset_frequeny` and sets it to `0`
(upstream ships `59.94`). Every observation `latency_steps` is `(camera_obs_latency -
robot_obs_latency) * dataset_frequeny` (`:27`, `:49`, `:57`, `:74`), so latency compensation
multiplies out to zero. A correctly spelled override adds a *new* key; you must override the typo.

**Known issue:** `train_diffusion_unet_image_workspace.yaml:3` and
`train_diffusion_unet_timm_robomimic_workspace.yaml:3` both declare `task: lift_image_abs`, but
`config/task/` contains no `lift_image_abs.yaml`. Both configs fail at Hydra composition time.

**Known issue:** the module docstring at `packages/diffusion_policy/train.py:1-5` suggests
`--config-name=train_diffusion_lowdim_workspace`, which does not exist anywhere in this fork. The
nearest surviving files are `config/legacy/train_diffusion_unet_lowdim_workspace.yaml` and
`config/legacy/train_diffusion_transformer_lowdim_workspace.yaml`. Ignore the docstring.

Full ranked list: [Known issues](./known-issues.md).

## 9. Going deeper

Internals of the workspace, policy, encoder and dataset classes: [Diffusion policy layers](./diffusion_policy_layers.md), which also names the eleven modules left dead when this fork dropped upstream's vendored gym environments.

---

**Next:** [Simulation and Docker](./simulation-and-docker.md) · [Data formats](./data-formats.md) · [Known issues](./known-issues.md)
