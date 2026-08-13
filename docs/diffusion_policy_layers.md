# Diffusion policy package layers

The layers inside `packages/diffusion_policy`, what each one owns, and which classes in each are still reachable in this fork.

**Read this if:** you are reading or extending the training package and need to know where a given behaviour lives.

**Before you start:** [Training and evaluating a diffusion policy](./training-and-eval.md) — the worked example this page deliberately does not repeat.

---

Every path below is relative to `packages/diffusion_policy/src/diffusion_policy/`. This fork dropped
upstream's vendored gym benchmarks, so parts of the tree that survived the trim no longer import;
each section says which parts those are.

## 1. Policy layer (`policy/`)

Defines how actions are predicted from observations.

- **Base classes**: `BaseImagePolicy` (vision), `BaseLowdimPolicy` (state only).
- **Live implementations**: `DiffusionUnetTimmPolicy` (the default — U-Net denoiser on a timm
  vision encoder), `DiffusionTransformerTimmPolicy`, `DiffusionUnetImagePolicy`,
  `DiffusionTransformerLowdimPolicy`, `DiffusionUnetVideoPolicy`, `BETLowdimPolicy` (Behavior
  Transformer).

The inference entry point, from `policy/base_image_policy.py:10`:

```python
def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                   fixed_action_prefix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    obs_dict: str -> B,To,*
    fixed_action_prefix: B,Tp,Da
    return: B,Ta,Da
    """
```

`fixed_action_prefix` is honoured only by `DiffusionUnetTimmPolicy`
(`policy/diffusion_unet_timm_policy.py:125`); the other policies take `obs_dict` alone.

Constructor arguments are supplied by Hydra from the workspace config, so a policy's `__init__`
signature and its config block have to agree. `DiffusionUnetImagePolicy.__init__`
(`policy/diffusion_unet_image_policy.py:15-32`) requires `shape_meta`, `noise_scheduler`,
`obs_encoder`, `horizon`, `n_action_steps` and `n_obs_steps`; the denoising U-Net
(`ConditionalUnet1D`, `model/diffusion/conditional_unet1d.py:69`) is built internally at `:49`, not
passed in.

## 2. Workspace layer (`workspace/`)

Owns the training lifecycle: model and EMA setup, the dataset and normalizer, the epoch loop,
checkpointing, and wandb logging.

- **`BaseWorkspace`** — checkpoint save/load and payload handling.
- **`TrainDiffusionUnetImageWorkspace`** — what every shipped UMI config targets.
- **`TrainDiffusionTransformerHybridWorkspace`**, **`TrainBETLowdimWorkspace`**, and the other
  `train_*_workspace.py` variants.

`train.py` does not instantiate the workspace recursively. It resolves the class and hands it the
whole config (`packages/diffusion_policy/train.py:30-32`):

```python
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace.run()
```

## 3. Configuration layer (`config/`)

Hydra configs, two levels deep.

- **Workspace configs** (`config/train_*.yaml`) — model, optimizer, training schedule, logging.
- **Task configs** (`config/task/*.yaml`) — `shape_meta`, horizons, dataset and env_runner targets.
  Present: `umi`, `umi_image`, `umi_bimanual`, `umi_ros2`, `umi_teleop_image`, `lift`, `square`,
  `tool_hang`, `tool_hang_abs`.
- **`config/legacy/`** — older configs kept for reference.

`shape_meta` is the contract between dataset, policy and runner. Excerpt from
`config/task/umi.yaml`:

```yaml
shape_meta: &shape_meta
  obs:
    camera0_rgb:
      shape: [3, 224, 224]
      horizon: ${task.img_obs_horizon}
      type: rgb
    robot0_eef_pos:
      shape: [3]
      horizon: ${task.low_dim_obs_horizon}
      type: low_dim
    robot0_eef_rot_axis_angle:
      raw_shape: [3]
      shape: [6]
      horizon: ${task.low_dim_obs_horizon}
      type: low_dim
      rotation_rep: rotation_6d
  action:
    shape: [10]
    horizon: ${task.action_horizon}
    rotation_rep: rotation_6d
```

`raw_shape` is the on-disk width, `shape` the width the policy sees after `rotation_rep`
conversion. The 10-D action is position(3) + 6-D rotation(6) + gripper width(1) — see
[Data formats](./data-formats.md).

The vision encoder is also configured here. The live encoder is `TimmObsEncoder`
(`model/vision/timm_obs_encoder.py:53`), selected by a timm `model_name`
(`train_diffusion_unet_timm_umi_workspace.yaml:33,47`). `MultiImageObsEncoder` still exists but
takes an already-instantiated `rgb_model` module, not a model name.

## 4. Dataset layer (`dataset/`)

Loads and preprocesses demonstration data.

- **Base classes**: `BaseDataset`, `BaseImageDataset`, `BaseLowdimDataset`
  (`dataset/base_dataset.py:8,32,55`).
- **Live datasets**: `UmiDataset` (targeted by `task/umi.yaml`, `umi_ros2.yaml` and
  `umi_bimanual.yaml`), `UmiImageDataset`, `UmiTeleopImageDataset`, `RobomimicReplayDataset`,
  `RealPushTImageDataset`.
- **Non-importable leftover**: `kitchen_mjl_lowdim_dataset.py` imports the removed `env/` package.

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Returns observation and action tensors for the given index"""

def get_normalizer(self) -> LinearNormalizer:
    """Returns the data normalizer for observations and actions"""
```

## 5. EnvRunner layer (`env_runner/`)

Executes a trained policy and collects metrics. `eval.py -e <import path>` picks a runner, or the
task config's `env_runner` block does.

- **`BaseImageRunner`** (`env_runner/base_image_runner.py:4`) — the interface, one method:

```python
def run(self, policy: BaseImagePolicy) -> Dict:
    """Execute policy in environment and return results"""
```

- **`RealPushTImageRunner`** — the default in `config/task/umi.yaml:92`. A stub: its `run()`
  returns an empty dict.
- **`ROS2Runner`** — targeted by `config/task/umi_ros2.yaml:92`, the only runner that drives a
  robot. See [ROS 2 integration design](./ros2_integration_design.md).

## 6. What this fork removed

Upstream's `env/` package — the vendored `block_pushing/`, `pusht/`, `kitchen/` and
`franka_assembly/` gym environments — is not in this repository and never has been. No gym
registration happens anywhere in the tree.

Ten modules were left behind that still import `diffusion_policy.env.*` at module level, so any
import of them raises `ModuleNotFoundError`: `env_runner/pusht_image_runner.py`,
`pusht_keypoints_runner.py`, `blockpush_lowdim_runner.py`, `robomimic_runner.py`,
`robomimic_image_runner.py`, `robomimic_lowdim_runner.py`, `franka_assembly_image_runner.py`,
`pick_and_place_cup_image_runner.py`, `dataset/kitchen_mjl_lowdim_dataset.py` and
`scripts/generate_bet_blockpush.py`. `env_runner/kitchen_lowdim_runner.py` imports it inside its
methods (`:56-57`), so it imports cleanly and fails when run. Treat all eleven as dead code, not
as examples to copy.

## Design patterns

- **Hierarchical configuration.** Task config and workspace config are separate files, composed by
  Hydra `defaults:` and overridable as `key.subkey=value` on the command line.
- **Instantiation by `_target_`.** Policies, encoders, datasets and runners are named by import
  path in YAML, so swapping one is a config edit. The signature has to match the config block.
- **Normalization in the dataset.** `get_normalizer()` returns a `LinearNormalizer` the workspace
  installs on the policy, so the checkpoint carries its own normalization statistics.
- **Checkpoints carry their config.** `BaseWorkspace` writes the config into the payload, which is
  why `eval.py` can rebuild the workspace from a `.ckpt` alone.

---

**Next:** [Training and evaluating a diffusion policy](./training-and-eval.md) · [Data formats](./data-formats.md) · [Known issues](./known-issues.md)
