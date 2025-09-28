# Diffusion Policy Package Layers Documentation

## Overview

The Diffusion Policy package implements a modular architecture for learning and executing robotic manipulation policies using diffusion models. This document provides a comprehensive overview of the main layers and components.

## Architecture Layers

### 1. Policy Layer (`src/diffusion_policy/policy/`)

The Policy layer contains the core policy implementations that define how actions are predicted from observations.

#### Key Components:
- **Base Policies**: Abstract base classes for different policy types
  - `BaseImagePolicy`: For vision-based policies
  - `BaseLowdimPolicy`: For state-based policies
- **Policy Implementations**:
  - `DiffusionUnetImagePolicy`: U-Net based diffusion policy for image inputs
  - `DiffusionTransformerLowdimPolicy`: Transformer-based policy for state inputs
  - `DiffusionUnetVideoPolicy`: Video-based policy implementation
  - `BETLowdimPolicy`: Behavioral Enaction Transformer policy

#### Core Method:
```python
def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                  fixed_action_prefix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    obs_dict: Dictionary with observation tensors
    fixed_action_prefix: Optional action prefix for conditioning
    returns: Dictionary with predicted actions
    """
```

### 2. Workspace Layer (`src/diffusion_policy/workspace/`)

The Workspace layer manages the training lifecycle, including model initialization, training loops, and checkpointing.

#### Key Components:
- **BaseWorkspace**: Abstract base class for all training workspaces
- **Training Workspaces**: Task-specific training implementations
  - `TrainDiffusionUnetImageWorkspace`: For vision-based diffusion policies
  - `TrainDiffusionTransformerHybridWorkspace`: For hybrid vision-state policies
  - `TrainBETLowdimWorkspace`: For BET policies

#### Core Features:
- Configuration management with Hydra
- Checkpoint saving/loading
- Training loop management
- Logging and monitoring
- Model initialization and optimization

### 3. Configuration Layer (`src/diffusion_policy/config/`)

The Configuration layer provides hierarchical configuration management for different tasks and model architectures.

#### Structure:
- **Main configs**: Workspace-level configurations (`train_*.yaml`)
- **Task configs**: Task-specific configurations (`task/*.yaml`)
- **Legacy configs**: Backward compatibility configurations

#### Key Configuration Files:
- `train_diffusion_transformer_umi_bimanual_workspace.yaml`: UMI bimanual task
- `task/square.yaml`: Square manipulation task
- `task/umi_image.yaml`: UMI vision-based task

#### Configuration Structure:
```yaml
name: task_name
shape_meta:
  obs:
    camera_image:
      shape: [3, 84, 84]
      type: rgb
      horizon: 2
    robot_state:
      shape: [7]
      type: low_dim
      horizon: 2
  action:
    shape: [10]
    horizon: 16
```

### 4. Dataset Layer (`src/diffusion_policy/dataset/`)

The Dataset layer provides interfaces for loading and preprocessing robotics demonstration data.

#### Key Components:
- **BaseDataset**: Abstract base class for all datasets
- **BaseImageDataset**: Base class for vision-based datasets
- **BaseLowdimDataset**: Base class for state-based datasets
- **Task-specific Datasets**:
  - `RobomimicReplayDataset`: Robomimic simulation data
  - `UmiImageDataset`: UMI vision datasets
  - `RealPushtImageDataset`: Real-world push-T data

#### Core Methods:
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Returns observation and action tensors for given index"""

def get_normalizer(self) -> LinearNormalizer:
    """Returns data normalizer for observations and actions"""
```

### 5. EnvRunner Layer (`src/diffusion_policy/env_runner/`)

The EnvRunner layer handles policy execution in different environments (simulation or real-world).

#### Key Components:
- **BaseImageRunner**: Base class for vision-based environment runners
- **Task-specific Runners**:
  - `PushtImageRunner`: Push-T environment execution
  - `RobomimicRunner`: Robomimic simulation environments
  - `BaseImageRunner`: Generic image-based execution

#### Core Method:
```python
def run(self, policy: BaseImagePolicy) -> Dict:
    """Execute policy in environment and return results"""
```

### 6. Environment Layer (`src/diffusion_policy/env/`)

The Environment layer contains implementations of various robotic environments and task oracles.

#### Key Components:
- **Task Environments**:
  - `block_pushing/`: Block manipulation tasks
  - `pusht/`: Push-T environment
  - `franka_assembly/`: Franka robot assembly tasks
  - `kitchen/`: Kitchen manipulation tasks
- **Environment Registration**: Gym environment registration

#### Example Environment:
```python
register(
    id='pusht-keypoints-v0',
    entry_point='envs.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)
```

## Working Example: Training a Diffusion Policy for Block Pushing

Here's a complete example of how to train a diffusion policy for a block pushing task:

### 1. Configuration File (`config/train_block_push_workspace.yaml`)

```yaml
defaults:
  - _self_
  - task: block_push

name: train_diffusion_unet_block_push
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace

task_name: ${task.name}
shape_meta: ${task.shape_meta}
exp_name: "block_push_experiment"

n_action_steps: 8
policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy

  shape_meta: ${shape_meta}

  noise_scheduler:
    _target_: diffusers.DDIMScheduler
    num_train_timesteps: 50
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2

  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    shape_meta: ${shape_meta}
    rgb_model_name: "resnet18"

  diffusion_model:
    _target_: diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1d
    input_dim: 10
    global_cond_dim: 512

dataset:
  _target_: diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset
  shape_meta: ${shape_meta}
  dataset_path: "data/block_push.hdf5"
  horizon: 16
  n_obs_steps: 2

training:
  batch_size: 64
  num_epochs: 100
  lr: 1e-4
  save_every_n_epochs: 10
  eval_every_n_epochs: 5
```

### 2. Task Configuration (`config/task/block_push.yaml`)

```yaml
name: block_push

low_dim_obs_horizon: 2
img_obs_horizon: 2
action_horizon: 16

shape_meta: &shape_meta
  obs:
    agentview_image:
      shape: [3, 84, 84]
      horizon: ${task.img_obs_horizon}
      type: rgb
    robot0_eef_pos:
      shape: [3]
      horizon: ${task.low_dim_obs_horizon}
      type: low_dim
    robot0_eef_quat:
      raw_shape: [4]
      shape: [6]
      horizon: ${task.low_dim_obs_horizon}
      type: low_dim
      rotation_rep: rotation_6d
    robot0_gripper_qpos:
      shape: [2]
      horizon: ${task.low_dim_obs_horizon}
      type: low_dim
  action:
    shape: [10]
    horizon: ${task.action_horizon}
    rotation_rep: rotation_6d
```

### 3. Training Script

```python
#!/usr/bin/env python3
"""
Example training script for diffusion policy on block pushing task.
"""

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np

@hydra.main(config_path="config", config_name="train_block_push_workspace")
def main(cfg):
    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Create workspace
    workspace = hydra.utils.instantiate(cfg)

    # Run training
    workspace.run()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

### 4. Running the Training

```bash
# Activate environment
conda activate diffusion_policy

# Run training
python train_block_push.py

# Or with Hydra directly
python -m hydra.main --config-path=config --config-name=train_block_push_workspace
```

## Key Design Patterns

### 1. Hierarchical Configuration
- Use Hydra for configuration management
- Separate task configurations from training configurations
- Support configuration inheritance and overrides

### 2. Modular Architecture
- Clear separation between policy, dataset, and environment components
- Abstract base classes for extensibility
- Plugin-style component instantiation

### 3. Data Normalization
- Built-in support for observation and action normalization
- Configurable normalization strategies
- Seamless integration with training pipeline

### 4. Checkpoint Management
- Automatic checkpoint saving and loading
- Support for best model selection
- Training state persistence

## Best Practices

1. **Configuration Management**: Always use task-specific configuration files
2. **Data Preprocessing**: Ensure proper data normalization and format consistency
3. **Environment Registration**: Register custom environments with Gym
4. **Checkpointing**: Save checkpoints regularly and monitor training progress
5. **Modularity**: Keep components modular and reusable across tasks

## Common Use Cases

1. **Training New Policies**: Create new workspace configurations for custom tasks
2. **Evaluating Policies**: Use EnvRunner components to test trained policies
3. **Data Collection**: Implement custom dataset classes for new data sources
4. **Environment Development**: Add new environments following the existing patterns