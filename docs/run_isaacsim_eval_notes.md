# Isaac Sim Evaluation Script

This document describes how to use `scripts/run_isaacsim_eval.py` to run diffusion policy evaluation in Isaac Sim.

## Quick Start

> **Note**: This script must be run inside the Isaac Sim container (v5.1.0).

1. Launch the container:
   ```bash
   docker compose run -it --rm isaac-sim /bin/bash
   ```

2. Run the evaluation script:
   ```bash
   python scripts/run_isaacsim_eval.py \
       --task kitchen \
       --checkpoint /path/to/checkpoint.ckpt \
       --output-dir /tmp/eval
   ```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--task` | Yes | - | Task name: `kitchen`, `dining-room`, or `living-room` |
| `--checkpoint` | Yes | - | Path to trained policy checkpoint (`.ckpt` file) |
| `--output-dir` | Yes | - | Directory to save evaluation logs |
| `--device` | No | `cuda:0` | Torch device for policy inference |
| `--headless` | No | `False` | Run simulation without GUI |

## Available Tasks

Each task maps to a specific runner class with task-specific success criteria:

| Task | Runner | Success Criteria |
|------|--------|------------------|
| `kitchen` | `KitchenIsaacSimAppRunner` | Blue cup stacked on pink cup (vertical order + XY alignment < 3cm) |
| `dining-room` | `DiningRoomIsaacSimAppRunner` | Fork on left, knife on right of plate (within 15cm) |
| `living-room` | `LivingRoomIsaacSimAppRunner` | All blocks (blue, green, red) inside target box |

## Examples

### Basic evaluation
```bash
python scripts/run_isaacsim_eval.py \
    --task kitchen \
    --checkpoint ./checkpoints/kitchen_policy.ckpt \
    --output-dir ./eval_results/kitchen
```

### Headless mode (for servers without display)
```bash
python scripts/run_isaacsim_eval.py \
    --task kitchen \
    --checkpoint ./checkpoints/kitchen_policy.ckpt \
    --output-dir ./eval_results/kitchen \
    --headless
```

### Using a different GPU
```bash
python scripts/run_isaacsim_eval.py \
    --task dining-room \
    --checkpoint ./checkpoints/dining_policy.ckpt \
    --output-dir ./eval_results/dining \
    --device cuda:1
```

## Output

The script saves evaluation results to `<output-dir>/eval_log.json`:

```json
{
  "episode_stats": [
    {"episode_idx": 0, "episode_length": 150, "success": true, "total_reward": 0.0},
    {"episode_idx": 1, "episode_length": 200, "success": false, "total_reward": 0.0}
  ],
  "total_episodes": 2,
  "avg_episode_length": 175.0,
  "success_rate": 0.5
}
```

---

## Action Post-Processing

The diffusion policy predicts actions in a **relative pose representation** by default. The evaluation script must convert these to absolute world-frame poses before sending to the IK solver.

### Pose Representation Modes

The checkpoint config specifies how poses are represented via `pose_repr`:

```yaml
pose_repr:
  obs_pose_repr: "rel"      # How observations are represented
  action_pose_repr: "rel"   # How actions are represented
```

| Mode | Description |
|------|-------------|
| `abs` | Absolute world-frame poses (no conversion needed) |
| `rel` | Relative to the last observation pose |
| `relative` | Proper SE(3) relative transformation |
| `delta` | Frame-to-frame deltas (cumulative) |

### Action Format

The policy outputs actions with shape `[N, 10]` where each row contains:

| Index | Field | Description |
|-------|-------|-------------|
| 0:3 | Position | Target EEF position (x, y, z) |
| 3:9 | Rotation | Target EEF rotation in 6D representation |
| 9 | Gripper | Gripper width |

### Conversion Pipeline (for `rel` mode)

When `action_pose_repr='rel'` (default), the script performs:

#### 1. Training (Forward Pass)
During training, absolute poses are converted to relative:
```
rel_pos = abs_pos - base_pos
rel_rot = abs_rot @ inv(base_rot)
```
Where `base_pos` and `base_rot` are from the **last observation** in the sequence.

#### 2. Inference (Backward Pass)
During inference, relative actions must be converted back to absolute:
```
abs_pos = rel_pos + base_pos
abs_rot = rel_rot @ base_rot
```

### Implementation Details

The conversion happens in `IsaacSimEnvironment._apply_action()`:

```python
def _apply_action(self, action: np.ndarray) -> None:
    rel_pos = action_step[:3]
    rel_rot6d = action_step[3:9]
    rel_rot_mat = rot6d_to_rotation_matrix(rel_rot6d)

    if self.action_pose_repr in ("rel", "relative"):
        # Convert relative to absolute
        abs_pos = rel_pos + self._last_eef_pos
        abs_rot_mat = rel_rot_mat @ self._last_eef_rot
    else:
        # Use as-is for absolute mode
        abs_pos = rel_pos
        abs_rot_mat = rel_rot_mat

    # Convert to quaternion for IK solver
    quat = rotation_matrix_to_quaternion(abs_rot_mat)
    
    # Send to IK solver
    ik_result = self.art_kine_solver.compute_inverse_kinematics(
        target_position=abs_pos,
        target_orientation=quat
    )
```

### Observation Conversion

Observations are also converted to relative format in `_stack_history()` to match training:

```python
def _convert_obs_to_relative(self, stacked):
    # Base is the LAST observation in the sequence
    base_pos = pos[-1]
    base_rot_mat = rot6d_to_rotation_matrix(rot6d[-1])
    
    for i in range(len(pos)):
        rel_pos[i] = pos[i] - base_pos
        rel_rot_mat = curr_rot_mat @ base_rot_mat.T  # @ inv(base)
        rel_rot6d[i] = rotation_matrix_to_rot6d(rel_rot_mat)
```

### Tracking the Base Pose

The script tracks `_last_eef_pos` and `_last_eef_rot` which are updated:

1. **On reset**: Initialized from the robot's starting pose
2. **After each step**: Updated from the actual EEF pose after action execution

This ensures actions are always relative to the most recent observation, matching the training data distribution.

### 6D Rotation Representation

The policy uses 6D rotation representation (first two columns of rotation matrix):

```python
def rot6d_to_rotation_matrix(rot6d):
    """Convert 6D rotation to 3x3 rotation matrix using Gram-Schmidt."""
    a1, a2 = rot6d[:3], rot6d[3:6]
    b1 = normalize(a1)
    b2 = normalize(a2 - dot(b1, a2) * b1)
    b3 = cross(b1, b2)
    return stack([b1, b2, b3], axis=1)

def rotation_matrix_to_rot6d(rot_mat):
    """Convert 3x3 rotation matrix to 6D representation."""
    return rot_mat[:, :2].flatten()  # First two columns
```

---

## Troubleshooting

### Policy outputs don't match expected behavior

1. **Check pose representation**: Verify checkpoint's `pose_repr` matches expectations
   ```python
   # The script prints this on startup:
   # "Pose representation config: {'obs_pose_repr': 'rel', 'action_pose_repr': 'rel'}"
   ```

2. **Verify coordinate frames**: Ensure Isaac Sim's coordinate system matches training data

3. **Check action scaling**: Actions should be in meters (position) and proper rotation units

### IK failures

If you see frequent "IK failed" messages:
- Target pose may be unreachable
- Check robot base pose calibration
- Verify URDF matches the simulated robot

### Success rate is always 0

- Check task-specific success criteria in `isaacsim_registry_runners.py`
- Verify object prim paths match the scene setup
- Ensure objects are loaded at expected positions
