# End-Effector Pose Frame Representation and Processing Flow

## Table of Contents
1. [Overview](#overview)
2. [Coordinate Frame Definitions](#coordinate-frame-definitions)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation/Inference Pipeline](#evaluationinference-pipeline)
5. [Pose Representation Conversion](#pose-representation-conversion)
6. [Configuration Reference](#configuration-reference)
7. [Code Reference Summary](#code-reference-summary)

---

## Overview

The Universal Manipulation Interface (UMI) uses different coordinate frames during data collection/training and evaluation/inference. Understanding these transformations is critical for proper policy deployment.

| Phase | Raw EEF Frame | Processed Frame for Policy |
|-------|---------------|----------------------------|
| **Data Collection (SLAM)** | SLAM Map Origin | ArUco Table Tag Frame |
| **Training** | ArUco Table Tag Frame | Relative to Current Pose |
| **Inference** | Robot Base Frame | Relative to Current Pose |

The key insight is that the **policy learns relative pose representations**, making it agnostic to absolute coordinate frames. This enables transfer from SLAM-based data collection to robot-based deployment.

---

## Coordinate Frame Definitions

### 1. SLAM Map Origin Frame
- **Definition**: The origin established by ORB_SLAM3 during mapping
- **Orientation**: Arbitrary (depends on where SLAM initializes)
- **Usage**: Raw camera trajectory output from SLAM

### 2. ArUco Table Tag Frame
- **Definition**: A world-fixed frame defined by an ArUco marker placed on the table surface
- **Orientation**: Z-axis pointing up (z-up convention)
- **Usage**: Consistent reference across different recording sessions
- **Evidence** (`scripts_slam_pipeline/06_generate_dataset_plan.py`, lines 421-424):
```python
pose_interp = pose_interp_from_df(df, 
    start_timestamp=row['start_timestamp'], 
    # build pose in tag frame (z-up)
    tx_base_slam=tx_tag_slam)
```

### 3. Robot Base Frame
- **Definition**: The origin frame of the robot arm (UR5/Franka)
- **Orientation**: Defined by robot manufacturer
- **Usage**: Real-time TCP pose reported by robot controller during inference

### 4. Relative Frame (Policy Input/Output)
- **Definition**: Pose expressed relative to a reference pose (typically current pose)
- **Orientation**: Reference pose becomes identity
- **Usage**: Both training data and inference observations

---

## Training Pipeline

### Step 1: Data Collection via SLAM

During data collection, GoPro cameras record video while ORB_SLAM3 tracks camera pose.

**File**: `scripts_slam_pipeline/06_generate_dataset_plan.py`

#### 1.1 Load SLAM-to-Tag Transform
```python
# Lines 111-119
if tx_slam_tag is None:
    path = demos_dir.joinpath('mapping', 'tx_slam_tag.json')
    assert path.is_file()
    tx_slam_tag = str(path)
tx_slam_tag = np.array(json.load(
    open(os.path.expanduser(tx_slam_tag), 'r')
    )['tx_slam_tag']
)
tx_tag_slam = np.linalg.inv(tx_slam_tag)
```

#### 1.2 Define Camera-to-TCP Transform
```python
# Lines 99-108
# tcp to camera transform
# all unit in meters
# y axis in camera frame
cam_to_center_height = 0.086 # constant for UMI
# optical center to mounting screw, positive is when optical center is in front of the mount
cam_to_mount_offset = 0.01465 # constant for GoPro Hero 9,10,11
cam_to_tip_offset = cam_to_mount_offset + tcp_offset

pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0])
tx_cam_tcp = pose_to_mat(pose_cam_tcp)
```

#### 1.3 Transform Camera Pose to TCP in Tag Frame
```python
# Lines 618-626
cam_pos = df[['x', 'y', 'z']].to_numpy()
cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
cam_pose[:,3,3] = 1
cam_pose[:,:3,3] = cam_pos
cam_pose[:,:3,:3] = cam_rot.as_matrix()
tx_slam_cam = cam_pose
tx_tag_cam = tx_tag_slam @ tx_slam_cam  # Transform to tag frame
```

```python
# Lines 687-688
# transform to tcp frame
tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
pose_tag_tcp = mat_to_pose(tx_tag_tcp)
```

#### 1.4 Store in Dataset
```python
# Lines 745-750 (dataset_plan output)
grippers.append({
    "tcp_pose": pose_tag_tcp,  # Stored in TAG FRAME
    "gripper_width": all_gripper_widths[cam_idx][start:end],
    "demo_start_pose": demo_start_poses[cam_idx],
    "demo_end_pose": demo_end_poses[cam_idx]
})
```

### Step 2: Replay Buffer Generation

**File**: `scripts_slam_pipeline/07_generate_replay_buffer.py`

The TCP pose (in tag frame) is split into position and rotation:
```python
# Lines 107-119
for gripper_id, gripper in enumerate(grippers):    
    eef_pose = gripper['tcp_pose']
    eef_pos = eef_pose[...,:3]
    eef_rot = eef_pose[...,3:]
    gripper_widths = gripper['gripper_width']
    
    robot_name = f'robot{gripper_id}'
    episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
    episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
    episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
```

**Data stored in `.zarr.zip`:**
- `robot{N}_eef_pos`: Position in **ArUco Tag Frame** (3D)
- `robot{N}_eef_rot_axis_angle`: Rotation in **ArUco Tag Frame** (axis-angle, 3D)
- `robot{N}_gripper_width`: Gripper opening width (1D)

### Step 3: Training Data Transformation

**File**: `diffusion_policy/dataset/umi_dataset.py`

#### 3.1 Convert to 4x4 Transformation Matrix
```python
# Lines 336-341
pose_mat = pose_to_mat(np.concatenate([
    obs_dict[f'robot{robot_id}_eef_pos'],
    obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
], axis=-1))
action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
```

#### 3.2 Convert to Relative Representation
```python
# Lines 343-353
# solve relative obs
obs_pose_mat = convert_pose_mat_rep(
    pose_mat, 
    base_pose_mat=pose_mat[-1],  # Current (latest) pose as reference
    pose_rep=self.obs_pose_repr,  # 'relative' from config
    backward=False)
action_pose_mat = convert_pose_mat_rep(
    action_mat, 
    base_pose_mat=pose_mat[-1],  # Same reference for action
    pose_rep=self.obs_pose_repr,
    backward=False)
```

#### 3.3 Convert to 10D Representation (3D pos + 6D rotation + 1D gripper)
```python
# Lines 355-360
# convert pose to pos + rot6d representation
obs_pose = mat_to_pose10d(obs_pose_mat)
action_pose = mat_to_pose10d(action_pose_mat)

action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
actions.append(np.concatenate([action_pose, action_gripper], axis=-1))
```

#### 3.4 Final Training Data Format
```python
# Lines 362-364
# generate data
obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]  # 6D rotation
```

**Training data format:**
- `robot{N}_eef_pos`: **Relative position** (3D) - relative to current pose
- `robot{N}_eef_rot_axis_angle`: **6D rotation representation** (6D) - relative to current pose
- `robot{N}_gripper_width`: Gripper width (1D) - absolute
- `action`: 10D per robot (3D pos + 6D rot + 1D gripper) - relative to current pose

---

## Evaluation/Inference Pipeline (Detailed)

The inference pipeline transforms EEF poses through several stages. This section provides an in-depth elaboration of each transformation with mathematical formulations and concrete examples.

**Key Files:**
- `eval_real.py` - Main evaluation loop
- `umi/real_world/real_inference_util.py` - Observation and action processing
- `umi/real_world/bimanual_umi_env.py` - Robot interface
- `umi/common/pose_util.py` - Pose conversion utilities
- `diffusion_policy/common/pose_repr_util.py` - Relative pose computation

---

### Overview: Inference Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE - DETAILED                                │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌─────────────────┐                                                               │
│  │ Robot Controller│──► ActualTCPPose (6D: xyz + axis-angle)                       │
│  │ (UR5/Franka)    │    Frame: Robot Base                                          │
│  └─────────────────┘                                                               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 1: Raw Observation Collection                              │               │
│  │ • Interpolate robot poses to observation timestamps             │               │
│  │ • Output: robot{N}_eef_pos (3D), robot{N}_eef_rot_axis_angle (3D)│              │
│  │ • Frame: Robot Base (ABSOLUTE)                                  │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 2: Convert to 4x4 Transformation Matrix                    │               │
│  │ • pose_to_mat(): 6D pose → 4x4 SE(3) matrix                     │               │
│  │ • T = [R | t]  where R ∈ SO(3), t ∈ ℝ³                         │               │
│  │       [0 | 1]                                                   │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 3: Convert to Relative Representation                      │               │
│  │ • T_rel = T_current⁻¹ @ T_obs                                   │               │
│  │ • Current pose becomes identity                                 │               │
│  │ • Historical poses expressed relative to current                │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 4: Convert to 10D Representation                           │               │
│  │ • mat_to_pose10d(): 4x4 matrix → 10D vector                     │               │
│  │ • [x, y, z, r1, r2, r3, r4, r5, r6]                             │               │
│  │ • Position (3D) + 6D Rotation (continuous representation)       │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 5: Policy Inference                                        │               │
│  │ • Input: obs_dict with relative poses + images                  │               │
│  │ • Output: action (10D × horizon × n_robots)                     │               │
│  │ • Action is in RELATIVE frame                                   │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 6: Convert Action to Absolute Pose                         │               │
│  │ • pose10d_to_mat(): 10D → 4x4 matrix                            │               │
│  │ • T_abs = T_current @ T_rel (backward transform)                │               │
│  │ • Frame: Robot Base (ABSOLUTE)                                  │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │ STEP 7: Convert to 6D Pose and Send to Robot                    │               │
│  │ • mat_to_pose(): 4x4 matrix → 6D (xyz + axis-angle)             │               │
│  │ • Send via servoL/exec_actions to robot controller              │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│           │                                                                        │
│           ▼                                                                        │
│  ┌─────────────────┐                                                               │
│  │ Robot Controller│◄── Target TCP Pose (Robot Base Frame)                         │
│  └─────────────────┘                                                               │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Raw Observation Collection

**File**: `umi/real_world/bimanual_umi_env.py`, lines 431-439

The robot controller (UR5 or Franka) continuously reports the TCP (Tool Center Point) pose at high frequency (125Hz for UR5, 500Hz for UR5e, 200Hz for Franka). This pose is in the **robot's base frame**.

```python
# bimanual_umi_env.py, lines 431-439
robot_pose_interpolator = PoseInterpolator(
    t=last_robot_data['robot_timestamp'], 
    x=last_robot_data['ActualTCPPose'])  # 6D: [x, y, z, rx, ry, rz]
robot_pose = robot_pose_interpolator(robot_obs_timestamps)
robot_obs = {
    f'robot{robot_idx}_eef_pos': robot_pose[...,:3],           # Position (3D)
    f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:] # Rotation (3D axis-angle)
}
```

**Data format at this stage:**
- `robot{N}_eef_pos`: Shape `(T, 3)` - XYZ position in robot base frame (meters)
- `robot{N}_eef_rot_axis_angle`: Shape `(T, 3)` - Axis-angle rotation in robot base frame (radians)

Where `T` is the observation horizon (typically 2 timesteps).

---

### Step 2: Convert to 4x4 Transformation Matrix

**File**: `umi/common/pose_util.py`, lines 24-30

The 6D pose (position + axis-angle) is converted to a 4x4 homogeneous transformation matrix for proper SE(3) operations.

```python
# pose_util.py, lines 24-30
def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])  # Axis-angle to Rotation object
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

# pos_rot_to_mat (lines 4-10)
def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos              # Translation vector
    mat[...,:3,:3] = rot.as_matrix() # Rotation matrix
    mat[...,3,3] = 1                 # Homogeneous coordinate
    return mat
```

**Mathematical representation:**

Given pose `p = [x, y, z, rx, ry, rz]`:

```
         ┌                     ┐
         │  R₁₁  R₁₂  R₁₃  x   │
T₄ₓ₄ =   │  R₂₁  R₂₂  R₂₃  y   │
         │  R₃₁  R₃₂  R₃₃  z   │
         │   0    0    0   1   │
         └                     ┘

Where R = Rotation matrix from axis-angle [rx, ry, rz]
```

**Usage in inference** (`real_inference_util.py`, lines 102-106):
```python
pose_mat = pose_to_mat(np.concatenate([
    env_obs[robot_prefix + '_eef_pos'],
    env_obs[robot_prefix + '_eef_rot_axis_angle']
], axis=-1))
# Shape: (T, 4, 4) where T is observation horizon
```

---

### Step 3: Convert to Relative Representation (Forward Transform)

**File**: `diffusion_policy/common/pose_repr_util.py`, lines 48-64

This is the **critical transformation** that enables the policy to work with relative poses.

```python
# pose_repr_util.py, lines 48-64
def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    if not backward:
        # Forward transform (for creating policy input)
        if pose_rep == 'relative':
            # T_relative = T_base⁻¹ @ T_pose
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
```

**Mathematical formulation:**

For observation horizon `[T₀, T₁, ..., T_{n-1}]` where `T_{n-1}` is the current (latest) pose:

```
T_rel[i] = T_current⁻¹ @ T_obs[i]

Specifically:
- T_rel[n-1] = T_current⁻¹ @ T_current = I (identity - current pose is origin)
- T_rel[n-2] = T_current⁻¹ @ T_{n-2}   (previous pose relative to current)
- ...
```

**Usage in inference** (`real_inference_util.py`, lines 108-117):
```python
# solve relative obs
obs_pose_mat = convert_pose_mat_rep(
    pose_mat,                    # Shape: (T, 4, 4) - observation poses
    base_pose_mat=pose_mat[-1],  # Shape: (4, 4) - CURRENT pose as reference
    pose_rep=obs_pose_repr,      # 'relative' from config
    backward=False)              # Forward transform for observation
```

**Concrete Example:**

Suppose the robot has two observation timesteps with poses:
```
T_prev (t-1): position [0.3, 0.1, 0.5], rotation R_prev
T_curr (t):   position [0.32, 0.12, 0.48], rotation R_curr
```

After relative transform with `base = T_curr`:
```
T_rel_prev = T_curr⁻¹ @ T_prev
           = [-0.02, -0.02, 0.02] in position (roughly)
           = R_curr⁻¹ @ R_prev in rotation

T_rel_curr = T_curr⁻¹ @ T_curr = Identity
           = [0, 0, 0] in position
           = Identity in rotation
```

The current pose becomes the origin `[0, 0, 0, I]`, and historical poses are expressed as displacements from current.

---

### Step 4: Convert to 10D Representation

**File**: `umi/common/pose_util.py`, lines 95-105

The 4x4 matrix is converted to a 10D vector: 3D position + 6D continuous rotation representation.

**Why 6D rotation?** The 6D rotation representation (first two columns of the rotation matrix) is continuous and avoids singularities present in Euler angles or discontinuities in quaternions.

```python
# pose_util.py, lines 95-105
def mat_to_rot6d(mat):
    """Extract first two columns of rotation matrix as 6D representation"""
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    """Convert 4x4 matrix to 10D: [x, y, z, r1, r2, r3, r4, r5, r6]"""
    pos = mat[...,:3,3]          # Extract position (3D)
    rotmat = mat[...,:3,:3]      # Extract rotation matrix
    d6 = mat_to_rot6d(rotmat)    # Convert to 6D rotation
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10
```

**6D Rotation Representation:**

```
         ┌             ┐
R₃ₓ₃ =   │ r₁  r₄  r₇ │
         │ r₂  r₅  r₈ │
         │ r₃  r₆  r₉ │
         └             ┘

6D representation = [r₁, r₂, r₃, r₄, r₅, r₆] (first two columns flattened)
```

**Usage in inference** (`real_inference_util.py`, lines 115-117):
```python
obs_pose = mat_to_pose10d(obs_pose_mat)
obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]              # Position (3D)
obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]   # Rotation (6D)
```

**Note**: Despite the key name `_eef_rot_axis_angle`, the actual data is **6D rotation**, not axis-angle. This naming is for compatibility with the training data schema.

---

### Step 5: Additional Relative Pose Computations

**File**: `umi/real_world/real_inference_util.py`, lines 119-169

Beyond the basic relative pose, additional relative representations are computed:

#### 5.1 Pose Relative to Other Robot (Bimanual)
```python
# Lines 119-147
for robot_id in range(n_robots):
    for other_robot_id in range(n_robots):
        if robot_id == other_robot_id:
            continue
        # Transform other robot's pose to this robot's frame
        tx_robota_robotb = tx_robot1_robot0  # Inter-robot transform
        if robot_id == 0:
            tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
        tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb
        
        # Compute relative pose
        rel_obs_pose_mat = convert_pose_mat_rep(
            tx_robota_tcpa,
            base_pose_mat=tx_robota_tcpb[-1],
            pose_rep='relative',
            backward=False)
```

#### 5.2 Pose Relative to Episode Start
```python
# Lines 149-169
if episode_start_pose is not None:
    for robot_id in range(n_robots):
        start_pose_mat = pose_to_mat(episode_start_pose[robot_id])
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]
```

---

### Step 6: Policy Inference

**File**: `eval_real.py`, lines 470-482

The observation dictionary (with relative poses and images) is fed to the policy.

```python
# eval_real.py, lines 470-482
with torch.no_grad():
    obs_dict_np = get_real_umi_obs_dict(
        env_obs=obs, 
        shape_meta=cfg.task.shape_meta, 
        obs_pose_repr=obs_pose_rep,           # 'relative'
        tx_robot1_robot0=tx_robot1_robot0,
        episode_start_pose=episode_start_pose)
    
    # Convert to torch tensors
    obs_dict = dict_apply(obs_dict_np, 
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    
    # Run inference
    result = policy.predict_action(obs_dict)
    raw_action = result['action_pred'][0].detach().to('cpu').numpy()
    # raw_action shape: (horizon, 10) for single robot, (horizon, 20) for bimanual
```

**Policy Input Format:**
```python
obs_dict = {
    'camera0_rgb': (1, T_img, C, H, W),           # Images
    'robot0_eef_pos': (1, T_lowdim, 3),           # Relative position
    'robot0_eef_rot_axis_angle': (1, T_lowdim, 6), # 6D relative rotation
    'robot0_gripper_width': (1, T_lowdim, 1),     # Absolute gripper width
    'robot0_eef_rot_axis_angle_wrt_start': (1, T_lowdim, 6),  # Rotation wrt episode start
    # ... additional keys for bimanual
}
```

**Policy Output Format:**
```python
# For single robot:
raw_action.shape = (horizon, 10)  # [x, y, z, r1, r2, r3, r4, r5, r6, gripper]

# For bimanual:
raw_action.shape = (horizon, 20)  # 10D per robot concatenated
```

---

### Step 7: Convert Action to Absolute Pose (Backward Transform)

**File**: `umi/real_world/real_inference_util.py`, lines 173-206

The policy outputs actions in **relative frame**. These must be converted back to **absolute poses** in the robot base frame.

```python
# real_inference_util.py, lines 173-206
def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):
    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    
    for robot_idx in range(n_robots):
        # Step 7.1: Get current pose in robot base frame
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))
        # pose_mat shape: (4, 4) - current TCP pose

        # Step 7.2: Extract action components
        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]    # Position (3D) + Rotation (6D)
        action_grip = action[..., start+9:start+10]    # Gripper width (1D)
        
        # Step 7.3: Convert 10D action to 4x4 matrix
        action_pose_mat = pose10d_to_mat(action_pose10d)
        # action_pose_mat shape: (horizon, 4, 4) - relative action poses

        # Step 7.4: Backward transform - relative to absolute
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,  # 'relative'
            backward=True)              # BACKWARD transform
        # action_mat shape: (horizon, 4, 4) - absolute poses in robot base frame

        # Step 7.5: Convert back to 6D pose
        action_pose = mat_to_pose(action_mat)
        # action_pose shape: (horizon, 6) - [x, y, z, rx, ry, rz]
        
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
    # Final shape: (horizon, 7) for single robot: [x, y, z, rx, ry, rz, gripper]
    #              (horizon, 14) for bimanual
```

**Backward Transform Mathematics:**

```python
# pose_repr_util.py, lines 94-96
elif pose_rep == 'relative':
    out = base_pose_mat @ pose_mat
    return out
```

**Mathematical formulation:**
```
T_absolute = T_current @ T_relative

Where:
- T_current: Current TCP pose in robot base frame (4x4)
- T_relative: Policy output in relative frame (4x4)
- T_absolute: Target TCP pose in robot base frame (4x4)
```

**Concrete Example:**

```
Current pose T_curr:
  Position: [0.32, 0.12, 0.48]
  Rotation: R_curr

Policy output T_rel (e.g., "move forward 5cm, rotate 10° around z"):
  Position: [0.05, 0, 0]  (relative)
  Rotation: Rz(10°)       (relative rotation)

Absolute action T_abs = T_curr @ T_rel:
  Position: R_curr @ [0.05, 0, 0] + [0.32, 0.12, 0.48]
          = [0.32 + 0.05*cos(θ), 0.12 + 0.05*sin(θ), 0.48] (approximately)
  Rotation: R_curr @ Rz(10°)
```

---

### Step 8: 10D to 4x4 Matrix Conversion (Inverse of Step 4)

**File**: `umi/common/pose_util.py`, lines 86-115

The 10D action is converted back to a 4x4 matrix.

```python
# pose_util.py, lines 86-93
def rot6d_to_mat(d6):
    """Convert 6D rotation to 3x3 rotation matrix via Gram-Schmidt"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

# pose_util.py, lines 107-115
def pose10d_to_mat(d10):
    """Convert 10D pose to 4x4 transformation matrix"""
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)  # Reconstruct rotation matrix
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out
```

**Gram-Schmidt Orthonormalization:**

The 6D representation may not be perfectly orthonormal after neural network inference. The `rot6d_to_mat` function applies Gram-Schmidt to ensure a valid rotation matrix:

```
Given 6D = [a1, a2] where a1, a2 ∈ ℝ³:

b1 = normalize(a1)
b2 = normalize(a2 - (b1·a2)b1)  # Orthogonalize
b3 = b1 × b2                     # Cross product

R = [b1 | b2 | b3]  # Valid SO(3) matrix
```

---

### Step 9: Matrix to 6D Pose Conversion

**File**: `umi/common/pose_util.py`, lines 12-33

Finally, the 4x4 matrix is converted back to a 6D pose for robot command.

```python
# pose_util.py, lines 12-22
def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()  # Convert to axis-angle
    return pose

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))
```

**Final output format:**
```
action_pose = [x, y, z, rx, ry, rz]

Where:
- x, y, z: Position in robot base frame (meters)
- rx, ry, rz: Axis-angle rotation in robot base frame (radians)
```

---

### Complete Inference Code Path Summary

```python
# 1. Get raw observation
obs = env.get_obs()
# obs['robot0_eef_pos']: (T, 3) in robot base frame
# obs['robot0_eef_rot_axis_angle']: (T, 3) axis-angle in robot base frame

# 2-4. Convert to relative 10D representation
obs_dict_np = get_real_umi_obs_dict(env_obs=obs, ...)
# obs_dict_np['robot0_eef_pos']: (T, 3) RELATIVE to current
# obs_dict_np['robot0_eef_rot_axis_angle']: (T, 6) 6D rotation RELATIVE to current

# 5. Policy inference
result = policy.predict_action(obs_dict)
raw_action = result['action_pred'][0]  # (horizon, 10) RELATIVE

# 6-9. Convert action to absolute pose
action = get_real_umi_action(raw_action, obs, action_pose_repr)
# action: (horizon, 7) = [x, y, z, rx, ry, rz, gripper] in robot base frame

# 10. Send to robot
env.exec_actions(actions=action, timestamps=...)
```

---

### Key Transformation Functions Reference

| Function | File | Input | Output | Purpose |
|----------|------|-------|--------|---------|
| `pose_to_mat` | `pose_util.py:29-30` | 6D pose `(N, 6)` | 4x4 matrix `(N, 4, 4)` | Convert pose to SE(3) |
| `mat_to_pose` | `pose_util.py:32-33` | 4x4 matrix `(N, 4, 4)` | 6D pose `(N, 6)` | Convert SE(3) to pose |
| `mat_to_pose10d` | `pose_util.py:100-105` | 4x4 matrix `(N, 4, 4)` | 10D pose `(N, 10)` | Convert to 10D (with 6D rot) |
| `pose10d_to_mat` | `pose_util.py:107-115` | 10D pose `(N, 10)` | 4x4 matrix `(N, 4, 4)` | Convert 10D to SE(3) |
| `convert_pose_mat_rep` | `pose_repr_util.py:48-111` | Pose + base + mode | Transformed pose | Relative ↔ Absolute |
| `rot6d_to_mat` | `pose_util.py:86-93` | 6D rotation `(N, 6)` | 3x3 matrix `(N, 3, 3)` | 6D to SO(3) |
| `mat_to_rot6d` | `pose_util.py:95-98` | 3x3 matrix `(N, 3, 3)` | 6D rotation `(N, 6)` | SO(3) to 6D |

---

## Pose Representation Conversion

**File**: `diffusion_policy/common/pose_repr_util.py`

The `convert_pose_mat_rep` function handles all pose representation conversions:

```python
def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    if not backward:
        # Training transform (forward pass)
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # Legacy buggy implementation (for compatibility)
            pos = pose_mat[...,:3,3] - base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ np.linalg.inv(base_pose_mat[:3,:3])
            ...
        elif pose_rep == 'relative':
            # Correct relative transformation
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == 'delta':
            # Delta between consecutive poses
            ...
    else:
        # Inference transform (backward pass)
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            pos = pose_mat[...,:3,3] + base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ base_pose_mat[:3,:3]
            ...
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            ...
```

### Supported Representations:

| `pose_rep` | Forward (Training) | Backward (Inference) |
|------------|-------------------|---------------------|
| `abs` | Identity | Identity |
| `rel` | Legacy: `pos - base_pos`, `rot @ inv(base_rot)` | Legacy: `pos + base_pos`, `rot @ base_rot` |
| `relative` | `inv(base) @ pose` | `base @ pose` |
| `delta` | Difference from previous pose | Cumulative sum |

**Recommended**: Use `relative` for proper SE(3) transformation.

---

## Configuration Reference

**File**: `diffusion_policy/config/task/umi.yaml`

```yaml
# Lines 87-89
pose_repr: &pose_repr
  obs_pose_repr: relative  # Options: abs, rel, relative, delta
  action_pose_repr: relative

# Lines 45-52 (rotation representation)
robot0_eef_rot_axis_angle:
  raw_shape: [3]
  shape: [6]  # 6D rotation representation
  ...
  rotation_rep: rotation_6d

# Lines 78-83 (action format)
action: 
  shape: [10]  # 3D pos + 6D rot + 1D gripper
  ...
  rotation_rep: rotation_6d
```

---

## Code Reference Summary

| File | Lines | Description |
|------|-------|-------------|
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | 99-108 | Camera-to-TCP transform definition |
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | 111-119 | SLAM-to-Tag transform loading |
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | 618-626 | Camera pose in tag frame |
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | 687-688 | TCP pose computation |
| `scripts_slam_pipeline/07_generate_replay_buffer.py` | 107-119 | Storing EEF data in replay buffer |
| `diffusion_policy/dataset/umi_dataset.py` | 336-364 | Training data transformation |
| `umi/real_world/bimanual_umi_env.py` | 431-439 | Raw robot observation |
| `umi/real_world/real_inference_util.py` | 62-171 | `get_real_umi_obs_dict` - Inference observation processing |
| `umi/real_world/real_inference_util.py` | 173-206 | `get_real_umi_action` - Action conversion |
| `diffusion_policy/common/pose_repr_util.py` | 48-112 | `convert_pose_mat_rep` - Core transformation function |
| `umi/common/pose_util.py` | 100-115 | `mat_to_pose10d` / `pose10d_to_mat` - 10D conversion |
| `eval_real.py` | 470-482 | Policy inference loop |
| `diffusion_policy/config/task/umi.yaml` | 87-89 | Pose representation config |

---

## Diagram: Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GoPro Camera ──► ORB_SLAM3 ──► Camera Pose (SLAM Frame)                   │
│                                        │                                    │
│                                        ▼                                    │
│                              tx_tag_slam @ tx_slam_cam                      │
│                                        │                                    │
│                                        ▼                                    │
│                              Camera Pose (Tag Frame)                        │
│                                        │                                    │
│                                        ▼                                    │
│                              tx_tag_cam @ tx_cam_tcp                        │
│                                        │                                    │
│                                        ▼                                    │
│                              TCP Pose (Tag Frame)  ──► Stored in .zarr.zip │
│                                        │                                    │
│                                        ▼                                    │
│                     inv(current_pose) @ pose_mat  (relative transform)     │
│                                        │                                    │
│                                        ▼                                    │
│                              mat_to_pose10d()                               │
│                                        │                                    │
│                                        ▼                                    │
│                     Policy Input: 3D pos + 6D rot (RELATIVE to current)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Robot Controller ──► TCP Pose (Robot Base Frame)                          │
│                                        │                                    │
│                                        ▼                                    │
│                     inv(current_pose) @ pose_mat  (relative transform)     │
│                                        │                                    │
│                                        ▼                                    │
│                              mat_to_pose10d()                               │
│                                        │                                    │
│                                        ▼                                    │
│                     Policy Input: 3D pos + 6D rot (RELATIVE to current)    │
│                                        │                                    │
│                                        ▼                                    │
│                              policy.predict_action()                        │
│                                        │                                    │
│                                        ▼                                    │
│                     Action: 3D pos + 6D rot + 1D grip (RELATIVE)           │
│                                        │                                    │
│                                        ▼                                    │
│                     current_pose @ action_mat  (backward transform)        │
│                                        │                                    │
│                                        ▼                                    │
│                     TCP Command (Robot Base Frame) ──► Robot Controller    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Frame Invariance**: The policy operates on **relative poses**, making it agnostic to the absolute coordinate frame. This is why training data (in tag frame) can be used for inference (in robot base frame).

2. **10D Representation**: The policy uses a 10-dimensional pose representation:
   - 3D position (relative to current)
   - 6D rotation (continuous rotation representation, relative to current)
   - 1D gripper width (absolute)

3. **Transformation Direction**:
   - **Training (forward)**: `relative_pose = inv(current) @ absolute_pose`
   - **Inference (backward)**: `absolute_pose = current @ relative_pose`

4. **Configuration**: The `pose_repr` in config controls the transformation type. Default is `relative` for both observation and action.
