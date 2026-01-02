# ROS2 Diffusion Policy Runner Notes

This document describes the ROS2 runner + environment implementation used to
execute diffusion policy inference, and captures the BaseImageRunner contract
for future extensions.

## BaseImageRunner contract
- File: `packages/diffusion_policy/src/diffusion_policy/env_runner/base_image_runner.py`
- Purpose: minimal runner abstraction used by image-based policies.
- Interface:
  - `__init__(output_dir)`: store output directory for logs and artifacts.
  - `run(policy: BaseImagePolicy) -> Dict`: required entry point for evaluation.
- Expectation: subclasses own environment lifecycle, policy reset, and the
  collection of episode metrics or outputs; `run` returns a dict of results.

## ROS2Runner (BaseImageRunner implementation)
- File: `packages/diffusion_policy/src/diffusion_policy/env_runner/ros2_runner.py`
- Role: execute a diffusion policy in a ROS2-backed environment and collect
  episode metrics.
- Environment creation:
  - Instantiates `ROS2Environment` with observation stacking (`n_obs_steps`).
  - Requires `urdf_path` to load the robot model in the environment.
- Observation processing:
  - Expects `ROS2Environment.get_obs()` to return stacked observations.
  - `camera0_rgb` is converted from `[n_steps, H, W, 3]` to
    `[1, n_steps, 3, H, W]` and cast to float tensor.
  - All other keys are treated as low-dim arrays and converted to
    `[1, n_steps, dim]` float tensors.
- Action execution:
  - Calls `policy.predict_action()` on the processed tensors.
  - Uses `action_dict["action"]` with shape `[action_horizon, action_dim]`.
  - Passes the numpy action array directly to `env.step(action)`.
- Episode loop:
  - Resets environment and policy at the start of each episode.
  - Steps until `done` or `max_steps_per_episode`.
  - Aggregates `episode_stats` and optional step data.
- Outputs:
  - Returns `episode_stats`, `total_episodes`, `avg_episode_length`,
    `success_rate`, and (optionally) `all_step_data`.

## ROS2Environment details
- File: `packages/diffusion_policy/src/diffusion_policy/environments/ros2_environment.py`
- Role: ROS2 interface layer that converts sensor messages into policy-ready
  observations and executes actions via IK + ROS2 publishers.
- ROS2 topics:
  - Subscribes to `rgb_topic`, `joint_states_topic` (Pose), and `gripper_topic`.
  - Publishes joint commands on `action_topic`.
  - Publishes gripper commands on `/gripper_command` (note: not `gripper_topic`).
- Robot model and kinematics:
  - Loads the URDF with IKPy `Chain.from_urdf_file`.
  - Uses an `active_joints_mask` for Franka Panda (7 active joints).
  - Maintains `last_joint_angles` for IK initial guess.
- Action format (critical):
  - Expects `action` shaped `[N, 10]` per step, with:
    - `0:3` position (x, y, z)
    - `3:9` rotation in 6D representation
    - `9` gripper width
  - Converts rotation 6D -> quaternion -> rotation matrix for IK target.
- Observation format:
  - `camera0_rgb`: RGB image array (normalized float, H x W x 3).
  - `robot0_eef_pos`: end-effector position.
  - `robot0_eef_rot_axis_angle`: rotation in 6D (via `mat_to_rot6d`).
  - `robot0_gripper_width`: scalar array of gripper width.
  - `robot0_eef_rot_axis_angle_wrt_start`: rotation relative to the initial pose.
- Stacking behavior:
  - Observations are appended to `obs_history`.
  - `get_obs(n_steps)` stacks the last `n_steps` entries and pads the front
    with the oldest observation if insufficient history exists.

## Practical guidance for junior developers
- Ensure `urdf_path` is valid and matches the robot used in ROS2.
- Validate topic names against your ROS2 setup; message types must match.
- Confirm policy action dimensions align with `[N, 10]` expectation.
- If observations are mis-shaped, verify:
  - ROS image encoding (`rgb8`) and normalization in `_process_raw_observations`.
  - `image_shape` (default `(3, 224, 224)`) matches the upstream camera output.
- The environment returns `done=False` and `reward=0.0` by default; success
  criteria are not implemented in this layer.
- `ROS2Runner` treats `done` as success; make sure environment or policy sets it
  if you need meaningful success rates.
