# ROS 2 integration design

The intended three-layer ROS 2 stack in `packages/diffusion_policy`, the real signatures behind it, and how far the code actually gets.

**Read this if:** you are reading the ROS 2 classes in the training package, or planning to make policy rollout on a real arm work.

**Before you start:** [Simulation and Docker](./simulation-and-docker.md) — `rclpy` and `cv_bridge` are declared in no `pyproject.toml`, so nothing on this page imports outside the `voilab-workspace` container.

---

> **Status: partly-working code with a design document wrapped around it.** The three layers exist
> and `ROS2Runner` is reachable from `eval.py`, but no pipeline stage or simulation script depends
> on this architecture, and several pieces below are broken. The defects are listed at the end of
> this page. Treat the layer split as the target design and the signatures as the current fact.

Every path below is relative to `packages/diffusion_policy/src/diffusion_policy/`.

## The three layers

```text
Runner        env_runner/ros2_runner.py        episode loop, policy inference, metrics
   |
Environment   environments/ros2_environment.py observations, inverse kinematics, action publishing
   |
Infrastructure infrastructure/ros2_infrastructure.py  subscriptions, publishers, thread-safe storage
```

The intent is that infrastructure holds no robot knowledge, the environment decides what it needs,
and the runner only sees a gym-like interface. Two places break that intent today; see
[Where design and code diverge](#where-design-and-code-diverge).

### Infrastructure layer

`ROS2Infrastructure` subclasses `rclpy.node.Node`, so it inherits `create_publisher` and
`create_subscription`. Its own wrappers are:

| Method | Signature (`ros2_infrastructure.py`) |
|---|---|
| `create_subscriber` | `(topic, msg_type, callback=None, qos_profile=None) -> str` (`:56`) |
| `add_publisher` | `(topic, msg_type, qos_profile=None)` (`:102`) |
| `get_data` | `(topic) -> Optional[dict]` (`:157`) |
| `publish_message` | `(topic, msg, msg_type=None)` (`:174`) |
| `wait_for_data` | `(topics, timeout=5.0) -> bool` (`:246`) |

`add_publisher` is deliberately named to avoid colliding with `Node.create_publisher`, whose
argument order is `(msg_type, topic, qos_profile)` — reversed relative to these wrappers.
`publish_message` takes a constructed ROS 2 message, not a dict, and on the first call for a topic
`msg_type` must be given or the publisher is created with `None` and the publish silently no-ops
(`:188`, `:195-196`).

`ROS2Manager` (`:349`) owns the node and the spin thread: `manager.initialize(node_name=...)`
returns the infrastructure object (`:358`).

### Environment layer

`ROS2Environment.__init__` (`environments/ros2_environment.py:25-34`):

```python
env = ROS2Environment(
    urdf_path='assets/franka_panda/franka_panda.urdf',   # required
    rgb_topic='/rgb',
    joint_states_topic='/eef_states',   # subscribes: end-effector Pose
    gripper_topic='/gripper_width',     # subscribes: Float64
    action_topic='/joint_states',       # publishes: JointState
    image_shape=(3, 224, 224),
    timeout=5.,
    n_obs_steps=2,
)
```

Note the direction of the topic names: `/eef_states` is an input and `/joint_states` is the
*output*. The constructor is not side-effect free — it loads the URDF into an ikpy chain, creates
the subscriptions, and then blocks in `_wait_for_initial_data`, raising `TimeoutError` after
`timeout` seconds if no publisher is live (`:83-87`). The environment layer cannot be exercised
without live topics.

Subscriptions are created by the environment, with a conversion callback as the third argument
(`:91-103`):

```python
self.infrastructure.create_subscriber(self.rgb_topic, Image, self._convert_message)
self.infrastructure.create_subscriber(self.joint_states_topic, Pose, self._convert_message)
self.infrastructure.create_subscriber(self.gripper_topic, Float64, self._convert_message)
```

That callback is load-bearing. Without it `_handle_message` stores `{'raw_message': msg, 'type':
'raw'}` (`ros2_infrastructure.py:138-141`), and `_process_raw_observations` then fails looking for
`data` / `position` / `orientation` keys that were never produced.

### Runner layer

`ROS2Runner.__init__` (`env_runner/ros2_runner.py:31-45`) requires `output_dir`, `shape_meta` and
`urdf_path`; the rest (`n_episodes=10`, `max_steps_per_episode=200`, `save_video`,
`save_observation_data`, `obs_latency_steps`, `n_obs_steps=2`, `n_action_steps=1`, `pose_repr`)
have defaults. It builds the environment eagerly at `:64` and asserts `rotation_rep` is present in
`shape_meta['action']` at `:103`.

Its methods are `__init__`, `_process_observation_for_policy` (`:114`), `run` (`:140`) and `close`
(`:206`), plus a module-level `create_ros2_runner` factory (`:213`). The episode loop — reset,
`predict_action`, `step`, metrics — is inlined in `run()`.

In practice the runner is constructed by `eval.py`, which injects the two required arguments
(`eval.py:58-62`):

```bash
uv run packages/diffusion_policy/eval.py \
  --checkpoint /path/to/latest.ckpt \
  --output_dir /path/to/eval_out \
  --env_runner_path diffusion_policy.env_runner.ros2_runner.ROS2Runner \
  --urdf_path assets/franka_panda/franka_panda.urdf
```

`run()` returns a flat dict — `episode_stats`, `total_episodes`, `avg_episode_length`,
`success_rate`, plus `all_step_data` when `save_observation_data` is set (`:194-202`). There is no
`aggregate_metrics` nesting.

## Observation format

`get_obs()` returns observations already stacked over `n_obs_steps` (`:236-268`):

```python
{
    "camera0_rgb":                        np.ndarray,  # [n_obs_steps, H, W, 3], float in [0,1]
    "robot0_eef_pos":                     np.ndarray,  # [n_obs_steps, 3]
    "robot0_eef_rot_axis_angle":          np.ndarray,  # [n_obs_steps, 6]  6-D rotation
    "robot0_gripper_width":               np.ndarray,  # [n_obs_steps, 1]
    "robot0_eef_rot_axis_angle_wrt_start":np.ndarray,  # [n_obs_steps, 6]  relative to first pose
}
```

The image is HWC as it leaves the environment; `ROS2Runner._process_observation_for_policy`
transposes it to `[n_obs_steps, 3, H, W]` and adds the batch axis (`ros2_runner.py:128-131`).
Despite the name, `robot0_eef_rot_axis_angle` holds a 6-D rotation, computed with `mat_to_rot6d`
(`:406`, `:420`) — the same convention the dataset uses ([Data formats](./data-formats.md)).

## Action format

Actions are **not** velocity twists. `step()` takes `(Ta, 10)` and rejects anything else with
`ValueError` (`:177-178`):

```python
# one row: [x, y, z, rot_6d(6), gripper_width]
obs, reward, done, info = env.step(action)   # action.shape == (n_action_steps, 10)
```

Per row the environment converts the 6-D rotation to a quaternion, solves inverse kinematics
against the URDF chain with ikpy (`:201-206`), publishes the seven active joint angles as a
`sensor_msgs/JointState` on `action_topic` (`:270-282`), and publishes the gripper width as a
`std_msgs/Float64` on the hard-coded topic `/gripper_command` (`:293`). `reward` is always `0.0`
and `done` always `False`.

## Robot support

`ROS2Environment` is Franka Panda specific in four hard-coded places: the neutral joint pose
(`:59-67`), the IK chain root `base_elements=["panda_link0"]` (`:122`), the nine-element
`active_joints_mask` (`:118`), and the published joint names `panda_joint1 … panda_joint7`
(`:276-279`). Supporting another arm means changing those, not only passing a different
`urdf_path`; a UR5 or any other arm is work to be done, not a configuration.

## Where design and code diverge

- **`ROS2EnvironmentFactory` does not exist.** `environments/ros2/ros2_env.py:16` and
  `examples/ros2_integration_example.py:20` import it from `environments/ros2_environment.py`,
  which defines only `ROS2Environment`. Both modules fail at import; the legacy `ROS2Env` shim is
  unusable and would in any case forward a `real_world` argument the constructor does not accept.
- **`create_ros2_runner` accepts a `real_world` argument** (`env_runner/ros2_runner.py:216`) that
  it silently drops; `ROS2Runner` has no such parameter. It passes nothing for the required
  `shape_meta` and `urdf_path` either (`:231-235`), so those have to arrive through `**kwargs` or
  the call raises `TypeError`. Construct `ROS2Runner` directly, or go through `eval.py`.
- **`get_observation_shapes()` (`:512`) disagrees with `get_obs()`** — it advertises
  `robot0_eef_quat` and omits both `robot0_eef_rot_axis_angle` keys.
- **`obs_history` grows without bound.** It is a plain list appended on every `get_obs()` call
  (`:73`, `:262`) while only the last `n_obs_steps` entries are read. Infrastructure storage, by
  contrast, is single-slot: one latest payload per topic, overwritten in place
  (`ros2_infrastructure.py:42`, `:144-145`).
- **The layer split leaks both ways.** `_convert_message` writes through three private members of
  the infrastructure object (`:341-343`), and `ROS2Infrastructure.shutdown` publishes a zero
  `Twist` to any publisher whose name contains `cmd_vel` (`ros2_infrastructure.py:329-331`) — a
  robot-control decision in the layer that is meant to hold none.
- **`reset()` and `close()` send `np.zeros(6)`** (`:146`, `:469`), which fails the `(N, 10)` shape
  check; the blanket `except` at `:231-234` swallows it, so the robot is never actually stopped.
- **`_publish_joint_states` calls `.tolist()`** on a Python list (`:281`), which raises
  `AttributeError` inside the same swallowed `try`.

## Debugging

```bash
ros2 topic list
ros2 topic info /rgb
ros2 topic echo /eef_states
```

A `TimeoutError` during construction means one of the three subscribed topics has no publisher. An
`{'error': ...}` value in the `info` dict returned by `step()` means the action was rejected or the
IK solve failed; the exception text is the only signal.

---

**Next:** [Simulation and Docker](./simulation-and-docker.md) · [Data formats](./data-formats.md) · [Known issues](./known-issues.md)
