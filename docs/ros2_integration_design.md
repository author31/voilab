# ROS2 Integration Design Documentation

## Overview

This document describes the architectural design and usage patterns for the ROS2 integration in the diffusion policy framework. The integration provides a clean, hierarchical interface for robotics tasks that bridges ROS2 communication with machine learning-based policy execution.

## Architecture Design

### Hierarchical Layer Structure

The ROS2 integration follows a strict 3-layer architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Runner Layer                             │
│  • Policy execution lifecycle                               │
│  • Episode management                                       │
│  • Results collection and metrics                           │
│  • Integration with diffusion policy framework              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Environment Layer                          │
│  • Manages subscriptions (decides what it wants)            │
│  • Processes raw sensor data into observations              │
│  • Implements business logic (rotation computation)         │
│  • Provides environment interface (reset, step, get_obs)    │
│  • Handles robot-specific configuration                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  • Generic ROS2 communication primitives                    │
│  • Topic subscription and publishing                        │
│  • Message conversion (ROS2 ↔ Python)                       │
│  • Thread-safe data access                                  │
│  • QoS profile management                                   │
│  • No business logic                                        │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### Infrastructure Layer (`ros2_infrastructure.py`)
- **Purpose**: Provide generic ROS2 communication primitives
- **Key Methods**:
  - `create_subscriber(topic, msg_type, callback, qos)` - Generic topic subscription
  - `create_publisher(topic, msg_type, qos)` - Generic topic publishing
  - `get_data(topic)` - Retrieve data from any topic
  - `publish_message(topic, data)` - Publish message to any topic
  - `wait_for_data(topics, timeout)` - Wait for data availability

#### Environment Layer (`ros2_environment.py`)
- **Purpose**: Manage subscriptions and implement robotics business logic
- **Key Methods**:
  - `_setup_subscriptions()` - Environment decides what topics to subscribe to
  - `_get_raw_sensor_data()` - Retrieve data from subscribed topics
  - `_publish_action(action)` - Handle action publishing
  - `_process_raw_observations()` - Convert raw data to structured observations
  - Standard gym interface: `reset()`, `step()`, `get_obs()`

#### Runner Layer (`ros2_runner.py`)
- **Purpose**: Handle policy execution and evaluation lifecycle
- **Key Methods**:
  - `_setup_environment()` - Environment configuration
  - `_process_observation_for_policy()` - Prepare observations for policy
  - `_execute_policy_step()` - Execute one policy step
  - `_evaluate_episode()` - Complete episode evaluation

## Key Design Principles

### 1. Environment Controls Subscriptions
The environment layer decides what data it needs and manages its own subscriptions:
```python
def _setup_subscriptions(self):
    """Set up subscriptions for all required topics."""
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32

    # Environment decides what to subscribe to
    self.infrastructure.create_subscriber(self.rgb_topic, Image)
    self.infrastructure.create_subscriber(self.joint_states_topic, JointState)
    self.infrastructure.create_subscriber(self.gripper_topic, Float32)
```

### 2. Infrastructure is Generic and Reusable
Infrastructure provides generic communication primitives without business logic:
```python
# Generic infrastructure methods work with any topic
infra.create_subscriber('/any/topic', AnyMessageType)
infra.create_publisher('/any/topic', AnyMessageType)
data = infra.get_data('/any/topic')
infra.publish_message('/any/topic', data)
```

### 3. Clear Data Flow
```
ROS2 Topics → Infrastructure (generic) → Environment (business logic) → Runner (policy) → Actions
```

## Usage Patterns

### Basic Environment Usage

```python
from diffusion_policy.environments.ros2_environment import ROS2Environment
from diffusion_policy.infrastructure.ros2_infrastructure import ROS2Manager

# Create environment with custom configuration
env = ROS2Environment(
    rgb_topic='/camera/color/image_raw',
    joint_states_topic='/joint_states',
    gripper_topic='/gripper_width',
    action_topic='/joint_commands',
    image_shape=(3, 224, 224),
    real_world=False
)

# Standard gym interface
obs = env.reset()
action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
obs, reward, done, info = env.step(action)
env.close()
```

### Policy Integration

```python
from diffusion_policy.env_runner.ros2_runner import ROS2Runner

# Create runner for policy evaluation
runner = ROS2Runner(
    output_dir='/tmp/evaluation_results',
    n_episodes=10,
    max_steps_per_episode=200,
    real_world=False,
    save_observation_data=True
)

# Run policy evaluation
results = runner.run(policy)
print(f"Success rate: {results['aggregate_metrics']['success_rate']}")
```

### Robot-Specific Configurations

#### Franka Robot
```python
franka_env = ROS2Environment(
    rgb_topic='/camera/color/image_raw',
    joint_states_topic='/joint_states',
    gripper_topic='/gripper_width',
    action_topic='/cartesian_velocity_controller/cmd_vel',
    image_shape=(3, 224, 224),
    real_world=False
)
```

#### UR5 Robot
```python
ur5_env = ROS2Environment(
    rgb_topic='/rgb/image',
    joint_states_topic='/ur5/joint_states',
    gripper_topic='/ur5/gripper_width',
    action_topic='/ur5/cmd_vel',
    image_shape=(3, 224, 224),
    real_world=False
)
```

## Observation Format

The environment provides structured observations compatible with diffusion policy:

```python
{
    "camera0_rgb": np.ndarray([3, 224, 224]),        # RGB image
    "robot0_eef_pos": np.ndarray([3]),              # End-effector position
    "robot0_eef_rot_axis_angle": np.ndarray([6]),   # Rotation (6D representation)
    "robot0_gripper_width": np.ndarray([1]),         # Gripper width
    "robot0_eef_rot_axis_angle_wrt_start": np.ndarray([6])  # Rotation relative to start
}
```

### Observation Processing

1. **Raw Data Collection**: Environment retrieves data from subscribed topics
2. **Business Logic**: Processes joint states to extract end-effector pose
3. **Rotation Computation**: Converts to 6D rotation representation
4. **Relative Transform**: Computes rotation relative to initial pose
5. **Structured Output**: Returns observations in policy-compatible format

## Action Format

Actions are 6-DOF vectors representing robot motion:
```python
action = np.array([linear_x, linear_y, linear_z, angular_x, angular_y, angular_z])
```

The environment converts actions to appropriate ROS2 messages based on the configured action topic.

## Error Handling and Robustness

### Timeout Management
```python
# Environment initialization with timeout
if not self._wait_for_initial_data(timeout):
    raise TimeoutError(f"Timeout waiting for sensor data after {timeout} seconds")
```

### Data Availability Checks
```python
# Check if all required data is available
if not self._is_data_available(raw_data):
    raise RuntimeError("Not all sensor data available for observation")
```

### Graceful Degradation
```python
# Return current observation even if action fails
return self.get_obs(), 0.0, False, {'error': str(e)}
```

## Testing and Debugging

### Layer-by-Layer Testing
Each layer can be tested independently:

1. **Infrastructure Test**: Verify generic communication primitives
2. **Environment Test**: Verify subscription management and observation processing
3. **Runner Test**: Verify policy execution lifecycle

### Example Test Pattern
```python
# Test infrastructure independently
manager = ROS2Manager()
infra = manager.initialize(node_name='test')

# Test generic capabilities
publisher = infra.create_publisher('/test_topic', Twist)
subscriber = infra.create_subscriber('/test_topic', Twist)

# Test environment independently
env = ROS2Environment(real_world=False)
shapes = env.get_observation_shapes()
```

## Performance Considerations

### Thread Safety
- Infrastructure uses thread-safe data access with locks
- Environment data retrieval is atomic and consistent

### Memory Management
- Circular buffers in infrastructure prevent memory leaks
- Proper cleanup in `close()` methods

### Real-time Considerations
- Configurable QoS profiles for different reliability requirements
- Small delays in policy execution to match real-time constraints

## Extending the Architecture

### Adding New Robot Types
1. Create robot-specific environment configuration
2. Adjust observation processing if needed
3. Update action publishing for robot-specific message types

### Adding New Sensor Types
1. Add subscription in `_setup_subscriptions()`
2. Update `_get_raw_sensor_data()` to include new sensor
3. Modify `_process_raw_observations()` to process new data

### Custom Message Types
1. Add message type support in infrastructure `_convert_message()`
2. Add message type support in infrastructure `_create_message()`
3. Update environment to handle new message format

## Migration Guide

### From Monolithic Design
1. **Separate Concerns**: Move business logic from infrastructure to environment
2. **Generic Infrastructure**: Remove hardcoded topic dependencies
3. **Environment Control**: Let environment manage its own subscriptions
4. **Test Independence**: Verify each layer works independently

### Backward Compatibility
The legacy `ros2_env.py` provides backward compatibility:
```python
# Old usage still works with deprecation warning
from diffusion_policy.environments.ros2.ros2_env import ROS2Env
env = ROS2Env(rgb_topic='/rgb', joint_states_topic='/joint_states')
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Missing ROS2 dependencies
   - Solution: Install ROS2 and required packages

2. **Timeout Errors**: No data on subscribed topics
   - Solution: Verify topics are active and publishing

3. **Message Type Errors**: Incorrect message types for topics
   - Solution: Verify message types match topic publishers

4. **Action Publishing Errors**: Wrong message format
   - Solution: Update `_publish_action()` for robot-specific requirements

### Debug Commands
```bash
# Check ROS2 topics
ros2 topic list

# Verify topic publishers
ros2 topic info /your/topic

# Monitor topic data
ros2 topic echo /your/topic
```

## Future Enhancements

### Planned Features
- Support for multiple cameras
- Depth image integration
- Force/torque sensor support
- Multi-robot environments
- Real-time performance monitoring

### Architecture Extensions
- Plugin system for robot types
- Configuration file support
- Dynamic topic discovery
- Enhanced error recovery
- Performance metrics collection

## Conclusion

The ROS2 integration architecture provides a clean, modular, and extensible framework for robotics tasks. By maintaining strict separation of concerns and clear layer responsibilities, the system enables easy testing, maintenance, and extension while supporting various robot configurations and use cases.
