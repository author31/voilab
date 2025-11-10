# ROS2 Threading Test Plan

## Overview
This test plan validates the threaded implementation of ROS2Manager and ROS2Environment to ensure non-blocking initialization and proper threading behavior.

## Test Objectives
1. Verify ROS2Environment initialization no longer blocks execution
2. Validate threaded ROS2Manager functionality
3. Test proper shutdown and cleanup
4. Ensure ROS2 callbacks work correctly in threaded environment

## Test Environment
- Python 3.10+
- ROS2 Humble/HawkEye
- Diffusion Policy package dependencies
- Test location: `packages/diffusion_policy/tests/`

## Test Cases

### Test 1: Non-blocking Initialization Test
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_non_blocking_initialization`

```python
def test_non_blocking_initialization():
    """Test that ROS2Environment initializes within reasonable time."""
    import time
    start_time = time.time()
    env = ROS2Environment(timeout=2.0)
    init_time = time.time() - start_time

    # Should initialize within 5 seconds (not block indefinitely)
    assert init_time < 5.0, f"Initialization took too long: {init_time}s"
    env.close()
```

### Test 2: Threaded Manager Basic Operations
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_threaded_manager_basic_ops`

```python
def test_threaded_manager_basic_ops():
    """Test basic operations of threaded ROS2Manager."""
    manager = ROS2Manager()
    infra = manager.initialize()

    # Verify infrastructure is created
    assert infra is not None
    assert manager.is_initialized == True
    assert manager.spin_thread is not None
    assert manager.spin_thread.is_alive() == True

    manager.shutdown()
    assert manager.is_initialized == False
```

### Test 3: Environment Operations Test
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_environment_operations`

```python
def test_environment_operations():
    """Test basic environment operations work with threading."""
    env = ROS2Environment(timeout=2.0)

    # Test getting observation
    obs = env.get_obs()
    assert isinstance(obs, dict)
    assert "camera0_rgb" in obs

    # Test step function
    action = np.zeros(6)
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Test reset
    obs = env.reset()
    assert isinstance(obs, dict)

    env.close()
```

### Test 4: Multiple Environments Test
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_multiple_environments`

```python
def test_multiple_environments():
    """Test creating multiple environments with shared manager."""
    manager = ROS2Manager()
    env1 = ROS2Environment(manager=manager, timeout=2.0)
    env2 = ROS2Environment(manager=manager, timeout=2.0)

    # Both environments should work with same manager
    assert env1.manager is env2.manager

    env1.close()
    env2.close()
    manager.shutdown()
```

### Test 5: Stress Test - Rapid Create/Destroy
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_rapid_create_destroy`

```python
def test_rapid_create_destroy():
    """Test rapid creation and destruction of environments."""
    for i in range(5):
        env = ROS2Environment(timeout=1.0)
        obs = env.get_obs()
        env.close()
```

### Test 6: Callback Thread Safety Test
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_callback_thread_safety`

```python
def test_callback_thread_safety():
    """Test that callbacks are thread-safe."""
    import threading
    callback_data = []

    def test_callback(topic, data):
        callback_data.append((threading.current_thread().name, topic))

    env = ROS2Environment(timeout=2.0)
    env.infrastructure.register_callback(env.rgb_topic, test_callback)

    # Wait for some callbacks
    time.sleep(1.0)

    # Verify callbacks were received
    assert len(callback_data) > 0

    env.close()
```

### Test 7: Shutdown Properly Test
**File**: `packages/diffusion_policy/tests/test_ros2_threading.py`
**Function**: `test_proper_shutdown`

```python
def test_proper_shutdown():
    """Test that shutdown is properly handled."""
    env = ROS2Environment(timeout=2.0)

    # Get thread ID before shutdown
    thread_id_before = env.manager.spin_thread.ident

    env.close()

    # Verify thread is properly shut down
    assert env.manager.is_initialized == False
    assert not env.manager.spin_thread.is_alive()
```

## Test Infrastructure Setup

### Required Mocking
Since these are unit tests, we'll need to mock ROS2 functionality:

```python
import unittest.mock as mock

@mock.patch('rclpy.init')
@mock.patch('rclpy.ok')
@mock.patch('rclpy.shutdown')
def test_with_mocked_ros2(mock_shutdown, mock_ok, mock_init):
    """Test with mocked ROS2 functions."""
    mock_ok.return_value = True

    # Test code here
```

### Test Dependencies
Create `packages/diffusion_policy/tests/conftest.py`:

```python
import pytest
import unittest.mock as mock
import numpy as np


@pytest.fixture
def mock_ros2():
    """Mock ROS2 functionality for testing."""
    with mock.patch('rclpy.init'), \
         mock.patch('rclpy.ok', return_value=True), \
         mock.patch('rclpy.shutdown'), \
         mock.patch('diffusion_policy.infrastructure.ros2_infrastructure.MultiThreadedExecutor') as mock_executor:

        # Mock the executor
        mock_executor_instance = mock.Mock()
        mock_executor.return_value = mock_executor_instance

        yield {
            'executor': mock_executor_instance,
            'executor_instance': mock_executor_instance
        }
```

## Test Execution

### Running Tests
```bash
# Run all ROS2 threading tests
cd packages/diffusion_policy
python -m pytest tests/test_ros2_threading.py -v

# Run specific test
python -m pytest tests/test_ros2_threading.py::test_non_blocking_initialization -v

# Run with coverage
python -m pytest tests/test_ros2_threading.py --cov=diffusion_policy.infrastructure.ros2_infrastructure
```

### Integration Test Script
Create `packages/diffusion_policy/tests/integration_test_ros2_threading.py`:

```python
#!/usr/bin/env python3
"""
Integration test for ROS2 threading functionality.
This test requires actual ROS2 environment running.
"""

import time
import numpy as np
from diffusion_policy.environments.ros2_environment import ROS2Environment


def integration_test_ros2_threading():
    """Integration test requiring actual ROS2."""
    print("Starting ROS2 threading integration test...")

    start_time = time.time()
    env = ROS2Environment(timeout=10.0)
    init_time = time.time() - start_time

    print(f"Initialization time: {init_time:.2f} seconds")

    # Test operations
    obs = env.get_obs()
    print(f"Observation keys: {list(obs.keys())}")

    action = np.zeros(6)
    obs, reward, done, info = env.step(action)
    print(f"Step completed: reward={reward}")

    obs = env.reset()
    print("Reset completed")

    env.close()
    print("Integration test completed successfully!")


if __name__ == "__main__":
    integration_test_ros2_threading()
```

## Success Criteria
1. All unit tests pass with mocked ROS2
2. Integration test passes with real ROS2 (when available)
3. Initialization completes within 5 seconds
4. No threading-related race conditions
5. Proper cleanup and shutdown verified
6. Multiple environments work correctly

## Known Issues and Considerations
1. Real ROS2 testing requires ROS2 environment to be running
2. Some tests may need to be skipped in CI/CD without ROS2
3. Thread timing can vary, tests should account for this
4. Memory usage should be monitored for rapid create/destroy cycles

## Future Enhancements
1. Add performance benchmarks
2. Test with different ROS2 QoS profiles
3. Add network failure simulation tests
4. Test with different message types and frequencies