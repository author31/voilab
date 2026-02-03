#!/usr/bin/env python3
"""
ROS2 Integration Example

This example demonstrates the proper hierarchical architecture:
1. Infrastructure layer handles ROS2 communication
2. Environment layer handles observation processing and business logic
3. Runner layer handles policy execution and evaluation

Usage:
    python ros2_integration_example.py
"""

import time
import numpy as np
from typing import Dict

# Import new architecture components
from diffusion_policy.infrastructure.ros2_infrastructure import ROS2Infrastructure, ROS2Manager
from diffusion_policy.environments.ros2_environment import ROS2Environment, ROS2EnvironmentFactory
from diffusion_policy.env_runner.ros2_runner import ROS2Runner, create_ros2_runner


class MockPolicy:
    """Mock policy for testing purposes."""

    def predict_action(self, obs_dict: Dict) -> Dict:
        """Mock action prediction."""
        # Return a simple forward movement action
        batch_size = next(iter(obs_dict.values())).shape[0]
        action = np.zeros((batch_size, 6))
        action[:, 0] = 0.1  # Move forward
        return {'action': action}


def test_infrastructure_layer():
    """Test the infrastructure layer independently."""
    print("=== Testing Infrastructure Layer ===")

    try:
        # Create manager and infrastructure
        manager = ROS2Manager()
        infra = manager.initialize(node_name='test_infrastructure')

        print(f"Infrastructure created: {type(infra).__name__}")

        # Test generic infrastructure capabilities
        print(f"Infrastructure initialized successfully")
        print(f"Node name: {infra.get_name()}")

        # Test generic publisher creation
        from geometry_msgs.msg import Twist
        publisher = infra.create_publisher('/test_cmd_vel', Twist)
        print(f"Generic publisher created: {type(publisher).__name__}")

        # Cleanup
        manager.shutdown()
        print("Infrastructure layer test completed successfully!\n")

    except Exception as e:
        print(f"Infrastructure layer test failed: {e}\n")


def test_environment_layer():
    """Test the environment layer independently."""
    print("=== Testing Environment Layer ===")

    try:
        # Create environment using factory
        env = ROS2EnvironmentFactory.create_default_environment(real_world=False)

        print(f"Environment created: {type(env).__name__}")
        print(f"Environment manages subscriptions: {hasattr(env, '_setup_subscriptions')}")

        # Test observation shapes (without requiring ROS2 data)
        shapes = env.get_observation_shapes()
        print(f"Observation shapes: {shapes}")

        # Test action space
        action_space = env.action_space
        print(f"Action space: {action_space}")

        # Test subscription management (without requiring ROS2)
        print(f"RGB topic: {env.rgb_topic}")
        print(f"Joint states topic: {env.joint_states_topic}")
        print(f"Gripper topic: {env.gripper_topic}")
        print(f"Action topic: {env.action_topic}")

        # Cleanup
        env.close()
        print("Environment layer test completed successfully!\n")

    except Exception as e:
        print(f"Environment layer test failed: {e}\n")


def test_runner_layer():
    """Test the runner layer with mock policy."""
    print("=== Testing Runner Layer ===")

    try:
        # Create mock policy
        policy = MockPolicy()

        # Create runner
        runner = create_ros2_runner(
            output_dir='/tmp/ros2_test',
            n_episodes=2,  # Short test
            max_steps_per_episode=10,  # Very short test
            real_world=False,
            save_observation_data=True
        )

        print(f"Runner created: {type(runner).__name__}")

        # Run evaluation
        results = runner.run(policy)

        print("Evaluation completed!")
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            print(f"Success rate: {metrics.get('success_rate', 'N/A')}")
            print(f"Average reward: {metrics.get('average_reward', 'N/A')}")

        # Cleanup
        runner.close()
        print("Runner layer test completed successfully!\n")

    except Exception as e:
        print(f"Runner layer test failed: {e}\n")


def test_layered_architecture():
    """Test the complete layered architecture."""
    print("=== Testing Complete Layered Architecture ===")

    try:
        # Step 1: Infrastructure layer
        print("Step 1: Creating infrastructure...")
        manager = ROS2Manager()
        infra = manager.initialize(node_name='layered_test')

        # Step 2: Environment layer
        print("Step 2: Creating environment...")
        env = ROS2Environment(manager=manager, real_world=False)

        # Step 3: Test environment configuration
        print("Step 3: Testing environment configuration...")
        print(f"Environment uses infrastructure: {env.infrastructure is infra}")
        print(f"Environment manages subscriptions: {hasattr(env, '_setup_subscriptions')}")

        # Test observation shapes (without requiring ROS2 data)
        shapes = env.get_observation_shapes()
        print(f"Environment provides observation shapes: {len(shapes)} components")

        # Step 4: Test action interface
        print("Step 4: Testing action interface...")
        action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"Action format compatible: {action.shape == (6,)}")

        # Step 5: Cleanup
        print("Step 5: Cleaning up...")
        env.close()

        print("Complete layered architecture test completed successfully!\n")

    except Exception as e:
        print(f"Layered architecture test failed: {e}\n")


def demonstrate_separation_of_concerns():
    """Demonstrate the separation of concerns in the new architecture."""
    print("=== Demonstrating Separation of Concerns ===")

    print("1. INFRASTRUCTURE LAYER RESPONSIBILITIES:")
    print("   - ROS2 node management")
    print("   - Topic subscription (/rgb, /joint_states, /gripper)")
    print("   - Topic publishing (/cmd_vel)")
    print("   - Message conversion (ROS2 <-> Python)")
    print("   - Thread-safe data access")
    print("   - QoS profile management")
    print()

    print("2. ENVIRONMENT LAYER RESPONSIBILITIES:")
    print("   - Observation processing and structuring")
    print("   - Business logic (rotation computation, relative poses)")
    print("   - Environment interface (reset, step, get_obs)")
    print("   - Robot-specific logic")
    print("   - Error handling and recovery")
    print()

    print("3. RUNNER LAYER RESPONSIBILITIES:")
    print("   - Policy execution lifecycle")
    print("   - Episode management")
    print("   - Results collection and metrics")
    print("   - Data saving and logging")
    print("   - Integration with diffusion policy framework")
    print()

    print("4. SEPARATION BENEFITS:")
    print("   ✓ Infrastructure can be tested independently")
    print("   ✓ Environment logic can be modified without changing infrastructure")
    print("   ✓ Runner can be reused with different environments")
    print("   ✓ Clear responsibilities make code maintainable")
    print("   ✓ Business logic separated from communication concerns")
    print()


def main():
    """Main function to run all tests."""
    print("ROS2 Integration Architecture Example")
    print("=" * 50)
    print()

    # Demonstrate the architecture concepts
    demonstrate_separation_of_concerns()

    # Test each layer independently
    print("Running layer-by-layer tests...\n")

    # Note: These tests require ROS2 to be running with appropriate topics
    # They will fail if ROS2 is not available, but demonstrate the architecture

    test_infrastructure_layer()
    test_environment_layer()
    test_runner_layer()
    test_layered_architecture()

    print("All tests completed!")
    print("\nNote: Some tests may fail if ROS2 is not running or topics are not available.")
    print("The architecture is designed to work when ROS2 environment is properly set up.")


if __name__ == '__main__':
    main()