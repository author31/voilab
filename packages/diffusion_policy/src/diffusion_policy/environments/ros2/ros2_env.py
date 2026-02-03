#!/usr/bin/env python3
"""
Legacy ROS2 Environment - DEPRECATED

This file is kept for backward compatibility.
New code should use:
- diffusion_policy.environments.ros2_environment.ROS2Environment
- diffusion_policy.infrastructure.ros2_infrastructure.ROS2Infrastructure

Example migration:
OLD: from diffusion_policy.environments.ros2.ros2_env import ROS2Env
NEW: from diffusion_policy.environments.ros2_environment import ROS2Environment
"""

import warnings
from diffusion_policy.environments.ros2_environment import ROS2Environment, ROS2EnvironmentFactory


def ROS2Env(*args, **kwargs):
    """
    DEPRECATED: Use ROS2Environment instead.

    This is a compatibility wrapper for the new architecture.
    """
    warnings.warn(
        "ROS2Env is deprecated. Use ROS2Environment from "
        "diffusion_policy.environments.ros2_environment instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old parameters to new environment
    env_kwargs = {}
    if 'rgb_topic' in kwargs:
        env_kwargs['rgb_topic'] = kwargs['rgb_topic']
    if 'joint_states_topic' in kwargs:
        env_kwargs['joint_states_topic'] = kwargs['joint_states_topic']
    if 'gripper_topic' in kwargs:
        env_kwargs['gripper_topic'] = kwargs['gripper_topic']
    if 'action_topic' in kwargs:
        env_kwargs['action_topic'] = kwargs['action_topic']
    if 'image_shape' in kwargs:
        env_kwargs['image_shape'] = kwargs['image_shape']
    if 'timeout' in kwargs:
        env_kwargs['timeout'] = kwargs['timeout']
    if 'real_world' in kwargs:
        env_kwargs['real_world'] = kwargs['real_world']

    return ROS2Environment(**env_kwargs)


def main():
    """Example usage demonstrating backward compatibility."""
    print("This script demonstrates backward compatibility with the deprecated ROS2Env.")
    print("New code should use ROS2Environment from ros2_environment module.")
    print()

    try:
        # This still works but shows deprecation warning
        from diffusion_policy.environments.ros2.ros2_env import ROS2Env as DeprecatedROS2Env

        # Create environment using deprecated interface
        env = DeprecatedROS2Env(
            rgb_topic='/rgb',
            joint_states_topic='/joint_states',
            gripper_topic='/gripper',
            action_topic='/cmd_vel',
            real_world=False
        )

        print("Created environment using deprecated interface (for testing)")
        obs = env.get_obs()
        print(f"Observation keys: {list(obs.keys())}")

        # Clean up
        env.close()

    except ImportError as e:
        print(f"Import error (expected if ROS2 dependencies not available): {e}")
    except Exception as e:
        print(f"Other error (expected if no ROS2 topics running): {e}")


if __name__ == '__main__':
    main()
