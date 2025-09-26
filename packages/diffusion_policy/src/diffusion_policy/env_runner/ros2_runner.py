import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.environments.ros2_environment import ROS2Environment, ROS2EnvironmentFactory


class ROS2Runner(BaseImageRunner):
    """
    Runner for executing diffusion policies in ROS2 environments.

    This runner handles the complete evaluation lifecycle:
    - Environment setup and management
    - Policy execution and observation processing
    - Episode evaluation and metrics collection
    - Results saving and logging
    """

    def __init__(self,
                 output_dir: str,
                 n_episodes: int = 10,
                 max_steps_per_episode: int = 200,
                 real_world: bool = False,
                 env_config: Optional[Dict] = None,
                 save_video: bool = False,
                 save_observation_data: bool = False):
        """
        Initialize ROS2 runner.

        Args:
            output_dir: Directory to save results
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            real_world: Whether to use real-world environment
            env_config: Additional environment configuration
            save_video: Whether to save video recordings
            save_observation_data: Whether to save observation data
        """
        super().__init__(output_dir)
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.real_world = real_world
        self.env_config = env_config or {}
        self.save_video = save_video
        self.save_observation_data = save_observation_data

        # Initialize environment
        self.env = None
        self._setup_environment()

        # Initialize results storage
        self.episode_results = []
        self.current_episode_data = []

    def _setup_environment(self):
        """Setup ROS2 environment with proper configuration."""
        env_config = {
            'real_world': self.real_world,
            **self.env_config
        }

        # Create environment using factory
        if 'robot_type' in env_config:
            robot_type = env_config['robot_type']
            if robot_type == 'franka':
                self.env = ROS2EnvironmentFactory.create_franka_environment(
                    real_world=self.real_world
                )
            elif robot_type == 'ur5':
                self.env = ROS2EnvironmentFactory.create_ur5_environment(
                    real_world=self.real_world
                )
            else:
                self.env = ROS2EnvironmentFactory.create_custom_environment(
                    real_world=self.real_world
                )
        else:
            self.env = ROS2EnvironmentFactory.create_default_environment(
                real_world=self.real_world
            )

        print(f"ROS2 Environment created: {type(self.env).__name__}")

    def _process_observation_for_policy(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Process environment observation for policy input.

        Args:
            obs: Raw observation from environment

        Returns:
            Processed observation ready for policy
        """
        policy_obs = {}

        # Process RGB image
        if 'camera0_rgb' in obs:
            # Add batch dimension and convert to tensor
            rgb_img = obs['camera0_rgb']
            policy_obs['camera0_rgb'] = torch.from_numpy(rgb_img).float().unsqueeze(0)

        # Process low-dimensional observations
        for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
            if key in obs:
                obs_data = obs[key]
                policy_obs[key] = torch.from_numpy(obs_data).float().unsqueeze(0)

        return policy_obs

    def _execute_policy_step(self, policy: BaseImagePolicy, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Execute one policy step.

        Args:
            policy: Policy to execute
            obs: Current observation

        Returns:
            Action predicted by policy
        """
        # Process observation for policy
        policy_obs = self._process_observation_for_policy(obs)

        # Get policy prediction
        with torch.no_grad():
            policy_output = policy.predict_action(policy_obs)

        # Extract action from policy output
        if isinstance(policy_output, dict):
            action = policy_output['action'].cpu().numpy()[0]  # Remove batch dimension
        else:
            action = policy_output.cpu().numpy()[0]

        return action

    def _evaluate_episode(self, policy: BaseImagePolicy, episode_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single episode.

        Args:
            policy: Policy to evaluate
            episode_idx: Episode index

        Returns:
            Episode results dictionary
        """
        print(f"Starting episode {episode_idx + 1}/{self.n_episodes}")

        # Reset environment
        obs = self.env.reset()

        episode_data = []
        total_reward = 0.0
        steps = 0
        success = False

        try:
            for step in range(self.max_steps_per_episode):
                # Execute policy step
                action = self._execute_policy_step(policy, obs)

                # Execute action in environment
                obs, reward, done, info = self.env.step(action)

                # Store step data
                step_data = {
                    'step': step,
                    'observation': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items()},
                    'action': action.tolist(),
                    'reward': reward,
                    'done': done,
                    'info': info
                }
                episode_data.append(step_data)

                if self.save_observation_data:
                    # Store raw observation data
                    step_data['raw_obs'] = {k: v for k, v in obs.items()}

                total_reward += reward
                steps += 1

                if done:
                    success = True
                    print(f"Episode {episode_idx + 1} completed successfully at step {steps}")
                    break

                # Small delay to match real-time execution
                time.sleep(0.01)

            if not done:
                print(f"Episode {episode_idx + 1} reached max steps ({self.max_steps_per_episode})")

        except Exception as e:
            print(f"Error in episode {episode_idx + 1}: {e}")
            info['error'] = str(e)

        # Compile episode results
        episode_results = {
            'episode_idx': episode_idx,
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'average_reward': total_reward / max(steps, 1),
            'data': episode_data if self.save_observation_data else None,
            'error': info.get('error', None)
        }

        return episode_results

    def _save_results(self, results: List[Dict[str, Any]]):
        """Save evaluation results to files."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save summary results
        summary = {
            'n_episodes': self.n_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'real_world': self.real_world,
            'episodes': results,
            'aggregate_metrics': self._compute_aggregate_metrics(results)
        }

        summary_path = os.path.join(self.output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save individual episode data if requested
        if self.save_observation_data:
            episodes_dir = os.path.join(self.output_dir, 'episode_data')
            os.makedirs(episodes_dir, exist_ok=True)

            for episode_result in results:
                if episode_result['data'] is not None:
                    episode_file = os.path.join(
                        episodes_dir,
                        f'episode_{episode_result["episode_idx"]:04d}.json'
                    )
                    with open(episode_file, 'w') as f:
                        json.dump(episode_result['data'], f, indent=2)

        print(f"Results saved to {self.output_dir}")

    def _compute_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate metrics across all episodes."""
        if not results:
            return {}

        metrics = {
            'total_episodes': len(results),
            'successful_episodes': sum(1 for r in results if r['success']),
            'success_rate': sum(1 for r in results if r['success']) / len(results),
            'average_reward': np.mean([r['total_reward'] for r in results]),
            'average_steps': np.mean([r['steps'] for r in results]),
            'std_reward': np.std([r['total_reward'] for r in results]),
            'max_reward': max([r['total_reward'] for r in results]),
            'min_reward': min([r['total_reward'] for r in results]),
        }

        return metrics

    def run(self, policy: BaseImagePolicy) -> Dict[str, Any]:
        """
        Run policy evaluation in ROS2 environment.

        Args:
            policy: Policy to evaluate

        Returns:
            Evaluation results dictionary
        """
        print(f"Starting policy evaluation in ROS2 environment")
        print(f"Configuration: {self.n_episodes} episodes, max {self.max_steps_per_episode} steps each")

        all_results = []

        try:
            # Evaluate each episode
            for episode_idx in range(self.n_episodes):
                episode_result = self._evaluate_episode(policy, episode_idx)
                all_results.append(episode_result)

                # Print progress
                success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
                avg_reward = np.mean([r['total_reward'] for r in all_results])
                print(f"Progress: {episode_idx + 1}/{self.n_episodes}, "
                      f"Success rate: {success_rate:.2f}, Avg reward: {avg_reward:.2f}")

            # Save results
            self._save_results(all_results)

            # Compute final metrics
            aggregate_metrics = self._compute_aggregate_metrics(all_results)

            print("\n=== Evaluation Results ===")
            print(f"Success rate: {aggregate_metrics['success_rate']:.2%}")
            print(f"Average reward: {aggregate_metrics['average_reward']:.2f}")
            print(f"Average steps: {aggregate_metrics['average_steps']:.1f}")
            print(f"Successful episodes: {aggregate_metrics['successful_episodes']}/{aggregate_metrics['total_episodes']}")

            return {
                'episodes': all_results,
                'aggregate_metrics': aggregate_metrics,
                'output_dir': self.output_dir
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                'error': str(e),
                'episodes': all_results,
                'aggregate_metrics': self._compute_aggregate_metrics(all_results) if all_results else {}
            }

        finally:
            # Clean up environment
            if self.env:
                self.env.close()

    def close(self):
        """Clean up runner resources."""
        if self.env:
            self.env.close()
            self.env = None


def create_ros2_runner(output_dir: str,
                      n_episodes: int = 10,
                      max_steps_per_episode: int = 200,
                      real_world: bool = False,
                      **kwargs) -> ROS2Runner:
    """
    Convenience function to create ROS2 runner with common configuration.

    Args:
        output_dir: Output directory for results
        n_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
        real_world: Real-world flag
        **kwargs: Additional runner configuration

    Returns:
        Configured ROS2Runner instance
    """
    return ROS2Runner(
        output_dir=output_dir,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        real_world=real_world,
        **kwargs
    )
