import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.environments.ros2_environment import ROS2Environment
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class ROS2Runner(BaseImageRunner):
    """
    Runner for executing diffusion policies in ROS2 environments.

    This runner handles the complete evaluation lifecycle:
    - Environment setup and management
    - Policy execution and observation processing
    - Episode evaluation and metrics collection
    - Results saving and logging
    """

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict,
        urdf_path: str,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200,
        save_video: bool = False,
        save_observation_data: bool = False,
        tqdm_interval_sec=5.0,
        obs_latency_steps=0,
        n_obs_steps: int = 2,
        n_action_steps: int = 1,
        pose_repr: dict = {},
    ):
        """
        Initialize ROS2 runner.

        Args:
            output_dir: Output directory for results
            shape_meta: Shape metadata for observations and actions
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Maximum steps per episode
            save_video: Whether to save video recordings
            save_observation_data: Whether to save observation data
            tqdm_interval_sec: Interval for tqdm updates
            obs_latency_steps: Observation latency steps
            n_obs_steps: Number of observation steps to stack
            n_action_steps: Number of action steps to execute
            pose_repr: Pose representation configuration
        """
        super().__init__(output_dir)
        # Initialize environment with observation stacking
        self.env = ROS2Environment(n_obs_steps=n_obs_steps, urdf_path=urdf_path)

        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_video = save_video
        self.save_observation_data = save_observation_data
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_latency_steps = obs_latency_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.shape_meta = shape_meta
        self.pose_repr = pose_repr

        # Initialize results storage
        self.episode_results = []
        self.current_episode_data = []
        self.rot_quat2mat = RotationTransformer(
            from_rep='quaternion',
            to_rep='matrix'
        )
        self.rot_aa2mat = RotationTransformer(
            from_rep='axis_angle',
            to_rep='matrix'
        )
        self.rot_mat2target = {}
        self.key_horizon = {}
        for key, attr in self.shape_meta['obs'].items():
            self.key_horizon[key] = self.shape_meta['obs'][key]['horizon']
            if 'rotation_rep' in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix',
                    to_rep=attr['rotation_rep']
                )

        max_obs_horizon = max(self.key_horizon.values())
        self.rot_quat2euler = RotationTransformer(
            from_rep='quaternion',
            to_rep='euler_angles'
        )
        assert 'rotation_rep' in self.shape_meta['action'], "Missing 'rotation_rep' from shape_meta"

        self.rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix',
            to_rep=self.shape_meta['action']['rotation_rep']
        )

        self.key_horizon['action'] = self.shape_meta['action']['horizon']
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')

    def _process_observation_for_policy(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Process environment observation for policy input.

        Args:
            obs: Raw observation from environment (already stacked)

        Returns:
            Processed observation ready for policy
        """
        policy_obs = {}

        # Process RGB image (shape: [n_steps, H, W, 3] from environment)
        if 'camera0_rgb' in obs:
            rgb_img = obs['camera0_rgb']  # [n_steps, H, W, 3]
            # Transpose from (n_steps, H, W, 3) to (n_steps, 3, H, W) then add batch dim
            rgb_img = rgb_img.transpose(0, 3, 1, 2)  # [n_steps, 3, H, W]
            policy_obs['camera0_rgb'] = torch.from_numpy(rgb_img).float().unsqueeze(0)  # [1, n_steps, 3, H, W]

        # Process low-dimensional observations (shape: [n_steps, dim])
        for k, v in obs.items():
            if k == "camera0_rgb": continue
            policy_obs[k] = torch.from_numpy(v).float().unsqueeze(0)  # [1, n_steps, dim]

        return policy_obs

    def run(self, policy: BaseImagePolicy) -> Dict:
        device = policy.device
        env = self.env
        if not env:
            raise RuntimeError("Environment is not initialized or has been closed.")

        # Initialize results storage
        all_results = []
        episode_stats = []

        for episode_idx in range(self.n_episodes):
            # start rollout
            obs = env.reset()
            policy.reset()

            episode_data = []
            step_count = 0
            done = False

            while not done and step_count < self.max_steps_per_episode:
                obs_dict = self._process_observation_for_policy(obs)
                obs_dict = dict_apply(
                    obs_dict,
                    lambda x: x.to(device=device)
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action = action_dict['action'].detach().cpu().numpy()[0]  # [action_horizon, action_dim]
                obs, reward, done, info = env.step(action)

                # Store step data
                step_data = {
                    'obs': obs_dict,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'info': info
                }
                episode_data.append(step_data)
                step_count += 1

            # Store episode results
            episode_stats.append({
                'episode_idx': episode_idx,
                'episode_length': step_count,
                'success': done,
                'total_reward': sum(step['reward'] for step in episode_data)
            })

            if self.save_observation_data:
                all_results.extend(episode_data)

        results = {
            'episode_stats': episode_stats,
            'total_episodes': self.n_episodes,
            'avg_episode_length': np.mean([ep['episode_length'] for ep in episode_stats]),
            'success_rate': np.mean([ep['success'] for ep in episode_stats])
        }

        if self.save_observation_data:
            results['all_step_data'] = all_results

        return results

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
        **kwargs
    )
