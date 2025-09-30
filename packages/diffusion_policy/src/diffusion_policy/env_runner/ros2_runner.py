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
        n_episodes: int = 10,
        max_steps_per_episode: int = 200,
        save_video: bool = False,
        save_observation_data: bool = False,
        tqdm_interval_sec=5.0,
        obs_latency_steps=0,
        n_obs_steps: int = 1,
        pose_repr: dict = {}
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
            pose_repr: Pose representation configuration
        """
        super().__init__(output_dir)
        # Initialize environment with observation stacking
        self.env = ROS2Environment(n_obs_steps=n_obs_steps)

        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_video = save_video
        self.save_observation_data = save_observation_data
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_latency_steps = obs_latency_steps
        self.n_obs_steps = n_obs_steps
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

        # Process RGB image (shape: [n_steps, 3, H, W])
        if 'camera0_rgb' in obs:
            # The environment already stacks observations, so we have shape [n_steps, 3, H, W]
            # Policy expects [batch, n_steps, 3, H, W]
            rgb_img = obs['camera0_rgb']  # [n_steps, 3, H, W]
            policy_obs['camera0_rgb'] = torch.from_numpy(rgb_img).float().unsqueeze(0)  # [1, n_steps, 3, H, W]

        # Process low-dimensional observations (shape: [n_steps, dim])
        for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
            if key in obs:
                obs_data = obs[key]  # [n_steps, dim]
                policy_obs[key] = torch.from_numpy(obs_data).float().unsqueeze(0)  # [1, n_steps, dim]

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

    def run(self, policy: BaseImagePolicy) -> Dict:
        device = policy.device
        env = self.env
        if not env:
            raise RuntimeError("Environment is not initialized or has been closed.")

        # start rollout
        obs = env.reset()
        policy.reset()

        prev_action = None
        done = False
        while not done:
            obs_dict = {}
            for key in obs.keys():
                slice_start = -(self.key_horizon[key] + self.obs_latency_steps)
                slice_end = None if self.obs_latency_steps == 0 else -self.obs_latency_steps
                obs_dict[key] = obs[key][:, slice_start: slice_end]

            current_pos = copy.copy(obs_dict['robot0_eef_pos'][:, -1:])
            current_rot_mat = copy.copy(self.rot_quat2mat.forward(obs_dict['robot0_eef_quat'][:, -1:]))

            # solve relative obs
            obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat'] = compute_relative_pose(
                pos=obs_dict['robot0_eef_pos'],
                rot=obs_dict['robot0_eef_quat'],
                base_pos=current_pos if self.obs_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                base_rot_mat=current_rot_mat if self.obs_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                rot_transformer_to_mat=self.rot_quat2mat,
                rot_transformer_to_target=self.rot_mat2target['robot0_eef_quat']
            )

            obs_dict = dict_apply(
                obs_dict, 
                lambda x: torch.from_numpy(x).to(device=device)
            )
            fixed_action_prefix = None
            if prev_action is not None:
                action_pos, action_rot = compute_relative_pose(
                    pos=prev_action[..., :3],
                    rot=prev_action[..., 3: -1],
                    base_pos=current_pos if self.action_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                    base_rot_mat=current_rot_mat if self.action_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                    rot_transformer_to_mat=self.rot_aa2mat,
                    rot_transformer_to_target=self.rot_mat2target['action']
                )
                action_gripper = prev_action[..., -1:]
                fixed_action_prefix = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)
                fixed_action_prefix = torch.from_numpy(fixed_action_prefix).to(device=device)

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict, fixed_action_prefix)

            # WARN: performance issue
            # device_transfer
            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().to('cpu').numpy()
            )
            action = np_action_dict['action']
            if not np.all(np.isfinite(action)):
                logger.error(action)
                raise RuntimeError("Nan or Inf action")

            # action rotation transformer
            action_pos, action_rot = compute_relative_pose(
                pos=action[..., :3],
                rot=action[..., 3: -1],
                base_pos=current_pos if self.action_pose_repr == 'rel' else np.zeros(3, dtype=np.float32),
                base_rot_mat=current_rot_mat if self.action_pose_repr == 'rel' else np.eye(3, dtype=np.float32),
                rot_transformer_to_mat=self.rot_aa2mat,
                rot_transformer_to_target=self.rot_mat2target['action'],
                backward=True
            )
            action_gripper = action[..., -1:]
            action_all = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)

            env_action = action_all[:, self.obs_latency_steps: self.obs_latency_steps + self.n_action_steps, :]
            prev_action = env_action[:, -self.obs_latency_steps:, :]


            obs, reward, done, info = env.step(env_action)
            done = np.all(done)
            _ = env.reset()

        return {}

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
