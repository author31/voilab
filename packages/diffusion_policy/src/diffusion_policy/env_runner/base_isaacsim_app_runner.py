import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class BaseIsaacSimAppRunner(BaseImageRunner, ABC):
    """
    Base runner for Isaac Sim inference with diffusion policies.

    Subclasses should manage environment-specific setup and IK using
    ArticulationKinematicsSolver as shown in `scripts/launch_isaacsim_workspace.py`.
    """

    def __init__(
        self,
        output_dir: str,
        sim_config: Dict[str, Any],
        n_episodes: int = 1,
        max_steps_per_episode: int = 200,
        n_obs_steps: int = 2,
        n_action_steps: int = 1,
        save_observation_data: bool = False,
    ):
        super().__init__(output_dir)
        from isaacsim import SimulationApp

        self.simulation_app = SimulationApp(sim_config)
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.save_observation_data = save_observation_data

    def _process_observation_for_policy(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert environment observations into policy-ready tensors.
        """
        policy_obs: Dict[str, torch.Tensor] = {}

        if "camera0_rgb" in obs:
            rgb_img = obs["camera0_rgb"]
            if rgb_img.ndim == 4:
                rgb_img = rgb_img.transpose(0, 3, 1, 2)
            elif rgb_img.ndim == 3:
                rgb_img = rgb_img.transpose(2, 0, 1)[None, ...]
            policy_obs["camera0_rgb"] = (
                torch.from_numpy(rgb_img).float().unsqueeze(0)
            )

        for key, value in obs.items():
            if key == "camera0_rgb":
                continue
            policy_obs[key] = torch.from_numpy(value).float().unsqueeze(0)

        return policy_obs

    def reset_env(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment and return the initial observation.
        Subclasses should override with Isaac Sim-specific logic.
        """
        raise NotImplementedError()

    def step_env(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Apply an action to the environment and return (obs, reward, done, info).
        Subclasses should override with Isaac Sim-specific logic.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_timeout(self, start_time: float, step_count: int) -> bool:
        """
        Return True if the current episode should timeout.
        """

    @abstractmethod
    def should_terminate(self, done: bool, info: Dict[str, Any]) -> bool:
        """
        Return True if the current episode should terminate early.
        """

    def run(self, policy: BaseImagePolicy) -> Dict[str, Any]:
        device = policy.device
        episode_stats = []
        all_results = []

        for episode_idx in range(self.n_episodes):
            if not self.simulation_app.is_running():
                break

            obs = self.reset_env()
            policy.reset()

            step_count = 0
            done = False
            episode_reward = 0.0
            episode_start_time = time.time()
            episode_data = []

            while self.simulation_app.is_running():
                if step_count >= self.max_steps_per_episode:
                    break
                if self.is_timeout(episode_start_time, step_count):
                    break

                obs_dict = self._process_observation_for_policy(obs)
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action = action_dict["action"].detach().cpu().numpy()[0]
                obs, reward, done, info = self.step_env(action)
                episode_reward += float(reward)

                if self.save_observation_data:
                    episode_data.append(
                        {
                            "obs": obs_dict,
                            "action": action,
                            "reward": reward,
                            "done": done,
                            "info": info,
                        }
                    )

                step_count += 1
                self.simulation_app.update()

                if self.should_terminate(done, info):
                    break

            episode_stats.append(
                {
                    "episode_idx": episode_idx,
                    "episode_length": step_count,
                    "success": bool(done),
                    "total_reward": episode_reward,
                }
            )

            if self.save_observation_data:
                all_results.extend(episode_data)

        results = {
            "episode_stats": episode_stats,
            "total_episodes": len(episode_stats),
            "avg_episode_length": float(
                np.mean([ep["episode_length"] for ep in episode_stats])
            )
            if episode_stats
            else 0.0,
            "success_rate": float(
                np.mean([ep["success"] for ep in episode_stats])
            )
            if episode_stats
            else 0.0,
        }

        if self.save_observation_data:
            results["all_step_data"] = all_results

        return results

    def close(self) -> None:
        if self.simulation_app:
            self.simulation_app.close()
            self.simulation_app = None
