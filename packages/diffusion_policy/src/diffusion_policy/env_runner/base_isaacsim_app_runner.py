import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import cv2
import os

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

BASE_SCENE_FP = "/workspace/voilab/assets/ED305_scene/ED305.usd"
FRANKA_PANDA_FP = "/workspace/voilab/assets/franka_panda/franka_panda_arm.usd"
FRANKA_PANDA_PRIM_PATH = "/World/Franka"
GOPRO_PRIM_PATH = "/World/Franka/panda/panda_link7/gopro_link"


class BaseIsaacSimAppRunner(BaseImageRunner, ABC):
    """
    Base runner for Isaac Sim inference with diffusion policies.

    Subclasses should manage environment-specific setup and IK using
    ArticulationKinematicsSolver as shown in `scripts/launch_isaacsim_workspace.py`.
    """

    def __init__(
        self,
        output_dir: str,
        sim_config: Optional[Dict[str, Any]] = None,
        shape_meta: Optional[Dict[str, Any]] = None,
        env=None,
        n_episodes: int = 1,
        n_obs_steps: int = 2,
        n_action_steps: int = 1,
        save_observation_data: bool = False,
        scene_config: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        max_steps_per_episode: Optional[int] = None,
    ):
        super().__init__(output_dir)
        self.shape_meta = shape_meta or {}
        self.scene_config = scene_config or getattr(env, "scene_config", {}) or {}
        self.env = env
        self.simulation_app = getattr(self.env, "simulation_app", None)
        self.n_episodes = n_episodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.save_observation_data = save_observation_data
        self.timeout_seconds = 60
        self.max_steps_per_episode = max_steps_per_episode

    def _process_observation_for_policy(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Prepare observation for policy inference.
        Ensures rollout obs have the same horizon / batch dims as training.
        """
        policy_obs: Dict[str, torch.Tensor] = {}

        #print("=== Start processing observations ===")
        #print("obs keys available:", list(obs.keys()))

        # 訓練 horizon
        horizon = 2  # default 2

        for key, meta in self.shape_meta['obs'].items():
            target_shape = meta['shape']
            if key not in obs:
                #print(f"Skipping missing key: {key}")
                continue
            data = obs[key]
            #print(f"\nProcessing key: '{key}' with meta: {meta}")
            #print(f"  Original data shape: {data.shape}, dtype: {data.dtype}")

            if meta['type'] == 'rgb':
                # RGB 統一成 (B,T,H,W,C)
                if data.ndim == 3:  # (H,W,C)
                    data = data[None, None, ...]  # (B=1,T=1,H,W,C)
                elif data.ndim == 4:
                    if data.shape[-1] in [1,3]:  # (T,H,W,C)
                        data = data[None, ...]     # (B=1,T,H,W,C)
                    elif data.shape[1] in [1,3]:  # (B,C,H,W)
                        data = data[:, None].transpose(0,1,3,4,2)  # (B,T,H,W,C)
                elif data.ndim == 5:  # (B,T,H,W,C)
                    pass
                else:
                    raise ValueError(f"Unexpected shape for RGB obs {key}: {data.shape}")

                B, T, H, W, C = data.shape
                target_C, target_H, target_W = target_shape
                resized = np.zeros((B, T, target_C, target_H, target_W), dtype=data.dtype)
                for b in range(B):
                    for t in range(T):
                        img = data[b, t]
                        if img.shape[2] != target_C:
                            img = img[:, :, :target_C]
                        resized[b, t] = cv2.resize(img, (target_W, target_H)).transpose(2,0,1)

                # tile 時間維度到訓練 horizon
                if T < horizon:
                    resized = np.tile(resized, (1, horizon//T, 1, 1, 1))
                    T = resized.shape[1]

                policy_obs[key] = torch.from_numpy(resized).float()  # (B,T,C,H,W)
                #print(f"  Final RGB tensor shape (B,T,C,H,W): {policy_obs[key].shape}")

            elif meta['type'] == 'low_dim':
                # 對齊 batch size
                rgb_keys = [k for k,m in self.shape_meta['obs'].items() if m['type']=='rgb' and k in obs]
                B = max([obs[k].shape[0] if obs[k].ndim>0 else 1 for k in rgb_keys], default=1)

                # 最後維度
                target_D = target_shape[0] if len(target_shape)==1 else np.prod(target_shape)

                if data.ndim == 1:  # (D,)
                    data = data[None, None, :]  # (1,1,D)
                elif data.ndim == 2:  # (T,D) or (B,D)
                    if data.shape[0] == B:
                        data = data[:, None, :]  # (B,T=1,D)
                    else:
                        data = data[None, :, :]  # (1,T,D)
                elif data.ndim == 3:  # (B,T,D)
                    pass

                # 最後維度補齊
                if data.shape[2] != target_D:
                    new_data = np.zeros((data.shape[0], data.shape[1], target_D), dtype=data.dtype)
                    min_D = min(data.shape[2], target_D)
                    new_data[:,:,:min_D] = data[:,:,:min_D]
                    data = new_data
                    #print(f"  Adjusted low_dim last dim to target shape: {data.shape}")

                # 對齊 batch size
                if data.shape[0] != B:
                    data = np.tile(data, (B,1,1))
                    #print(f"  Tiled low_dim to match RGB batch size: {data.shape}")

                # tile 時間維度到訓練 horizon
                T = data.shape[1]
                if T < horizon:
                    data = np.tile(data, (1, horizon//T, 1))
                    T = data.shape[1]

                policy_obs[key] = torch.from_numpy(data).float()
                #print(f"  Final low_dim tensor shape (B,T,D): {policy_obs[key].shape}")

        #print("=== Finished processing observations ===")
        return policy_obs



    def reset_env(self, episode_index: int = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment and return the initial observation.
        Subclasses should override with Isaac Sim-specific logic.
        """
        if self.env is None:
            raise RuntimeError("IsaacSim environment is not initialized.")
        return self.env.reset(options={"episode_index": episode_index})

    def step_env(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Apply an action to the environment and return (obs, reward, done, info).
        Subclasses should override with Isaac Sim-specific logic.
        """
        if self.env is None:
            raise RuntimeError("IsaacSim environment is not initialized.")
        return self.env.step(action)

    def is_timeout(self, start_time: float, step_count: int) -> bool:
        """
        Return True if the current episode should timeout.

        Checks both wall-clock time (timeout_seconds) and step count
        (max_steps_per_episode). Returns True if either limit is exceeded.
        """
        if self.timeout_seconds is not None:
            if (time.time() - start_time) >= self.timeout_seconds:
                return True
        if self.max_steps_per_episode is not None:
            if step_count >= self.max_steps_per_episode:
                return True
        return False

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
            if self.simulation_app is not None:
                if not self.simulation_app.is_running():
                    break
            elif self.env is not None and not self.env.is_running():
                break

            obs = self.reset_env(episode_index=episode_idx)
            policy.reset()

            step_count = 0
            done = False
            episode_reward = 0.0
            episode_start_time = time.time()
            episode_data = []

            while True:
                if self.simulation_app is not None:
                    if not self.simulation_app.is_running():
                        break
                elif self.env is not None and not self.env.is_running():
                    break

                # Dynamic termination check
                current_max_steps = self.max_steps_per_episode
                if hasattr(policy, "get_current_episode_length"):
                    current_max_steps = policy.get_current_episode_length()

                if current_max_steps is not None and step_count >= current_max_steps:
                    break
                
                if self.timeout_seconds is not None and (time.time() - episode_start_time) >= self.timeout_seconds:
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
                if self.simulation_app is not None:
                    self.simulation_app.update()
                elif self.env is not None:
                    self.env.update()

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
            "success_rate": float(np.mean([ep["success"] for ep in episode_stats]))
            if episode_stats
            else 0.0,
        }

        if self.save_observation_data:
            results["all_step_data"] = all_results

        return results

    def close(self) -> None:
        if self.env:
            self.env.close()
        self.env = None
        self.simulation_app = None
