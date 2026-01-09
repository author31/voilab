import sys
import time
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_isaacsim_app_runner import (
    BaseIsaacSimAppRunner,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Add scripts directory to path for registry imports
sys.path.insert(0, "/workspace/voilab/scripts")


def _get_object_pose(object_prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight copy of scripts.utils.get_object_pose to avoid importing scripts.
    """
    import isaacsim.core.utils.prims as prims_utils
    import isaacsim.core.utils.xforms as xforms_utils

    prim = prims_utils.get_prim_at_path(object_prim_path)
    pos, rot_quat_wxyz = xforms_utils.get_world_pose(prims_utils.get_prim_path(prim))
    return np.asarray(pos), np.asarray(rot_quat_wxyz)


@runtime_checkable
class IsaacSimTaskEnv(Protocol):
    """
    Minimal protocol for an Isaac Sim task environment.
    """

    def reset(self) -> Dict[str, np.ndarray]: ...

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]: ...


def _get_registry_class(task_name: str):
    """
    Get registry class for the given task name.
    Imports from scripts/registry to share configuration.
    """
    import registry

    return registry.get_task_registry(task_name)


class RegistryIsaacSimAppRunner(BaseIsaacSimAppRunner):
    """
    Base runner for Isaac Sim tasks that handles common episode bookkeeping.
    Uses the shared registry system for task configuration.
    """

    # Subclasses should set this to their task name
    TASK_NAME: Optional[str] = None

    def __init__(
        self,
        output_dir: str,
        sim_config: Optional[Dict[str, Any]] = None,
        shape_meta: Optional[Dict[str, Any]] = None,
        env: Optional[IsaacSimTaskEnv] = None,
        task_name: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
        n_episodes: int = 1,
        n_obs_steps: int = 2,
        n_action_steps: int = 1,
        save_observation_data: bool = False,
    ):
        super().__init__(
            output_dir=output_dir,
            sim_config=sim_config,
            shape_meta=shape_meta,
            env=env,
            n_episodes=n_episodes,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            save_observation_data=save_observation_data,
        )

        # Determine task name from argument or class attribute
        self._task_name = task_name or self.TASK_NAME
        self._registry_config: Optional[Dict[str, Any]] = None

        if self._task_name is not None:
            self._load_registry_config()

        self.task_env: Optional[IsaacSimTaskEnv] = None
        self.timeout_seconds = timeout_seconds

        if env is not None:
            self.set_task_env(env)
        elif self.env is not None:
            self.set_task_env(self.env)

    def _load_registry_config(self) -> None:
        """Load configuration from registry based on task name."""
        if self._task_name is None:
            return
        registry_class = _get_registry_class(self._task_name)
        self._registry_config = registry_class.get_config()

    @property
    def registry_config(self) -> Dict[str, Any]:
        """Get registry configuration, loading if necessary."""
        if self._registry_config is None:
            if self._task_name is None:
                return {}
            self._load_registry_config()
        return self._registry_config or {}

    @property
    def environment_vars(self) -> Dict[str, Any]:
        """Get environment_vars from registry config."""
        return self.registry_config.get("environment_vars", {})

    @property
    def preload_objects(self) -> list:
        """Get PRELOAD_OBJECTS from registry config."""
        return self.environment_vars.get("PRELOAD_OBJECTS", [])

    def _get_preload_prim_path(self, object_name: str) -> str:
        """Get prim_path for a preload object by name."""
        for entry in self.preload_objects:
            if entry.get("name") == object_name:
                prim_path = entry.get("prim_path")
                if prim_path:
                    return prim_path
        raise ValueError(f"Missing PRELOAD_OBJECTS prim_path for {object_name}")

    def set_task_env(self, task_env: IsaacSimTaskEnv) -> None:
        self.task_env = task_env

    def reset_env(self) -> Dict[str, np.ndarray]:
        if self.task_env is None:
            raise RuntimeError("task_env is not set; call set_task_env first.")
        return self.task_env.reset()

    def step_env(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.task_env is None:
            raise RuntimeError("task_env is not set; call set_task_env first.")
        return self.task_env.step(action)

    def is_timeout(self, start_time: float, step_count: int) -> bool:
        if self.timeout_seconds is None:
            return False
        return (time.time() - start_time) >= self.timeout_seconds

    def should_terminate(self, done: bool, info: Dict[str, Any]) -> bool:
        return bool(done)

    def run(self, policy: BaseImagePolicy) -> Dict[str, Any]:
        device = policy.device
        episode_stats = []
        all_results = []

        print("Running")

        for episode_idx in range(self.n_episodes):
            if self.simulation_app is not None:
                if not self.simulation_app.is_running():
                    break
            elif self.env is not None and not self.env.is_running():
                break

            obs = self.reset_env()
            policy.reset()

            step_count = 0
            done = False
            completion_triggered = False
            episode_reward = 0.0
            episode_start_time = time.time()
            episode_data = []
            info: Dict[str, Any] = {}

            while True:
                if self.simulation_app is not None:
                    if not self.simulation_app.is_running():
                        break
                elif self.env is not None and not self.env.is_running():
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
                if self.simulation_app is not None:
                    self.simulation_app.update()
                elif self.env is not None:
                    self.env.update()

                completion_triggered = self.should_terminate(done, info)
                if done or completion_triggered:
                    break

            episode_success = bool(
                done
                or completion_triggered
                or (isinstance(info, dict) and info.get("episode_completed", False))
            )

            episode_stats.append(
                {
                    "episode_idx": episode_idx,
                    "episode_length": step_count,
                    "success": episode_success,
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


class KitchenIsaacSimAppRunner(RegistryIsaacSimAppRunner):
    """
    Runner for kitchen task (cup stacking).
    Configuration loaded from KitchenTaskRegistry.
    """

    TASK_NAME = "kitchen"

    def __init__(
        self,
        output_dir: str,
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            **kwargs,
        )

    def should_terminate(self, done: bool, info: Dict[str, Any]) -> bool:
        if done:
            return True

        try:
            blue_cup_pos, _ = _get_object_pose(self._get_preload_prim_path("blue cup"))
            pink_cup_pos, _ = _get_object_pose(self._get_preload_prim_path("pink cup"))

            vertical_order_ok = blue_cup_pos[2] > pink_cup_pos[2]

            xy_dist = np.linalg.norm(blue_cup_pos[:2] - pink_cup_pos[:2])
            xy_alignment_ok = xy_dist < 0.03

            completed = vertical_order_ok and xy_alignment_ok
        except Exception as exc:
            if isinstance(info, dict):
                info["episode_completion_error"] = str(exc)
            return False

        if completed and isinstance(info, dict):
            info.setdefault("episode_completed", True)

        return completed


class DiningRoomIsaacSimAppRunner(RegistryIsaacSimAppRunner):
    """
    Runner for dining room task (table setting).
    Configuration loaded from DiningRoomTaskRegistry.
    """

    TASK_NAME = "dining-room"

    def __init__(
        self,
        output_dir: str,
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            **kwargs,
        )

    @property
    def fork_path(self) -> str:
        return self.environment_vars.get("FORK_PATH", "/World/fork")

    @property
    def knife_path(self) -> str:
        return self.environment_vars.get("KNIFE_PATH", "/World/knife")

    @property
    def plate_path(self) -> str:
        return self.environment_vars.get("PLATE_PATH", "/plate")

    def should_terminate(self, done: bool, info: Dict[str, Any]) -> bool:
        if done:
            return True

        try:
            plate_pos, _ = _get_object_pose(self.plate_path)
            fork_pos, _ = _get_object_pose(self.fork_path)
            knife_pos, _ = _get_object_pose(self.knife_path)

            max_dist_xy = 0.15

            fork_dist_xy = np.linalg.norm(fork_pos[:2] - plate_pos[:2])
            knife_dist_xy = np.linalg.norm(knife_pos[:2] - plate_pos[:2])

            fork_near_plate = fork_dist_xy <= max_dist_xy
            knife_near_plate = knife_dist_xy <= max_dist_xy

            fork_on_left = fork_pos[1] > plate_pos[1]
            knife_on_right = knife_pos[1] < plate_pos[1]

            completed = (
                fork_near_plate and knife_near_plate and fork_on_left and knife_on_right
            )
        except Exception as exc:
            if isinstance(info, dict):
                info["episode_completion_error"] = str(exc)
            return False

        if completed and isinstance(info, dict):
            info.setdefault("episode_completed", True)

        return completed


class LivingRoomIsaacSimAppRunner(RegistryIsaacSimAppRunner):
    """
    Runner for living room task (block sorting).
    Configuration loaded from LivingRoomTaskRegistry.
    """

    TASK_NAME = "living-room"

    # Default box bounds (can be overridden by registry if needed)
    BOX_MIN = np.array([1.41298, 10.87098, 0.70])
    BOX_MAX = np.array([1.61221, 11.08388, 0.81762])

    def __init__(
        self,
        output_dir: str,
        **kwargs,
    ):
        # Remove sim_config from kwargs if present to avoid duplicate argument
        kwargs.pop("sim_config", None)
        super().__init__(
            output_dir=output_dir,
            **kwargs,
        )

    @property
    def blue_block_path(self) -> str:
        return self.environment_vars.get("BLUE_BLOCK_PATH", "/World/cylinder")

    @property
    def green_block_path(self) -> str:
        return self.environment_vars.get("GREEN_BLOCK_PATH", "/World/bridge")

    @property
    def red_block_path(self) -> str:
        return self.environment_vars.get("RED_BLOCK_PATH", "/World/triangle")

    def should_terminate(self, done: bool, info: Dict[str, Any]) -> bool:
        if done:
            return True

        try:
            blue_block_pos, _ = _get_object_pose(self.blue_block_path)
            green_block_pos, _ = _get_object_pose(self.green_block_path)
            red_block_pos, _ = _get_object_pose(self.red_block_path)

            def in_box(pos: np.ndarray) -> bool:
                return (
                    self.BOX_MIN[0] <= pos[0] <= self.BOX_MAX[0]
                    and self.BOX_MIN[1] <= pos[1] <= self.BOX_MAX[1]
                    and self.BOX_MIN[2] <= pos[2] <= self.BOX_MAX[2]
                )

            completed = (
                in_box(blue_block_pos)
                and in_box(green_block_pos)
                and in_box(red_block_pos)
            )
        except Exception as exc:
            if isinstance(info, dict):
                info["episode_completion_error"] = str(exc)
            return False

        if completed and isinstance(info, dict):
            info.setdefault("episode_completed", True)

        return completed
