import copy
from typing import Dict

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()


class IsaacsimReplayDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
    ):
        with zarr.ZipStore(dataset_path, mode="r") as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore(),
            )

        rgb_keys = []
        lowdim_keys = []
        key_horizon = {}
        key_latency_steps = {}
        key_down_sample_steps = {}
        robot_prefixes = []

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

            key_horizon[key] = attr.get("horizon", 1)
            key_latency_steps[key] = attr.get("latency_steps", 0)
            key_down_sample_steps[key] = attr.get("down_sample_steps", 1)

            if key.endswith("_eef_pos"):
                robot_prefix = key[: -len("_eef_pos")]
                if robot_prefix not in robot_prefixes:
                    robot_prefixes.append(robot_prefix)

        action_shape_meta = shape_meta["action"]
        key_horizon["action"] = action_shape_meta.get("horizon", 1)
        key_latency_steps["action"] = action_shape_meta.get("latency_steps", 0)
        key_down_sample_steps["action"] = action_shape_meta.get("down_sample_steps", 1)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.robot_prefixes = robot_prefixes
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.temporally_independent_normalization = temporally_independent_normalization
        self.sampler = sampler
        self.threadpool_limits_is_applied = False

        self.action_array = self._get_action_array().astype(np.float32)

    def _get_action_array(self) -> np.ndarray:
        if "action" in self.replay_buffer.keys():
            return self.replay_buffer["action"][:]

        actions = []
        for robot_prefix in self.robot_prefixes:
            for suffix in ["_eef_pos", "_eef_rot_axis_angle", "_gripper_width"]:
                key = f"{robot_prefix}{suffix}"
                if key not in self.replay_buffer:
                    raise RuntimeError(f"Missing action key {key} in replay buffer")
                actions.append(self.replay_buffer[key][:])
        return np.concatenate(actions, axis=-1)

    def _get_data_cache(self) -> Dict[str, np.ndarray]:
        if not self.temporally_independent_normalization:
            data_cache: Dict[str, np.ndarray] = {
                key: self.replay_buffer[key][:] for key in self.lowdim_keys
            }
            data_cache["action"] = self.action_array
            return data_cache

        list_cache: Dict[str, list[np.ndarray]] = {
            key: [] for key in self.lowdim_keys + ["action"]
        }
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=0,
        )
        for batch in dataloader:
            for key in self.lowdim_keys:
                list_cache[key].append(batch["obs"][key].numpy())
            list_cache["action"].append(batch["action"].numpy())
        self.sampler.ignore_rgb(False)

        data_cache = {
            key: np.concatenate(value, axis=0) for key, value in list_cache.items()
        }
        return data_cache

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
        )
        val_set.val_mask = self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        data_cache = self._get_data_cache()
        normalizer = LinearNormalizer()

        num_robot = max(len(self.robot_prefixes), 1)
        action_dim = data_cache["action"].shape[-1] // num_robot
        action_normalizers = []
        for robot_idx in range(num_robot):
            start = robot_idx * action_dim
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., start : start + 3])
                )
            )
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., start + 3 : start + 6])
                )
            )
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., start + 6 : start + 7])
                )
            )
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key.endswith("_eef_pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("_eef_rot_axis_angle"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("_gripper_width"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported observation key {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.action_array)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True

        data = self.sampler.sample_sequence(idx)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data
