#!/usr/bin/env python3
"""
Local UMI profiling harness.

Examples:
python packages/diffusion_policy/profile_umi.py \
  --config-name train_diffusion_transformer_umi_workspace \
  --dataset-path /data/umi/dataset.zarr.zip \
  --mode dataloader \
  --num-workers 0

python packages/diffusion_policy/profile_umi.py \
  --config-name train_diffusion_transformer_umi_workspace \
  --dataset-path /data/umi/dataset.zarr.zip \
  --mode train-step \
  --num-workers 4 \
  --batch-size 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import shutil
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_ROOT = PACKAGE_ROOT / "src" / "diffusion_policy" / "config"
UMI_SRC_ROOT = REPO_ROOT / "packages" / "umi" / "src"

for path in (REPO_ROOT, PACKAGE_ROOT / "src", UMI_SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile UMI dataset and train-step bottlenecks."
    )
    parser.add_argument(
        "--config-name",
        default="train_diffusion_transformer_umi_workspace",
        help="Hydra config name under packages/diffusion_policy/src/diffusion_policy/config.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the UMI dataset (.zarr.zip or .zarr). If omitted, uses the config value.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional override for task.dataset.cache_dir. Use this to avoid in-memory dataset copies.",
    )
    parser.add_argument(
        "--mode",
        choices=("init", "dataloader", "train-step", "normalizer"),
        default="dataloader",
        help="What to time.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader num_workers. Defaults to config for dataloader mode and 0 for train-step mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override DataLoader batch size. Defaults to config for dataloader mode and min(config, 2) for train-step mode.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Number of measured batches or steps after warmup.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Warmup iterations before collecting timing stats.",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=10,
        help="Same-batch replay steps for train-step mode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override for train-step mode. Defaults to cfg.training.device.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass conservative memory and disk safety checks.",
    )
    parser.add_argument(
        "--allow-full-normalizer",
        action="store_true",
        help="Allow mode=normalizer, which runs dataset.get_normalizer() and can be expensive.",
    )
    parser.add_argument(
        "--use-dataset-normalizer",
        action="store_true",
        help="In train-step mode, compute and attach the real dataset normalizer instead of a cheap identity normalizer.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override. May be passed multiple times.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the collected metrics as JSON.",
    )
    return parser.parse_args()


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def read_mem_available_bytes() -> int | None:
    meminfo_path = pathlib.Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    for line in meminfo_path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            return int(parts[1]) * 1024
    return None


def dataset_size_bytes(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def resolve_existing_path(raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((pathlib.Path.cwd() / path).resolve())
        candidates.append((PACKAGE_ROOT / path).resolve())
        candidates.append((REPO_ROOT / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Path does not exist: {raw_path}")


def resolve_output_path(raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (pathlib.Path.cwd() / path).resolve()


def nearest_existing_parent(path: pathlib.Path) -> pathlib.Path:
    candidate = path
    while not candidate.exists():
        candidate = candidate.parent
        if candidate == candidate.parent:
            return pathlib.Path("/")
    return candidate


def resolve_config_name(config_name: str) -> str:
    path = pathlib.Path(config_name)
    if path.suffix == ".yaml":
        return path.stem
    return config_name


def load_cfg(args: argparse.Namespace):
    config_name = resolve_config_name(args.config_name)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=config_name, overrides=args.override)
    return cfg


def apply_dataset_overrides(
    cfg, args: argparse.Namespace
) -> Tuple[pathlib.Path, pathlib.Path | None]:
    dataset_path_raw = args.dataset_path or cfg.task.dataset.dataset_path
    if dataset_path_raw is None:
        raise ValueError(
            "No dataset path was provided and the config does not define task.dataset.dataset_path."
        )

    dataset_path = resolve_existing_path(str(dataset_path_raw))
    cfg.task.dataset.dataset_path = str(dataset_path)

    cache_dir_path = None
    if args.cache_dir is not None:
        cache_dir_path = resolve_output_path(args.cache_dir)
        cfg.task.dataset.cache_dir = str(cache_dir_path)
    else:
        cache_dir = cfg.task.dataset.get("cache_dir", None)
        if cache_dir is not None:
            cache_dir_path = resolve_output_path(str(cache_dir))
            cfg.task.dataset.cache_dir = str(cache_dir_path)

    return dataset_path, cache_dir_path


def system_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "mem_available_bytes": read_mem_available_bytes(),
        "repo_disk_free_bytes": shutil.disk_usage(REPO_ROOT).free,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        summary["gpu_name"] = device.name
        summary["gpu_total_memory_bytes"] = device.total_memory
    return summary


def print_system_summary(summary: Dict[str, Any]) -> None:
    print("System summary:")
    print(f"  CPU threads: {summary['cpu_count']}")
    if summary["mem_available_bytes"] is not None:
        print(f"  RAM available: {format_bytes(summary['mem_available_bytes'])}")
    print(f"  Repo disk free: {format_bytes(summary['repo_disk_free_bytes'])}")
    if summary["cuda_available"]:
        print(
            f"  GPU: {summary['gpu_name']} ({format_bytes(summary['gpu_total_memory_bytes'])})"
        )
    else:
        print("  GPU: unavailable")


def run_preflight(
    args: argparse.Namespace,
    dataset_path: pathlib.Path,
    cache_dir: pathlib.Path | None,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_bytes = dataset_size_bytes(dataset_path)
    mem_available = summary.get("mem_available_bytes")

    target_kind = "memory" if cache_dir is None else "disk cache"
    if cache_dir is None:
        cache_parent = None
        target_free = mem_available
        target_free_label = "RAM"
    else:
        cache_parent = nearest_existing_parent(cache_dir.parent)
        target_free = shutil.disk_usage(cache_parent).free
        target_free_label = f"disk at {cache_parent}"

    report = {
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": dataset_bytes,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "target_kind": target_kind,
        "target_free_bytes": target_free,
    }

    print("Data preflight:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Dataset size: {format_bytes(dataset_bytes)}")
    print(f"  Dataset target: {target_kind}")
    if target_free is not None:
        print(f"  Available {target_free_label}: {format_bytes(target_free)}")

    if (
        target_free is not None
        and dataset_bytes > int(target_free * 0.70)
        and not args.force
    ):
        raise RuntimeError(
            "Dataset is large relative to available target capacity. "
            "Refusing to continue without --force. "
            "Either provide --cache-dir for disk-backed loading or free more RAM/disk."
        )

    return report


def effective_loader_cfg(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    loader_cfg = OmegaConf.to_container(copy.deepcopy(cfg.dataloader), resolve=True)
    assert isinstance(loader_cfg, dict)

    if args.num_workers is not None:
        loader_cfg["num_workers"] = args.num_workers
    elif args.mode == "train-step":
        loader_cfg["num_workers"] = 0

    if args.batch_size is not None:
        loader_cfg["batch_size"] = args.batch_size
    elif args.mode == "train-step":
        loader_cfg["batch_size"] = min(int(loader_cfg["batch_size"]), 2)

    if loader_cfg.get("num_workers", 0) == 0:
        loader_cfg["persistent_workers"] = False
        loader_cfg.pop("prefetch_factor", None)

    return loader_cfg


def build_dataloader(dataset, loader_cfg: Dict[str, Any]) -> DataLoader:
    return DataLoader(dataset, **loader_cfg)


def stats_from_timings(values: List[float]) -> Dict[str, float]:
    values_np = np.array(values, dtype=np.float64)
    return {
        "count": int(values_np.size),
        "mean_s": float(values_np.mean()),
        "stdev_s": float(values_np.std(ddof=0)),
        "min_s": float(values_np.min()),
        "p50_s": float(np.percentile(values_np, 50)),
        "p95_s": float(np.percentile(values_np, 95)),
        "max_s": float(values_np.max()),
    }


def print_timing_stats(title: str, values: List[float]) -> None:
    stats = stats_from_timings(values)
    print(f"{title}:")
    print(f"  count: {stats['count']}")
    print(f"  mean: {format_seconds(stats['mean_s'])}")
    print(f"  p50: {format_seconds(stats['p50_s'])}")
    print(f"  p95: {format_seconds(stats['p95_s'])}")
    print(f"  min: {format_seconds(stats['min_s'])}")
    print(f"  max: {format_seconds(stats['max_s'])}")


def iter_tensors(tree: Any) -> Iterable[Tuple[str, torch.Tensor]]:
    stack = [("batch", tree)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, torch.Tensor):
            yield prefix, node
        elif isinstance(node, dict):
            for key, value in reversed(list(node.items())):
                stack.append((f"{prefix}.{key}", value))
        elif isinstance(node, (list, tuple)):
            for idx in reversed(range(len(node))):
                stack.append((f"{prefix}[{idx}]", node[idx]))


def summarize_batch(batch: Any) -> Dict[str, Any]:
    specs = []
    total_bytes = 0
    for path, tensor in iter_tensors(batch):
        nbytes = tensor.numel() * tensor.element_size()
        total_bytes += nbytes
        specs.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": nbytes,
            }
        )
    return {
        "tensor_count": len(specs),
        "total_tensor_bytes": total_bytes,
        "tensors": specs,
    }


def print_batch_summary(summary: Dict[str, Any]) -> None:
    print("Batch summary:")
    print(f"  Tensors: {summary['tensor_count']}")
    print(f"  Total tensor bytes: {format_bytes(summary['total_tensor_bytes'])}")
    for spec in summary["tensors"]:
        print(
            "  "
            f"{spec['path']}: shape={spec['shape']} dtype={spec['dtype']} size={format_bytes(spec['bytes'])}"
        )


def fetch_next_batch(loader: DataLoader, iterator):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def sync_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device | None = None):
    sync_device(device)
    start = time.perf_counter()
    result = fn()
    sync_device(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


def instantiate_dataset(cfg):
    return hydra.utils.instantiate(cfg.task.dataset)


def instantiate_workspace(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg)


def build_identity_normalizer_from_batch(batch: Dict[str, Any]):
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    normalizer = LinearNormalizer()

    def create_identity_field(tensor: torch.Tensor):
        flat_dim = int(np.prod(tensor.shape[2:]))
        scale = np.ones(flat_dim, dtype=np.float32)
        offset = np.zeros(flat_dim, dtype=np.float32)
        stats = {
            "min": np.zeros(flat_dim, dtype=np.float32),
            "max": np.ones(flat_dim, dtype=np.float32),
            "mean": np.zeros(flat_dim, dtype=np.float32),
            "std": np.ones(flat_dim, dtype=np.float32),
        }
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=stats,
        )

    for key, tensor in batch["obs"].items():
        normalizer[key] = create_identity_field(tensor)
    normalizer["action"] = create_identity_field(batch["action"])
    return normalizer


def prepare_model_normalizer(
    model, dataset, first_batch: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    if args.use_dataset_normalizer:
        normalizer, normalizer_time = timed_call(lambda: dataset.get_normalizer())
        normalizer_kind = "dataset"
    else:
        normalizer, normalizer_time = timed_call(
            lambda: build_identity_normalizer_from_batch(first_batch)
        )
        normalizer_kind = "identity-batch"

    model.set_normalizer(normalizer)
    return {
        "normalizer_kind": normalizer_kind,
        "normalizer_prepare_time_s": normalizer_time,
    }


def profile_init(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "init"}
    dataset, dataset_time = timed_call(lambda: instantiate_dataset(cfg))
    report["dataset_init_time_s"] = dataset_time
    report["dataset_len"] = len(dataset)

    workspace_time = None
    if cfg._target_.endswith("Workspace"):
        _, workspace_time = timed_call(lambda: instantiate_workspace(cfg))
        report["workspace_init_time_s"] = workspace_time

    print(f"Dataset init: {format_seconds(dataset_time)}")
    if workspace_time is not None:
        print(f"Workspace init: {format_seconds(workspace_time)}")
    return report


def profile_dataloader(
    cfg, loader_cfg: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "mode": "dataloader",
        "loader_cfg": copy.deepcopy(loader_cfg),
    }

    dataset, dataset_time = timed_call(lambda: instantiate_dataset(cfg))
    loader, loader_time = timed_call(lambda: build_dataloader(dataset, loader_cfg))

    print(f"Dataset init: {format_seconds(dataset_time)}")
    print(f"DataLoader init: {format_seconds(loader_time)}")
    print(f"Dataset length: {len(dataset)}")
    print("Loader config:")
    for key, value in loader_cfg.items():
        print(f"  {key}: {value}")

    iterator = iter(loader)
    total_iters = args.warmup_batches + args.max_batches
    measured_times: List[float] = []
    first_batch_time = None
    batch_summary = None

    for idx in range(total_iters):
        (batch, iterator), elapsed = timed_call(
            lambda: fetch_next_batch(loader, iterator)
        )
        if first_batch_time is None:
            first_batch_time = elapsed
            batch_summary = summarize_batch(batch)
        if idx >= args.warmup_batches:
            measured_times.append(elapsed)

    assert first_batch_time is not None
    assert batch_summary is not None

    print(f"First batch: {format_seconds(first_batch_time)}")
    print_batch_summary(batch_summary)
    print_timing_stats("Steady-state next(dataloader)", measured_times)

    report.update(
        {
            "dataset_init_time_s": dataset_time,
            "dataloader_init_time_s": loader_time,
            "dataset_len": len(dataset),
            "first_batch_time_s": first_batch_time,
            "batch_summary": batch_summary,
            "steady_state_next_stats": stats_from_timings(measured_times),
        }
    )
    return report


def select_device(cfg, args: argparse.Namespace) -> torch.device:
    device_str = args.device or cfg.training.device
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_str}")
    return device


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return dict_apply(batch, lambda x: x.to(device, non_blocking=True))


def run_model_step(
    model, optimizer, batch: Dict[str, Any], device: torch.device
) -> Dict[str, float]:
    step_report: Dict[str, float] = {}

    _, zero_grad_time = timed_call(
        lambda: optimizer.zero_grad(set_to_none=True), device
    )
    loss, forward_time = timed_call(lambda: model(batch), device)
    _, backward_time = timed_call(lambda: loss.backward(), device)
    _, step_time = timed_call(lambda: optimizer.step(), device)

    step_report["zero_grad_s"] = zero_grad_time
    step_report["forward_s"] = forward_time
    step_report["backward_s"] = backward_time
    step_report["optimizer_step_s"] = step_time
    step_report["loss"] = float(loss.detach().cpu().item())
    return step_report


def profile_train_step(
    cfg, loader_cfg: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "mode": "train-step",
        "loader_cfg": copy.deepcopy(loader_cfg),
    }

    device = select_device(cfg, args)
    dataset, dataset_time = timed_call(lambda: instantiate_dataset(cfg))
    loader, loader_time = timed_call(lambda: build_dataloader(dataset, loader_cfg))
    workspace, workspace_time = timed_call(lambda: instantiate_workspace(cfg))

    model = workspace.model
    optimizer = workspace.optimizer
    model.to(device)
    optimizer_to(optimizer, device)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Dataset init: {format_seconds(dataset_time)}")
    print(f"DataLoader init: {format_seconds(loader_time)}")
    print(f"Workspace init: {format_seconds(workspace_time)}")
    print(f"Device: {device}")
    print("Loader config:")
    for key, value in loader_cfg.items():
        print(f"  {key}: {value}")

    iterator = iter(loader)
    total_iters = args.warmup_batches + args.max_batches
    measured = {
        "next_batch_s": [],
        "host_to_device_s": [],
        "zero_grad_s": [],
        "forward_s": [],
        "backward_s": [],
        "optimizer_step_s": [],
        "loss": [],
    }
    batch_summary = None
    static_batch_device = None
    normalizer_report = None

    try:
        for idx in range(total_iters):
            (batch_cpu, iterator), next_time = timed_call(
                lambda: fetch_next_batch(loader, iterator)
            )
            if batch_summary is None:
                batch_summary = summarize_batch(batch_cpu)
                print_batch_summary(batch_summary)
                normalizer_report = prepare_model_normalizer(
                    model=model,
                    dataset=dataset,
                    first_batch=batch_cpu,
                    args=args,
                )
                print(
                    "Model normalizer: "
                    f"{normalizer_report['normalizer_kind']} "
                    f"({format_seconds(normalizer_report['normalizer_prepare_time_s'])})"
                )

            batch_device, h2d_time = timed_call(
                lambda: move_batch_to_device(batch_cpu, device), device
            )
            if static_batch_device is None:
                static_batch_device = batch_device

            step = run_model_step(model, optimizer, batch_device, device)

            if idx >= args.warmup_batches:
                measured["next_batch_s"].append(next_time)
                measured["host_to_device_s"].append(h2d_time)
                measured["zero_grad_s"].append(step["zero_grad_s"])
                measured["forward_s"].append(step["forward_s"])
                measured["backward_s"].append(step["backward_s"])
                measured["optimizer_step_s"].append(step["optimizer_step_s"])
                measured["loss"].append(step["loss"])
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA out of memory during train-step profiling. "
                "Retry with a smaller --batch-size, fewer --max-batches, or --device cpu."
            ) from exc
        raise

    assert batch_summary is not None
    assert static_batch_device is not None
    assert normalizer_report is not None

    same_batch = {
        "zero_grad_s": [],
        "forward_s": [],
        "backward_s": [],
        "optimizer_step_s": [],
        "loss": [],
    }
    for _ in range(args.replay_steps):
        step = run_model_step(model, optimizer, static_batch_device, device)
        same_batch["zero_grad_s"].append(step["zero_grad_s"])
        same_batch["forward_s"].append(step["forward_s"])
        same_batch["backward_s"].append(step["backward_s"])
        same_batch["optimizer_step_s"].append(step["optimizer_step_s"])
        same_batch["loss"].append(step["loss"])

    print_timing_stats("Steady-state next(dataloader)", measured["next_batch_s"])
    print_timing_stats("Steady-state host->device", measured["host_to_device_s"])
    print_timing_stats("Steady-state forward", measured["forward_s"])
    print_timing_stats("Steady-state backward", measured["backward_s"])
    print_timing_stats("Steady-state optimizer.step", measured["optimizer_step_s"])
    print_timing_stats("Same-batch replay forward", same_batch["forward_s"])
    print_timing_stats("Same-batch replay backward", same_batch["backward_s"])
    print_timing_stats(
        "Same-batch replay optimizer.step", same_batch["optimizer_step_s"]
    )

    report.update(
        {
            "dataset_init_time_s": dataset_time,
            "dataloader_init_time_s": loader_time,
            "workspace_init_time_s": workspace_time,
            "device": str(device),
            "batch_summary": batch_summary,
            **normalizer_report,
            "steady_state": {
                key: stats_from_timings(values)
                for key, values in measured.items()
                if key != "loss"
            },
            "steady_state_loss_mean": statistics.mean(measured["loss"]),
            "same_batch_replay": {
                key: stats_from_timings(values)
                for key, values in same_batch.items()
                if key != "loss"
            },
            "same_batch_loss_mean": statistics.mean(same_batch["loss"]),
        }
    )
    return report


def profile_normalizer(cfg, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.allow_full_normalizer:
        raise RuntimeError(
            "mode=normalizer is disabled by default because it runs the full dataset scan in "
            "UmiDataset.get_normalizer(). Re-run with --allow-full-normalizer if you want to time it."
        )

    report: Dict[str, Any] = {"mode": "normalizer"}
    dataset, dataset_time = timed_call(lambda: instantiate_dataset(cfg))
    _, normalizer_time = timed_call(lambda: dataset.get_normalizer())

    print(f"Dataset init: {format_seconds(dataset_time)}")
    print(f"get_normalizer(): {format_seconds(normalizer_time)}")

    report["dataset_init_time_s"] = dataset_time
    report["normalizer_time_s"] = normalizer_time
    report["dataset_len"] = len(dataset)
    return report


def maybe_write_json(report: Dict[str, Any], json_out: str | None) -> None:
    if json_out is None:
        return
    output_path = resolve_output_path(json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"JSON report written to {output_path}")


def run() -> int:
    args = parse_args()
    summary = system_summary()
    print_system_summary(summary)
    sys.stdout.flush()

    cfg = load_cfg(args)
    dataset_path, cache_dir = apply_dataset_overrides(cfg, args)
    preflight = run_preflight(args, dataset_path, cache_dir, summary)

    report: Dict[str, Any] = {
        "config_name": resolve_config_name(args.config_name),
        "mode": args.mode,
        "overrides": args.override,
        "system": summary,
        "preflight": preflight,
    }

    if args.mode == "init":
        report.update(profile_init(cfg, args))
    elif args.mode == "dataloader":
        loader_cfg = effective_loader_cfg(cfg, args)
        report.update(profile_dataloader(cfg, loader_cfg, args))
    elif args.mode == "train-step":
        loader_cfg = effective_loader_cfg(cfg, args)
        report.update(profile_train_step(cfg, loader_cfg, args))
    elif args.mode == "normalizer":
        report.update(profile_normalizer(cfg, args))
    else:
        raise AssertionError(f"Unsupported mode: {args.mode}")

    maybe_write_json(report, args.json_out)
    return 0


def main() -> int:
    try:
        return run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
