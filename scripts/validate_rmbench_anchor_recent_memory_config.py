#!/usr/bin/env python3
"""Fail-closed validation for the RMBench anchor+recent memory contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from lightwam.utils.config_resolvers import register_default_resolvers


EXPECTED_MODEL = {
    "enabled": True,
    "mode": "anchor_recent_fixed_slots_v1",
    "apply_to": "action_branch_only",
    "min_size": 2,
    "max_size": 5,
    "raw_stride": 1,
    "anchor_size": 1,
    "anchor_stride": 1,
    "recent_min_size": 1,
    "recent_max_size": 4,
    "recent_stride": 1,
}
EXPECTED_DATA = {
    "history_enabled": True,
    "history_max_size": 5,
    "history_raw_stride": 1,
    "history_anchor_size": 1,
    "history_anchor_stride": 1,
    "history_recent_min_size": 1,
    "history_recent_max_size": 4,
    "history_recent_stride": 1,
}
EXPECTED_EXPERIMENT = {
    "policy_memory": "anchor_recent_raw_observation",
    "action_horizon": 32,
    "trained_action_prefix": 24,
    "evaluation_replan_steps": 24,
}


def _assert_mapping(node: DictConfig, expected: dict[str, object], path: str) -> None:
    for key, expected_value in expected.items():
        actual = node.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Memory contract violation: {path}.{key}={actual!r}, "
                f"expected {expected_value!r}."
            )


def _validate(cfg: DictConfig) -> None:
    if cfg.get("model") is None or cfg.model.get("history_memory") is None:
        raise ValueError("Memory contract violation: model.history_memory is missing.")
    _assert_mapping(cfg.model.history_memory, EXPECTED_MODEL, "model.history_memory")

    for split in ("train", "val"):
        node = cfg.data.get(split)
        if node is None:
            raise ValueError(f"Memory contract violation: data.{split} is missing.")
        _assert_mapping(node, EXPECTED_DATA, f"data.{split}")
        if not bool(node.get("use_latent_cache", False)):
            raise ValueError(
                f"Memory contract violation: data.{split}.use_latent_cache must be true."
            )
        cache_dir = node.get("latent_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(
                f"Memory contract violation: data.{split}.latent_cache_dir is missing."
            )
        if int(node.get("num_frames", -1)) != 33:
            raise ValueError(f"Memory contract violation: data.{split}.num_frames must be 33.")
        if int(node.get("global_sample_stride", -1)) != 1:
            raise ValueError(
                f"Memory contract violation: data.{split}.global_sample_stride must be 1."
            )
        if int(node.get("action_video_freq_ratio", -1)) != 4:
            raise ValueError(
                f"Memory contract violation: data.{split}.action_video_freq_ratio must be 4."
            )
        if int(node.get("processor", {}).get("history_num_frames", -1)) != 5:
            raise ValueError(
                f"Memory contract violation: data.{split}.processor.history_num_frames must be 5."
            )

    if str(cfg.data.train.latent_cache_dir) == str(cfg.data.val.latent_cache_dir):
        raise ValueError("Train and validation must use separate episode-packed caches.")

    experiment = cfg.get("experiment")
    if experiment is None:
        raise ValueError("Memory contract violation: experiment metadata is missing.")
    _assert_mapping(experiment, EXPECTED_EXPERIMENT, "experiment")

    action_weighting = cfg.model.get("loss", {}).get("action_temporal_weighting")
    if action_weighting is None:
        raise ValueError(
            "Memory contract violation: model.loss.action_temporal_weighting is missing."
        )
    expected_action_weighting = {
        "enabled": True,
        "num_prefix_steps": 24,
        "prefix_weight": 1.0,
        "tail_weight": 0.0,
    }
    _assert_mapping(
        action_weighting,
        expected_action_weighting,
        "model.loss.action_temporal_weighting",
    )


def _find_training_config(checkpoint: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(parent / "config.yaml" for parent in list(checkpoint.parents)[:4])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Could not locate the checkpoint's training config.yaml. Set "
        "TRAINING_CONFIG_PATH explicitly."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="Hydra experiment name under configs/experiment")
    source.add_argument("--checkpoint", help="Checkpoint whose saved run config must be memory-aware")
    parser.add_argument("--training-config", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    register_default_resolvers()
    if args.experiment:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={args.experiment}", *args.overrides],
            )
        source = f"experiment={args.experiment}"
    else:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        training_config = _find_training_config(checkpoint, args.training_config)
        cfg = OmegaConf.load(training_config)
        source = str(training_config)

    _validate(cfg)
    print(f"[rmbench-anchor-recent-memory] contract OK: {source}")


if __name__ == "__main__":
    main()
