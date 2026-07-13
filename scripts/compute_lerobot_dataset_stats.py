#!/usr/bin/env python3
"""Compute Light-WAM normalization statistics for a configured LeRobot dataset."""

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lightwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from lightwam.utils.config_resolvers import register_default_resolvers
from lightwam.utils.logging_config import setup_logging


register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required")

    output_cfg = cfg.get("output_stats_path")
    if output_cfg in (None, "", "null"):
        output_path = Path(str(cfg.output_dir)).expanduser().resolve() / "dataset_stats.json"
    else:
        output_path = Path(str(output_cfg)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    processor_cfg = OmegaConf.create(train_cfg["processor"])
    train_cfg["processor"] = None
    train_cfg["pretrained_norm_stats"] = None
    train_cfg["use_latent_cache"] = False
    train_cfg["latent_cache_dir"] = None
    train_cfg["video_only"] = False
    train_cfg["is_training_set"] = True

    dataset = instantiate(train_cfg)
    processor = instantiate(processor_cfg).eval()
    stats_source = getattr(dataset, "lerobot_dataset", dataset)
    if not hasattr(stats_source, "get_dataset_stats"):
        raise TypeError(
            f"Configured dataset {type(dataset).__name__} does not expose "
            "get_dataset_stats() directly or through `.lerobot_dataset`"
        )
    stats = stats_source.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, str(output_path))
    print(f"[dataset-stats] saved={output_path}")
    print(f"[dataset-stats] num_episodes={stats.get('num_episodes')}")
    print(f"[dataset-stats] num_transition={stats.get('num_transition')}")


if __name__ == "__main__":
    main()
