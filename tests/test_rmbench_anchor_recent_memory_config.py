from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.launch_rmbench_experiment import _missing_inputs
from scripts.validate_rmbench_anchor_recent_memory_config import _validate


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "rmbench/put_back_block_anchor_recent_memory_5k"


def _compose_memory_experiment():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train", overrides=[f"experiment={EXPERIMENT}"])


def _compose_memory_sim():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="sim_rmbench_put_back_block_anchor_recent_memory_5k")


def test_rmbench_anchor_recent_memory_contract_and_schedule() -> None:
    cfg = _compose_memory_experiment()
    _validate(cfg)

    assert cfg.max_steps == 5000
    assert cfg.warmup_steps == 1000
    assert cfg.save_every == 1000
    assert cfg.launcher.num_processes == 4
    assert cfg.batch_size * cfg.gradient_accumulation_steps * cfg.launcher.num_processes == 32
    assert cfg.experiment.action_horizon == 32
    assert cfg.experiment.trained_action_prefix == 24
    assert cfg.experiment.evaluation_replan_steps == 24
    assert cfg.model.loss.action_temporal_weighting.num_prefix_steps == 24
    assert cfg.wandb.project == "RMBench_post-train"
    assert cfg.memory_monitor.enabled is True
    assert cfg.memory_monitor.shuffled_probe_every == 500


def test_rmbench_memory_contract_rejects_training_eval_horizon_drift() -> None:
    cfg = _compose_memory_experiment()
    cfg.model.loss.action_temporal_weighting.num_prefix_steps = 16

    with pytest.raises(ValueError, match="num_prefix_steps"):
        _validate(cfg)


def test_rmbench_anchor_recent_memory_uses_split_episode_caches() -> None:
    cfg = _compose_memory_experiment()
    train_cache = str(cfg.data.train.latent_cache_dir)
    val_cache = str(cfg.data.val.latent_cache_dir)

    assert train_cache != val_cache
    assert "episode_packed" in train_cache
    assert "episode_packed" in val_cache
    assert cfg.data.train.history_anchor_size == 1
    assert cfg.data.train.history_recent_min_size == 1
    assert cfg.data.train.history_recent_max_size == 4
    assert cfg.data.train.history_recent_stride == 1


def test_rmbench_anchor_recent_memory_sim_uses_every_natural_frame() -> None:
    cfg = _compose_memory_sim()

    assert cfg.model.history_memory.anchor_size == 1
    assert cfg.model.history_memory.recent_min_size == 1
    assert cfg.model.history_memory.recent_max_size == 4
    assert cfg.model.history_memory.raw_stride == 1
    assert cfg.model.history_memory.recent_stride == 1
    assert cfg.data.train.processor.history_num_frames == 5
    assert cfg.EVALUATION.skip_get_obs_within_replan is True


def test_rmbench_preflight_checks_both_split_caches(tmp_path: Path) -> None:
    cfg = _compose_memory_experiment()
    dataset_dir = tmp_path / "dataset"
    text_cache = tmp_path / "text"
    stats_path = tmp_path / "stats.json"
    train_cache = tmp_path / "train_cache"
    val_cache = tmp_path / "val_cache"
    dataset_dir.mkdir()
    text_cache.mkdir()
    stats_path.write_text("{}")
    train_cache.mkdir()
    val_cache.mkdir()
    (train_cache / "index.pt").touch()

    for split in ("train", "val"):
        cfg.data[split].dataset_dirs = [str(dataset_dir)]
        cfg.data[split].pretrained_norm_stats = str(stats_path)
        cfg.data[split].text_embedding_cache_dir = str(text_cache)
    cfg.data.train.latent_cache_dir = str(train_cache)
    cfg.data.val.latent_cache_dir = str(val_cache)

    missing = _missing_inputs(cfg)
    assert missing == [f"val latent cache index: {val_cache / 'index.pt'}"]
