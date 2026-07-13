from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.rmbench.lightwam_policy import deploy_policy as rmbench_policy
from experiments.rmbench.run_official_eval import _normalize_official_metrics
from experiments.robotwin.lightwam_policy import deploy_policy as robotwin_policy


ROOT = Path(__file__).resolve().parents[1]


def _forbidden_paths(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "memory" in str(key).lower() or "history" in str(key).lower():
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def test_rmbench_policy_reuses_robotwin_implementation() -> None:
    assert rmbench_policy.WorldActionRobotWinPolicy is robotwin_policy.WorldActionRobotWinPolicy
    assert rmbench_policy.encode_obs is robotwin_policy.encode_obs
    assert rmbench_policy.eval is robotwin_policy.eval
    assert rmbench_policy.reset_model is robotwin_policy.reset_model


def test_rmbench_get_model_forwards_real_checkpoint(monkeypatch) -> None:
    captured = {}

    def fake_get_model(args):
        captured.update(args)
        return object()

    monkeypatch.setattr(robotwin_policy, "get_model", fake_get_model)
    rmbench_policy.get_model(
        {"ckpt_setting": "short-result-tag", "checkpoint_path": "/tmp/real-checkpoint.pt"}
    )
    assert captured["ckpt_setting"] == "/tmp/real-checkpoint.pt"


def test_rmbench_eval_config_is_no_memory_robotwin_protocol() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="sim_rmbench_no_memory")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert cfg.EVALUATION.task_config == "demo_clean"
    assert cfg.EVALUATION.action_horizon == 32
    assert cfg.EVALUATION.replan_steps == 24
    assert cfg.EVALUATION.dataset_stats_path == "./data/rmbench/dataset_stats.json"
    assert cfg.data.train._target_.endswith("RobotVideoDataset")
    assert _forbidden_paths(resolved) == []


def test_short_eval_metrics_survive_official_hardcoded_denominator() -> None:
    successes, reward_sum = _normalize_official_metrics(2, 7.5, episode_limit=5)
    assert successes / 100 == 2 / 5
    assert reward_sum / 100 == 7.5 / 5
