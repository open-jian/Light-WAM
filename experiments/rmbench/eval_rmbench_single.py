#!/usr/bin/env python3
"""Launch one official RM-Bench task with the shared RoboTwin Light-WAM policy."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from experiments.robotwin.eval_robotwin_single import (
    _append_override,
    _ensure_policy_symlink,
    _resolve_ckpt_tag,
    _resolve_dataset_stats_path,
    _resolve_path,
    _resolve_training_config_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "lightwam_policy"


def _resolve_official_result_tag(cfg: DictConfig, ckpt_tag: str) -> str:
    """Return a safe, per-evaluation directory component for official outputs."""
    configured_tag = cfg.EVALUATION.get("official_result_tag")
    tag = str(configured_tag).strip() if configured_tag is not None else f"{ckpt_tag}_seed{cfg.seed}"
    if not tag or tag in {".", ".."} or Path(tag).name != tag:
        raise ValueError(
            "`EVALUATION.official_result_tag` must be a non-empty single directory name"
        )
    return tag


def _new_result_directories(result_parent: Path, before: set[Path]) -> list[Path]:
    if not result_parent.is_dir():
        return []
    return sorted(
        (path for path in result_parent.iterdir() if path.is_dir() and path not in before),
        key=lambda path: path.name,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench_no_memory.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None")
    if int(cfg.EVALUATION.eval_num_episodes) <= 0:
        raise ValueError("`EVALUATION.eval_num_episodes` must be positive")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    result_tag = _resolve_official_result_tag(cfg, ckpt_tag)

    rmbench_root = _resolve_path(str(cfg.EVALUATION.rmbench_root), base=PROJECT_ROOT)
    if not (rmbench_root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"Invalid RM-Bench root: {rmbench_root}")

    policy_source = (PROJECT_ROOT / "experiments" / "rmbench" / POLICY_NAME).resolve()
    _ensure_policy_symlink(robotwin_root=rmbench_root, policy_source_dir=policy_source)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / (
        f"eval_{cfg.EVALUATION.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    use_run_config = bool(cfg.EVALUATION.use_training_run_config)
    training_config = _resolve_training_config_path(cfg, ckpt_path) if use_run_config else None
    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_rmbench_no_memory.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    overrides: list[str] = []
    for key, value in (
        ("task_name", cfg.EVALUATION.task_name),
        ("task_config", cfg.EVALUATION.task_config),
        ("ckpt_setting", result_tag),
        ("checkpoint_path", str(ckpt_path)),
        ("seed", cfg.seed),
        ("policy_name", POLICY_NAME),
        ("instruction_type", cfg.EVALUATION.instruction_type),
        ("eval_num_episodes", int(cfg.EVALUATION.eval_num_episodes)),
        ("sim_cfg_path", str(sim_cfg_path)),
        ("sim_task", sim_task),
        ("use_training_run_config", use_run_config),
        ("training_config_path", None if training_config is None else str(training_config)),
        ("mixed_precision", cfg.mixed_precision),
        ("device", cfg.EVALUATION.device),
        ("dataset_stats_path", str(stats_path)),
        ("action_horizon", int(cfg.EVALUATION.action_horizon)),
        ("replan_steps", int(cfg.EVALUATION.replan_steps)),
        ("num_inference_steps", int(cfg.EVALUATION.num_inference_steps)),
        ("sigma_shift", cfg.EVALUATION.sigma_shift),
        ("text_cfg_scale", float(cfg.EVALUATION.text_cfg_scale)),
        ("negative_prompt", cfg.EVALUATION.negative_prompt),
        ("rand_device", cfg.EVALUATION.rand_device),
        ("tiled", bool(cfg.EVALUATION.tiled)),
        ("timing_enabled", bool(cfg.EVALUATION.timing_enabled)),
        ("skip_get_obs_within_replan", bool(cfg.EVALUATION.skip_get_obs_within_replan)),
    ):
        _append_override(overrides, key, value)

    result_parent = (
        rmbench_root
        / "eval_result"
        / str(cfg.EVALUATION.task_name)
        / POLICY_NAME
        / str(cfg.EVALUATION.task_config)
        / result_tag
    )
    before = set(result_parent.iterdir()) if result_parent.is_dir() else set()
    command = [
        sys.executable,
        "-u",
        str((PROJECT_ROOT / "experiments" / "rmbench" / "run_official_eval.py").resolve()),
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    )

    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(rmbench_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
    finally:
        new_results = _new_result_directories(result_parent, before)
        if len(new_results) > 1:
            raise RuntimeError(f"RM-Bench produced multiple result directories: {new_results}")
        if new_results:
            destination = output_dir / "official_result"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite result destination: {destination}")
            shutil.move(str(new_results[0]), str(destination))

    if return_code != 0:
        raise RuntimeError(f"RM-Bench evaluation failed with return code {return_code}. Log: {log_path}")

    OmegaConf.save(cfg, output_dir / f"eval_config_{cfg.EVALUATION.task_name}.yaml")
    print(f"RM-Bench evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
