#!/usr/bin/env python3
"""Launch a file-defined RM-Bench training experiment with preflight checks."""

from __future__ import annotations

import argparse
import os
import shlex
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


def _project_path(value: object) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _missing_inputs(cfg: DictConfig) -> list[str]:
    missing: list[str] = []
    for split in ("train", "val"):
        split_cfg = cfg.data.get(split)
        if split_cfg is None:
            missing.append(f"data.{split} configuration")
            continue

        for dataset_dir in split_cfg.dataset_dirs:
            path = _project_path(dataset_dir)
            if not path.is_dir():
                missing.append(f"{split} task dataset directory: {path}")

        stats_path = _project_path(split_cfg.pretrained_norm_stats)
        if not stats_path.is_file():
            missing.append(f"{split} normalization stats file: {stats_path}")

        text_cache = _project_path(split_cfg.text_embedding_cache_dir)
        if not text_cache.is_dir():
            missing.append(f"{split} text embedding cache directory: {text_cache}")

        if bool(split_cfg.use_latent_cache):
            latent_cache = _project_path(split_cfg.latent_cache_dir)
            if not (latent_cache / "index.pt").is_file():
                missing.append(
                    f"{split} latent cache index: {latent_cache / 'index.pt'}"
                )
    return missing


def _compose(experiment: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *overrides],
        )


def _launch_command(cfg: DictConfig, experiment: str, overrides: list[str]) -> tuple[list[str], str]:
    launcher = cfg.launcher
    num_processes = int(launcher.num_processes)
    global_batch = (
        num_processes
        * int(cfg.batch_size)
        * int(cfg.gradient_accumulation_steps)
    )
    expected = int(launcher.expected_global_batch_size)
    if global_batch != expected:
        raise ValueError(
            "Configured global batch mismatch: "
            f"{num_processes} processes * {cfg.batch_size} micro batch * "
            f"{cfg.gradient_accumulation_steps} accumulation = {global_batch}, "
            f"expected {expected}. Change the experiment YAML, not launcher environment variables."
        )

    gpu_ids = ",".join(str(gpu_id) for gpu_id in launcher.gpu_ids)
    if len(launcher.gpu_ids) != num_processes:
        raise ValueError(
            f"launcher.gpu_ids has {len(launcher.gpu_ids)} entries but "
            f"launcher.num_processes is {num_processes}"
        )

    accelerate_config = _project_path(launcher.accelerate_config_file)
    if not accelerate_config.is_file():
        raise FileNotFoundError(f"Accelerate config not found: {accelerate_config}")

    accelerate = Path(sys.executable).with_name("accelerate")
    if not accelerate.is_file():
        raise FileNotFoundError(
            f"Accelerate executable not found beside the active Python: {accelerate}"
        )

    command = [
        str(accelerate),
        "launch",
        "--config_file",
        str(accelerate_config),
        "--num_processes",
        str(num_processes),
        "--main_process_port",
        str(launcher.main_process_port),
        "scripts/train.py",
        f"experiment={experiment}",
        *overrides,
    ]
    return command, gpu_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="rmbench/put_back_block_robotwin_style_5k",
        help="Hydra experiment config name under configs/experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compose, validate, and print the launch without starting training",
    )
    parser.add_argument(
        "--skip-input-preflight",
        action="store_true",
        help="only for inspecting a config before its dataset/caches are prepared",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="optional Hydra overrides (prefer a new experiment file for persistent changes)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _compose(args.experiment, args.overrides)
    command, gpu_ids = _launch_command(cfg, args.experiment, args.overrides)

    if not args.skip_input_preflight:
        missing = _missing_inputs(cfg)
        if missing:
            details = "\n".join(f"  - {item}" for item in missing)
            print(
                "RM-Bench experiment inputs are incomplete:\n"
                f"{details}\n"
                "Prepare the task-local conversion, stats, text cache, and latent cache before training.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"CUDA_VISIBLE_DEVICES={gpu_ids} {shlex.join(command)}")
    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    )
    env.setdefault("HYDRA_FULL_ERROR", "1")
    os.chdir(PROJECT_ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
