#!/usr/bin/env python3
"""Best-effort replay of sampled RM-Bench HDF qpos targets for diagnostics.

The HDF sequence is not the native dense planner trajectory, so a failed task
does not by itself indicate a data or policy contract failure.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import yaml


def run_renderer_preflight(callback: Callable[[], None]) -> None:
    try:
        callback()
    except SystemExit as exc:
        raise RuntimeError("RM-Bench renderer preflight exited before replay") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML object in {path}, got {type(payload)}")
    return payload


def resolve_demo_seed(demo_dir: Path, episode: int) -> int:
    seeds = [
        int(value)
        for value in (demo_dir / "seed.txt").read_text(encoding="utf-8").split()
    ]
    if episode < 0 or episode >= len(seeds):
        raise IndexError(f"Episode {episode} is outside seed list [0, {len(seeds)})")
    return seeds[episode]


def build_env_args(
    benchmark_root: Path, task_name: str, task_config: str, episode: int, seed: int
) -> dict[str, Any]:
    args = _load_yaml(benchmark_root / "task_config" / f"{task_config}.yml")
    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "now_ep_num": episode,
            "seed": seed,
            "save_path": str(benchmark_root / "data" / task_name / task_config),
            "render_freq": 0,
            "save_data": False,
            "eval_video_log": False,
            "eval_video_save_dir": None,
            "eval_mode": True,
            "need_plan": False,
            "dual_arm": True,
        }
    )
    embodiment = args.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) not in {1, 3}:
        raise ValueError(f"Unsupported embodiment config: {embodiment}")
    registry = _load_yaml(benchmark_root / "task_config" / "_embodiment_config.yml")

    def robot_path(name: str) -> Path:
        path = Path(str(registry[name]["file_path"]))
        return path if path.is_absolute() else (benchmark_root / path).resolve()

    if len(embodiment) == 1:
        left_path = right_path = robot_path(embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    else:
        left_path, right_path = robot_path(embodiment[0]), robot_path(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    args["left_robot_file"], args["right_robot_file"] = str(left_path), str(right_path)
    args["left_embodiment_config"] = _load_yaml(left_path / "config.yml")
    args["right_embodiment_config"] = _load_yaml(right_path / "config.yml")
    cameras = _load_yaml(benchmark_root / "task_config" / "_camera_config.yml")
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = int(cameras[head_type]["h"])
    args["head_camera_w"] = int(cameras[head_type]["w"])
    return args


def load_qpos(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hdf:
        qpos = np.asarray(hdf["joint_action/vector"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 14 or qpos.shape[0] < 2:
        raise ValueError(f"Expected qpos [T>=2,14], got {qpos.shape} in {path}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"Non-finite qpos in {path}")
    return qpos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/data2/jian/benchmark/RMBench/RMBench"),
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--initial-drive-target-tolerance", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-success", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_json = cli.output_json.expanduser().resolve()
    output_json.unlink(missing_ok=True)
    root = cli.benchmark_root.expanduser().resolve()
    demo_dir = root / "data" / cli.task_name / cli.task_config
    episode_path = demo_dir / "data" / f"episode{cli.episode}.hdf5"
    seed = resolve_demo_seed(demo_dir, cli.episode)
    qpos = load_qpos(episode_path)
    env_args = build_env_args(root, cli.task_name, cli.task_config, cli.episode, seed)

    os.chdir(root)
    sys.path.insert(0, str(root))
    from script.test_render import Sapien_TEST

    run_renderer_preflight(Sapien_TEST)
    env_module = importlib.import_module(f"envs.{cli.task_name}")
    env = getattr(env_module, cli.task_name)()
    initialized = False
    report: dict[str, Any] = {
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "episode": cli.episode,
        "seed": seed,
        "episode_path": str(episode_path),
        "qpos_frames": int(qpos.shape[0]),
        "action_offset": 1,
        "replay_contract": "sampled HDF drive targets reparameterized by take_action",
    }
    try:
        env.setup_demo(is_test=False, **env_args)
        initialized = True
        initial = np.asarray(env.get_obs()["joint_action"]["vector"], dtype=np.float32)
        initial_max_abs = float(np.abs(initial - qpos[0]).max())
        initial_match = initial_max_abs <= cli.initial_drive_target_tolerance
        report.update(
            {
                "initial_drive_target_max_abs": initial_max_abs,
                "initial_drive_target_tolerance": cli.initial_drive_target_tolerance,
                "initial_drive_target_matches": initial_match,
                "step_limit": int(env.step_lim),
            }
        )
        if not initial_match:
            raise RuntimeError(
                "Initial drive-target mismatch: "
                f"{initial_max_abs} > {cli.initial_drive_target_tolerance}"
            )

        executed = 0
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            for target in qpos[1:]:
                with contextlib.redirect_stdout(devnull):
                    env.take_action(target, action_type="qpos")
                executed += 1
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
                if cli.progress_every > 0 and executed % cli.progress_every == 0:
                    print(
                        f"[gt-replay] task={cli.task_name} executed={executed}/{len(qpos)-1}",
                        flush=True,
                    )
        replay_success = bool(env.eval_success or env.check_success())
        report.update(
            {
                "targets_requested": int(qpos.shape[0] - 1),
                "targets_executed": executed,
                "take_action_count": int(env.take_action_cnt),
                "success": replay_success,
            }
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if cli.require_success and not replay_success:
            raise SystemExit(1)
    finally:
        if initialized:
            env.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
