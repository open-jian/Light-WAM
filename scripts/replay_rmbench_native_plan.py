#!/usr/bin/env python3
"""Replay one RM-Bench native plan with its saved seed and drive targets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.replay_rmbench_gt_qpos import (
    build_env_args,
    load_qpos,
    resolve_demo_seed,
    run_renderer_preflight,
)


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
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_json = cli.output_json.expanduser().resolve()
    output_json.unlink(missing_ok=True)
    root = cli.benchmark_root.expanduser().resolve()
    demo_dir = root / "data" / cli.task_name / cli.task_config
    episode_path = demo_dir / "data" / f"episode{cli.episode}.hdf5"
    trajectory_path = demo_dir / "_traj_data" / f"episode{cli.episode}.pkl"
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
    report = {
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "episode": cli.episode,
        "seed": seed,
        "episode_path": str(episode_path),
        "trajectory_path": str(trajectory_path),
        "replay_contract": "native planned positions and velocities",
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
            }
        )
        if not initial_match:
            raise RuntimeError(
                "Initial drive-target mismatch: "
                f"{initial_max_abs} > {cli.initial_drive_target_tolerance}"
            )

        trajectory = env.load_tran_data(cli.episode)
        env.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": trajectory["left_joint_path"],
                "right_joint_path": trajectory["right_joint_path"],
            }
        )
        env.play_once()
        left_segments = len(trajectory["left_joint_path"])
        right_segments = len(trajectory["right_joint_path"])
        left_consumed = int(env.left_cnt)
        right_consumed = int(env.right_cnt)
        complete_plan_consumption = (
            left_consumed == left_segments and right_consumed == right_segments
        )
        success = bool(
            env.plan_success and complete_plan_consumption and env.check_success()
        )
        report.update(
            {
                "left_plan_segments": left_segments,
                "right_plan_segments": right_segments,
                "left_plan_segments_consumed": left_consumed,
                "right_plan_segments_consumed": right_consumed,
                "complete_plan_consumption": complete_plan_consumption,
                "plan_success": bool(env.plan_success),
                "success": success,
            }
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not success:
            raise SystemExit(1)
    finally:
        if initialized:
            env.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
