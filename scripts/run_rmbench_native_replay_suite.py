#!/usr/bin/env python3
"""Run one seed-matched native-plan replay for every runnable RM-Bench task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RMBENCH_TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)

# These downloaded datasets are present in the official RMBench release, but the
# official repository has no matching env module, so they cannot be simulated.
RMBENCH_DATA_ONLY_TASKS = ("classify_blocks", "storage_blocks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/data2/jian/benchmark/RMBench/RMBench"),
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/rmbench_no_memory/step4/native_replay")
    )
    parser.add_argument("--tasks", nargs="+", choices=RMBENCH_TASKS, default=RMBENCH_TASKS)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    replay_script = Path(__file__).with_name("replay_rmbench_native_plan.py")
    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.unlink(missing_ok=True)
    for task in cli.tasks:
        (output_dir / f"{task}.json").unlink(missing_ok=True)

    reports = []
    for index, task in enumerate(cli.tasks, start=1):
        report_path = output_dir / f"{task}.json"
        print(f"[native-suite] {index}/{len(cli.tasks)} task={task}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(replay_script),
                "--benchmark-root",
                str(cli.benchmark_root.expanduser().resolve()),
                "--task-name",
                task,
                "--task-config",
                cli.task_config,
                "--episode",
                str(cli.episode),
                "--output-json",
                str(report_path),
            ],
            cwd=project_root,
            check=True,
        )
        if not report_path.is_file():
            raise RuntimeError(f"Native replay produced no fresh report for {task}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("task_name") != task or report.get("episode") != cli.episode:
            raise RuntimeError(f"Native replay report identity mismatch for {task}: {report}")
        if not report.get("initial_drive_target_matches") or not report.get("success"):
            raise RuntimeError(f"Native replay did not pass for {task}: {report}")
        reports.append(report)

    summary = {
        "passed": True,
        "episode": cli.episode,
        "task_config": cli.task_config,
        "tasks_requested": len(cli.tasks),
        "tasks_passed": len(reports),
        "data_only_tasks_not_replayed": list(RMBENCH_DATA_ONLY_TASKS),
        "max_initial_drive_target_abs": max(
            report["initial_drive_target_max_abs"] for report in reports
        ),
        "tasks": reports,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
