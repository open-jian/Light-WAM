#!/usr/bin/env python3
"""Run RM-Bench's official evaluator with a configurable episode count."""

from __future__ import annotations

import sys
from pathlib import Path


OFFICIAL_RESULT_DENOMINATOR = 100


def _normalize_official_metrics(successes, reward_sum, episode_limit: int):
    """Compensate for RM-Bench main() dividing returned totals by hard-coded 100."""

    scale = OFFICIAL_RESULT_DENOMINATOR / float(episode_limit)
    return successes * scale, reward_sum * scale


def main() -> None:
    rmbench_root = Path.cwd().resolve()
    script_root = rmbench_root / "script"
    for path in (rmbench_root, script_root, rmbench_root / "policy"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from test_render import Sapien_TEST
    import eval_policy as official

    Sapien_TEST()
    usr_args = official.parse_args_and_config()
    episode_limit = int(usr_args.get("eval_num_episodes", 100))
    if episode_limit <= 0:
        raise ValueError(f"eval_num_episodes must be positive, got {episode_limit}")

    original_eval_policy = official.eval_policy

    def eval_policy_with_limit(*args, **kwargs):
        kwargs["test_num"] = episode_limit
        next_seed, successes, reward_sum = original_eval_policy(*args, **kwargs)
        successes, reward_sum = _normalize_official_metrics(
            successes, reward_sum, episode_limit
        )
        return next_seed, successes, reward_sum

    official.eval_policy = eval_policy_with_limit
    official.main(usr_args)


if __name__ == "__main__":
    main()
