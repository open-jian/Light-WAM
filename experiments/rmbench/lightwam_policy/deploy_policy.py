"""Thin RM-Bench facade over the existing RoboTwin Light-WAM policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.robotwin.lightwam_policy import deploy_policy as _robotwin_policy


WorldActionRobotWinPolicy = _robotwin_policy.WorldActionRobotWinPolicy
encode_obs = _robotwin_policy.encode_obs
eval = _robotwin_policy.eval
reset_model = _robotwin_policy.reset_model


def get_model(usr_args: dict[str, Any]):
    """Load the shared RoboTwin policy while keeping eval paths checkpoint-safe."""

    forwarded = dict(usr_args)
    checkpoint_path = forwarded.get("checkpoint_path")
    if checkpoint_path not in (None, "", "none", "None", "null"):
        forwarded["ckpt_setting"] = checkpoint_path
    return _robotwin_policy.get_model(forwarded)
