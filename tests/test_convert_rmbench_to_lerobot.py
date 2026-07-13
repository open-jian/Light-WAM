from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_rmbench_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("convert_rmbench_to_lerobot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_state_action_uses_next_qpos() -> None:
    qpos = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    state, action = MODULE.build_state_action(qpos)
    np.testing.assert_array_equal(state, qpos[:-1])
    np.testing.assert_array_equal(action, qpos[1:])
    assert state.shape == action.shape == (4, 14)


def test_decode_legacy_rmbench_jpeg_restores_simulator_rgb() -> None:
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 220, 80, 15
    buffer = BytesIO()
    Image.fromarray(rgb[..., ::-1], mode="RGB").save(buffer, format="JPEG", quality=100)
    decoded = MODULE.decode_legacy_rmbench_rgb_jpeg(buffer.getvalue())
    assert decoded.dtype == np.uint8
    assert decoded.shape == rgb.shape
    assert np.abs(decoded.astype(np.int16) - rgb.astype(np.int16)).mean() < 2.0
