import h5py
import numpy as np
import pytest

from scripts.replay_rmbench_gt_qpos import (
    load_qpos,
    resolve_demo_seed,
    run_renderer_preflight,
)
from scripts.run_rmbench_native_replay_suite import RMBENCH_DATA_ONLY_TASKS, RMBENCH_TASKS


def test_resolve_demo_seed_matches_saved_seed_list(tmp_path) -> None:
    (tmp_path / "seed.txt").write_text("4 8 15 16 23 42\n", encoding="utf-8")
    assert resolve_demo_seed(tmp_path, 3) == 16


def test_load_qpos_is_finite_14d(tmp_path) -> None:
    path = tmp_path / "episode0.hdf5"
    expected = np.arange(42, dtype=np.float32).reshape(3, 14)
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("joint_action/vector", data=expected)
    qpos = load_qpos(path)
    np.testing.assert_array_equal(qpos, expected)
    assert qpos.dtype == np.float32
    assert np.isfinite(qpos).all()


def test_replay_suite_separates_runnable_and_data_only_tasks() -> None:
    assert len(RMBENCH_TASKS) == 10
    assert len(set(RMBENCH_TASKS)) == len(RMBENCH_TASKS)
    assert RMBENCH_DATA_ONLY_TASKS == ("classify_blocks", "storage_blocks")
    assert not set(RMBENCH_TASKS).intersection(RMBENCH_DATA_ONLY_TASKS)


def test_renderer_preflight_converts_bare_exit_to_failure() -> None:
    def bare_exit() -> None:
        raise SystemExit

    with pytest.raises(RuntimeError, match="renderer preflight exited"):
        run_renderer_preflight(bare_exit)
