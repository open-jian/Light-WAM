#!/usr/bin/env python3
"""Run RM-Bench's official evaluator with a configurable episode count."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OFFICIAL_RESULT_DENOMINATOR = 100
MULTI_VIEW_VIDEO_SIZE = (640, 480)


def _as_rgb_frame(frame: object, *, size: tuple[int, int]) -> np.ndarray:
    """Return one conventional uint8 RGB frame resized to ``size`` (width, height)."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB frame, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    if image.size != size:
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 104, 17), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _compose_multi_view_frame(observation: dict) -> np.ndarray:
    """Build a labeled 2x2 diagnostic frame without changing policy inputs."""

    camera_obs = observation["observation"]
    panel_size = (MULTI_VIEW_VIDEO_SIZE[0] // 2, MULTI_VIEW_VIDEO_SIZE[1] // 2)
    panels = (
        (observation["third_view_rgb"], "third view"),
        (camera_obs["head_camera"]["rgb"], "head"),
        (camera_obs["left_camera"]["rgb"], "left wrist"),
        (camera_obs["right_camera"]["rgb"], "right wrist"),
    )
    labeled = [
        _label_frame(_as_rgb_frame(frame, size=panel_size), label)
        for frame, label in panels
    ]
    return np.ascontiguousarray(
        np.concatenate(
            [np.concatenate(labeled[:2], axis=1), np.concatenate(labeled[2:], axis=1)],
            axis=0,
        )
    )


def _install_multi_view_recorder() -> None:
    """Patch only the recorded third-view frame; policy camera observations stay intact."""

    from envs._base_task import Base_Task

    original_get_obs = Base_Task.get_obs

    def get_obs_with_multi_view(self):
        observation = original_get_obs(self)
        if getattr(self, "eval_video_path", None) is not None:
            composite = _compose_multi_view_frame(observation)
            observation["third_view_rgb"] = composite
            self.now_obs["third_view_rgb"] = composite
        return observation

    Base_Task.get_obs = get_obs_with_multi_view


def _install_put_back_block_diagnostics() -> None:
    """Log the benchmark's own button event without changing its state transitions."""

    from envs.put_back_block import put_back_block

    original_check_success = put_back_block.check_success

    def check_success_with_diagnostics(self):
        previous_press_count = int(self.press_cnt)
        previous_stage = int(self.stage_id)
        result = original_check_success(self)
        if int(self.press_cnt) > previous_press_count:
            print(
                "[RM-Bench diagnostic] button_pressed=true "
                f"button_qpos={self.get_current_button_value('button'):.6f} "
                f"block_in_center={self.check_block_in_center()} "
                f"stage={previous_stage}->{self.stage_id}",
                flush=True,
            )
        return result

    put_back_block.check_success = check_success_with_diagnostics


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

    video_layout = str(usr_args.get("video_layout", "third_view")).strip().lower()
    if video_layout not in {"third_view", "multi_view"}:
        raise ValueError(
            f"Unsupported video_layout={video_layout!r}; expected 'third_view' or 'multi_view'"
        )
    if video_layout == "multi_view":
        _install_multi_view_recorder()
    if str(usr_args.get("task_name", "")) == "put_back_block":
        _install_put_back_block_diagnostics()

    original_eval_policy = official.eval_policy

    def eval_policy_with_limit(*args, **kwargs):
        kwargs["test_num"] = episode_limit
        if video_layout == "multi_view":
            kwargs["video_size"] = f"{MULTI_VIEW_VIDEO_SIZE[0]}x{MULTI_VIEW_VIDEO_SIZE[1]}"
        next_seed, successes, reward_sum = original_eval_policy(*args, **kwargs)
        successes, reward_sum = _normalize_official_metrics(
            successes, reward_sum, episode_limit
        )
        return next_seed, successes, reward_sum

    official.eval_policy = eval_policy_with_limit
    official.main(usr_args)


if __name__ == "__main__":
    main()
