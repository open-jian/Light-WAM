#!/usr/bin/env python3
"""Convert native RM-Bench demonstrations to Light-WAM's RoboTwin LeRobot format.

The native RM-Bench HDF5 files contain observed joint positions but no separate
command stream.  RoboTwin's LeRobot contract stores the next observed joint
position as the action at the current frame.  Consequently, a source episode
with ``T`` observations becomes ``T - 1`` LeRobot frames:

    observation.state[t] = qpos[t]
    action[t] = qpos[t + 1]

RM-Bench's existing JPEGs were produced by passing simulator RGB arrays to
OpenCV's BGR encoder.  They are decoded here with PIL and have their red/blue
channels reversed exactly once before being written as conventional RGB video.
No RM-Bench-specific color correction is needed downstream.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


CAMERA_MAPPING = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}

MOTOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


@dataclass(frozen=True)
class SourceEpisode:
    task: str
    setting: str
    source_episode: int
    hdf_path: Path
    instruction_path: Path


def _episode_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode"):
        raise ValueError(f"Unexpected RM-Bench episode filename: {path}")
    suffix = stem.removeprefix("episode")
    if not suffix.isdigit():
        raise ValueError(f"Unexpected RM-Bench episode filename: {path}")
    return int(suffix)


def discover_source_episodes(
    source_root: Path,
    *,
    setting: str,
    tasks: Sequence[str] | None,
    max_episodes_per_task: int | None,
) -> list[SourceEpisode]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"RM-Bench data root does not exist: {source_root}")

    requested_tasks = set(tasks) if tasks else None
    discovered: list[SourceEpisode] = []
    task_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for task_dir in task_dirs:
        task = task_dir.name
        if requested_tasks is not None and task not in requested_tasks:
            continue
        setting_dir = task_dir / setting
        data_dir = setting_dir / "data"
        instruction_dir = setting_dir / "instructions"
        if not data_dir.is_dir():
            continue

        hdf_paths = sorted(data_dir.glob("episode*.hdf5"), key=_episode_number)
        if max_episodes_per_task is not None:
            hdf_paths = hdf_paths[:max_episodes_per_task]
        for hdf_path in hdf_paths:
            source_episode = _episode_number(hdf_path)
            instruction_path = instruction_dir / f"episode{source_episode}.json"
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Missing instruction for {task}/{setting}/episode{source_episode}: "
                    f"{instruction_path}"
                )
            discovered.append(
                SourceEpisode(
                    task=task,
                    setting=setting,
                    source_episode=source_episode,
                    hdf_path=hdf_path,
                    instruction_path=instruction_path,
                )
            )

    if requested_tasks is not None:
        found_tasks = {episode.task for episode in discovered}
        missing_tasks = sorted(requested_tasks - found_tasks)
        if missing_tasks:
            raise FileNotFoundError(
                f"No {setting!r} demonstrations found for requested tasks: {missing_tasks}"
            )
    if not discovered:
        raise FileNotFoundError(
            f"No RM-Bench demonstrations found below {source_root} for setting {setting!r}"
        )
    return discovered


def load_instruction(path: Path, split: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instructions = payload.get(split)
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"Instruction split {split!r} is empty or invalid in {path}")
    instruction = instructions[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Instruction split {split!r} contains an invalid value in {path}")
    return instruction.strip()


def decode_legacy_rmbench_rgb_jpeg(payload: object) -> np.ndarray:
    """Decode one legacy RM-Bench JPEG into conventional HWC uint8 RGB."""

    encoded = bytes(payload)
    with Image.open(BytesIO(encoded)) as image:
        decoded_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # The JPEG was created by treating simulator RGB bytes as OpenCV BGR.
    return np.ascontiguousarray(decoded_rgb[..., ::-1])


def build_state_action(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != len(MOTOR_NAMES):
        raise ValueError(f"Expected qpos [T,14], got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"RM-Bench episode must contain at least two qpos frames, got {qpos.shape[0]}")
    if not np.isfinite(qpos).all():
        raise ValueError("RM-Bench qpos contains non-finite values")
    return np.ascontiguousarray(qpos[:-1]), np.ascontiguousarray(qpos[1:])


def _numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
    }


def _episode_stats(
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    global_index: np.ndarray,
    task_index: np.ndarray,
) -> dict[str, dict[str, list[float] | list[int]]]:
    return {
        "observation.state": _numeric_stats(state),
        "action": _numeric_stats(action),
        "timestamp": _numeric_stats(timestamp),
        "frame_index": _numeric_stats(frame_index),
        "episode_index": _numeric_stats(episode_index),
        "index": _numeric_stats(global_index),
        "task_index": _numeric_stats(task_index),
    }


def _video_output_path(root: Path, episode_index: int, camera_key: str) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{camera_key}"
        / f"episode_{episode_index:06d}.mp4"
    )


def _parquet_output_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _encode_camera_video(
    encoded_frames: Iterable[object],
    *,
    frame_count: int,
    output_path: Path,
    fps: int,
    crf: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    stderr_file = tempfile.TemporaryFile()
    width = height = -1
    written = 0
    try:
        for payload in encoded_frames:
            if written >= frame_count:
                break
            frame = decode_legacy_rmbench_rgb_jpeg(payload)
            if process is None:
                height, width = frame.shape[:2]
                command = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr_file)
            elif frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Camera resolution changed within an episode: "
                    f"expected {(height, width)}, got {frame.shape[:2]}"
                )
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
            written += 1

        if process is None:
            raise ValueError("Camera stream contains no frames")
        assert process.stdin is not None
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            stderr_file.seek(0)
            detail = stderr_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed for {output_path}: {detail}")
        if written != frame_count:
            raise ValueError(f"Expected {frame_count} camera frames, encoded {written}")
    except BaseException:
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
            process.wait()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        stderr_file.close()
    return height, width


def _write_parquet(
    output_path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    global_index: np.ndarray,
    task_index: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )
    pq.write_table(table, output_path, compression="zstd")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _dataset_info(
    *,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    height: int,
    width: int,
) -> dict[str, object]:
    features: dict[str, object] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(MOTOR_NAMES)],
            "names": [MOTOR_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(MOTOR_NAMES)],
            "names": [MOTOR_NAMES],
        },
    }
    video_info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    for camera_key in CAMERA_MAPPING:
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "rgb"],
            "info": video_info,
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAMERA_MAPPING),
        "total_chunks": (total_episodes + 999) // 1000,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }


def convert_dataset(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_episodes = discover_source_episodes(
        source_root,
        setting=args.setting,
        tasks=args.tasks,
        max_episodes_per_task=args.max_episodes_per_task,
    )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    instructions = sorted(
        {load_instruction(episode.instruction_path, args.instruction_split) for episode in source_episodes}
    )
    instruction_to_index = {instruction: index for index, instruction in enumerate(instructions)}

    episode_rows: list[dict[str, object]] = []
    episode_stats_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    total_frames = 0
    output_height = output_width = -1

    try:
        for output_episode, source_episode in enumerate(source_episodes):
            instruction = load_instruction(source_episode.instruction_path, args.instruction_split)
            with h5py.File(source_episode.hdf_path, "r") as hdf:
                state, action = build_state_action(hdf["joint_action/vector"][()])
                frame_count = state.shape[0]
                camera_group = hdf["observation"]
                for camera_key, source_camera in CAMERA_MAPPING.items():
                    encoded_frames = camera_group[source_camera]["rgb"]
                    if len(encoded_frames) < frame_count:
                        raise ValueError(
                            f"{source_episode.hdf_path}: camera {source_camera} has "
                            f"{len(encoded_frames)} frames, expected at least {frame_count}"
                        )
                    height, width = _encode_camera_video(
                        encoded_frames,
                        frame_count=frame_count,
                        output_path=_video_output_path(output_root, output_episode, camera_key),
                        fps=args.fps,
                        crf=args.video_crf,
                    )
                    if output_height < 0:
                        output_height, output_width = height, width
                    elif (height, width) != (output_height, output_width):
                        raise ValueError(
                            f"Dataset camera resolution changed from "
                            f"{(output_height, output_width)} to {(height, width)}"
                        )

            frame_index = np.arange(frame_count, dtype=np.int64)
            episode_index = np.full(frame_count, output_episode, dtype=np.int64)
            global_index = np.arange(total_frames, total_frames + frame_count, dtype=np.int64)
            task_index = np.full(frame_count, instruction_to_index[instruction], dtype=np.int64)
            timestamp = frame_index.astype(np.float32) / float(args.fps)
            _write_parquet(
                _parquet_output_path(output_root, output_episode),
                state=state,
                action=action,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index,
                global_index=global_index,
                task_index=task_index,
            )

            episode_rows.append(
                {
                    "episode_index": output_episode,
                    "tasks": [instruction],
                    "length": frame_count,
                    "raw_file_name": str(source_episode.hdf_path.relative_to(source_root)),
                }
            )
            episode_stats_rows.append(
                {
                    "episode_index": output_episode,
                    "stats": _episode_stats(
                        state=state,
                        action=action,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        episode_index=episode_index,
                        global_index=global_index,
                        task_index=task_index,
                    ),
                }
            )
            provenance_rows.append(
                {
                    "output_episode": output_episode,
                    "task": source_episode.task,
                    "setting": source_episode.setting,
                    "source_episode": source_episode.source_episode,
                    "source_hdf": str(source_episode.hdf_path.relative_to(source_root)),
                    "source_instruction": str(
                        source_episode.instruction_path.relative_to(source_root)
                    ),
                    "frames": frame_count,
                    "instruction": instruction,
                }
            )
            total_frames += frame_count
            print(
                f"[{output_episode + 1}/{len(source_episodes)}] "
                f"{source_episode.task}/episode{source_episode.source_episode}: {frame_count} frames",
                flush=True,
            )

        _write_json(
            output_root / "meta" / "info.json",
            _dataset_info(
                fps=args.fps,
                total_episodes=len(source_episodes),
                total_frames=total_frames,
                total_tasks=len(instructions),
                height=output_height,
                width=output_width,
            ),
        )
        _write_jsonl(
            output_root / "meta" / "tasks.jsonl",
            (
                {"task_index": index, "task": instruction}
                for index, instruction in enumerate(instructions)
            ),
        )
        _write_jsonl(output_root / "meta" / "episodes.jsonl", episode_rows)
        _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episode_stats_rows)
        manifest = {
            "schema_version": 1,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "setting": args.setting,
            "instruction_split": args.instruction_split,
            "fps": args.fps,
            "image_encoding": "legacy_cv2_encoded_simulator_rgb",
            "output_color_space": "RGB",
            "camera_mapping": CAMERA_MAPPING,
            "state_contract": "observation.state[t] = source qpos[t]",
            "action_contract": "action[t] = source qpos[t+1]",
            "memory_enabled": False,
            "total_episodes": len(source_episodes),
            "total_frames": total_frames,
            "total_tasks": len(instructions),
            "episodes": provenance_rows,
        }
        _write_json(output_root / "meta" / "rmbench_conversion.json", manifest)
        return manifest
    except BaseException:
        if not args.keep_partial:
            shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data2/jian/benchmark/RMBench/RMBench/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/rmbench/lerobot_no_memory"),
    )
    parser.add_argument("--setting", default="demo_clean")
    parser.add_argument("--instruction-split", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-partial", action="store_true")
    args = parser.parse_args()
    if args.max_episodes_per_task is not None and args.max_episodes_per_task <= 0:
        parser.error("--max-episodes-per-task must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be in [0, 51]")
    return args


def main() -> None:
    args = parse_args()
    manifest = convert_dataset(args)
    print(
        "Conversion complete: "
        f"episodes={manifest['total_episodes']} frames={manifest['total_frames']} "
        f"tasks={manifest['total_tasks']} output={manifest['output_root']}"
    )


if __name__ == "__main__":
    main()
