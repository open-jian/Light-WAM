#!/usr/bin/env python3
"""Fail-closed parity gate for the converted RM-Bench no-Memory baseline."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import av
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from experiments.rmbench.lightwam_policy.deploy_policy import WorldActionRobotWinPolicy
from lightwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from lightwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from scripts.convert_rmbench_to_lerobot import CAMERA_MAPPING, decode_legacy_rmbench_rgb_jpeg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_video_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"Video frame {frame_index} unavailable in {path}")


def _decode_uncorrected_jpeg(payload: object) -> np.ndarray:
    with Image.open(BytesIO(bytes(payload))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mae(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32)).mean())


def _forbidden_config_paths(value, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "memory" in str(key).lower() or "history" in str(key).lower():
                found.append(path)
            found.extend(_forbidden_config_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_config_paths(child, f"{prefix}[{index}]"))
    return found


def _build_online_policy(processor, video_size: tuple[int, int]):
    policy = object.__new__(WorldActionRobotWinPolicy)
    policy.processor = processor
    policy.concat_multi_camera = "robotwin"
    policy.video_size = video_size
    policy.resize_transform = ResizeSmallestSideAspectPreserving(
        args={"img_w": video_size[1], "img_h": video_size[0]}
    )
    policy.crop_transform = CenterCrop(args={"img_w": video_size[1], "img_h": video_size[0]})
    policy.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})
    policy.model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32)
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/rmbench/lerobot_no_memory"))
    parser.add_argument("--stats", type=Path, default=Path("data/rmbench/dataset_stats.json"))
    parser.add_argument("--output-json", type=Path, default=Path("runs/rmbench_no_memory/step4/parity.json"))
    parser.add_argument("--rgb-mae-threshold", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_json = cli.output_json.expanduser().resolve()
    output_json.unlink(missing_ok=True)
    dataset_root = cli.dataset_root.resolve()
    manifest = json.loads((dataset_root / "meta" / "rmbench_conversion.json").read_text())
    if manifest.get("memory_enabled") is not False:
        raise RuntimeError("Converted dataset does not declare memory_enabled=false")

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["task=rmbench_no_memory_3cam_384_1e-4"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    forbidden = _forbidden_config_paths(resolved)
    if forbidden:
        raise RuntimeError(f"Memory/history keys found in resolved config: {forbidden}")

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    train_cfg.dataset_dirs = [str(dataset_root)]
    train_cfg.pretrained_norm_stats = str(cli.stats.resolve())
    train_cfg.val_set_proportion = 0.0
    train_cfg.use_latent_cache = False
    train_cfg.latent_cache_dir = None
    dataset = instantiate(train_cfg)

    eval_processor = instantiate(cfg.data.train.processor).eval()
    eval_processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(cli.stats.resolve())))
    video_size = tuple(int(value) for value in cfg.data.train.video_size)
    online_policy = _build_online_policy(eval_processor, video_size)
    inner = dataset.lerobot_dataset.multi_dataset._datasets[0]

    first_by_task = {}
    action_checked = 0
    for row in manifest["episodes"]:
        first_by_task.setdefault(row["task"], row)
        parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{row['output_episode']:06d}.parquet"
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        source_path = Path(manifest["source_root"]) / row["source_hdf"]
        with h5py.File(source_path, "r") as hdf:
            qpos = np.asarray(hdf["joint_action/vector"], dtype=np.float32)
        if not np.array_equal(state, qpos[:-1]) or not np.array_equal(action, qpos[1:]):
            raise RuntimeError(f"Next-qpos alignment failure: {source_path}")
        action_checked += 1

    task_reports = []
    for task, row in sorted(first_by_task.items()):
        episode = int(row["output_episode"])
        video_keys = [f"observation.images.{key}" for key in CAMERA_MAPPING]
        raw = inner._query_videos({key: [0.0] for key in video_keys}, ep_idx=episode)
        converted = {
            key.removeprefix("observation.images."): (
                raw[key].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
            )
            for key in video_keys
        }

        repeated = {
            key: torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).repeat(33, 1, 1, 1)
            for key, image in converted.items()
        }
        train_pixels = dataset.processor.build_pixel_values_from_episode_images({"images": repeated})
        train_video, _ = dataset._finalize_video_tensor(
            train_pixels, torch.zeros(33, dtype=torch.bool)
        )
        observation = {
            "observation": {
                source_key: {"rgb": converted[output_key]}
                for output_key, source_key in CAMERA_MAPPING.items()
            }
        }
        online = online_policy._build_robotwin_image_tensor(observation)[0]
        tensor_max_abs = float((train_video[:, 0].float() - online.float()).abs().max())
        if tensor_max_abs != 0.0:
            raise RuntimeError(f"Train/eval tensor mismatch for {task}: max_abs={tensor_max_abs}")

        source_path = Path(manifest["source_root"]) / row["source_hdf"]
        camera_mae = {}
        with h5py.File(source_path, "r") as hdf:
            for output_key, source_key in CAMERA_MAPPING.items():
                payload = hdf[f"observation/{source_key}/rgb"][0]
                expected = decode_legacy_rmbench_rgb_jpeg(payload)
                value = _mae(converted[output_key], expected)
                if value > cli.rgb_mae_threshold:
                    raise RuntimeError(f"Converted RGB MAE too high for {task}/{source_key}: {value}")
                camera_mae[output_key] = value
            head_payload = hdf["observation/head_camera/rgb"][0]
            corrected_head = decode_legacy_rmbench_rgb_jpeg(head_payload)
            uncorrected_head = _decode_uncorrected_jpeg(head_payload)
        source_video = Path(manifest["source_root"]) / task / row["setting"] / "video" / f"episode{row['source_episode']}.mp4"
        source_head = _read_video_frame(source_video)
        corrected_source_mae = _mae(corrected_head, source_head)
        uncorrected_source_mae = _mae(uncorrected_head, source_head)
        if corrected_source_mae >= uncorrected_source_mae:
            raise RuntimeError(f"Legacy RGB correction is not beneficial for {task}")
        task_reports.append(
            {
                "task": task,
                "output_episode": episode,
                "camera_rgb_mae": camera_mae,
                "corrected_head_source_mae": corrected_source_mae,
                "uncorrected_head_source_mae": uncorrected_source_mae,
                "train_eval_tensor_max_abs": tensor_max_abs,
            }
        )

    report = {
        "passed": True,
        "memory_enabled": False,
        "episodes_action_checked": action_checked,
        "tasks_checked": len(task_reports),
        "camera_order": list(CAMERA_MAPPING),
        "action_contract": manifest["action_contract"],
        "train_eval_tensor_global_max_abs": max(
            item["train_eval_tensor_max_abs"] for item in task_reports
        ),
        "converted_rgb_global_max_mae": max(
            max(item["camera_rgb_mae"].values()) for item in task_reports
        ),
        "tasks": task_reports,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
