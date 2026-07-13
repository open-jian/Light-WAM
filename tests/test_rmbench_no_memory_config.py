from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def _forbidden_paths(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "memory" in str(key).lower() or "history" in str(key).lower():
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def test_rmbench_no_memory_config_reuses_robotwin_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["task=rmbench_no_memory_3cam_384_1e-4"])
        robotwin = compose(config_name="train", overrides=["task=robotwin_uncond_3cam_384_1e-4"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert cfg.data.train._target_.endswith("RobotVideoDataset")
    assert cfg.data.train.num_frames == 33
    assert cfg.data.train.action_video_freq_ratio == 4
    assert list(cfg.data.train.video_size) == [384, 320]
    assert cfg.data.train.concat_multi_camera == "robotwin"
    assert cfg.data.train.processor.num_output_cameras == 3
    assert [item.key for item in cfg.data.train.shape_meta.images] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert cfg.model.loss.action_temporal_weighting.enabled is False
    for key in (
        "batch_size",
        "learning_rate",
        "lr_scheduler_type",
        "num_epochs",
        "gradient_accumulation_steps",
        "weight_decay",
    ):
        assert cfg[key] == robotwin[key]
    for key in (
        "num_frames",
        "global_sample_stride",
        "action_video_freq_ratio",
        "video_size",
        "concat_multi_camera",
    ):
        assert cfg.data.train[key] == robotwin.data.train[key]
    assert _forbidden_paths(resolved) == []
