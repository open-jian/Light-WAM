import torch
import numpy as np
from collections import deque
from pathlib import Path

from hydra import compose, initialize_config_dir

from lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from lightwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from lightwam.models.wan22.lightwam import LightWAM
from lightwam.models.wan22.wan_video_dit import WanVideoDiT
from lightwam.trainer import Wan22Trainer
from experiments.robotwin.lightwam_policy.deploy_policy import WorldActionRobotWinPolicy


ROOT = Path(__file__).resolve().parents[1]


def _assert_block(mask, query_slice, key_slice, expected):
    block = mask[query_slice, key_slice]
    assert bool(block.all().item()) is expected


def test_recent_history_mask_prevents_future_leakage_and_padding_reads():
    # Three history frames (2 tokens each), then current/future/future (3 tokens each).
    valid = torch.tensor([[False, True, True]], dtype=torch.bool)
    mask = LightWAM._build_recent_history_attention_mask(
        history_valid_mask=valid,
        history_tokens_per_frame=2,
        main_num_frames=3,
        main_tokens_per_frame=3,
        device=torch.device("cpu"),
    )[0, 0]

    h0, h1, h2 = slice(0, 2), slice(2, 4), slice(4, 6)
    current, future = slice(6, 9), slice(9, 15)

    _assert_block(mask, h0, h0, True)  # isolated padding query; avoids all-masked rows
    _assert_block(mask, h1, h0, False)
    _assert_block(mask, h1, h1, True)
    _assert_block(mask, h1, h2, False)
    _assert_block(mask, h2, h1, True)
    _assert_block(mask, h2, h2, True)
    _assert_block(mask, h2, current, False)

    _assert_block(mask, current, h0, False)
    _assert_block(mask, current, h1, True)
    _assert_block(mask, current, h2, True)
    _assert_block(mask, current, current, True)
    _assert_block(mask, current, future, False)

    _assert_block(mask, future, h0, False)
    _assert_block(mask, future, h1, True)
    _assert_block(mask, future, current, True)
    _assert_block(mask, future, future, True)


def test_processor_splits_fixed_history_candidates_before_video_clip():
    processor = object.__new__(LightWAMProcessor)
    processor.history_num_frames = 2
    processor.num_obs_steps = 3
    pixels = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1, 1)
    padding = torch.tensor([True, False, False, False, True])

    history, current, history_is_pad, current_is_pad = processor._split_history_from_image_data(
        pixels,
        padding,
    )

    assert history[:, :, 0, 0, 0].tolist() == [[0.0, 1.0]]
    assert current[:, :, 0, 0, 0].tolist() == [[2.0, 3.0, 4.0]]
    assert history_is_pad.tolist() == [True, False]
    assert current_is_pad.tolist() == [False, False, True]


def test_cached_history_never_crosses_episode_and_does_not_repeat_first_frame():
    dataset = object.__new__(RobotVideoDataset)
    dataset.history_enabled = True
    dataset.history_max_size = 3
    dataset.history_frame_offsets = [-12, -8, -4]
    dataset.lerobot_dataset = type(
        "FakeBaseDataset",
        (),
        {
            "global_sample_stride": 1,
            "episode_data_index": {
                "from": torch.tensor([5]),
                "to": torch.tensor([20]),
            },
        },
    )()
    loaded_indices = []

    def load_cached(sample_idx):
        loaded_indices.append(sample_idx)
        return torch.full((1, 3, 1, 1), float(sample_idx))

    dataset._load_cached_video_latents = load_cached
    history, valid = dataset._load_cached_history_latents(
        sample_idx=10,
        history_valid_mask=torch.ones(3, dtype=torch.bool),
        reference_latents=torch.zeros(1, 3, 1, 1),
    )

    assert loaded_indices == [6]
    assert valid.tolist() == [False, False, True]
    assert history[:, :, 0, 0].tolist() == [[0.0, 0.0, 6.0]]


def test_mixed_resolution_rope_aligns_libero_history_to_current_grid():
    dit = WanVideoDiT(
        hidden_dim=12,
        in_dim=1,
        ffn_dim=24,
        out_dim=1,
        text_dim=4,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        attn_head_dim=12,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )
    timestep = torch.zeros(1)
    context = torch.zeros(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)

    lowres = dit.pre_dit(
        x=torch.zeros(1, 1, 1, 14, 28),
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        frame_position_offset=3,
        spatial_position_scale=2,
    )
    highres = dit.pre_dit(
        x=torch.zeros(1, 1, 1, 28, 56),
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        frame_position_offset=3,
        spatial_position_scale=1,
    )

    assert lowres["meta"]["tokens_per_frame"] == 98
    assert highres["meta"]["tokens_per_frame"] == 392
    # Low-res patch coordinate (6,13) maps to high-res coordinate (12,26).
    torch.testing.assert_close(lowres["freqs"][97], highres["freqs"][362])


def test_random_window_size_stays_within_configured_bounds():
    model = object.__new__(LightWAM)
    torch.nn.Module.__init__(model)
    model.history_enabled = True
    model.history_min_size = 4
    model.history_max_size = 12
    model.device = torch.device("cpu")
    model.train()

    sampled = {model._sample_history_window_size() for _ in range(100)}
    assert sampled <= set(range(4, 13))
    assert len(sampled) > 1
    model.eval()
    assert model._sample_history_window_size() == 12


def test_robotwin_policy_samples_real_observations_every_four_steps_after_inference():
    policy = object.__new__(WorldActionRobotWinPolicy)
    policy.history_enabled = True
    policy.history_raw_stride = 4
    policy.history_max_size = 12
    policy.recent_observations = deque(maxlen=12)
    policy.pending_actions = deque()
    policy.step_count = 0
    policy.timing_enabled = False
    policy._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    observed_histories = []

    policy._build_robotwin_image_tensor = lambda observation: torch.full(
        (1, 3, 16, 16), float(observation["id"])
    )

    def fill_action_queue(observation, instruction, image_tensor=None):
        observed_histories.append([float(frame[0, 0, 0]) for frame in policy.recent_observations])
        policy.pending_actions.append(np.zeros(2, dtype=np.float32))

    policy._fill_action_queue = fill_action_queue

    class FakeEnv:
        def get_instruction(self):
            return "task"

        def take_action(self, action, action_type):
            assert action_type == "qpos"

    env = FakeEnv()
    policy.step(env, {"id": 0})
    assert observed_histories == [[]]
    assert [float(frame[0, 0, 0]) for frame in policy.recent_observations] == [0.0]

    policy.pending_actions.append(np.zeros(2, dtype=np.float32))
    policy.step_count = 4
    assert policy.should_request_observation()
    policy.step(env, {"id": 4})
    assert observed_histories == [[]]  # no replan; only remember the real observation
    assert [float(frame[0, 0, 0]) for frame in policy.recent_observations] == [0.0, 4.0]


def test_future_output_backpropagates_to_old_history_without_leaking_back_to_current():
    dit = WanVideoDiT(
        hidden_dim=12,
        in_dim=1,
        ffn_dim=24,
        out_dim=1,
        text_dim=4,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=1,
        attn_head_dim=12,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    ).eval()
    model = object.__new__(LightWAM)
    torch.nn.Module.__init__(model)
    model.video_expert = dit
    model.set_memory_gradient_monitoring(True)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-3)
    weight_before = dit.patch_embedding.weight.detach().clone()

    context = torch.zeros(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    history = torch.randn(1, 1, 2, 4, 4, requires_grad=True)
    current = torch.randn(1, 1, 1, 4, 4)

    def run(future):
        history_pre = dit.pre_dit(
            x=history,
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        main_pre = dit.pre_dit(
            x=torch.cat([current, future], dim=2),
            timestep=torch.ones(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            frame_position_offset=2,
        )
        merged, main_slice = model._merge_history_and_main_pre(
            history_pre=history_pre,
            main_pre=main_pre,
            history_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            monitor_branch="video",
        )
        output = dit.forward_backbone(merged)
        current_slice = slice(main_slice.start, main_slice.start + 4)
        future_slice = slice(main_slice.start + 4, main_slice.stop)
        return output[:, current_slice], output[:, future_slice]

    current_a, future_a = run(torch.zeros(1, 1, 1, 4, 4))
    current_b, _ = run(torch.ones(1, 1, 1, 4, 4))
    torch.testing.assert_close(current_a, current_b, atol=1e-6, rtol=1e-6)

    future_a.square().mean().backward()
    assert history.grad is not None
    assert float(history.grad[:, :, 0].abs().sum().item()) > 0.0
    memory_stats = model.pop_memory_gradient_stats()
    for key in (
        "video/history_grad_rms",
        "video/current_grad_rms",
        "video/age_01_grad_rms",
        "video/age_02_grad_rms",
    ):
        assert key in memory_stats
        value_sum, value_count = memory_stats[key]
        assert float(value_count.item()) > 0.0
        assert float(value_sum.item()) > 0.0
    optimizer.step()
    assert not torch.equal(weight_before, dit.patch_embedding.weight.detach())


def test_shuffled_memory_probe_rolls_history_and_mask_together():
    sample = {
        "video": torch.tensor([[10.0], [20.0], [30.0]]),
        "history_video": torch.tensor([[1.0], [2.0], [3.0]]),
        "history_valid_mask": torch.tensor([[True], [False], [True]]),
    }

    shuffled = Wan22Trainer._build_shuffled_history_sample(sample)

    assert shuffled is not None
    assert shuffled["video"] is sample["video"]
    assert shuffled["history_video"].flatten().tolist() == [3.0, 1.0, 2.0]
    assert shuffled["history_valid_mask"].flatten().tolist() == [True, True, False]


def test_libero_memory_monitor_is_enabled_and_uses_mem_metric_group():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["task=libero_uncond_2cam224_1e-4"])
    assert cfg.memory_monitor.enabled is True
    assert cfg.memory_monitor.shuffled_probe_every == 500
    assert cfg.memory_monitor.probe_batch_size == 2

    trainer = object.__new__(Wan22Trainer)
    trainer._memory_gradient_accumulator = {
        "video/history_grad_rms": (torch.tensor(6.0), torch.tensor(3.0)),
        "video/current_grad_rms": (torch.tensor(2.0), torch.tensor(2.0)),
    }
    trainer.accelerator = type(
        "FakeAccelerator",
        (),
        {
            "device": torch.device("cpu"),
            "gather": staticmethod(lambda value: value),
        },
    )()

    metrics = trainer._consume_global_memory_gradient_metrics()

    assert metrics["mem/video/history_grad_rms"] == 2.0
    assert metrics["mem/video/current_grad_rms"] == 1.0
    assert metrics["mem/video/history_to_current_grad_ratio"] == 2.0
    assert all(key.startswith("mem/") for key in metrics)
