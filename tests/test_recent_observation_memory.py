import torch
import numpy as np
import json
import pytest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from lightwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from lightwam.memory import (
    apply_recent_window_mask,
    build_anchor_recent_history_indices,
)
from lightwam.models.wan22.lightwam import DisabledActionExpert, LightWAM
from lightwam.models.wan22.mot import MoT
from lightwam.models.wan22.state_fusion_action_expert import StateFusionActionExpert
from lightwam.models.wan22.wan_video_dit import WanVideoDiT
from lightwam.trainer import Wan22Trainer
from experiments.robotwin.lightwam_policy.deploy_policy import WorldActionRobotWinPolicy
from scripts.precompute_video_latents import (
    _build_history_from_episode_main_latents,
    _encode_independent_history_latents,
    _validate_existing_memory_cache_meta,
)


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


def test_video_only_precompute_returns_main_and_max_history():
    dataset = object.__new__(RobotVideoDataset)
    dataset.history_enabled = True
    dataset.history_max_size = 3
    dataset.num_frames = 3
    dataset.video_only = True
    dataset.use_latent_cache = False

    class FakeProcessor:
        @staticmethod
        def _build_pixel_values_from_images_impl(sample, expected_num_obs_steps):
            assert expected_num_obs_steps == 6
            return sample["images"]["image"].unsqueeze(0)

    dataset.processor = FakeProcessor()
    def finalize(*, video, image_is_pad, temporal_indices=None):
        if temporal_indices is None:
            temporal_indices = [0, 1, 2]
        return (
            video[0, temporal_indices].permute(1, 0, 2, 3).contiguous(),
            image_is_pad[temporal_indices],
        )

    dataset._finalize_video_tensor = finalize
    sample = {
        "images": {
            "image": torch.arange(18, dtype=torch.float32).reshape(6, 3, 1, 1),
        },
        "image_is_pad": torch.tensor([True, False, False, False, False, True]),
    }

    video, history, valid = dataset._build_video_and_history_for_cache(sample)

    assert valid.tolist() == [False, True, True]
    assert history[:, 0].eq(0).all()
    assert video.shape == (3, 3, 1, 1)
    assert video[:, 0, 0, 0].tolist() == [9.0, 10.0, 11.0]


def test_memory_cache_entry_requires_fixed_history_latents_and_mask():
    dataset = object.__new__(RobotVideoDataset)
    dataset.history_enabled = True
    dataset.history_max_size = 3
    entry = dataset._validate_cached_latent_entry(
        {
            "video_latents": torch.zeros(16, 3, 2, 2),
            "history_video_latents": torch.ones(16, 3, 2, 2),
            "history_valid_mask": torch.tensor([False, True, True]),
        },
        "fake.pt",
    )

    assert entry["video_latents"].shape == (16, 3, 2, 2)
    assert entry["history_video_latents"].shape == (16, 3, 2, 2)
    assert entry["history_valid_mask"].tolist() == [False, True, True]


def test_history_cache_encoder_uses_independent_t1_inputs_and_zeroes_padding():
    calls = []

    class FakeModel:
        device = torch.device("cpu")
        torch_dtype = torch.float32

        @staticmethod
        def _encode_video_latents(video, tiled):
            assert tiled is False
            assert video.shape[2] == 1
            calls.append(video[:, 0, 0, 0, 0].tolist())
            return video[:, :1]

    history_video = torch.zeros(1, 3, 3, 1, 1)
    history_video[0, 0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    history = _encode_independent_history_latents(
        model=FakeModel(),
        history_video=history_video,
        history_valid_mask=torch.tensor([[False, True, True]]),
        tiled=False,
        cache_dtype=torch.float32,
        encode_batch_size=2,
    )

    assert calls == [[1.0, 2.0], [3.0]]
    assert history.shape == (1, 1, 3, 1, 1)
    assert history.flatten().tolist() == [0.0, 2.0, 3.0]


def test_episode_cache_reuses_earlier_samples_first_latent_for_history():
    video_latents = torch.zeros(6, 1, 3, 1, 1)
    video_latents[:, 0, 0, 0, 0] = torch.arange(6, dtype=torch.float32)
    history, valid = _build_history_from_episode_main_latents(
        video_latents=video_latents,
        sample_indices=list(range(10, 16)),
        episode_sample_start=10,
        episode_sample_end=16,
        history_frame_offsets=[-4, -2],
    )

    assert valid.tolist() == [
        [False, False],
        [False, False],
        [False, True],
        [False, True],
        [True, True],
        [True, True],
    ]
    assert history[:, 0, :, 0, 0].tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [1.0, 3.0],
    ]


def test_episode_cache_derives_anchor_recent_history_without_stored_copies(tmp_path):
    video_latents = torch.zeros(6, 1, 3, 1, 1)
    video_latents[:, 0, 0, 0, 0] = torch.arange(6, dtype=torch.float32)
    torch.save(
        {
            "sample_indices": torch.arange(10, 16, dtype=torch.int64),
            "video_latents": video_latents,
        },
        tmp_path / "episode.pt",
    )
    dataset = object.__new__(RobotVideoDataset)
    dataset.latent_cache_dir = str(tmp_path)
    dataset._latent_cache_format = "episode_packed_v1"
    dataset._latent_cache_shard_paths = ["episode.pt"]
    dataset._latent_cache_sample_to_shard = torch.zeros(16, dtype=torch.int64)
    dataset._latent_cache_sample_to_offset = torch.zeros(16, dtype=torch.int64)
    dataset._latent_cache_sample_to_offset[15] = 5
    dataset._latent_cache_last_shard_relpath = None
    dataset._latent_cache_last_shard_payload = None
    dataset.history_enabled = True
    dataset.history_max_size = 4
    dataset.history_anchor_size = 2
    dataset.history_anchor_stride = 1
    dataset.history_recent_max_size = 2
    dataset.history_recent_stride = 1

    entry = dataset._load_cached_latent_entry_from_shard(15)

    assert entry["history_valid_mask"].tolist() == [True, True, True, True]
    assert entry["history_video_latents"][0, :, 0, 0].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_memory_cache_resume_rejects_relabeling_another_split(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"split_contract": {"is_training_set": True}}))

    with pytest.raises(ValueError, match="does not match"):
        _validate_existing_memory_cache_meta(
            meta_path=meta_path,
            expected_meta={"split_contract": {"is_training_set": False}},
        )


def test_memory_cache_reader_rejects_train_val_split_mismatch(tmp_path):
    torch.save(
        {
            "storage_format": "episode_packed_v1",
            "num_samples": 2,
            "shard_paths": ["episode.pt"],
            "sample_to_shard": torch.zeros(2, dtype=torch.int64),
            "sample_to_offset": torch.arange(2, dtype=torch.int64),
        },
        tmp_path / "index.pt",
    )
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "split_contract": {"name": "train"},
                "history_cache": {
                    "enabled": True,
                    "max_size": 4,
                    "raw_stride": 1,
                    "anchor_size": 2,
                    "anchor_stride": 1,
                    "recent_min_size": 1,
                    "recent_max_size": 2,
                    "recent_stride": 1,
                    "encoding": "derived_from_episode_main_first_latent_t1_no_duplication",
                },
            }
        )
    )

    dataset = object.__new__(RobotVideoDataset)
    dataset.latent_cache_dir = str(tmp_path)
    dataset.lerobot_dataset = type("FakeDataset", (), {"__len__": lambda self: 2})()
    dataset.history_enabled = True
    dataset.history_max_size = 4
    dataset.history_raw_stride = 1
    dataset.history_anchor_size = 2
    dataset.history_anchor_stride = 1
    dataset.history_recent_min_size = 1
    dataset.history_recent_max_size = 2
    dataset.history_recent_stride = 1
    dataset.get_latent_cache_split_contract = lambda: {"name": "val"}

    with pytest.raises(ValueError, match="episode split mismatch"):
        dataset._init_latent_cache_reader()


def test_latent_cache_reader_rejects_incomplete_split_index(tmp_path):
    torch.save(
        {
            "storage_format": "episode_packed_v1",
            "num_samples": 1,
            "shard_paths": ["episode.pt"],
            "sample_to_shard": torch.zeros(1, dtype=torch.int64),
            "sample_to_offset": torch.zeros(1, dtype=torch.int64),
        },
        tmp_path / "index.pt",
    )
    dataset = object.__new__(RobotVideoDataset)
    dataset.latent_cache_dir = str(tmp_path)
    dataset.lerobot_dataset = type("FakeDataset", (), {"__len__": lambda self: 2})()

    with pytest.raises(ValueError, match="sample count mismatch"):
        dataset._init_latent_cache_reader()


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


def test_independent_history_observations_can_share_t0_rope():
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
    pre = dit.pre_dit(
        x=torch.zeros(1, 1, 3, 4, 4),
        timestep=torch.zeros(1),
        context=torch.zeros(1, 2, 4),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
        temporal_position_ids=torch.zeros(3, dtype=torch.long),
    )
    tokens_per_frame = pre["meta"]["tokens_per_frame"]
    torch.testing.assert_close(
        pre["freqs"][:tokens_per_frame],
        pre["freqs"][tokens_per_frame : 2 * tokens_per_frame],
    )
    torch.testing.assert_close(
        pre["freqs"][:tokens_per_frame],
        pre["freqs"][2 * tokens_per_frame :],
    )


def test_random_window_size_stays_within_configured_bounds():
    model = object.__new__(LightWAM)
    torch.nn.Module.__init__(model)
    model.history_enabled = True
    model.history_min_size = 2
    model.history_max_size = 5
    model.history_anchor_size = 1
    model.history_recent_min_size = 1
    model.history_recent_max_size = 4
    model.device = torch.device("cpu")
    model.video_expert = torch.nn.Identity()
    # This is the actual adapter-training state: wrapper eval, expert train.
    model.eval()
    model.video_expert.train()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        sampled = {model._sample_history_window_size() for _ in range(100)}
    assert sampled == set(range(2, 6))
    model.video_expert.eval()
    assert model._sample_history_window_size() == 5


def test_anchor_recent_slots_keep_start_and_recent_without_duplicates():
    indices, valid = build_anchor_recent_history_indices(
        target_local_indices=[0, 3, 8],
        episode_length=80,
        history_max_size=5,
        anchor_size=1,
        anchor_stride=1,
        recent_max_size=4,
        recent_stride=1,
    )
    assert valid[0].sum().item() == 0
    assert indices[1].tolist() == [0, 0, 0, 1, 2]
    assert valid[1].tolist() == [True, False, False, True, True]
    assert indices[2].tolist() == [0, 4, 5, 6, 7]
    assert valid[2].all()
    assert not bool((indices[2][valid[2]] == 8).any().item())

    short = apply_recent_window_mask(valid[2:3], anchor_size=1, recent_window_size=2)
    assert short[0, :1].all()
    assert not short[0, 1:3].any()
    assert short[0, 3:].all()


def test_robotwin_policy_samples_every_natural_frame_after_inference():
    policy = object.__new__(WorldActionRobotWinPolicy)
    policy.history_enabled = True
    policy.history_raw_stride = 1
    policy.history_max_size = 5
    policy.history_anchor_size = 1
    policy.history_recent_max_size = 4
    policy.anchor_observation_latents = []
    policy.recent_observation_latents = deque(maxlen=4)
    policy.pending_actions = deque()
    policy.step_count = 0
    policy.timing_enabled = False
    policy._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    policy.tiled = False
    observed_histories = []

    policy._build_robotwin_image_tensor = lambda observation: torch.full(
        (1, 3, 16, 16), float(observation["id"])
    )

    def fill_action_queue(observation, instruction, image_tensor=None):
        observed_histories.append(
            [float(frame[0, 0, 0]) for frame in policy.anchor_observation_latents]
            + [float(frame[0, 0, 0]) for frame in policy.recent_observation_latents]
        )
        policy.pending_actions.append(np.zeros(2, dtype=np.float32))
        return image_tensor[0]

    policy._fill_action_queue = fill_action_queue
    policy.model = type(
        "FakeModel",
        (),
        {"encode_observation_latent": staticmethod(lambda image, tiled: image[0].cpu())},
    )()

    class FakeEnv:
        def get_instruction(self):
            return "task"

        def take_action(self, action, action_type):
            assert action_type == "qpos"

    env = FakeEnv()
    policy.step(env, {"id": 0})
    assert observed_histories == [[]]
    assert [float(frame[0, 0, 0]) for frame in policy.anchor_observation_latents] == [0.0]

    policy.pending_actions.append(np.zeros(2, dtype=np.float32))
    policy.step_count = 1
    assert policy.should_request_observation()
    policy.step(env, {"id": 1})
    assert observed_histories == [[]]  # no replan; only remember the real observation
    assert [float(frame[0, 0, 0]) for frame in policy.anchor_observation_latents] == [0.0]
    assert [float(frame[0, 0, 0]) for frame in policy.recent_observation_latents] == [1.0]


def test_online_memory_keeps_frame_zero_and_four_latest_frames():
    policy = object.__new__(WorldActionRobotWinPolicy)
    policy.history_enabled = True
    policy.history_max_size = 5
    policy.history_anchor_size = 1
    policy.history_recent_max_size = 4
    policy.anchor_observation_latents = []
    policy.recent_observation_latents = deque(maxlen=4)
    for value in range(8):
        policy._remember_observation_latent(
            torch.full((1, 1, 1), float(value), dtype=torch.float32)
        )

    packed, valid = policy._pack_observation_memory()

    assert valid.all()
    assert packed[0, :, 0, 0].tolist() == [0.0, 4.0, 5.0, 6.0, 7.0]


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


def test_memory_gradient_monitor_enables_frozen_input_boundary_gradients():
    model = object.__new__(LightWAM)
    torch.nn.Module.__init__(model)
    model.video_expert = torch.nn.Identity()
    model.set_memory_gradient_monitoring(True)
    # Adapter training leaves the parent in eval while the video expert trains.
    model.eval()
    model.video_expert.train()

    history_tokens = torch.randn(1, 2, 4)
    main_tokens = torch.randn(1, 3, 4)
    model._register_memory_gradient_hooks(
        branch="video",
        history_tokens=history_tokens,
        main_tokens=main_tokens,
        history_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        history_tokens_per_frame=1,
        main_tokens_per_frame=1,
    )

    assert history_tokens.requires_grad
    assert main_tokens.requires_grad
    torch.cat([history_tokens, main_tokens], dim=1).square().mean().backward()
    stats = model.pop_memory_gradient_stats()
    assert "video/history_grad_rms" in stats
    assert "video/current_grad_rms" in stats


def test_tiny_memory_training_step_updates_slot_embeddings_and_not_video_loss():
    class FakeVAE:
        temporal_downsample_factor = 4
        upsampling_factor = 1
        z_dim = 1
        model = SimpleNamespace(z_dim=1)

        def to(self, *args, **kwargs):
            return self

    video_expert = WanVideoDiT(
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
        use_wam_adapter=True,
        adapter_layer_indices=[0],
        adapter_dim=4,
    )
    action_expert = DisabledActionExpert(action_dim=2)
    state_fusion = StateFusionActionExpert(
        video_hidden_dim=12,
        action_dim=2,
        num_fusion_layers=1,
        per_layer_dim=8,
        trunk_dim=16,
        num_trunk_blocks=1,
        step_pos_dim=8,
        token_pooling_type="mean",
        token_pooling_num_heads=1,
        feature_sources=["adapted"],
    )
    model = LightWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=MoT({"video": video_expert}, mot_checkpoint_mixed_attn=False),
        vae=FakeVAE(),
        text_dim=4,
        device="cpu",
        torch_dtype=torch.float32,
        video_latent_spatial_downsample_factor=2,
        history_enabled=True,
        history_min_size=2,
        history_max_size=3,
        history_raw_stride=4,
        history_anchor_size=1,
        history_anchor_stride=4,
        history_recent_min_size=1,
        history_recent_max_size=2,
        history_recent_stride=4,
        use_wam_adapter=True,
        freeze_backbone=True,
        remove_original_action_expert=True,
        state_fusion_action_expert=state_fusion,
    )
    model.configure_trainable_modules()
    sample = {
        "video_latents": torch.randn(2, 1, 3, 4, 4),
        "history_video_latents": torch.randn(2, 1, 3, 4, 4),
        "history_valid_mask": torch.ones(2, 3, dtype=torch.bool),
        "image_is_pad": torch.zeros(2, 9, dtype=torch.bool),
        "action": torch.randn(2, 8, 2),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "context": torch.randn(2, 3, 4),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        loss, losses = model.training_loss(sample)
    changed_history = dict(sample)
    changed_history["history_video_latents"] = sample["history_video_latents"] + 10.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        _, changed_losses = model.training_loss(changed_history)

    assert losses["loss_video_raw"] == changed_losses["loss_video_raw"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    before = model.history_slot_embedding.weight.detach().clone()
    loss.backward()
    assert float(model.history_slot_embedding.weight.grad.abs().sum().item()) > 0.0
    assert float(model.history_role_embedding.weight.grad.abs().sum().item()) > 0.0
    optimizer.step()
    assert not torch.equal(before, model.history_slot_embedding.weight.detach())


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


def test_validation_keeps_cached_history_tensors():
    sample = {
        "video_latents": torch.zeros(16, 3, 2, 2),
        "history_video_latents": torch.zeros(16, 12, 2, 2),
        "history_valid_mask": torch.ones(12, dtype=torch.bool),
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action": torch.zeros(32, 14),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "proprio": torch.zeros(32, 14),
        "context": torch.zeros(8, 4),
        "context_mask": torch.ones(8, dtype=torch.bool),
        "prompt": "task",
    }

    batched = Wan22Trainer._to_batched_eval_sample(sample)

    assert batched["video_latents"].shape == (1, 16, 3, 2, 2)
    assert batched["history_video_latents"].shape == (1, 16, 12, 2, 2)
    assert batched["history_valid_mask"].shape == (1, 12)
    assert batched["action_horizon"] == 32


def test_cached_validation_restores_selective_adapter_train_mode():
    sample = {
        "video_latents": torch.zeros(1, 3, 2, 2),
        "history_video_latents": torch.zeros(1, 3, 2, 2),
        "history_valid_mask": torch.ones(3, dtype=torch.bool),
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action": torch.zeros(8, 2),
        "action_is_pad": torch.zeros(8, dtype=torch.bool),
        "context": torch.zeros(2, 4),
        "context_mask": torch.ones(2, dtype=torch.bool),
        "prompt": "task",
    }

    class FakeModel:
        def __init__(self):
            self.dit = torch.nn.Identity().eval()
            self.video_expert = torch.nn.Identity().train()

        def eval(self):
            self.dit.eval()
            self.video_expert.eval()
            return self

        @staticmethod
        def training_loss(_sample):
            return torch.tensor(1.25), {}

    model = FakeModel()
    trainer = object.__new__(Wan22Trainer)
    trainer.val_dataset = [sample]
    trainer.model = model
    trainer.global_step = 0
    trainer.accelerator = type(
        "FakeAccelerator",
        (),
        {
            "process_index": 0,
            "device": torch.device("cpu"),
            "unwrap_model": staticmethod(lambda value: value),
            "autocast": staticmethod(nullcontext),
            "gather_for_metrics": staticmethod(lambda value: value),
        },
    )()
    restored = []
    trainer._set_dit_only_train_mode = lambda: restored.append(True)

    result = trainer.evaluate()

    assert result == {"val_loss": 1.25}
    assert restored == [True]


def test_latent_only_validation_log_does_not_require_video_metrics():
    description, payload = Wan22Trainer._build_eval_log(
        500,
        {"val_loss": 1.25},
    )

    assert description == "[eval] step=500 val_loss=1.2500"
    assert payload == {"eval/val_loss": 1.25}


def test_memory_monitor_uses_mem_metric_group():
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


def test_memory_window_size_uses_mem_wandb_group():
    assert (
        Wan22Trainer._wandb_key_for_train_metric("memory_recent_window_size")
        == "mem/memory_recent_window_size"
    )
    assert Wan22Trainer._wandb_key_for_train_metric("loss_action") == "train/loss_action"
