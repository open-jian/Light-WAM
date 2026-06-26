import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from lightwam.models.wan22.helpers.loader import (
    apply_video_backbone_preset,
    resolve_video_backbone_type,
)
from lightwam.models.wan22.state_fusion_action_expert import StateFusionActionExpert
from lightwam.models.wan22.wan_video_dit import WanVideoDiT
from lightwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _cfg_get(cfg: DictConfig | dict | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    return cfg.get(key, default)


def _to_plain_dict(node: Any) -> dict[str, Any]:
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True)
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    raise TypeError(f"Expected dict-like config node, got {type(node)}")


def _shape_of(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _shape_of(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_of(v) for v in value]
    if isinstance(value, slice):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _num_params(module: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters(recurse=True):
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def _resolve_effective_model_cfg(cfg: DictConfig) -> dict[str, Any]:
    if cfg.get("model") is None:
        raise ValueError(
            "Missing model config. Pass a task, for example: "
            "`task=libero_uncond_2cam224_1e-4`."
        )
    if cfg.get("data") is None:
        raise ValueError(
            "Missing data config. Pass a task, for example: "
            "`task=libero_uncond_2cam224_1e-4`."
        )

    model_cfg = cfg.model
    video_backbone_type = resolve_video_backbone_type(
        str(model_cfg.get("video_backbone_type", "wan2_2_ti2v"))
    )
    video_dit_config = _to_plain_dict(model_cfg.video_dit_config)
    video_dit_config = apply_video_backbone_preset(
        dit_config=video_dit_config,
        video_backbone_type=video_backbone_type,
    )

    wam_adapter_cfg = _to_plain_dict(model_cfg.get("wam_adapter", {}))
    use_wam_adapter = bool(wam_adapter_cfg.get("use_wam_adapter", False))
    use_backbone_lora = bool(wam_adapter_cfg.get("use_backbone_lora", False))
    if use_wam_adapter:
        video_dit_config["use_wam_adapter"] = True
        video_dit_config["adapter_layer_indices"] = wam_adapter_cfg.get("adapter_layer_indices")
        video_dit_config["adapter_dim"] = int(wam_adapter_cfg.get("adapter_dim", 128))
        video_dit_config["adapter_scale"] = float(wam_adapter_cfg.get("adapter_scale", 1.0))
    if use_backbone_lora:
        video_dit_config["use_backbone_lora"] = True
        video_dit_config["lora_layer_indices"] = wam_adapter_cfg.get("lora_layer_indices")
        video_dit_config["lora_target_modules"] = wam_adapter_cfg.get(
            "lora_target_modules",
            ["ffn.0", "ffn.2"],
        )
        video_dit_config["lora_rank"] = int(wam_adapter_cfg.get("lora_rank", 16))
        video_dit_config["lora_alpha"] = float(wam_adapter_cfg.get("lora_alpha", 16.0))
        video_dit_config["lora_dropout"] = float(wam_adapter_cfg.get("lora_dropout", 0.0))

    data_cfg = cfg.data.train
    image_h = int(data_cfg.shape_meta.images[0].shape[1])
    image_w_single = int(data_cfg.shape_meta.images[0].shape[2])
    num_cameras = int(data_cfg.processor.num_output_cameras)
    concat_mode = str(data_cfg.get("concat_multi_camera", "horizontal"))
    if concat_mode == "horizontal":
        image_w = image_w_single * num_cameras
        image_h_total = image_h
    elif concat_mode == "vertical":
        image_w = image_w_single
        image_h_total = image_h * num_cameras
    else:
        image_w = image_w_single
        image_h_total = image_h

    raw_window_steps = int(data_cfg.num_frames)
    action_video_freq_ratio = int(data_cfg.action_video_freq_ratio)
    video_model_frames = ((raw_window_steps - 1) // action_video_freq_ratio) + 1
    action_horizon = raw_window_steps - 1
    action_dim = int(data_cfg.processor.action_output_dim)
    proprio_dim = int(data_cfg.processor.proprio_output_dim)
    context_len = int(data_cfg.get("context_len", 128))
    include_proprio_token = proprio_dim > 0
    context_len_model = context_len + (1 if include_proprio_token else 0)

    # Wan2.1 uses z_dim=16/up=8; Wan2.2 TI2V uses z_dim=48/up=16.
    if video_backbone_type == "wan2_1_t2v":
        vae_z_dim = 16
        vae_spatial_downsample = 8
    else:
        vae_z_dim = 48
        vae_spatial_downsample = 16
    vae_temporal_downsample = 4

    latent_t = ((video_model_frames - 1) // vae_temporal_downsample) + 1
    latent_h = image_h_total // vae_spatial_downsample
    latent_w = image_w // vae_spatial_downsample
    future_factor = int(model_cfg.get("video_latent_spatial_downsample_factor", 1))
    future_latent_h = latent_h // future_factor
    future_latent_w = latent_w // future_factor
    patch_t, patch_h, patch_w = [int(x) for x in video_dit_config["patch_size"]]
    future_grid = [
        latent_t // patch_t,
        future_latent_h // patch_h,
        future_latent_w // patch_w,
    ]
    action_grid = [
        1 // patch_t,
        latent_h // patch_h,
        latent_w // patch_w,
    ]

    return {
        "video_backbone_type": video_backbone_type,
        "video_dit_config": video_dit_config,
        "state_fusion_config": _to_plain_dict(
            model_cfg.get("state_fusion_action_expert_config", {})
        ),
        "data": {
            "batch_size": int(_cfg_get(cfg.get("trace"), "batch_size", 1)),
            "raw_window_steps": raw_window_steps,
            "video_model_frames": video_model_frames,
            "action_video_freq_ratio": action_video_freq_ratio,
            "num_cameras": num_cameras,
            "frame_shape": [3, image_h_total, image_w],
            "video_shape": [int(_cfg_get(cfg.get("trace"), "batch_size", 1)), 3, video_model_frames, image_h_total, image_w],
            "action_shape": [int(_cfg_get(cfg.get("trace"), "batch_size", 1)), action_horizon, action_dim],
            "proprio_shape": [int(_cfg_get(cfg.get("trace"), "batch_size", 1)), action_horizon, proprio_dim],
            "context_shape": [
                int(_cfg_get(cfg.get("trace"), "batch_size", 1)),
                context_len_model,
                int(video_dit_config["text_dim"]),
            ],
        },
        "vae": {
            "z_dim": vae_z_dim,
            "spatial_downsample": vae_spatial_downsample,
            "temporal_downsample": vae_temporal_downsample,
            "input_latents": [
                int(_cfg_get(cfg.get("trace"), "batch_size", 1)),
                vae_z_dim,
                latent_t,
                latent_h,
                latent_w,
            ],
            "future_latents": [
                int(_cfg_get(cfg.get("trace"), "batch_size", 1)),
                vae_z_dim,
                latent_t,
                future_latent_h,
                future_latent_w,
            ],
            "observation_latents": [
                int(_cfg_get(cfg.get("trace"), "batch_size", 1)),
                vae_z_dim,
                1,
                latent_h,
                latent_w,
            ],
        },
        "tokens": {
            "future_grid": future_grid,
            "future_tokens": future_grid[0] * future_grid[1] * future_grid[2],
            "future_tokens_per_frame": future_grid[1] * future_grid[2],
            "action_grid": action_grid,
            "action_tokens": action_grid[0] * action_grid[1] * action_grid[2],
            "action_tokens_per_frame": action_grid[1] * action_grid[2],
            "hidden_dim": int(video_dit_config["hidden_dim"]),
        },
    }


def _register_hooks(
    video_expert: WanVideoDiT,
    state_expert: StateFusionActionExpert,
    records: list[dict[str, Any]],
) -> list[Any]:
    wanted_names = {
        "video.patch_embedding",
        "video.head",
        "state.layer_compressors.0",
        "state.layer_compressors.1",
        "state.layer_compressors.2",
        "state.fused_proj",
        "state.trunk.0",
        "state.step_pos_proj",
        "state.output",
    }
    for idx in (0, 8, 16, 24, 29):
        wanted_names.add(f"video.blocks.{idx}")
    for idx in (8, 16, 24):
        wanted_names.add(f"video.wam_adapters.{idx}")
    for idx in (0, 1, 2):
        wanted_names.add(f"state.layer_poolers.{idx}.adapted")

    name_to_module: dict[str, torch.nn.Module] = {}
    for name, module in video_expert.named_modules():
        name_to_module[f"video.{name}" if name else "video"] = module
    for name, module in state_expert.named_modules():
        name_to_module[f"state.{name}" if name else "state"] = module

    handles = []

    def make_hook(name: str, module: torch.nn.Module):
        def hook(_module, inputs, output):
            total, trainable = _num_params(module)
            records.append(
                {
                    "name": name,
                    "class": module.__class__.__name__,
                    "input": _shape_of(inputs),
                    "output": _shape_of(output),
                    "params": total,
                    "trainable_params": trainable,
                }
            )

        return hook

    for name in sorted(wanted_names):
        module = name_to_module.get(name)
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name, module)))
    return handles


def _run_hook_trace(effective: dict[str, Any], trace_cfg: DictConfig | dict | None) -> list[dict[str, Any]]:
    batch_size = int(_cfg_get(trace_cfg, "batch_size", 1))
    device_text = str(_cfg_get(trace_cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_text)
    dtype_text = str(
        _cfg_get(
            trace_cfg,
            "dtype",
            "bf16" if device.type == "cuda" and torch.cuda.is_available() else "fp32",
        )
    ).lower()
    if dtype_text in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_text in {"fp16", "float16"}:
        dtype = torch.float16
    elif dtype_text in {"fp32", "float32"}:
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported trace dtype: {dtype_text}")

    branch = str(_cfg_get(trace_cfg, "branch", "both")).lower()
    if branch not in {"both", "future", "action"}:
        raise ValueError("trace.branch must be one of: both, future, action")

    video_expert = WanVideoDiT(**effective["video_dit_config"]).to(device=device, dtype=dtype)
    video_expert.eval()
    video_expert.requires_grad_(False)
    if hasattr(video_expert, "enable_backbone_lora_training"):
        video_expert.enable_backbone_lora_training()
    if hasattr(video_expert, "wam_adapters"):
        video_expert.wam_adapters.train()
        video_expert.wam_adapters.requires_grad_(True)
    if hasattr(video_expert, "head"):
        video_expert.head.train()
        video_expert.head.requires_grad_(True)

    state_fusion_config = dict(effective["state_fusion_config"])
    action_dim = int(effective["data"]["action_shape"][2])
    num_fusion_layers = len(getattr(video_expert, "adapter_layer_indices", ()))
    if num_fusion_layers <= 0:
        raise ValueError("Hook tracing requires WAM adapters to be enabled.")
    state_expert = StateFusionActionExpert(
        video_hidden_dim=int(effective["video_dit_config"]["hidden_dim"]),
        action_dim=action_dim,
        num_fusion_layers=num_fusion_layers,
        **state_fusion_config,
    ).to(device=device, dtype=dtype)
    state_expert.train()
    state_expert.requires_grad_(True)

    records: list[dict[str, Any]] = []
    handles = _register_hooks(video_expert, state_expert, records)

    context_shape = list(effective["data"]["context_shape"])
    context_shape[0] = batch_size
    context = torch.randn(context_shape, device=device, dtype=dtype)
    context_mask = torch.ones((batch_size, context_shape[1]), device=device, dtype=torch.bool)
    timestep = torch.zeros((batch_size,), device=device, dtype=dtype)

    try:
        with torch.no_grad():
            if branch in {"both", "future"}:
                future_shape = list(effective["vae"]["future_latents"])
                future_shape[0] = batch_size
                latents = torch.randn(future_shape, device=device, dtype=dtype)
                pre = video_expert.pre_dit(
                    x=latents,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=True,
                )
                records.append(
                    {
                        "name": "future.pre_dit",
                        "class": "dict",
                        "input": {
                            "latents": _shape_of(latents),
                            "context": _shape_of(context),
                            "context_mask": _shape_of(context_mask),
                        },
                        "output": _shape_of(pre),
                        "params": 0,
                        "trainable_params": 0,
                    }
                )
                tokens = video_expert.forward_backbone(pre)
                pred = video_expert.post_dit(tokens, pre)
                records.append(
                    {
                        "name": "future.output",
                        "class": "Tensor",
                        "input": {"tokens": _shape_of(tokens)},
                        "output": _shape_of(pred),
                        "params": 0,
                        "trainable_params": 0,
                    }
                )

            if branch in {"both", "action"}:
                obs_shape = list(effective["vae"]["observation_latents"])
                obs_shape[0] = batch_size
                obs = torch.randn(obs_shape, device=device, dtype=dtype)
                pre = video_expert.pre_dit(
                    x=obs,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=True,
                )
                records.append(
                    {
                        "name": "action.pre_dit",
                        "class": "dict",
                        "input": {
                            "latents": _shape_of(obs),
                            "context": _shape_of(context),
                            "context_mask": _shape_of(context_mask),
                        },
                        "output": _shape_of(pre),
                        "params": 0,
                        "trainable_params": 0,
                    }
                )
                _ = video_expert.forward_backbone(pre)
                layer_states = []
                for layer_idx, backbone_tokens, adapted_tokens in video_expert.get_wam_action_fusion_layer_states(
                    selected_layers=list(getattr(video_expert, "adapter_layer_indices", ()))
                ):
                    layer_states.append(
                        {
                            "layer_idx": int(layer_idx),
                            "backbone": backbone_tokens,
                            "adapted": adapted_tokens,
                            "delta": adapted_tokens - backbone_tokens,
                        }
                    )
                pred_action = state_expert(
                    layer_states=layer_states,
                    action_horizon=int(effective["data"]["action_shape"][1]),
                )
                records.append(
                    {
                        "name": "action.output",
                        "class": "Tensor",
                        "input": {"layer_states": _shape_of(layer_states)},
                        "output": _shape_of(pred_action),
                        "params": 0,
                        "trainable_params": 0,
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    return records


def _build_static_report(effective: dict[str, Any]) -> list[dict[str, Any]]:
    vcfg = effective["video_dit_config"]
    batch_size = int(effective["data"]["batch_size"])
    hidden = int(vcfg["hidden_dim"])
    action_horizon = int(effective["data"]["action_shape"][1])
    action_dim = int(effective["data"]["action_shape"][2])
    state_cfg = effective["state_fusion_config"]
    per_layer_dim = int(state_cfg.get("per_layer_dim", 4608))
    trunk_dim = int(state_cfg.get("trunk_dim", 6144))
    num_fusion_layers = len(vcfg.get("adapter_layer_indices") or [])
    if num_fusion_layers <= 0:
        num_fusion_layers = 3

    return [
        {
            "name": "data.video",
            "class": "Input",
            "output": {"shape": effective["data"]["video_shape"], "dtype": "float"},
        },
        {
            "name": "vae.input_latents",
            "class": "WanVideoVAE.encode",
            "output": {"shape": effective["vae"]["input_latents"], "dtype": "float"},
        },
        {
            "name": "future.video_tokens",
            "class": "WanVideoDiT.patch_embedding",
            "output": {
                "shape": [batch_size, effective["tokens"]["future_tokens"], hidden],
                "dtype": "model_dtype",
            },
        },
        {
            "name": "action.observation_tokens",
            "class": "WanVideoDiT.patch_embedding",
            "output": {
                "shape": [batch_size, effective["tokens"]["action_tokens"], hidden],
                "dtype": "model_dtype",
            },
        },
        {
            "name": "adapter.layer_states",
            "class": "ResidualAdapter caches",
            "output": {
                "layers": vcfg.get("adapter_layer_indices"),
                "backbone": [batch_size, effective["tokens"]["action_tokens"], hidden],
                "adapted": [batch_size, effective["tokens"]["action_tokens"], hidden],
                "delta": [batch_size, effective["tokens"]["action_tokens"], hidden],
            },
        },
        {
            "name": "state_fusion.pooling",
            "class": "LearnedQueryPooler",
            "output": {
                "per_layer_input": [batch_size, effective["tokens"]["action_tokens"], hidden],
                "num_queries": int(state_cfg.get("token_pooling_num_queries", 16)),
                "pooled_per_layer": [batch_size, hidden],
            },
        },
        {
            "name": "state_fusion.layer_compressors",
            "class": "LayerFusionCompressor",
            "output": {
                "per_layer": [batch_size, per_layer_dim],
                "concat": [batch_size, per_layer_dim * num_fusion_layers],
            },
        },
        {
            "name": "state_fusion.trunk",
            "class": "MLP trunk",
            "output": {"shape": [batch_size, trunk_dim], "dtype": "model_dtype"},
        },
        {
            "name": "action.pred_action",
            "class": "Output MLP",
            "output": {"shape": [batch_size, action_horizon, action_dim], "dtype": "model_dtype"},
        },
    ]


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Light-WAM Shape Trace")
    lines.append("")
    lines.append(f"Mode: `{payload['mode']}`")
    lines.append("")
    lines.append("## Effective Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload["effective"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Records")
    lines.append("")
    for idx, record in enumerate(payload["records"], start=1):
        lines.append(f"### {idx}. `{record['name']}`")
        lines.append("")
        lines.append(f"- class: `{record.get('class', '')}`")
        if "params" in record:
            lines.append(f"- params: `{record.get('params')}`")
            lines.append(f"- trainable params: `{record.get('trainable_params')}`")
        lines.append("")
        lines.append("```json")
        compact = {k: v for k, v in record.items() if k not in {"name", "class", "params", "trainable_params"}}
        lines.append(json.dumps(compact, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    trace_cfg = cfg.get("trace")
    mode = str(_cfg_get(trace_cfg, "mode", "static")).lower()
    if mode not in {"static", "hooks"}:
        raise ValueError("trace.mode must be one of: static, hooks")

    effective = _resolve_effective_model_cfg(cfg)
    records = _build_static_report(effective)
    if mode == "hooks":
        records.extend(_run_hook_trace(effective, trace_cfg))

    payload = {
        "mode": mode,
        "effective": effective,
        "records": records,
    }

    output_dir = Path(str(_cfg_get(trace_cfg, "output_dir", "./architecture_traces")))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lightwam_shape_trace.json"
    md_path = output_dir / "lightwam_shape_trace.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
