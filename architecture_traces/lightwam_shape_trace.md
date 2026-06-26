# Light-WAM Shape Trace

Mode: `static`

## Effective Config

```json
{
  "video_backbone_type": "wan2_1_t2v",
  "video_dit_config": {
    "has_image_input": false,
    "patch_size": [
      1,
      2,
      2
    ],
    "in_dim": 16,
    "hidden_dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "attn_head_dim": 128,
    "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": true,
    "require_clip_embedding": false,
    "require_vae_embedding": false,
    "fuse_vae_embedding_in_latents": true,
    "use_gradient_checkpointing": false,
    "video_attention_mask_mode": "first_frame_causal",
    "action_conditioned": false,
    "action_dim": 7,
    "action_group_causal_mask_mode": "group_diagonal",
    "use_wam_adapter": true,
    "adapter_layer_indices": [
      8,
      16,
      24
    ],
    "adapter_dim": 256,
    "adapter_scale": 1.0,
    "use_backbone_lora": true,
    "lora_layer_indices": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29
    ],
    "lora_target_modules": [
      "self_attn.q",
      "self_attn.k",
      "self_attn.v",
      "self_attn.o",
      "cross_attn.q",
      "cross_attn.k",
      "cross_attn.v",
      "cross_attn.o",
      "ffn.0",
      "ffn.2"
    ],
    "lora_rank": 64,
    "lora_alpha": 128.0,
    "lora_dropout": 0.0
  },
  "state_fusion_config": {
    "per_layer_dim": 4608,
    "trunk_dim": 6144,
    "num_trunk_blocks": 1,
    "step_pos_dim": 256,
    "token_pooling_type": "learned_query",
    "token_pooling_num_queries": 16,
    "token_pooling_num_heads": 8,
    "token_pooling_merge_dim": null,
    "token_pooling_merge_num_slots": 2,
    "feature_sources": [
      "adapted"
    ],
    "layer_feature_sources": null
  },
  "data": {
    "batch_size": 1,
    "raw_window_steps": 33,
    "video_model_frames": 9,
    "action_video_freq_ratio": 4,
    "num_cameras": 2,
    "frame_shape": [
      3,
      224,
      448
    ],
    "video_shape": [
      1,
      3,
      9,
      224,
      448
    ],
    "action_shape": [
      1,
      32,
      7
    ],
    "proprio_shape": [
      1,
      32,
      8
    ],
    "context_shape": [
      1,
      129,
      4096
    ]
  },
  "vae": {
    "z_dim": 16,
    "spatial_downsample": 8,
    "temporal_downsample": 4,
    "input_latents": [
      1,
      16,
      3,
      28,
      56
    ],
    "future_latents": [
      1,
      16,
      3,
      14,
      28
    ],
    "observation_latents": [
      1,
      16,
      1,
      28,
      56
    ]
  },
  "tokens": {
    "future_grid": [
      3,
      7,
      14
    ],
    "future_tokens": 294,
    "future_tokens_per_frame": 98,
    "action_grid": [
      1,
      14,
      28
    ],
    "action_tokens": 392,
    "action_tokens_per_frame": 392,
    "hidden_dim": 1536
  }
}
```

## Records

### 1. `data.video`

- class: `Input`

```json
{
  "output": {
    "shape": [
      1,
      3,
      9,
      224,
      448
    ],
    "dtype": "float"
  }
}
```

### 2. `vae.input_latents`

- class: `WanVideoVAE.encode`

```json
{
  "output": {
    "shape": [
      1,
      16,
      3,
      28,
      56
    ],
    "dtype": "float"
  }
}
```

### 3. `future.video_tokens`

- class: `WanVideoDiT.patch_embedding`

```json
{
  "output": {
    "shape": [
      1,
      294,
      1536
    ],
    "dtype": "model_dtype"
  }
}
```

### 4. `action.observation_tokens`

- class: `WanVideoDiT.patch_embedding`

```json
{
  "output": {
    "shape": [
      1,
      392,
      1536
    ],
    "dtype": "model_dtype"
  }
}
```

### 5. `adapter.layer_states`

- class: `ResidualAdapter caches`

```json
{
  "output": {
    "layers": [
      8,
      16,
      24
    ],
    "backbone": [
      1,
      392,
      1536
    ],
    "adapted": [
      1,
      392,
      1536
    ],
    "delta": [
      1,
      392,
      1536
    ]
  }
}
```

### 6. `state_fusion.pooling`

- class: `LearnedQueryPooler`

```json
{
  "output": {
    "per_layer_input": [
      1,
      392,
      1536
    ],
    "num_queries": 16,
    "pooled_per_layer": [
      1,
      1536
    ]
  }
}
```

### 7. `state_fusion.layer_compressors`

- class: `LayerFusionCompressor`

```json
{
  "output": {
    "per_layer": [
      1,
      4608
    ],
    "concat": [
      1,
      13824
    ]
  }
}
```

### 8. `state_fusion.trunk`

- class: `MLP trunk`

```json
{
  "output": {
    "shape": [
      1,
      6144
    ],
    "dtype": "model_dtype"
  }
}
```

### 9. `action.pred_action`

- class: `Output MLP`

```json
{
  "output": {
    "shape": [
      1,
      32,
      7
    ],
    "dtype": "model_dtype"
  }
}
```
