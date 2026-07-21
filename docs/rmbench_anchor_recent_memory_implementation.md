# RMBench Anchor + Recent Memory

This is the implemented fixed-latent-slot memory path. It is separate from the
persistent layer-KV proposal in `single_frame_observation_kv_memory_plan.md`.

## Contract

- Current observation: one raw frame, encoded as an independent `T=1` VAE latent.
- Memory: 12 fixed slots: 4 episode-start anchors (`0,4,8,12`) followed by 8 recent slots.
- Recent training window: uniformly sampled `4..8`; anchors are never dropped.
- Current observation is excluded from its own history.
- Memory affects only the StateFusion direct-MSE action branch. Video loss is unchanged.
- Action horizon / execution / supervised prefix: `32 / 24 / 24`.

Offline episode-packed cache stores each sample's ordinary video latent once.
History is gathered from source samples' first causal latent at read time; no
`[target, history]` copies are stored.

Online inference stores the first 4 sampled observation latents permanently and
keeps the latest 8 thereafter. Each observation is VAE-encoded once. A current
latent is committed only after action inference succeeds.

The fixed-slot action attention is still recomputed at each replan. Persistent
per-layer KV reuse is a separate architecture and is not mixed into this run.

## Fixed defects

1. Random windows use the trainable video expert's train/eval state, so adapter
   training really samples `4..8`; W&B logs `train/memory_recent_window_size`.
2. Independent memory observations use temporal RoPE position 0 plus learned
   slot/type embeddings. The causal future-video branch never reads memory.
3. Training, validation, cache precompute, RoboTwin/RMBench online policy, and
   official RMBench launch all use the same anchor/recent layout.
4. Cache resume and loading verify the exact train/validation episode split;
   mismatched or incomplete caches fail before training.

Every 500 steps the trainer also compares correct memory against cross-sample
shuffled memory. `mem/probe/action_loss_gap` should become positive while
`mem/probe/video_loss_gap` should remain zero for the action-only design.

## RMBench put_back_block

```bash
# 1. Build separate train/validation episode caches.
bash scripts/precompute_rmbench_anchor_recent_memory.sh

# 2. Post-train from the completed no-memory 5k checkpoint.
bash scripts/train_rmbench_anchor_recent_memory.sh

# 3. Evaluate a memory checkpoint.
CKPT=/absolute/path/to/step_005000.pt \
  bash scripts/eval_rmbench_anchor_recent_memory.sh
```

The training profile is 4 GPUs, micro-batch 2, accumulation 4 (global batch
32), 5,000 steps, 1,000 warmup steps, cosine decay, and a checkpoint every
1,000 steps. Existing `rmbench_no_memory` configs and scripts are unchanged.
