# Light-WAM vs Fast-WAM: memory research notes

这份笔记只看本地代码实现，不把 Light-WAM 当成“已经有 memory”的模型。结论是：Light-WAM 当前仍主要从当前观测出 action，但它比 Fast-WAM 更适合作为 memory 研究底座，因为 action decoder 已经从原来的 ActionDiT/MoT diffusion 路径中拆出来，变成了一个读取视频骨干中间状态的 StateFusionActionExpert。

## 1. 总体结构差异

| 维度 | Fast-WAM | Light-WAM | 对 memory 研究的意义 |
| --- | --- | --- | --- |
| 视频基座 | 默认 Wan2.2-TI2V-5B | 默认 Wan2.1-T2V-1.3B | Light-WAM 更轻，实验成本更低。 |
| 训练方式 | MoT 里同时跑 video tokens 和 noisy action tokens | 冻结视频骨干，训练 adapter/LoRA/future head/state-fusion action head | memory 实验可以集中改小模块，不必全量动大 DiT。 |
| action decoder | ActionDiT diffusion head，action 也要多步去噪 | StateFusionActionExpert 直接从视频骨干/adapter 状态回归 action chunk | memory 对 action 的影响更直接、更容易解释。 |
| 视频监督 | future video flow-matching | 保留 future video 监督，并支持低分辨率 latent 监督 | 仍保留 WAM 的 future-video 约束，但更省。 |
| 数据管道 | 在线 VAE encode 视频 | 支持 offline video latent cache | 做 history/memory ablation 会快很多。 |

## 2. Fast-WAM 原始 action 路径

Fast-WAM 的核心是 `src/fastwam/models/wan22/fastwam.py`：

1. 视频帧经过 VAE 得到 video latents。
2. 当前/条件视频 token 进入 video expert。
3. action 先加噪，变成 noisy action tokens。
4. video tokens 和 action tokens 一起进入 MoT。
5. action expert 对输出 action tokens 做 post-DiT，预测 action flow target。
6. inference 时 action 需要 diffusion denoise loop；为加速会 prefill video KV cache。

这个结构强，但对 memory 不友好：如果想把历史帧作为 memory，历史既会增加 video token 数，也会影响 attention mask、KV cache、ActionDiT denoise loop，以及训练的 condition-video 对齐逻辑。我们之前直接加 history frame，在 LIBERO 上跑出来效果差，本质上就是这种改法很容易变成“给大 diffusion action head 硬塞更多 token”。

## 3. Light-WAM 做了哪些关键改动

### 3.1 换成轻量视频骨干和 PEFT

`configs/model/lightwam.yaml` 默认使用：

- `model_id: Wan-AI/Wan2.1-T2V-1.3B`
- `wam_adapter.use_wam_adapter: true`
- `wam_adapter.freeze_backbone: true`
- `wam_adapter.use_backbone_lora: true`
- adapter 层默认是 `[8, 16, 24]`
- LoRA 可以覆盖 self-attn、cross-attn、FFN 等模块

Fast-WAM 的 `configs/model/fastwam.yaml` 默认是：

- `model_id: Wan-AI/Wan2.2-TI2V-5B`
- 需要预生成 ActionDiT backbone
- action/video 两个 expert 都放进 MoT

这意味着 Light-WAM 的可训练部分天然更集中：adapter、LoRA、future head、StateFusionActionExpert。对于 memory 实验，我们可以先冻结大多数视频骨干，只动 memory adapter/pooler/head。

### 3.2 原 ActionDiT 可以被完全移除

Light-WAM 在 state-fusion 模式下设置：

- `remove_original_action_expert: true`
- `action_expert = DisabledActionExpert(...)`
- `mot = MoT(mixtures={"video": video_expert})`

也就是说，action 不再走原来的 ActionDiT diffusion 分支。checkpoint 里也单独保存 `state_fusion_action_expert`。

这点非常关键：memory 可以进入 action head 的输入状态，而不是必须参与 action denoising diffusion。

### 3.3 视频骨干中间层会缓存多层状态

Light-WAM 在 `wan_video_dit.py` 里给选定层加 adapter，并缓存：

- `backbone_tokens`
- `adapted_tokens`
- `delta_tokens = adapted - backbone`

`StateFusionActionExpert` 再从多层状态里读特征。配置里默认只用 `feature_sources: [adapted]`，但代码也支持：

- `backbone`
- `adapted`
- `delta`
- 每层不同 `layer_feature_sources`

这非常适合做 memory：我们可以把 memory 当成新的 feature source、额外 token bank、或者 adapter residual 的条件，而不是把所有历史帧直接拼到 observation video 里。

### 3.4 action head 是 direct state reader

`StateFusionActionExpert` 做的是：

1. 从多个 adapter 层拿 token。
2. 对每层 token 做 pooling，默认支持 learned-query pooling。
3. 压缩每层特征。
4. 拼接多层特征。
5. 加 action step positional embedding。
6. 直接输出 `[B, action_horizon, action_dim]`。

训练 action loss 是 action MSE，不是 action diffusion target。这样做 memory 实验时，指标更干净：

- 不受 action denoise step 数影响。
- 不需要每步重新跑 action diffusion。
- 更容易做 “有/无 memory” 的可解释对比。

### 3.5 保留 future-video 监督，但降低成本

Light-WAM 仍保留未来视频监督。不同点是它支持：

- `video_latent_spatial_downsample_factor`
- `use_first_frame_residual_video_target`
- offline `video_latents`

这让我们仍然可以研究 “world model imagination 是否帮助 action”，同时把训练成本压低。

### 3.6 数据管道更适合长序列实验

Light-WAM 的 processor 新增了：

- `build_pixel_values_from_episode_images`
- `preprocess_without_images`

数据 config 新增：

- `use_latent_cache`
- `latent_cache_dir`

并有 `scripts/precompute_video_latents.py`。这说明它已经支持把全 episode 图像预先变成 latent cache。做 memory/history 时，长窗口数据会更现实，因为不用每次训练都重复 VAE encode。

## 4. 为什么 Light-WAM 更适合作为 memory 底座

我认为最核心是三点：

1. action decoder 已经模块化

Fast-WAM 的 action decoder 和 MoT/action diffusion 绑得很紧；Light-WAM 的 StateFusionActionExpert 是一个独立模块。我们可以在这个模块前后加 memory，不需要先重写整个 MoT。

2. memory 可以接在“状态特征”层，而不是“原始图像拼接”层

Fast-WAM 上加历史帧，最直接的方法是增加 condition video frames，但这会大幅增加 token、显存和 attention mask 复杂度。Light-WAM 里可以先把历史编码成 compact memory tokens，再通过 learned-query pooling 或 cross-attention 融到 state-fusion head。

3. 实验迭代成本低

Light-WAM 有更小 backbone、冻结 backbone、adapter/LoRA、latent cache、direct action head。memory 研究本身需要大量 ablation，例如历史长度 1/4/8/12、learned memory vs FIFO memory、只接 action head vs 同时接 future head。Light-WAM 的结构更能承受这种实验量。

## 5. 它现在还不是什么

Light-WAM 现在不是一个真正的 long-horizon memory policy。当前 state-fusion inference 仍要求单帧 `input_image`，`_predict_state_fusion_action_from_observation` 也检查 observation latent 必须是单帧。也就是说：

- 它没有现成的 episode-level recurrent memory。
- 没有自动维护 memory bank。
- 没有训练可变历史长度。
- 没有显式区分短/长 horizon task 的 memory 机制。

它更像是一个适合我们继续改的底座。

## 6. 建议的 memory 改造路线

我建议优先不要再走“直接拼历史帧到 video input”的路线。更稳的路线是：

1. 保留 Light-WAM 的 state-fusion action head。
2. 数据侧采样可变长度历史窗口，例如 `history_frames in {1, 4, 8, 12}`。
3. 用 frozen video backbone 或单独 memory encoder 把历史帧编码成 compact memory tokens。
4. 在 StateFusionActionExpert 中增加 memory cross-attention：
   - query: 当前 state-fusion pooled tokens 或 action-step tokens
   - key/value: history memory tokens
5. 先只让 memory 影响 action head，不影响 future-video branch。
6. 做 ablation：
   - no memory
   - raw history concat
   - mean pooled memory
   - learned-query memory
   - recurrent memory bank
7. 最后再测试 memory 是否也应该调节 future-video prediction。

这个路线的优点是：先把 memory 的收益/损害隔离在 action 预测上，不让视频生成损失和 action diffusion loop 把结果搅混。

## 7. 当前最值得改的文件

- `src/lightwam/models/wan22/state_fusion_action_expert.py`
  - 加 memory token 输入、memory pooling、memory cross-attention。
- `src/lightwam/models/wan22/lightwam.py`
  - 放开 single-frame observation 限制，构造 memory inputs，调用 action head。
- `src/lightwam/datasets/lerobot/robot_video_dataset.py`
  - 采样历史帧/历史 latent。
- `src/lightwam/datasets/lerobot/processors/lightwam_processor.py`
  - 处理 history image/history latent，不影响现有 current observation。
- `configs/data/libero_2cam.yaml`
  - 加 history 窗口和 latent cache 配置。
- `configs/model/lightwam.yaml`
  - 加 memory head 开关和维度配置。

## 8. 一句话结论

Fast-WAM 更像是“video/action 共同 diffusion 的 WAM”；Light-WAM 更像是“冻结视频世界模型 + 可训练状态读出器”。memory 研究更需要后者，因为 memory 最自然的插入点不是原始像素，也不是 noisy action diffusion token，而是 compact state representation 到 action decoder 之间。
