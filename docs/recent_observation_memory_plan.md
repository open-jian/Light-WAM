# Light-WAM Recent-Observation Memory 实现计划

## 目标与固定设计

- 在现有每个训练样本前增加 `H` 张真实历史观测，`H` 每个 batch 从 `[4, 12]` 随机一次。
- 历史帧只从 VAE 四倍时间降采样网格取：`t-4H, ..., t-8, t-4`。
- 每张历史图独立以 `T=1` 编码；当前帧和原有 future video/action 标签不变。
- history 始终使用当前数据分辨率对应的低分辨率 video token 网格；LIBERO `224×448` 下是 98 token。action 分支的 current 保持原生高分辨率网格，LIBERO 下是 392 token。
- memory 只保存真实观测，不保存生成的 future，不引入 action token、gist 或跨 batch memory。

## 实现步骤

### 1. 配置与数据

- 在目标 task override 增加 history 开关及参数，避免污染 `rmbench_no_memory`：`enabled`、`min_size=4`、`max_size=12`、`raw_stride=4`。
- 在 LeRobot dataset/processor 中按 episode 边界读取历史图和原始 frame id，禁止跨 episode。
- episode 开头历史不足时左侧 padding，并生成 `history_valid_mask`；不要重复第一帧。
- 同一 batch 使用相同 `H`；多卡时由 rank 0 采样并 broadcast。
- 默认 4 卡训练改为每卡 batch 4、梯度累积 4，global batch 仍为 64；`rmbench_no_memory` 保持 16×1。
- 优先预计算/缓存独立 `T=1` history latent，避免训练时重复跑 VAE。

### 2. 拼接与两类时间

模型输入按以下顺序拼接：

```text
[H 个 clean history] + [1 个 clean current] + [原有 noisy future clip]
```

- 3D RoPE 时间位置按 latent step 递增；history 原始帧相差 4，对应 RoPE 相差 1。
- diffusion timestep：history/current 全为 `0`，future 全为本次采样的 `tau`。
- history 用独立 clean pre-state（diffusion timestep 全为 `0`），current+future 沿用原 pre-state（current 为 `0`、future 为 `tau`），再按 token 拼接。
- action 分支不能沿 latent 时间维拼接，因为 history 和 current 使用不同空间分辨率；两者 patchify 后再拼接。
- action 分支把低分辨率 history 的空间 RoPE 坐标按 2 倍映射到 current 的高分辨率网格。
- 第一版整窗重算时使用相对 RoPE 位置；后续若加入持久 KV cache，再使用 episode 内单调递增的 latent-step position。

### 3. Attention mask

在 `wan_video_dit.py` 增加专用 history/future mask：

- history query：只能看有效的自己及更早 history；current query 可以看有效 history 和自己。
- future query：可以看全部有效 history/current。
- 同一次预测的 future tokens 之间保持双向 attention。
- history/current 不能看 future；padding 不能被任何有效 token 看见。

不要直接复用 `first_frame_causal`，也不要把整个 future clip 改成逐帧 causal。

### 4. 训练与 loss

- 在 `lightwam.py::_training_loss_state_fusion` 的 future-video pass 前追加 history latent。
- 只对原 current+future 输出 slice 沿用原有 video flow loss；history 不增加重建 loss。
- action pass 同样输入 `history + current`；StateFusionActionExpert 只读取已经融合历史的 current-frame token slice。
- 保留现有 action 标签、video/action loss 权重和 spatial-downsample 策略；video 输出只 decode 原 current+future token slice。
- history 不 `detach`；整个窗口一次 forward、一次 backward。

### 5. 推理

- 在 `deploy_policy.py` 维护最多 12 张真实观测的 deque，episode reset 时清空。
- 按训练一致的每 4 个环境 step 采集一次观测；即使尚未 replan，也要将观测留给下一次推理。
- 第一版每次 replan 重算完整窗口，先保证与训练完全一致。
- KV cache 作为第二阶段优化：只缓存真实 observation KV，淘汰最老记录，不提交生成的 future KV。

## 必要验证

1. `history_enabled=false` 时数值和当前 baseline 一致。
2. 检查 frame index 始终位于四倍网格且不跨 episode。
3. 改动 future token 后，history/current hidden state 不应变化，确认无未来泄漏。
4. oldest valid history latent 能从 video loss 获得非零梯度。
5. 分别以 `H=4`、`H=12` 跑一个 optimizer-step smoke test，记录峰值显存和速度。
6. 推理验证 deque 的采样间隔、逐步填满、淘汰和 episode reset。
7. 若实现 KV cache，增加“整窗重算 vs streaming cache”输出一致性测试。

## 训练中监控

- 所有 W&B 指标使用 `mem/...` 前缀，单独归入 `mem` 分类。
- 每次正常 backward 记录 video/action 两路的 history/current token gradient RMS、两者比例，以及 `age_01` 到 `age_12` 的逐帧梯度；`age_01` 表示最近历史帧。
- 每 500 optimizer step，从当前 batch 取 2 个样本，以完全相同的 diffusion noise/timestep 分别运行正确历史和 batch 内打乱历史，记录 `video_loss_gap`、`action_loss_gap` 和 `total_loss_gap`。
- gap 定义为 `shuffled - correct`；稳定为正说明正确历史比错误历史更有用。梯度非零只表示训练信号到达 memory，不能单独证明语义已学会。

## 主要风险

- 当前部署只在 action queue 为空时请求 observation，必须补上每 4 step 的采集，否则训练/推理历史频率不一致。
- 当前 timestep embedding 和 video loss 都默认只有第一帧是 condition，需统一改为前 `H+1` 帧。
- StateFusionActionExpert 若不切 current token，会错误地直接 pooling history/future。
- `H=12` 会显著增加 token 数，长跑前必须先做显存 smoke test；activation checkpointing 可以省显存，但不要用 `detach` 截断窗口内梯度。
