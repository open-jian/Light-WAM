# Light-WAM 单帧 Observation KV Memory 实施计划

## 1. 目标

在不改变当前 Light-WAM 单帧推理接口的前提下，为模型增加可训练的历史观测 memory：

```text
当前 raw frame
  -> 独立 T=1 VAE observation latent
  -> Video DiT 在第 8/16/24 层读取历史 KV
  -> 当前帧的 adapted states
  -> 原 StateFusion direct-MSE action head
```

训练和推理必须使用完全相同的 memory 单位：每张真实观测都独立进行 `T=1` VAE 编码。这里的 latent 只代表一张观测，不是从 episode 第 0 帧连续编码得到的 causal video latent-flow。

## 2. 固定边界

第一版必须保持以下行为不变：

- `infer_action()` 仍接收一张当前 raw frame。
- 当前帧和历史帧都独立进行 `T=1` VAE 编码。
- 每个历史观测必须使用它被写入 memory 时对应的 proprio，不能复制当前 target 的 proprio。
- 原 future-video latent cache、video flow loss 和 future-video forward 保持不变。
- 原 StateFusion pooling、compressor、trunk、action step embedding 和 direct-MSE output 保持不变。
- 原 action horizon、有效监督前缀、action normalization 和 proprio 输入保持不变。
- Memory 仅保存真实历史观测，不保存生成视频、action token、action history 或跨 episode 状态。
- `memory_enabled=false` 时必须走当前原始路径，输出和 loss 不得改变。
- 现有 `rmbench_no_memory` 配置保持严格无 memory；memory 实验使用新的独立配置。

第一版明确不做：

- LingBot-VA 式 streaming causal VAE；
- 连续 episode video latent-flow memory；
- gist token、语义检索、RNN/TTT memory；
- 在 StateFusion 内新增 memory cross-attention；
- 跨 optimizer step 保存训练 KV；
- 单独的 memory reconstruction/contrastive loss。

## 3. Memory 架构

### 3.1 Memory 写入和读取位置

只修改 Video DiT 的第 `8/16/24` 层。这三层与当前 WAM adapter 和 StateFusion 的特征来源完全一致。

对于当前观测 `o_t`：

1. 独立编码得到 `z_t = VAE_T1(o_t)`。
2. `z_t` 正常通过 Video DiT。
3. 在第 `8/16/24` 层，当前 query 同时读取：
   - 该层已经缓存的历史 key/value；
   - 当前观测自己的 key/value。
4. 当前观测经过 memory attention 后继续通过原 adapter。
5. StateFusion 只读取当前观测的 `adapted` tokens。
6. action 成功产生后，将本次三个层的 pending KV 提交到 memory bank。

非 memory 层仍按原来的单帧 self-attention 运行。历史 tokens 不会直接进入 StateFusion；它们只能通过第 `8/16/24` 层改变当前 tokens。

### 3.2 Online memory state

每个 episode 维护一个独立 `ObservationKVMemoryState`：

```text
next_event_id
layer 8:  [{event_id, raw_frame_id, K, V}, ...]
layer 16: [{event_id, raw_frame_id, K, V}, ...]
layer 24: [{event_id, raw_frame_id, K, V}, ...]
```

约束：

- KV 使用 inference dtype，存放在当前模型 GPU。
- 三层必须以同一个 `event_id` 原子写入。
- episode reset 时三层和 `next_event_id` 一起清空。
- 当前 forward 完成前只产生 pending KV，不能提前污染 cache。
- 同一次 action prediction 只能 commit 一次；benchmark、profiling 或一致性检查不能重复写入。
- runtime memory state 不注册为 module buffer，不进入 checkpoint/state_dict。
- memory 层使用单调递增的 memory event position 做 temporal RoPE。
- 非 memory 层继续使用原单帧 temporal position `0`。

### 3.3 Retention policy

按两个阶段实现，不能一开始就同时加入压缩策略。

**阶段 A：full-history KV 能力上界**

- 保留 episode 内所有 replan observation KV。
- 目的只是验证模型能否利用历史，不作为最终效率方案。
- 当前 RMBench 在 `replan_steps=24` 下每个 episode只有十几个左右的memory events，第一阶段优先保证等价性，不提前做淘汰优化。

**阶段 B：anchor + random recent window**

- 前 `2` 个 memory events 作为 full-resolution anchor，整个 episode 不淘汰。
- 推理保留最近 `12` 个 full-resolution memory events。
- 训练时每个样本独立采样 `H ~ UniformInteger(4, 12)`，只允许读取 anchor 和最近 `H` 个 memory events。
- anchor 不计入 `H`，并始终可见；这对 `put_back_block` 的初始位置记忆是硬约束。
- event 去重后按时间排序，当前观测不能作为自己的历史 entry。

只有阶段 B 已证明有效且 cache 成本成为瓶颈后，才考虑把中间历史换成 temporal sparse KV 或 8 个 gist tokens。

## 4. Observation 频率合同

单帧输入并不等于每个 simulator step 都必须写 memory。训练必须严格模拟部署采用的 observation trigger。

第一版推荐：

```yaml
observation_trigger: replan_only
```

含义：

- 只有 action queue 为空、模型需要重新预测 action 时，才读取一张真实观测并产生一个 memory event。
- action queue 尚未执行完时不请求额外观测、不额外运行 VAE/DiT。
- 当前 RMBench 若执行前 `24` 个 action，则相邻 memory events 的部署间隔就是约 `24` 个 simulator steps。
- 训练数据中的历史 observation 索引必须使用同一 replan 间隔，不能训练每 4 帧一条 memory、推理却每 24 步才写一次。

若后续实验要使用 `observation_stride=4` 的密集 memory，必须同时修改 evaluator，使其在 action queue 未清空时也请求真实观测并执行 writer forward；这属于另一个显式配置，不得静默启用。

## 5. 单帧 latent sidecar

### 5.1 为什么不能直接使用旧 history cache

当前 put-back cache 的每个 entry 是一个独立 anchor 对应的重叠 future clip。它不是 episode stream，但它的第一个 temporal latent满足：

```text
cached_clip[:, 0] == 对该 anchor raw frame 独立执行 T=1 VAE 的 latent
```

因此可以复用每个 entry 的首 latent作为 observation latent；不能使用其余 temporal latent作为历史 observation，也不能把多个首 latent称为连续 causal latent-flow。

旧 `raw_frame_as_mem_rand_win_size` 分支给每个 target 重复保存最多 12 个 history latent，落盘重复严重，不应继续使用这种布局。

### 5.2 新 sidecar 格式

新增去重的 episode-packed sidecar；每个原始 observation anchor 只存一次：

```text
single_observation_latents_v1/
  meta.json
  index.pt
  episodes/
    episode_<source_episode_id>.pt
```

`meta.json` 至少记录：

- format/version；
- dataset 根目录与 manifest fingerprint；
- train/val split seed、比例和选中的 source episode IDs；
- camera layout、resize/crop/normalization fingerprint；
- VAE model ID、权重 fingerprint、latent dtype/shape；
- `encoding_mode: independent_single_observation_t1`；
- proprio normalization stats、维度和 dtype；
- action/video/replan stride；
- 创建代码 commit。

每个 episode payload 至少包含：

```text
source_episode_id
global_sample_indices [N]
raw_frame_ids         [N]
timestamps            [N]
observation_latents   [N, C, H, W]
normalized_proprio    [N, D]
```

`index.pt` 至少包含：

```text
sample_to_episode
sample_to_episode_offset
episode_paths
episode_start/end
```

Reader 必须校验 dataset fingerprint、split、VAE/preprocess fingerprint、proprio normalization stats 和 sample 数；任一项不匹配都 fail closed，不能只根据长度猜测 cache 可复用。

### 5.3 生成方式

优先从现有 future-video cache 提取每个 entry 的首 latent并重排为 episode-packed sidecar，避免重新运行 VAE。生成脚本必须抽样验证：

```text
sidecar latent
vs 当前 raw frame 在线独立 T=1 VAE latent
```

两者在相同 dtype 下必须满足约定 tolerance。

## 6. Memory training dataset

新增 opt-in dataset wrapper，不改原 `RobotVideoDataset` 的 no-memory 返回合同。

### 6.1 样本定义

一个训练 item 仍然对应当前 Light-WAM 的一个 target `t`：

- 原 future-video clip、action target、proprio、prompt 和 padding mask完全不变。
- 根据 `t` 定位 source episode 和绝对 raw frame ID。
- 第一版只保留与部署严格对齐的 target：若 `replan_steps=R`，则 episode 内 target 为 `t=0,R,2R,...`；replan phase 固定从 episode 起点开始，不随机平移。
- 按同一个 `R` 构造从 episode 开始到 `t` 的 memory-event 序列。
- 最后一个 event 是当前 observation；之前的 events 才是历史。
- 读取 sidecar 中对应的独立 T=1 latents。
- full-history 阶段令 `H` 等于当前 target 之前的全部有效 events。
- random-window 阶段才逐样本采样 `H`；固定 seed 下由 `epoch + global_sample_idx` 可复现。

建议返回：

```text
memory_compute_latents   [P, C, 1, H, W]
memory_event_ids         [P]
memory_raw_frame_ids     [P]
memory_frame_mask        [P]
memory_proprio           [P, D]
memory_visible_H         scalar
memory_target_position   scalar
```

batch collate 在本地 batch 内 pad 到最大 `P`；不同 GPU 可以拥有不同 `H/P`，不需要把窗口长度 broadcast 成全局相同值。

### 6.2 Episode 安全

必须验证：

- 所有历史和当前 observation 来自同一 source episode；
- raw frame IDs 严格递增且去重；
- 绝不读取 `t` 之后的观测；
- episode 开头自然产生短历史，不复制第一帧填充；
- padded latent 为零且永远不可作为 attention key；
- local episode index 与 source episode ID 不得混用。

### 6.3 随机窗口的正确实现

随机性放在 dataset/sampler，不放在模型的 `self.training` 分支中。

旧实现的 `_sample_history_window_size()` 检查顶层 `self.training`；实际 PEFT trainer 会让顶层模型保持 eval，导致所谓随机窗口训练始终使用 `H=12`。新实现必须增加统计测试，验证足够多样本中 `H=4...12` 均能出现。

## 7. Packed training forward

训练时不能把 inference KV 跨 optimizer step 保存，因为旧参数产生的 KV 会与更新后的参数不一致，并切断历史梯度。

每个 item 应在一个 autograd graph 中执行：

```text
[episode memory events ... current]
        -> packed Video DiT forward
        -> 只 gather 最后一个有效 current frame
        -> 原 StateFusion
        -> 原 action MSE
```

### 7.1 Attention 规则

对 packed sequence 中每个 query event `i`：

- 非 memory 层：只能看自己的 observation tokens。
- memory 层 `8/16/24`：
  - 可以看自己的 tokens；
  - 可以看位于自己之前的 anchor events；
  - 可以看自己之前最近 `H` 个普通 events；
  - 不能看未来 event；
  - 不能看 padding。
- memory 层使用绝对 memory event ID 的 temporal RoPE。
- 非 memory 层每个 observation 都使用原始 `T=1` temporal position。

为了使历史 KV 本身也与在线逐步 KV inference 一致，packed forward 应从该 episode 的第一个 memory event开始计算；一个样本只采一个 `H`，该样本内所有 query event 都使用同一 retention 容量。随机 `H` 只改变 attention 可见性，不通过随机裁掉 VAE 起点来实现。

### 7.2 每个 memory event 的 conditioning

在线写入某个 observation KV 时，Video DiT 同时接收该时刻的 text context 和 proprio。因此 packed training 中每个 event 也必须使用自己的 proprio：

- prompt/text embedding 在同一 episode 内可以共享；
- 从原始 `sample["context"]` 构造尚未追加 proprio 的 base context；
- 对 `memory_proprio[:, i]` 使用现有 proprio encoder；
- 将第 `i` 个 proprio token只追加到第 `i` 个 event 的 context；
- 当前 target event使用当前时刻 proprio，历史 event使用其被观测时的 proprio。

当前 `build_inputs()` 返回的 `context` 已经追加了 target proprio，不能把它直接复制给整个 packed history，也不能在其上再次追加 proprio。实现时应拆出“纯 text context”和“为单个 event追加 proprio”两个明确 helper；原 future-video branch仍可继续使用当前 `build_inputs()` 的结果。

### 7.3 StateFusion 接口

在第 `8/16/24` 层保存 packed `backbone/adapted` states 后：

- 根据每个样本的 `memory_target_position` gather 当前 observation 的空间 tokens；
- 构造当前已有的 `{layer_idx, backbone, adapted, delta}`；
- 传给未修改的 `StateFusionActionExpert`；
- 历史 frame tokens 不允许被 StateFusion pooler直接看到。

### 7.4 Loss

Future-video branch 保持当前实现，仍使用原 `sample["video_latents"]`：

```text
loss_total = lambda_video * loss_video + lambda_action * loss_action
```

第一版不改变：

- video scheduler/noise target；
- video loss mask；
- action target和 normalization；
- action temporal prefix weighting；
- video/action loss权重。

Memory 通过当前 action loss 对历史 K/V 生产路径的反向传播学习。第一版不增加 memory loss，也不把 shuffled memory 配上原 action label进行训练；那样反而会教模型忽略 memory。

## 8. Online inference

新增 memory-aware model subclass或严格隔离的 forward path，保留 `infer_action(prompt, input_image, ...)` 参数合同。

部署 policy 另外维护单调的 `(episode_id, replan_id)`，作为一次 memory transaction 的身份。cache 必须拒绝重复 commit同一身份；不能仅依靠“函数被调用了几次”判断是否应写入。

单次 replan 的顺序固定为：

1. 将当前 raw frame按原 preprocessing处理。
2. 使用当前 `_encode_input_image_latents_tensor()` 独立编码 T=1 latent。
3. 将当前时刻 proprio追加到当前 text context，读取 episode 内已经提交的历史 KV。
4. 当前 observation 通过 Video DiT；在第 `8/16/24` 层同时得到 pending KV。
5. 当前 memory-enriched states进入原 StateFusion，产生 action chunk。
6. action成功产生后原子提交 pending KV。
7. 执行配置规定的 action prefix。

必须处理：

- `reset_model()` 清空 action queue和 memory；
- 第一帧 cache为空时与原单帧路径等价；
- action inference抛异常时不提交 pending KV；
- 同一帧的 debug/benchmark重复 forward使用临时 cache副本或关闭 commit；
- 重试相同 `(episode_id, replan_id)` 时不得产生第二份cache entry；
- batch size在一个 episode 内固定为 1；
- episode 之间不允许复用任何 KV。

## 9. 配置隔离

建议新增独立 selector：

```yaml
model:
  training_memory_type: single_observation_kv_random_window
  eval_memory_type: single_observation_kv
  observation_memory:
    enabled: true
    memory_unit: independent_single_observation_t1
    layer_indices: [8, 16, 24]
    observation_trigger: replan_only
    replan_steps: 24
    anchor_events: 2
    recent_events_train_min: 4
    recent_events_train_max: 12
    recent_events_eval: 12
    history_sampling: uniform_integer_per_sample
    packed_training: true
    persistent_training_cache: false
    gradient_through_history: true
```

建议文件布局：

```text
configs/model/memory_component/single_observation_kv/
  full_history.yaml
  anchor_recent_random_window.yaml
```

配置加载必须校验：

- memory layers 与 adapter layers 都是 `[8,16,24]`；
- train/eval memory unit一致；
- dataset history stride 与 evaluator observation trigger一致；
- sidecar不是原 future-video latent cache目录；
- memory checkpoint不被 no-memory evaluator静默加载；
- memory config缺字段时直接报错，不回退到无 memory。

## 10. 初始化与训练调度原则

- 从经过 RMBench task post-train、基础动作已经学会的最佳 no-memory checkpoint初始化。
- 只加载模型权重；新 memory post-train使用新的 optimizer、scheduler和 step 计数。
- 第一版没有新增 trainable memory参数；训练集合仍是现有 LoRA、WAM adapters、StateFusion和video head。
- 首次运行不擅自改变 learning rate、action prefix或loss权重。
- packed history会增加显存，真实 batch size必须由 `H=4` 和 `H=12` smoke决定，再用 gradient accumulation保持批准的 global batch。
- 先训练 full-history能力上界，再训练 random-window策略；不要同时改变 memory结构和训练预算。

## 11. 必须完成的测试

### 11.1 数据与 cache

1. sidecar latent与在线 T=1 VAE编码一致。
2. episode映射、source episode ID、raw frame ID正确。
3. 历史不跨 episode、不包含未来、padding不可见。
4. 固定 seed可复现；不同 epoch会改变 `H`。
5. 统计确认 `H=4...12` 的覆盖，而不是恒定12。
6. cache fingerprint或split变化时读取失败。
7. 每个 memory event 的 normalized proprio与同一 raw frame严格对齐。

### 11.2 模型数值测试

1. `memory_enabled=false` 的 loss/output与当前 baseline一致。
2. 空 cache下 memory forward与原单帧 forward在约定 tolerance内一致。
3. 修改未来 observation后，较早 query输出不变，证明没有未来泄漏。
4. 历史 observation变化时，当前 memory-layer输出可以变化。
5. history对应的Q/K/V LoRA、adapter和StateFusion获得有限且非零梯度。
6. 只修改某个历史 event 的 proprio时，只有该 event及其之后允许读取它的states可以变化。
7. 保存checkpoint/state_dict时不包含runtime KV cache。

### 11.3 Packed/online-KV 等价测试

冻结同一组权重，准备一段短的固定 T=1 observation序列、逐帧 proprio和同一个prompt，比较：

```text
A. online inference逐步forward并commit KV
B. training一次packed causal forward
```

逐项比较：

- 第 `8/16/24` 层当前 observation states；
- 选中的历史 KV；
- 最终 action输出。

该测试不通过，不允许启动正式 memory训练。

### 11.4 Memory 使用测试

对完全相同的当前 observation、proprio和prompt比较：

```text
correct history
shuffled history（来自另一 episode）
cleared history
```

shuffled实验必须将历史latent与其对应proprio作为一个整体替换，同时保持当前observation、当前proprio、prompt、history长度和event positions不变。记录action MSE和最终任务成功率。训练后的模型必须表现为正确memory优于shuffled/cleared memory；仅仅观察到历史梯度非零不能证明模型学会了memory语义。

### 11.5 资源与 rollout

1. `H=4`、`H=12` 和当前数据最大full-history各跑一个optimizer-step smoke，记录峰值显存和step time。
2. 跑一个完整 episode，检查 write、read、evict和reset顺序。
3. 确认每个 replan只增加一个 memory event。
4. 使用显式`episode_id/replan_id`验证重试同一replan不会重复写入。
5. 视频仍写入 `Outputs/`，日志和结果写入 `runs/`。

## 12. 分步实施顺序

### Step 1：配置和 fail-closed dispatch

- 新增独立 memory selector和配置。
- 保证所有旧配置resolve结果不变。
- 增加 train/eval memory合同校验。

### Step 2：去重的单帧 latent sidecar

- 从现有 cache首 latent生成 episode-packed sidecar。
- 同时保存每帧对应的normalized proprio。
- 加强dataset/split/preprocess/VAE/proprio-normalization fingerprint。
- 完成 cached-vs-online T=1 等价测试。

### Step 3：episode-safe memory dataset

- 构造 replan-aligned历史序列。
- 先实现full-history模式，再增加逐样本随机`H`模式；两者共享padding和绝对event/frame IDs。
- 完成边界、随机性和无未来泄漏测试。

### Step 4：full-history persistent-KV inference

- 移植并整理已有三层 KV cache骨架。
- 实现 pending/atomic commit和episode reset。
- 先做 training-free rollout，确认数据流和性能上界。

### Step 5：packed training forward

- 实现三层 memory attention和其他层frame-local attention。
- gather当前 frame进入未修改的StateFusion。
- 保留原future-video branch和loss。
- 完成packed/online-KV等价测试。

### Step 6：full-history post-train基线

- 从最佳no-memory task checkpoint做weights-only初始化。
- 先完成单步和短程smoke，再决定真实batch/accumulation。
- 只回答“当前架构是否能学会使用历史”。

### Step 7：random-window post-train

- 启用`anchor=2`和`H=4...12`逐样本随机窗口。
- 训练步数和其他超参数与full-history实验分开记录。
- 用correct/shuffled/cleared三组评测证明memory依赖。

### Step 8：效率优化

只有前述实验成功后，再依次比较：

- anchor + recent-only；
- temporal sparse middle KV；
- spatial pooled KV；
- trainable gist tokens。

每次只改变一个 retention策略，并保持同一checkpoint初始化、训练预算和评测种子。

## 13. 旧代码复用边界

提交 `c779234` 已经包含最接近本计划的原型，实施时应逐文件移植并重新验证，不要整分支merge：

- `src/lightwam/memory_training/dataset.py`：复用episode-safe索引和padding骨架，改为replan-aligned events。
- `src/lightwam/memory_training/single_frame_latents.py`：复用T=1 episode-packed sidecar、fingerprint和逐帧proprio合同。
- `src/lightwam/memory_training/packed_attention.py`：复用frame-level causal mask和memory/local两套RoPE骨架。
- `src/lightwam/memory_training/model.py`：复用“future-video branch不变，只替换action observation branch”的隔离方式。
- `src/lightwam/models/wan22/training_free_memory.py`：复用在线三层KV、pending commit和reset骨架。
- 对应tests：复用packed/online-KV等价、cache格式和无未来泄漏测试思路。

移植时必须删除或改写旧原型中的raw-frame stride-4、固定`anchor/recent/middle_stride`假设，第一阶段统一为`replan_only + full_history`。

不得整合`origin/raw_frame_as_mem_rand_win_size`最新分支。该分支使用`t-4,t-8...` raw history、按target重复存history、真实训练中`H`恒为12，并把history注入future-video branch，与本计划的固定边界冲突。

## 14. Definition of Done

满足以下条件才算第一版完成：

- 当前单帧raw-frame推理接口未改变；
- 训练和推理都使用独立T=1 observation latent；
- 原future-video branch和StateFusion direct-MSE head未改变；
- memory仅作用于Video DiT第8/16/24层；
- packed training与online persistent-KV inference通过数值等价测试；
- no-memory配置没有回归；
- random `H`已被统计证明真实生效；
- full-history与random-window均有独立实验记录；
- correct memory明确优于shuffled/cleared memory；
- 完整rollout中cache能正确write、evict和episode reset。
