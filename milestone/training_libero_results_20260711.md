# Light-WAM `training_libero` 训练与评测结果

更新时间：2026-07-11（America/Chicago）

## 结论

`training_libero` 的四套最终 checkpoint 已在本机完成 LIBERO r5 快速验证：195/200 成功，总成功率 97.5%。其中 LIBERO-Object 为 50/50，成功率 100%。

    Suite             Steps    Success  Episodes  Rate
    LIBERO-Object     12,500       50        50   100.0%
    LIBERO-Spatial    60,000       48        50    96.0%
    LIBERO-Goal       60,000       49        50    98.0%
    LIBERO-10         80,000       48        50    96.0%
    TOTAL                  -      195       200    97.5%

## 评测协议

- 分支：`training_libero`
- 评测 commit：`b4819a6ab0c1de4dcef58d9f1529535e4bf6f730`
- MuJoCo 后端：OSMesa
- 每个 suite 包含 10 个任务，每个任务评测前 5 个初始状态，共 50 episodes
- 四个 suite 合计 200 episodes
- `EVALUATION.use_training_run_config=true`，评测读取各训练 run 自己的配置和 dataset stats
- Action horizon 为 32，推理 `replan_steps=10`
- 所有评测 worker 和四个汇总器退出状态均为 0
- 评测时间：2026-07-11 12:49:47 至 13:16:18 CDT

> 这是每任务 5 次的快速回归验证，不是每任务 50 次的正式论文评测。97.5% 可用于确认 checkpoint 和当前分支逻辑工作正常，不应直接作为最终论文数字。

## Checkpoint 与训练概况

所有训练的 effective global batch 均为 64，warmup 均为 1,000 steps，scheduler 均为 cosine。下表中的 final loss 是训练日志最后一个 step 的单 batch 数值，不是全数据集均值。

    Suite       Checkpoint       GPUxBatch  LR    Loss    Action  Samples/s
    Object      step_012500.pt      4x16    1e-4  0.0332  0.0041      70.80
    Spatial     step_060000.pt      4x16    2e-4  0.0176  0.0007      69.93
    Goal        step_060000.pt       8x8    2e-4  0.0220  0.0006     130.86
    LIBERO-10   step_080000.pt       8x8    1e-4  0.0241  0.0007     131.21

Checkpoint 路径：

- Object：`lightwam_libero_paper_repro/paper_20260628_004401_object_12500steps/checkpoints/weights/step_012500.pt`
- Spatial：`lightwam_libero_paper_repro/paper_20260628_004401_spatial_60000steps/checkpoints/weights/step_060000.pt`
- Goal：`lightwam_libero_paper_repro/paper_20260628_195529_goal_60000steps/checkpoints/weights/step_060000.pt`
- LIBERO-10：`lightwam_libero_paper_repro/paper_20260628_195529_libero_10_80000steps/checkpoints/weights/step_080000.pt`

四个 checkpoint 的内部 step 与文件名一致；`mot`、`proprio_encoder` 和 `state_fusion_action_expert` 加载完整，没有 missing/unexpected keys。

## 失败分布

200 个 episode 中共有 5 个失败：

1. Spatial task 4, trial 4：从顶层抽屉取出 black bowl 并放到 plate。未夹起 bowl，空手移动后超时。
2. Spatial task 8, trial 4：拿起 plate 旁的 black bowl 并放到 plate。抓取失败，目标留在源位并超时。
3. Goal task 8, trial 4：把 bowl 放到 plate。未抓起 bowl，停滞至超时。
4. LIBERO-10 task 2, trial 1：打开 stove 并把 moka pot 放上去。stove 已打开，但 moka pot 抓取失败并被碰倒。
5. LIBERO-10 task 6, trial 4：mug 放到 plate，再放置 chocolate pudding。第一子任务成功，第二个物体抓取失败。

失败集中在物体 acquisition 后无法恢复，没有发现成功判定假阴性、checkpoint 加载错误或渲染异常。

未列出的逐任务成功率均为 100%；低于 100% 的任务为：

- `libero_spatial_4`：4/5，80%
- `libero_spatial_8`：4/5，80%
- `libero_goal_8`：4/5，80%
- `libero_10_2`：4/5，80%
- `libero_10_6`：4/5，80%

## 结果文件

- 完整归档：`lightwam_libero_paper_repro/training_libero_20260711_archive/RESULTS.md`
- 机器可读汇总：`lightwam_libero_paper_repro/training_libero_20260711_archive/r5_results.tsv`
- Checkpoint SHA-256：`lightwam_libero_paper_repro/training_libero_20260711_archive/final_checkpoints.sha256`
- 各 suite 的详细结果：对应训练目录下的 `eval/r5/summary.json`
- 逐任务成功率：对应训练目录下的 `eval/r5/task_success_rates.csv`
- 失败 rollout：对应训练目录下的 `eval/r5/*/videos/`，文件名包含 `success=False`

## 使用边界

本结果属于原始 `training_libero` 路径，不包含后来实验的 memory、latent-flow 或 temporal-granularity 改造。此前其他分支得到的 Object 90% 等结果不能与本次 100% Object 混为同一配置。
