# Light-WAM Project Instructions

## Scope and repository safety

- These instructions apply to the entire Light-WAM repository.
- Work in the current branch and worktree. Do not create, remove, or switch branches or worktrees unless the user explicitly asks. If another branch is required, state its name and reason first.
- Do not edit the sibling FastWAM repository when the task is scoped to Light-WAM.
- Treat `third_party/` as vendored simulator source. Avoid modifying it unless the task specifically requires a vendor patch.
- Preserve existing user changes, datasets, checkpoints, and run artifacts. Never clean or delete them merely to recover disk space.

## Environment and entrypoints

- Use the `lightwam` environment for training and preprocessing. Follow the dedicated LIBERO or RoboTwin evaluation environments documented in `README.md` when simulator dependencies require them.
- Prefer the maintained scripts in `scripts/` over retyping long Hydra commands.
- Common entrypoints:
  - RoboTwin training: `bash scripts/train_robotwin.sh`
  - RoboTwin evaluation: `CKPT=/path/to/weights.pt bash scripts/eval_robotwin.sh`
  - RMBench no-memory training: `bash scripts/train_rmbench_no_memory.sh`
  - RMBench no-memory evaluation: `CKPT=/path/to/weights.pt TASK_NAME=<task> bash scripts/eval_rmbench_no_memory.sh`

## Data and output placement

- Put every generated video under `/data2/jian/project/Light-WAM/Outputs/`. This includes training visualizations, evaluation rollouts, predicted videos, replay videos, and diagnostic clips.
- Organize videos below `Outputs/` by experiment or run name; do not place videos anywhere else.
- Put generated checkpoints, logs, non-video evaluation results, diagnostics, and temporary experiment artifacts under this repository's `runs/` directory.
- When evaluating a checkpoint from an existing training run, place non-video evaluation outputs under that training run when practical, while keeping its videos under `Outputs/`.
- Never write experiment outputs into `data/`, `third_party/RoboTwin/`, an external RMBench checkout, or another project's directory.
- Dataset directories and simulator checkouts are inputs, not result directories. Keep them clean.
- Use short W&B run names containing only the task, essential method/config identifier, and step budget.

## Training and evaluation

- Before a long multi-GPU run, verify the resolved task, dataset, checkpoint, normalization statistics, GPU count, step/episode count, and output directory.
- Run a minimal smoke test before a long training or evaluation job unless the user explicitly asks to skip it.
- Do not silently change official evaluation cameras, task configs, instruction splits, or domain settings to improve scores.
- The RMBench `no_memory` pipeline must remain free of memory/history features unless the user explicitly requests a different experiment.
- Report the exact checkpoint and dataset statistics loaded, the final output directory, completed episode count, success rate, and video location.

## Verification

- Run targeted tests for touched code. Use `pytest -q` for broad shared changes when feasible.
- Run `git diff --check` before handing off code changes.
- For training changes, verify trainable/frozen parameter groups and perform at least one optimizer step when practical.
- For evaluation changes, verify one complete episode and confirm that videos are written under `Outputs/` and non-video result files under `runs/`.
