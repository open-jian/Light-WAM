# RM-Bench no-Memory baseline contract

This baseline adapts RM-Bench to the existing RoboTwin Light-WAM pipeline. It
does not add image history, proprioceptive history, task phase, executed-action
history, or any other memory signal.

## Dataset contract

- Native RM-Bench demonstrations remain read-only.
- The converted dataset uses the LeRobot v2.1 layout consumed by
  `RobotVideoDataset`.
- Camera order is `head_camera -> cam_high`, `left_camera -> cam_left_wrist`,
  and `right_camera -> cam_right_wrist`.
- Legacy RM-Bench JPEGs are converted once to conventional RGB. Training and
  evaluation must not apply an RM-Bench-specific channel swap.
- A source episode with `T` observations produces `T - 1` samples:
  `observation.state[t] = qpos[t]` and `action[t] = qpos[t + 1]`.
- The 14 action/state dimensions use RoboTwin's dual-arm ALOHA ordering.
- Dataset FPS defaults to 15, matching RM-Bench's supplied GO1 conversion.

## Experimental invariant

The conversion manifest records `memory_enabled: false`. Later training and
evaluation must reuse the current RoboTwin model and preprocessing path and
must fail closed if image memory or proprioceptive history is enabled.

Generated datasets, statistics, latent caches, checkpoints, and videos belong
under ignored `data/` or `runs/` directories and are not Git artifacts.
