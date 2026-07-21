from __future__ import annotations

import torch


def validate_anchor_recent_layout(
    *,
    history_max_size: int,
    anchor_size: int,
    anchor_stride: int,
    recent_max_size: int,
    recent_stride: int,
) -> None:
    """Validate the fixed ``[anchors, recent]`` observation-memory layout."""
    history_max_size = int(history_max_size)
    anchor_size = int(anchor_size)
    anchor_stride = int(anchor_stride)
    recent_max_size = int(recent_max_size)
    recent_stride = int(recent_stride)
    if anchor_size < 0 or recent_max_size < 0:
        raise ValueError("Anchor/recent sizes must be non-negative.")
    if anchor_size + recent_max_size != history_max_size:
        raise ValueError(
            "History layout must satisfy anchor_size + recent_max_size == history_max_size, "
            f"got {anchor_size} + {recent_max_size} != {history_max_size}."
        )
    if anchor_size > 0 and anchor_stride <= 0:
        raise ValueError(f"`anchor_stride` must be positive, got {anchor_stride}.")
    if recent_max_size > 0 and recent_stride <= 0:
        raise ValueError(f"`recent_stride` must be positive, got {recent_stride}.")


def build_anchor_recent_history_indices(
    *,
    target_local_indices: torch.Tensor | list[int],
    episode_length: int,
    history_max_size: int,
    anchor_size: int,
    anchor_stride: int,
    recent_max_size: int,
    recent_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-slot source indices and validity for episode observations.

    Slots are ``[episode-start anchors, right-aligned recent observations]``.
    The current observation is never included, and a recent observation that is
    already present in an anchor slot is masked to avoid duplicate memory tokens.
    Invalid slots contain a safe clamped index and must be ignored using the mask.
    """
    validate_anchor_recent_layout(
        history_max_size=history_max_size,
        anchor_size=anchor_size,
        anchor_stride=anchor_stride,
        recent_max_size=recent_max_size,
        recent_stride=recent_stride,
    )
    episode_length = int(episode_length)
    if episode_length <= 0:
        raise ValueError(f"`episode_length` must be positive, got {episode_length}.")
    targets = torch.as_tensor(target_local_indices, dtype=torch.int64).view(-1)
    if targets.numel() == 0:
        return (
            torch.empty((0, int(history_max_size)), dtype=torch.int64),
            torch.empty((0, int(history_max_size)), dtype=torch.bool),
        )
    if bool(((targets < 0) | (targets >= episode_length)).any().item()):
        raise ValueError(
            f"Target indices must lie in [0, {episode_length}), got {targets.tolist()}."
        )

    parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    anchor_sources = torch.empty((targets.numel(), 0), dtype=torch.int64)
    anchor_valid = torch.empty((targets.numel(), 0), dtype=torch.bool)
    if int(anchor_size) > 0:
        anchor_steps = torch.arange(int(anchor_size), dtype=torch.int64) * int(anchor_stride)
        anchor_sources = anchor_steps.unsqueeze(0).expand(targets.numel(), -1)
        anchor_valid = (anchor_sources < targets.unsqueeze(1)) & (
            anchor_sources < episode_length
        )
        parts.append(anchor_sources)
        valid_parts.append(anchor_valid)

    if int(recent_max_size) > 0:
        recent_offsets = torch.arange(
            -int(recent_max_size) * int(recent_stride),
            0,
            int(recent_stride),
            dtype=torch.int64,
        )
        recent_sources = targets.unsqueeze(1) + recent_offsets.unsqueeze(0)
        recent_valid = (recent_sources >= 0) & (recent_sources < targets.unsqueeze(1))
        if int(anchor_size) > 0:
            duplicates = (
                recent_sources.unsqueeze(2) == anchor_sources.unsqueeze(1)
            ) & anchor_valid.unsqueeze(1)
            recent_valid &= ~duplicates.any(dim=2)
        parts.append(recent_sources)
        valid_parts.append(recent_valid)

    sources = torch.cat(parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)
    if tuple(sources.shape) != (targets.numel(), int(history_max_size)):
        raise RuntimeError(
            f"Internal history layout mismatch: got {tuple(sources.shape)}."
        )
    return sources.clamp(min=0, max=episode_length - 1).contiguous(), valid.contiguous()


def apply_recent_window_mask(
    valid_mask: torch.Tensor,
    *,
    anchor_size: int,
    recent_window_size: int,
) -> torch.Tensor:
    """Keep every anchor and only the newest ``recent_window_size`` recent slots."""
    if valid_mask.ndim != 2:
        raise ValueError(f"`valid_mask` must be [B,H], got {tuple(valid_mask.shape)}.")
    anchor_size = int(anchor_size)
    recent_window_size = int(recent_window_size)
    recent_size = int(valid_mask.shape[1]) - anchor_size
    if anchor_size < 0 or recent_size < 0:
        raise ValueError("Invalid anchor size for history mask.")
    if recent_window_size < 0 or recent_window_size > recent_size:
        raise ValueError(
            f"Recent window must be in [0, {recent_size}], got {recent_window_size}."
        )
    output = valid_mask.to(dtype=torch.bool).clone()
    drop_recent = recent_size - recent_window_size
    if drop_recent > 0:
        output[:, anchor_size : anchor_size + drop_recent] = False
    return output.contiguous()
