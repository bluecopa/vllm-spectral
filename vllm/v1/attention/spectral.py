"""
SpectralQuant KV cache compression for vLLM.

Rotates K/V cache activations into spectral basis (eigenvectors of
activation covariance) before storing. Signal concentrates in first
d_eff dimensions; noise goes to remaining dimensions.

Phase 1: Rotation only — cache stays full head_dim but FP8 is much
more effective in spectral basis (signal dims get large values with
good FP8 precision, noise dims get near-zero values).

Phase 2: Dimensional reduction — truncate to d_eff dims, massive
memory savings (50-100x for Gemma 4).

The rotation preserves exact attention scores because V is orthogonal:
  score = (V^T q)^T (V^T k) = q^T V V^T k = q^T k

Usage:
  1. Generate sidecar: python phase9_calibrate_sidecar.py
  2. Place spectral_sidecar.pt alongside model
  3. Serve: vllm serve <model> --spectral-calibration spectral_sidecar.pt

Reference: SpectralQuant (Dynamis Labs, 2025)
           "3 Is All You Need" (nanothoughts, 2025)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.compiler

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global registry: maps model path -> SpectralCalibration
_SPECTRAL_REGISTRY: dict[str, "SpectralCalibration"] = {}

# Global flag for whether spectral rotation is enabled
_SPECTRAL_ENABLED = False

# Spectral rank for Phase 2 truncation (None = Phase 1 rotation only)
_SPECTRAL_RANK: int | None = None


@dataclass
class LayerSpectralConfig:
    """Spectral calibration data for one transformer layer."""
    # Rotation matrices: (num_kv_heads, head_dim, head_dim)
    # Columns are eigenvectors sorted by eigenvalue descending.
    # V^T rotates into spectral basis; V rotates back.
    k_rotation: torch.Tensor
    v_rotation: torch.Tensor
    # Effective dimensionality per KV head
    k_d_eff: torch.Tensor  # (num_kv_heads,)
    v_d_eff: torch.Tensor  # (num_kv_heads,)
    head_dim: int
    num_kv_heads: int
    layer_type: str  # "local" or "global"


class SpectralCalibration:
    """Holds spectral calibration data for all layers of a model."""

    def __init__(self, sidecar_path: str, device: torch.device | str = "cuda"):
        logger.info("Loading spectral calibration from %s", sidecar_path)
        raw = torch.load(sidecar_path, map_location="cpu", weights_only=True)

        self.num_layers = raw["num_layers"]
        self.layers: dict[int, LayerSpectralConfig] = {}

        for layer_idx, layer_data in raw["layers"].items():
            li = int(layer_idx)
            self.layers[li] = LayerSpectralConfig(
                k_rotation=layer_data["k_rotation"].to(device).float(),
                v_rotation=layer_data["v_rotation"].to(device).float(),
                k_d_eff=layer_data["k_d_eff"],
                v_d_eff=layer_data["v_d_eff"],
                head_dim=int(layer_data["head_dim"]),
                num_kv_heads=int(layer_data["num_kv_heads"]),
                layer_type=str(layer_data["layer_type"]),
            )

        # Build layer_name -> layer_idx mapping
        # vLLM layer names: "model.language_model.layers.{idx}.self_attn.attn"
        # or "model.layers.{idx}.self_attn.attn"
        # We extract the index at runtime from the layer_name.
        self._device = device

        # Log summary
        total_k_d_eff = sum(
            lc.k_d_eff.sum().item() for lc in self.layers.values()
        )
        total_heads = sum(lc.num_kv_heads for lc in self.layers.values())
        avg_k = total_k_d_eff / max(total_heads, 1)
        logger.info(
            "SpectralQuant loaded: %d layers, avg K d_eff=%.1f, %d total KV heads",
            self.num_layers, avg_k, total_heads,
        )

    def get_layer(self, layer_idx: int) -> LayerSpectralConfig | None:
        return self.layers.get(layer_idx)


def _extract_layer_index(layer_name: str) -> int:
    """Extract numeric layer index from vLLM layer name.

    Handles patterns like:
      "model.language_model.layers.5.self_attn.attn"
      "model.layers.5.self_attn.attn"
    """
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    raise ValueError(f"Cannot extract layer index from: {layer_name}")


def init_spectral(
    sidecar_path: str,
    spectral_rank: int | None = None,
    device: str = "cuda",
) -> None:
    """Initialize spectral calibration from sidecar file.

    Called during vLLM model loading when --spectral-calibration is set.

    Args:
        sidecar_path: Path to calibration .pt file.
        spectral_rank: If set, truncate cache to this many dims per head
            (Phase 2). If None, rotation only (Phase 1).
        device: Device to load rotation matrices onto.
    """
    global _SPECTRAL_ENABLED, _SPECTRAL_RANK
    if sidecar_path in _SPECTRAL_REGISTRY:
        logger.info("SpectralQuant already loaded for %s", sidecar_path)
        _SPECTRAL_ENABLED = True
        _SPECTRAL_RANK = spectral_rank
        return

    calibration = SpectralCalibration(sidecar_path, device=device)
    _SPECTRAL_REGISTRY[sidecar_path] = calibration
    _SPECTRAL_ENABLED = True
    _SPECTRAL_RANK = spectral_rank

    if spectral_rank is not None:
        logger.info(
            "SpectralQuant Phase 2 enabled: truncating cache to %d dims/head",
            spectral_rank,
        )
    else:
        logger.info("SpectralQuant Phase 1 enabled: rotation only")


def get_calibration() -> SpectralCalibration | None:
    """Get the active spectral calibration (first registered)."""
    if not _SPECTRAL_REGISTRY:
        return None
    return next(iter(_SPECTRAL_REGISTRY.values()))


@torch.compiler.disable
def is_enabled() -> bool:
    """Check if spectral rotation is active."""
    return _SPECTRAL_ENABLED


def is_truncating() -> bool:
    """Check if Phase 2 truncation is active."""
    return _SPECTRAL_ENABLED and _SPECTRAL_RANK is not None


def get_spectral_rank() -> int | None:
    """Get the configured spectral rank (None if Phase 1 only)."""
    return _SPECTRAL_RANK


def get_spectral_head_size(layer_name: str) -> int | None:
    """Get the truncated head_size for cache allocation.

    Returns spectral_rank if truncation is enabled, None otherwise.
    Used by Attention.get_kv_cache_spec() to allocate smaller cache blocks.
    """
    if not is_truncating():
        return None
    return _SPECTRAL_RANK


@torch.compiler.disable
def rotate_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate K and V tensors into spectral basis before cache storage.

    If Phase 2 (truncation) is active, only the first spectral_rank
    columns of the rotation matrix are used, producing truncated output.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        Phase 1: (rotated_key, rotated_value) same shapes as input
        Phase 2: (truncated_key, truncated_value) with last dim = spectral_rank
    """
    cal = get_calibration()
    if cal is None:
        return key, value

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return key, value

    orig_dtype = key.dtype
    rank = _SPECTRAL_RANK  # None for Phase 1

    # Use truncated rotation matrices if Phase 2
    # V_k: (H, D, D) → V_k[:, :, :rank]: (H, D, rank)
    k_rot = lc.k_rotation if rank is None else lc.k_rotation[:, :, :rank]
    v_rot = lc.v_rotation if rank is None else lc.v_rotation[:, :, :rank]

    # key: (T, H, D) → transpose to (H, T, D)
    # matmul: (H, T, D) @ (H, D, rank_or_D) → (H, T, rank_or_D)
    k_float = key.transpose(0, 1).float()
    v_float = value.transpose(0, 1).float()

    k_rotated = torch.bmm(k_float, k_rot).transpose(0, 1)  # (T, H, rank_or_D)
    v_rotated = torch.bmm(v_float, v_rot).transpose(0, 1)

    if rank is None:
        # Phase 1: same shape — copy in-place to preserve tensor identity
        # This is critical for torch.compile which may trace tensor pointers
        key.copy_(k_rotated.to(orig_dtype))
        value.copy_(v_rotated.to(orig_dtype))
        return key, value
    else:
        return k_rotated.to(orig_dtype), v_rotated.to(orig_dtype)


@torch.compiler.disable
def rotate_q(
    query: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Rotate Q tensor into spectral basis to match cached K rotation.

    If Phase 2 (truncation) is active, Q is also truncated to spectral_rank
    dims so it matches the truncated K cache for attention score computation.

    For GQA: each query head group shares the same K rotation.

    Args:
        query: (num_tokens, num_q_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        Phase 1: rotated query with same shape
        Phase 2: truncated query with last dim = spectral_rank
    """
    cal = get_calibration()
    if cal is None:
        return query

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return query

    num_q_heads = query.shape[1]
    num_kv_heads = lc.num_kv_heads
    group_size = num_q_heads // num_kv_heads
    rank = _SPECTRAL_RANK

    # Truncated rotation: (H_kv, D, D) → (H_kv, D, rank)
    k_rot = lc.k_rotation if rank is None else lc.k_rotation[:, :, :rank]

    orig_dtype = query.dtype

    if group_size == 1:
        q_float = query.transpose(0, 1).float()  # (H, T, D)
        q_rotated = torch.bmm(q_float, k_rot).transpose(0, 1)
    else:
        expanded_rotation = k_rot.repeat_interleave(group_size, dim=0)
        q_float = query.transpose(0, 1).float()  # (num_q_heads, T, D)
        q_rotated = torch.bmm(q_float, expanded_rotation).transpose(0, 1)

    if rank is None:
        # In-place copy to preserve tensor identity for torch.compile
        query.copy_(q_rotated.to(orig_dtype))
        return query
    else:
        return q_rotated.to(orig_dtype)


@torch.compiler.disable
def unrotate_output(
    output: torch.Tensor,
    layer_name: str,
    full_head_dim: int | None = None,
) -> torch.Tensor:
    """
    Rotate attention output back from spectral basis.

    Phase 1: output has full head_dim, unrotate with V_v^T.
    Phase 2: output has spectral_rank dims, unrotate with V_v[:, :rank]^T
             which expands back to full head_dim.

    Math: out @ V_v[:, :rank]^T = out @ V_v[:rank, :] → (T, H, full_D)

    Args:
        output: (num_tokens, num_q_heads, spectral_rank_or_head_dim)
        layer_name: vLLM layer identifier
        full_head_dim: original head dim (needed for Phase 2 expansion)

    Returns:
        unrotated output with last dim = full_head_dim
    """
    cal = get_calibration()
    if cal is None:
        return output

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return output

    num_q_heads = output.shape[1]
    num_kv_heads = lc.num_kv_heads
    group_size = num_q_heads // num_kv_heads
    rank = _SPECTRAL_RANK

    orig_dtype = output.dtype

    # Unrotation matrix:
    # Phase 1: V_v^T full (D, D) — just transpose
    # Phase 2: V_v[:, :rank]^T = (rank, D) — maps rank dims back to full D
    if rank is not None:
        # (H_kv, D, rank) → transpose → (H_kv, rank, D)
        v_unrot = lc.v_rotation[:, :, :rank].transpose(-2, -1)
    else:
        v_unrot = lc.v_rotation.transpose(-2, -1)  # (H_kv, D, D)

    if group_size == 1:
        o_float = output.transpose(0, 1).float()  # (H, T, rank_or_D)
        o_unrotated = torch.bmm(o_float, v_unrot).transpose(0, 1)
    else:
        expanded_unrot = v_unrot.repeat_interleave(group_size, dim=0)
        o_float = output.transpose(0, 1).float()
        o_unrotated = torch.bmm(o_float, expanded_unrot).transpose(0, 1)

    return o_unrotated.to(orig_dtype)
