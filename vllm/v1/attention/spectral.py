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

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global registry: maps model path -> SpectralCalibration
_SPECTRAL_REGISTRY: dict[str, "SpectralCalibration"] = {}

# Global flag for whether spectral rotation is enabled
_SPECTRAL_ENABLED = False


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


def init_spectral(sidecar_path: str, device: str = "cuda") -> None:
    """Initialize spectral calibration from sidecar file.

    Called during vLLM model loading when --spectral-calibration is set.
    """
    global _SPECTRAL_ENABLED
    if sidecar_path in _SPECTRAL_REGISTRY:
        logger.info("SpectralQuant already loaded for %s", sidecar_path)
        _SPECTRAL_ENABLED = True
        return

    calibration = SpectralCalibration(sidecar_path, device=device)
    _SPECTRAL_REGISTRY[sidecar_path] = calibration
    _SPECTRAL_ENABLED = True
    logger.info("SpectralQuant enabled")


def get_calibration() -> SpectralCalibration | None:
    """Get the active spectral calibration (first registered)."""
    if not _SPECTRAL_REGISTRY:
        return None
    return next(iter(_SPECTRAL_REGISTRY.values()))


def is_enabled() -> bool:
    """Check if spectral rotation is active."""
    return _SPECTRAL_ENABLED


def rotate_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate K and V tensors into spectral basis before cache storage.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        (rotated_key, rotated_value) with same shapes

    Math: k_rotated = k @ V_k  (row-vector convention)
    This is equivalent to V_k^T @ k_col for column vectors.
    """
    cal = get_calibration()
    if cal is None:
        return key, value

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return key, value

    # key: (T, H, D), V_k: (H, D, D)
    # Batched matmul: transpose to (H, T, D) @ (H, D, D) -> (H, T, D)
    orig_dtype = key.dtype
    k_float = key.transpose(0, 1).float()  # (H, T, D)
    v_float = value.transpose(0, 1).float()

    k_rotated = torch.bmm(k_float, lc.k_rotation).transpose(0, 1)  # (T, H, D)
    v_rotated = torch.bmm(v_float, lc.v_rotation).transpose(0, 1)

    return k_rotated.to(orig_dtype), v_rotated.to(orig_dtype)


def rotate_q(
    query: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Rotate Q tensor into spectral basis to match cached K rotation.

    For GQA: each query head group shares the same K rotation.

    Args:
        query: (num_tokens, num_q_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        rotated query with same shape
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

    orig_dtype = query.dtype

    if group_size == 1:
        # MHA or 1:1 mapping — simple batched matmul
        q_float = query.transpose(0, 1).float()  # (H, T, D)
        q_rotated = torch.bmm(q_float, lc.k_rotation).transpose(0, 1)
    else:
        # GQA: expand K rotation to match Q heads
        # k_rotation: (num_kv_heads, D, D)
        # Repeat each KV head's rotation for its query group
        # (num_kv_heads, D, D) -> (num_q_heads, D, D)
        expanded_rotation = lc.k_rotation.repeat_interleave(group_size, dim=0)
        q_float = query.transpose(0, 1).float()  # (num_q_heads, T, D)
        q_rotated = torch.bmm(q_float, expanded_rotation).transpose(0, 1)

    return q_rotated.to(orig_dtype)


def unrotate_output(
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Rotate attention output back from spectral basis.

    The attention output is: out_rot = softmax(scores) @ V_cached
    where V_cached is in spectral basis (V_v^T @ v).
    So out_rot = V_v^T @ true_output, and we need:
    true_output = V_v @ out_rot = out_rot @ V_v^T

    For GQA: each query head group shares the same V rotation.

    Args:
        output: (num_tokens, num_q_heads, head_dim_v)
        layer_name: vLLM layer identifier

    Returns:
        unrotated output with same shape
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

    orig_dtype = output.dtype

    # Unrotation: out @ V_v^T (transpose of the rotation matrix)
    # V_v: (num_kv_heads, D, D) — columns are eigenvectors
    # V_v^T: (num_kv_heads, D, D) — rows are eigenvectors
    v_rotation_T = lc.v_rotation.transpose(-2, -1)  # (H_kv, D, D)

    if group_size == 1:
        o_float = output.transpose(0, 1).float()  # (H, T, D)
        o_unrotated = torch.bmm(o_float, v_rotation_T).transpose(0, 1)
    else:
        expanded_rotation_T = v_rotation_T.repeat_interleave(group_size, dim=0)
        o_float = output.transpose(0, 1).float()
        o_unrotated = torch.bmm(o_float, expanded_rotation_T).transpose(0, 1)

    return o_unrotated.to(orig_dtype)
