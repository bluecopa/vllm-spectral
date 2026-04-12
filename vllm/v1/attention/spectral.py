"""
SpectralQuant KV cache compression for vLLM.

Rotates K/V cache activations into spectral basis (eigenvectors of
activation covariance) before storing. Signal concentrates in first
d_eff dimensions; noise goes to remaining dimensions.

Phase 1: Rotation only — cache stays full head_dim but FP8 is much
more effective in spectral basis (signal dims get large values with
good FP8 precision, noise dims get near-zero values).

Phase 2 (non-uniform quantization): After rotation, quantize with
more bits for high-variance "semantic" dims and fewer bits for
low-variance "tail" dims using Lloyd-Max optimal codebooks. The cache
stores uint8 indices per dim; norms are stored in a side buffer.
Attention runs in rotated basis (inner products preserved by
orthogonal rotation); only the V output needs unrotation.

The rotation preserves exact attention scores because V is orthogonal:
  score = (V^T q)^T (V^T k) = q^T V V^T k = q^T k

Usage:
  1. Generate sidecar: python phase9_calibrate_sidecar.py
  2. Place spectral_sidecar.pt alongside model
  3. Phase 1: vllm serve <model> --spectral-calibration spectral_sidecar.pt
  4. Phase 2: vllm serve <model> --spectral-calibration spectral_sidecar_chat_v2.pt --spectral-quantize

Reference: SpectralQuant (Dynamis Labs, 2025)
           "3 Is All You Need" (nanothoughts, 2025)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
import torch.compiler

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global registry: maps model path -> SpectralCalibration
_SPECTRAL_REGISTRY: dict[str, "SpectralCalibration"] = {}

# Global flag for whether spectral rotation is enabled
_SPECTRAL_ENABLED = False

# Spectral rank for Phase 2 compressed-cache mode (old trunk-based approach).
_SPECTRAL_RANK: int | None = None

# Phase 2 non-uniform quantization flag
_SPECTRAL_QUANTIZE = False

# Default bit allocation for Phase 2. Global layers use these; local layers
# stay on Phase 1 rotation-only.
_PHASE2_B_HIGH = 6
_PHASE2_B_LOW = 4

# Phase 2 non-uniform quantization layer types. Controlled by env var.
_COMPRESSED_LAYER_TYPES = (
    {"global", "local"} if os.environ.get("SPECTRAL_ALL_LAYERS", "1") == "1"
    else {"global"}
)


# ---------------------------------------------------------------------------
# Per-layer codebooks for Phase 2
# ---------------------------------------------------------------------------

@dataclass
class LayerCodebooks:
    """Lloyd-Max codebooks for one layer's non-uniform quantization."""
    # Centroid tensors on device: (num_kv_heads, 2^b_high)
    k_semantic_centroids: torch.Tensor
    k_tail_centroids: torch.Tensor
    v_semantic_centroids: torch.Tensor
    v_tail_centroids: torch.Tensor
    # Per-head d_eff (int rounded from participation ratio)
    k_d_eff_int: list[int]
    v_d_eff_int: list[int]
    b_high: int
    b_low: int


# Per-layer codebook registry: layer_idx -> LayerCodebooks
_LAYER_CODEBOOKS: dict[int, LayerCodebooks] = {}

# Per-layer packed dim registry: layer_idx -> packed_dim.
# Packed_dim is the max required width within that layer across K/V heads.
_PACKED_DIMS: dict[int, int] = {}  # layer_idx -> packed_dim

# Per-layer allocation dim: defaults to packed_dim. It may be padded above
# packed_dim when SPECTRAL_SHARED_ALLOC=1 to preserve shared-tensor allocation.
_ALLOC_DIMS: dict[int, int] = {}  # layer_idx -> alloc_dim (>= packed_dim)

# Vectorized pack/unpack maps: precomputed index tensors for batch bit-packing.
# Keyed by (layer_idx, 'k'|'v').
_PACK_MAPS: dict[tuple, tuple] = {}
_UNPACK_MAPS: dict[tuple, tuple] = {}

# Dequant temp buffers for Phase 2 → Triton attention path.
# Compact buffer: only max_active_blocks rows, block table remapped at runtime.
# Per-layer-type buffers because global/local may have different block_elems.
_DEQUANT_KEY_BUF: dict[str, torch.Tensor] = {}  # layer_type -> (max_active, block_elems) bf16
_DEQUANT_VAL_BUF: dict[str, torch.Tensor] = {}
_DEQUANT_REMAP: dict[str, torch.Tensor] = {}  # layer_type -> (num_total_blocks,) int32
_DEQUANT_ACTIVE_MASK: dict[str, torch.Tensor] = {}  # layer_type -> (num_total_blocks,) int32
_DEQUANT_BLOCK_LIST: dict[str, torch.Tensor] = {}   # layer_type -> (MAX_BLOCKS,) int64
_DEQUANT_MAX_BLOCKS: int = 0
_DEQUANT_VIEWS: dict[str, tuple[int, int, int]] = {}  # layer_type -> (block_size, H, D)

# Pre-allocated rotation output buffers for Phase 2 deferred compress.
# Keyed by layer_type (global/local) since dimensions differ.
_ROTATE_BUF_K: dict[str, torch.Tensor] = {}  # layer_type -> (max_batch_tokens, H, D) bf16
_ROTATE_BUF_V: dict[str, torch.Tensor] = {}
_ROTATE_NORMS_K: dict[str, torch.Tensor] = {}  # layer_type -> (max_batch_tokens, H) f32
_ROTATE_NORMS_V: dict[str, torch.Tensor] = {}

# Secondary CUDA stream for async compress (Phase 2 deferred compress).
_COMPRESS_STREAM: torch.cuda.Stream | None = None
_ASYNC_COMPRESS = os.environ.get("SPECTRAL_ASYNC_COMPRESS", "0") == "1"

# Deferred compress: skip compress_kv, pass raw K/V to attention for
# direct injection into dequant buffer. Gated by SPECTRAL_DEFERRED_COMPRESS.
_DEFERRED_COMPRESS = os.environ.get("SPECTRAL_DEFERRED_COMPRESS", "0") == "1"

# Per-layer stash for raw K/V + slot_mapping (deferred compress path).
# Set by stash_kv_for_deferred_compress(), consumed by
# spectral_phase2_triton_attention().
_DEFERRED_KV_STASH: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def is_deferred_compress() -> bool:
    """Check if deferred compress path is active."""
    return _SPECTRAL_QUANTIZE and _DEFERRED_COMPRESS


def stash_kv_for_deferred_compress(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    slot_mapping: torch.Tensor,
) -> None:
    """Stash raw K/V for later injection in spectral_phase2_triton_attention."""
    layer_idx = _extract_layer_index(layer_name)
    _DEFERRED_KV_STASH[layer_idx] = (key, value, slot_mapping)


def pop_deferred_kv(layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pop stashed K/V for this layer. Returns (key, value, slot_mapping) or None."""
    return _DEFERRED_KV_STASH.pop(layer_idx, None)


def _build_pack_map(
    d_effs: list[int], head_dim: int, packed_dim: int, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build vectorized pack index map for all heads.

    Returns (hi_src, lo_src, is_sem, has_lo, valid) each (H, packed_dim).
    """
    H = len(d_effs)
    hi = [[0] * packed_dim for _ in range(H)]
    lo = [[0] * packed_dim for _ in range(H)]
    sem = [[False] * packed_dim for _ in range(H)]
    has = [[False] * packed_dim for _ in range(H)]
    val = [[False] * packed_dim for _ in range(H)]

    for h in range(H):
        d_eff = d_effs[h]
        n_tail = head_dim - d_eff
        n_pairs = n_tail // 2
        for p in range(d_eff):
            hi[h][p] = p
            sem[h][p] = True
            val[h][p] = True
        for j in range(n_pairs):
            p = d_eff + j
            hi[h][p] = d_eff + 2 * j
            lo[h][p] = d_eff + 2 * j + 1
            has[h][p] = True
            val[h][p] = True
        if n_tail % 2 == 1:
            p = d_eff + n_pairs
            hi[h][p] = head_dim - 1
            val[h][p] = True

    return (
        torch.tensor(hi, dtype=torch.long, device=device),
        torch.tensor(lo, dtype=torch.long, device=device),
        torch.tensor(sem, dtype=torch.bool, device=device),
        torch.tensor(has, dtype=torch.bool, device=device),
        torch.tensor(val, dtype=torch.bool, device=device),
    )


def _build_unpack_map(
    d_effs: list[int], head_dim: int, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build vectorized unpack index map for all heads.

    Returns (src, is_sem, is_high) each (H, head_dim).
    """
    H = len(d_effs)
    src = [[0] * head_dim for _ in range(H)]
    sem = [[False] * head_dim for _ in range(H)]
    high = [[False] * head_dim for _ in range(H)]

    for h in range(H):
        d_eff = d_effs[h]
        for d in range(d_eff):
            src[h][d] = d
            sem[h][d] = True
        for j in range(head_dim - d_eff):
            d = d_eff + j
            src[h][d] = d_eff + j // 2
            high[h][d] = (j % 2 == 0)

    return (
        torch.tensor(src, dtype=torch.long, device=device),
        torch.tensor(sem, dtype=torch.bool, device=device),
        torch.tensor(high, dtype=torch.bool, device=device),
    )


def _pack_all_heads(
    indices: torch.Tensor,
    hi_src: torch.Tensor,
    lo_src: torch.Tensor,
    is_sem: torch.Tensor,
    has_lo: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Vectorized bit-pack: (T, H, D) uint8 indices → (T, H, packed_dim) uint8."""
    T = indices.shape[0]
    v_hi = torch.gather(indices, -1, hi_src.unsqueeze(0).expand(T, -1, -1))
    v_lo = torch.gather(indices, -1, lo_src.unsqueeze(0).expand(T, -1, -1))
    v_lo_safe = torch.where(has_lo.unsqueeze(0), v_lo, torch.zeros_like(v_lo))
    computed = torch.where(
        is_sem.unsqueeze(0), v_hi, (v_hi << 4) | v_lo_safe,
    )
    return torch.where(valid.unsqueeze(0), computed, torch.zeros_like(computed))


def _unpack_all_heads(
    packed: torch.Tensor,
    src: torch.Tensor,
    is_sem: torch.Tensor,
    is_high: torch.Tensor,
) -> torch.Tensor:
    """Vectorized bit-unpack: (S, H, packed_dim) uint8 → (S, H, D) uint8."""
    S = packed.shape[0]
    raw = torch.gather(packed, -1, src.unsqueeze(0).expand(S, -1, -1))
    return torch.where(
        is_sem.unsqueeze(0),
        raw,
        torch.where(is_high.unsqueeze(0), (raw >> 4) & 0x0F, raw & 0x0F),
    )

# Norm side buffer: stores (k_norm, v_norm) per token per Phase 2 KV head.
# Shape: (max_tokens, num_phase2_kv_heads, 2) in float16.
# Indexed by absolute cache slot position.
_NORM_BUFFER: torch.Tensor | None = None
_NORM_BUFFER_LAYER_OFFSETS: dict[int, int] = {}  # layer_idx -> head offset


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
    # Eigenvalues per KV head (v2 sidecar only, None for v1)
    k_eigenvalues: torch.Tensor | None = None  # (num_kv_heads, head_dim)
    v_eigenvalues: torch.Tensor | None = None  # (num_kv_heads, head_dim)
    # Pre-computed f32 inverse rotation for unrotate_output
    v_unrot_f32: torch.Tensor | None = field(default=None, repr=False)
    # Lazy cache for GQA-expanded f32 rotations (keyed by num_q_heads)
    _q_rot_cache: dict = field(default_factory=dict, repr=False)
    _v_unrot_cache: dict = field(default_factory=dict, repr=False)

    def _init_cached_rotations(self) -> None:
        """Pre-compute transposed V rotation and make contiguous."""
        self.v_unrot_f32 = self.v_rotation.transpose(
            -2, -1
        ).contiguous()

    def get_q_rot(self, num_q_heads: int) -> torch.Tensor:
        """Get f32 K rotation expanded for Q heads (GQA), cached."""
        if num_q_heads not in self._q_rot_cache:
            group_size = num_q_heads // self.num_kv_heads
            if group_size <= 1:
                self._q_rot_cache[num_q_heads] = self.k_rotation
            else:
                self._q_rot_cache[num_q_heads] = (
                    self.k_rotation.repeat_interleave(group_size, dim=0)
                )
        return self._q_rot_cache[num_q_heads]

    def get_v_unrot(self, num_q_heads: int) -> torch.Tensor:
        """Get f32 V inverse rotation expanded for Q heads (GQA), cached."""
        if num_q_heads not in self._v_unrot_cache:
            group_size = num_q_heads // self.num_kv_heads
            if group_size <= 1:
                self._v_unrot_cache[num_q_heads] = self.v_unrot_f32
            else:
                self._v_unrot_cache[num_q_heads] = (
                    self.v_unrot_f32.repeat_interleave(group_size, dim=0)
                )
        return self._v_unrot_cache[num_q_heads]


class SpectralCalibration:
    """Holds spectral calibration data for all layers of a model."""

    def __init__(self, sidecar_path: str, device: torch.device | str = "cuda"):
        logger.info("Loading spectral calibration from %s", sidecar_path)
        raw = torch.load(sidecar_path, map_location="cpu", weights_only=True)

        self.num_layers = raw["num_layers"]
        self.version = raw.get("version", 1)
        self.layers: dict[int, LayerSpectralConfig] = {}

        for layer_idx, layer_data in raw["layers"].items():
            li = int(layer_idx)

            # v2 sidecars include eigenvalues
            k_evals = layer_data.get("k_eigenvalues")
            v_evals = layer_data.get("v_eigenvalues")
            if k_evals is not None:
                k_evals = k_evals.to(device).float()
            if v_evals is not None:
                v_evals = v_evals.to(device).float()

            lc = LayerSpectralConfig(
                k_rotation=layer_data["k_rotation"].to(device).float(),
                v_rotation=layer_data["v_rotation"].to(device).float(),
                k_d_eff=layer_data["k_d_eff"],
                v_d_eff=layer_data["v_d_eff"],
                head_dim=int(layer_data["head_dim"]),
                num_kv_heads=int(layer_data["num_kv_heads"]),
                layer_type=str(layer_data["layer_type"]),
                k_eigenvalues=k_evals,
                v_eigenvalues=v_evals,
            )
            lc._init_cached_rotations()
            self.layers[li] = lc

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
            "SpectralQuant loaded: %d layers, avg K d_eff=%.1f, %d total KV heads, version=%d",
            self.num_layers, avg_k, total_heads, self.version,
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


# ---------------------------------------------------------------------------
# Lloyd-Max codebook computation
# ---------------------------------------------------------------------------

# Cache of unit-variance Lloyd-Max centroids: n_bits -> centroids for N(0,1)
_UNIT_CENTROIDS_CACHE: dict[int, torch.Tensor] = {}


def _get_unit_centroids(n_bits: int) -> torch.Tensor:
    """Get Lloyd-Max centroids for N(0, 1). Computed once and cached."""
    if n_bits in _UNIT_CENTROIDS_CACHE:
        return _UNIT_CENTROIDS_CACHE[n_bits]

    n_levels = 1 << n_bits
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    data = torch.randn(50000, generator=gen, dtype=torch.float32)

    d_min, d_max = float(data.min()), float(data.max())
    centroids = torch.linspace(d_min, d_max, n_levels, dtype=torch.float32)

    for _ in range(200):
        dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)

        new_centroids = centroids.clone()
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = data[mask].mean()

        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if shift < 1e-7:
            break

    centroids = centroids.sort().values
    _UNIT_CENTROIDS_CACHE[n_bits] = centroids
    logger.info("Lloyd-Max unit centroids computed for %d bits (%d levels)", n_bits, n_levels)
    return centroids


def _solve_lloyd_max_gaussian(
    sigma: float,
    n_bits: int,
) -> torch.Tensor:
    """Compute Lloyd-Max centroids for N(0, sigma^2).

    Exploits the fact that optimal centroids for N(0, σ²) are exactly
    σ × centroids_for_N(0,1). Only 2 solves needed total (one per bit width).
    """
    return _get_unit_centroids(n_bits) * sigma


def _compute_layer_codebooks(
    lc: LayerSpectralConfig,
    b_high: int,
    b_low: int,
    device: torch.device | str,
) -> LayerCodebooks:
    """Compute per-head Lloyd-Max codebooks from eigenvalues."""
    H = lc.num_kv_heads
    D = lc.head_dim
    k_evals = lc.k_eigenvalues  # (H, D) or None
    v_evals = lc.v_eigenvalues

    if k_evals is None or v_evals is None:
        raise RuntimeError(
            "Cannot compute Phase 2 codebooks: sidecar lacks eigenvalues. "
            "Re-run phase9_calibrate_sidecar.py to generate v2 sidecar."
        )

    k_d_eff_int = []
    v_d_eff_int = []
    k_sem_centroids_list = []
    k_tail_centroids_list = []
    v_sem_centroids_list = []
    v_tail_centroids_list = []

    for h in range(H):
        # K codebooks
        d_eff_k = max(1, min(round(float(lc.k_d_eff[h].item())), D - 1))
        k_d_eff_int.append(d_eff_k)

        ev_k = k_evals[h].cpu().float()
        # Eigenvalues are from unnormalized data. Since we quantize NORMALIZED
        # vectors (divided by L2 norm), the per-dim variance shrinks by
        # sum(eigenvalues) = E[||x||^2]. Divide sigma by expected_norm to
        # match the actual distribution of the normalized data.
        k_norm_sq = float(ev_k.sum().clamp(min=1e-8).item())
        k_expected_norm = math.sqrt(k_norm_sq)
        sigma_k_high = float(ev_k[:d_eff_k].clamp(min=1e-8).sqrt().mean().item()) / k_expected_norm
        sigma_k_low = float(ev_k[d_eff_k:].clamp(min=1e-8).sqrt().mean().item()) / k_expected_norm
        sigma_k_low = max(sigma_k_low, sigma_k_high * 1e-4)

        k_sem_centroids_list.append(
            _solve_lloyd_max_gaussian(sigma_k_high, b_high).to(device)
        )
        k_tail_centroids_list.append(
            _solve_lloyd_max_gaussian(sigma_k_low, b_low).to(device)
        )

        # V codebooks
        d_eff_v = max(1, min(round(float(lc.v_d_eff[h].item())), D - 1))
        v_d_eff_int.append(d_eff_v)

        ev_v = v_evals[h].cpu().float()
        v_norm_sq = float(ev_v.sum().clamp(min=1e-8).item())
        v_expected_norm = math.sqrt(v_norm_sq)
        sigma_v_high = float(ev_v[:d_eff_v].clamp(min=1e-8).sqrt().mean().item()) / v_expected_norm
        sigma_v_low = float(ev_v[d_eff_v:].clamp(min=1e-8).sqrt().mean().item()) / v_expected_norm
        sigma_v_low = max(sigma_v_low, sigma_v_high * 1e-4)

        v_sem_centroids_list.append(
            _solve_lloyd_max_gaussian(sigma_v_high, b_high).to(device)
        )
        v_tail_centroids_list.append(
            _solve_lloyd_max_gaussian(sigma_v_low, b_low).to(device)
        )

    return LayerCodebooks(
        k_semantic_centroids=torch.stack(k_sem_centroids_list),  # (H, 2^b_high)
        k_tail_centroids=torch.stack(k_tail_centroids_list),      # (H, 2^b_low)
        v_semantic_centroids=torch.stack(v_sem_centroids_list),
        v_tail_centroids=torch.stack(v_tail_centroids_list),
        k_d_eff_int=k_d_eff_int,
        v_d_eff_int=v_d_eff_int,
        b_high=b_high,
        b_low=b_low,
    )


def _compute_packed_dim(codebooks: LayerCodebooks, head_dim: int) -> int:
    """Max packed dim across all heads and K/V.

    Per head: d_eff bytes (semantic, full uint8) + ceil((D - d_eff) / 2) bytes
    (tail, nibble-packed two per byte).
    """
    max_packed = 0
    for h in range(len(codebooks.k_d_eff_int)):
        for d_eff in (codebooks.k_d_eff_int[h], codebooks.v_d_eff_int[h]):
            packed = d_eff + (head_dim - d_eff + 1) // 2
            max_packed = max(max_packed, packed)
    return max_packed


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_spectral(
    sidecar_path: str,
    spectral_rank: int | None = None,
    spectral_quantize: bool = False,
    b_high: int = 6,
    b_low: int = 4,
    device: str = "cuda",
) -> None:
    """Initialize spectral calibration from sidecar file.

    Called during vLLM model loading when --spectral-calibration is set.

    Args:
        sidecar_path: Path to calibration .pt file.
        spectral_rank: Reduced rank for compressed-cache Phase 2 (old approach).
        spectral_quantize: Enable Phase 2 non-uniform quantization for globals.
        b_high: Bits for semantic regime (default 6).
        b_low: Bits for tail regime (default 4).
        device: Device to load rotation matrices onto.
    """
    global _SPECTRAL_ENABLED, _SPECTRAL_RANK, _SPECTRAL_QUANTIZE
    global _PHASE2_B_HIGH, _PHASE2_B_LOW

    if sidecar_path in _SPECTRAL_REGISTRY:
        logger.info("SpectralQuant already loaded for %s", sidecar_path)
        _SPECTRAL_ENABLED = True
        _SPECTRAL_RANK = spectral_rank
        _SPECTRAL_QUANTIZE = spectral_quantize
        _PHASE2_B_HIGH = b_high
        _PHASE2_B_LOW = b_low
        return

    calibration = SpectralCalibration(sidecar_path, device=device)
    _SPECTRAL_REGISTRY[sidecar_path] = calibration
    _SPECTRAL_ENABLED = True
    _SPECTRAL_RANK = spectral_rank
    _SPECTRAL_QUANTIZE = spectral_quantize
    _PHASE2_B_HIGH = b_high
    _PHASE2_B_LOW = b_low

    if spectral_quantize:
        # Compute codebooks for all global layers
        _init_phase2_codebooks(calibration, b_high, b_low, device)
        logger.info(
            "SpectralQuant Phase 2 enabled: non-uniform quantization, "
            "b_high=%d, b_low=%d, layer_types=%s",
            b_high, b_low, sorted(_COMPRESSED_LAYER_TYPES),
        )
    elif spectral_rank is not None:
        logger.info(
            "SpectralQuant Phase 2 enabled: compressed cache, rank=%d, layer_types=%s",
            spectral_rank,
            sorted(_COMPRESSED_LAYER_TYPES),
        )
    else:
        logger.info("SpectralQuant Phase 1 enabled: rotation only")


def _init_phase2_codebooks(
    calibration: SpectralCalibration,
    b_high: int,
    b_low: int,
    device: str,
) -> None:
    """Compute and register Lloyd-Max codebooks for all Phase 2 layers."""
    _LAYER_CODEBOOKS.clear()
    _PACKED_DIMS.clear()
    _ALLOC_DIMS.clear()
    _PACK_MAPS.clear()
    _UNPACK_MAPS.clear()

    # Optional: skip specific layers (keep them on Phase 1 rotation+fp8).
    # For Eagle3 compatibility, skip layers feeding aux hidden states.
    _skip_raw = os.environ.get("SPECTRAL_SKIP_LAYERS", "")
    skip_layers: set[int] = set()
    if _skip_raw:
        skip_layers = {int(x) for x in _skip_raw.split(",") if x.strip()}
        logger.info("Phase 2 skipping layers %s (keeping Phase 1)", sorted(skip_layers))

    for layer_idx, lc in calibration.layers.items():
        if layer_idx in skip_layers:
            continue
        if lc.layer_type not in _COMPRESSED_LAYER_TYPES:
            continue
        if lc.k_eigenvalues is None:
            logger.warning(
                "Layer %d has no eigenvalues, skipping Phase 2 codebook init",
                layer_idx,
            )
            continue

        codebooks = _compute_layer_codebooks(lc, b_high, b_low, device)
        _LAYER_CODEBOOKS[layer_idx] = codebooks
        logger.debug(
            "Phase 2 codebooks for layer %d: K d_eff=%s, V d_eff=%s",
            layer_idx, codebooks.k_d_eff_int, codebooks.v_d_eff_int,
        )

    # Compute each layer's packed_dim. We also keep the old per-head_dim maxima
    # only for logging/diagnostics; allocation should not use them by default
    # because one outlier layer can otherwise inflate every layer in the group.
    per_hdim_packed: dict[int, int] = {}
    per_layer_packed: dict[int, int] = {}
    for layer_idx, codebooks in _LAYER_CODEBOOKS.items():
        lc = calibration.get_layer(layer_idx)
        if lc is not None:
            packed = _compute_packed_dim(codebooks, lc.head_dim)
            per_layer_packed[layer_idx] = packed
            per_hdim_packed[lc.head_dim] = max(
                per_hdim_packed.get(lc.head_dim, 0), packed,
            )

    for layer_idx in _LAYER_CODEBOOKS:
        if layer_idx in per_layer_packed:
            _PACKED_DIMS[layer_idx] = per_layer_packed[layer_idx]

    # Default to true per-layer allocation. The old padded/shared allocation
    # mode is still available for A/B testing allocator utilization.
    if os.environ.get("SPECTRAL_SHARED_ALLOC", "0") == "1":
        _compute_alloc_dims(calibration)
    else:
        _ALLOC_DIMS.update(_PACKED_DIMS)

    # Precompute vectorized pack/unpack index maps for all layers.
    for layer_idx, codebooks in _LAYER_CODEBOOKS.items():
        lc = calibration.get_layer(layer_idx)
        if lc is None:
            continue
        pd = _PACKED_DIMS[layer_idx]
        dev = codebooks.k_semantic_centroids.device
        _PACK_MAPS[(layer_idx, "k")] = _build_pack_map(
            codebooks.k_d_eff_int, lc.head_dim, pd, dev,
        )
        _PACK_MAPS[(layer_idx, "v")] = _build_pack_map(
            codebooks.v_d_eff_int, lc.head_dim, pd, dev,
        )
        _UNPACK_MAPS[(layer_idx, "k")] = _build_unpack_map(
            codebooks.k_d_eff_int, lc.head_dim, dev,
        )
        _UNPACK_MAPS[(layer_idx, "v")] = _build_unpack_map(
            codebooks.v_d_eff_int, lc.head_dim, dev,
        )

    logger.info(
        "Phase 2 codebooks initialized for %d layers, "
        "packed_dim per layer=%s, alloc_dim per layer=%s "
        "(per-head_dim maxima: %s, shared_alloc=%s)",
        len(_LAYER_CODEBOOKS),
        {k: v for k, v in sorted(per_layer_packed.items())},
        {k: v for k, v in sorted(_ALLOC_DIMS.items())},
        {k: v for k, v in sorted(per_hdim_packed.items())},
        os.environ.get("SPECTRAL_SHARED_ALLOC", "0") == "1",
    )


def _compute_alloc_dims(calibration: "SpectralCalibration") -> None:
    """Compute padded head_sizes so page_sizes are divisible across layer types.

    vLLM's ``unify_kv_cache_spec_page_size`` requires
    ``max_page_size % min_page_size == 0``.  Since
    ``page_size ∝ num_kv_heads * head_size`` (block_size and dtype_size
    cancel in the ratio), we pad ``packed_dim`` to the smallest value
    >= actual packed_dim such that the effective sizes
    ``H * padded_dim`` satisfy divisibility.

    With uniform page_sizes the allocator creates shared tensors
    (one per group_size, shared by num_groups layers) instead of
    60 individual per-layer tensors, giving ~4-6× better block
    utilisation for hybrid models.
    """
    # Group by head_dim -> (num_kv_heads, max packed_dim in that head_dim group)
    groups: dict[int, tuple[int, int]] = {}
    for layer_idx, packed_dim in _PACKED_DIMS.items():
        lc = calibration.get_layer(layer_idx)
        if lc is None:
            continue
        if lc.head_dim not in groups:
            groups[lc.head_dim] = (lc.num_kv_heads, packed_dim)
        else:
            H, group_packed_dim = groups[lc.head_dim]
            groups[lc.head_dim] = (H, max(group_packed_dim, packed_dim))

    if len(groups) <= 1:
        # Single head_dim — page sizes already uniform.
        _ALLOC_DIMS.update(_PACKED_DIMS)
        return

    # Find group with the largest effective_size = H * packed_dim.
    eff_sizes = {hd: H * pd for hd, (H, pd) in groups.items()}
    max_eff = max(eff_sizes.values())

    # For each group, find smallest padded_dim >= packed_dim such that
    # max_eff % (H * padded_dim) == 0.
    padded: dict[int, int] = {}
    success = True
    for hd, (H, pd) in groups.items():
        if eff_sizes[hd] == max_eff:
            padded[hd] = pd  # largest group — no padding
            continue
        if max_eff % H != 0:
            # H doesn't divide max_eff — can't satisfy divisibility.
            success = False
            break
        # Need p | (max_eff / H) and p >= pd.
        target = max_eff // H
        found = False
        for p in range(pd, pd * 4):
            if target % p == 0:
                padded[hd] = p
                found = True
                break
        if not found:
            success = False
            break

    if not success:
        logger.warning(
            "Could not compute uniform alloc_dims for page_size unification; "
            "falling back to packed_dims (per-layer allocation)."
        )
        _ALLOC_DIMS.update(_PACKED_DIMS)
        return

    # Verify: max_eff % (H * padded) == 0 for all groups.
    for hd, (H, _pd) in groups.items():
        if max_eff % (H * padded[hd]) != 0:
            logger.warning(
                "Alloc dim verification failed for head_dim=%d; "
                "falling back to packed_dims.",
                hd,
            )
            _ALLOC_DIMS.update(_PACKED_DIMS)
            return

    # Store per-layer alloc dims.
    for layer_idx in _PACKED_DIMS:
        lc = calibration.get_layer(layer_idx)
        if lc is not None:
            _ALLOC_DIMS[layer_idx] = padded[lc.head_dim]

    logger.info(
        "Phase 2 alloc_dim padding: %s (effective H*dim: %s)",
        {hd: padded[hd] for hd in sorted(padded)},
        {hd: groups[hd][0] * padded[hd] for hd in sorted(padded)},
    )


def init_norm_buffer(max_slots: int, device: str = "cuda") -> None:
    """Pre-allocate the norm side buffer for Phase 2.

    Called from gpu_model_runner after KV cache allocation is known.

    The buffer stores (k_norm, v_norm) per KV head per slot as float16.
    """
    global _NORM_BUFFER, _NORM_BUFFER_LAYER_OFFSETS

    if not _SPECTRAL_QUANTIZE:
        return

    cal = get_calibration()
    if cal is None:
        return

    # Count total Phase 2 KV heads and build offset map
    total_phase2_kv_heads = 0
    _NORM_BUFFER_LAYER_OFFSETS.clear()
    for layer_idx, lc in sorted(cal.layers.items()):
        if lc.layer_type in _COMPRESSED_LAYER_TYPES and layer_idx in _LAYER_CODEBOOKS:
            _NORM_BUFFER_LAYER_OFFSETS[layer_idx] = total_phase2_kv_heads
            total_phase2_kv_heads += lc.num_kv_heads

    if total_phase2_kv_heads == 0:
        return

    # Shape: (max_slots, total_phase2_kv_heads, 2) — last dim is [k_norm, v_norm]
    _NORM_BUFFER = torch.zeros(
        max_slots, total_phase2_kv_heads, 2,
        dtype=torch.float16, device=device,
    )
    logger.info(
        "Phase 2 norm buffer allocated: (%d, %d, 2) = %.1f MB",
        max_slots, total_phase2_kv_heads,
        _NORM_BUFFER.numel() * 2 / (1024 * 1024),
    )


def init_dequant_buffer(
    kv_caches: dict[str, torch.Tensor],
    device: str = "cuda",
) -> None:
    """Pre-allocate compact bf16 dequant buffers for Phase 2 Triton attention.

    Called from gpu_model_runner after KV cache allocation.
    Uses a small fixed-capacity buffer (default 2048 blocks) shared across
    all layers. At runtime, block_table is remapped to index into this
    compact buffer instead of the full cache.
    """
    global _DEQUANT_MAX_BLOCKS

    if not _SPECTRAL_QUANTIZE:
        return

    cal = get_calibration()
    if cal is None:
        return

    # Find unique layer types and their cache geometry.
    seen: dict[str, tuple[int, int, int, int]] = {}
    for layer_name, cache in kv_caches.items():
        layer_idx = _extract_layer_index(layer_name)
        lc = cal.get_layer(layer_idx)
        if lc is None or layer_idx not in _LAYER_CODEBOOKS:
            continue
        lt = lc.layer_type
        if lt in seen:
            continue
        if cache.ndim == 5 and cache.shape[1] == 2:
            num_blocks = cache.shape[0]
            block_size = cache.shape[2]
        else:
            continue
        seen[lt] = (num_blocks, block_size, lc.num_kv_heads, lc.head_dim)

    if not seen:
        return

    max_active = int(os.environ.get("SPECTRAL_DEQUANT_BLOCKS", "2048"))
    _DEQUANT_MAX_BLOCKS = max_active

    _DEQUANT_KEY_BUF.clear()
    _DEQUANT_VAL_BUF.clear()
    _DEQUANT_REMAP.clear()
    _DEQUANT_ACTIVE_MASK.clear()
    _DEQUANT_BLOCK_LIST.clear()
    _DEQUANT_VIEWS.clear()
    total_mem = 0
    for lt, (nb, bs, H, D) in seen.items():
        block_elems = bs * H * D
        _DEQUANT_KEY_BUF[lt] = torch.empty(
            max_active, block_elems, dtype=torch.bfloat16, device=device,
        )
        _DEQUANT_VAL_BUF[lt] = torch.empty(
            max_active, block_elems, dtype=torch.bfloat16, device=device,
        )
        _DEQUANT_REMAP[lt] = torch.empty(nb, dtype=torch.int32, device=device)
        _DEQUANT_ACTIVE_MASK[lt] = torch.zeros(nb, dtype=torch.int32, device=device)
        _DEQUANT_BLOCK_LIST[lt] = torch.full(
            (max_active,), -1, dtype=torch.int64, device=device,
        )
        _DEQUANT_VIEWS[lt] = (bs, H, D)
        total_mem += max_active * block_elems * 2 * 2  # K+V, bf16

    # --- Rotation output buffers for deferred compress ---
    # Pre-allocate (max_batch_tokens, H, D) bf16 + (max_batch_tokens, H) f32 norms.
    # max_batch_tokens is small during decode (1-4 for Eagle3), but we allocate
    # for the max_num_seqs config to handle prefill warmup.
    max_batch_tokens = int(os.environ.get("SPECTRAL_MAX_BATCH_TOKENS", "512"))
    _ROTATE_BUF_K.clear()
    _ROTATE_BUF_V.clear()
    _ROTATE_NORMS_K.clear()
    _ROTATE_NORMS_V.clear()
    for lt, (nb, bs, H, D) in seen.items():
        _ROTATE_BUF_K[lt] = torch.empty(
            max_batch_tokens, H, D, dtype=torch.bfloat16, device=device,
        )
        _ROTATE_BUF_V[lt] = torch.empty(
            max_batch_tokens, H, D, dtype=torch.bfloat16, device=device,
        )
        _ROTATE_NORMS_K[lt] = torch.empty(
            max_batch_tokens, H, dtype=torch.float32, device=device,
        )
        _ROTATE_NORMS_V[lt] = torch.empty(
            max_batch_tokens, H, dtype=torch.float32, device=device,
        )
        total_mem += max_batch_tokens * H * D * 2 * 2  # K+V bf16
        total_mem += max_batch_tokens * H * 4 * 2       # K+V norms f32

    # --- Secondary CUDA stream for async compress ---
    global _COMPRESS_STREAM
    if _ASYNC_COMPRESS:
        _COMPRESS_STREAM = torch.cuda.Stream(device=device)
        logger.info("Phase 2 async compress stream created")

    mem_mb = total_mem / (1024 ** 2)
    logger.info(
        "Phase 2 dequant buffer: max_active=%d K+V=%.0f MB "
        "remap_sizes=%s views=%s rotate_bufs=%d tokens",
        max_active, mem_mb,
        {lt: nb for lt, (nb, _, _, _) in seen.items()},
        dict(_DEQUANT_VIEWS),
        max_batch_tokens,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_calibration() -> SpectralCalibration | None:
    """Get the active spectral calibration (first registered)."""
    if not _SPECTRAL_REGISTRY:
        return None
    return next(iter(_SPECTRAL_REGISTRY.values()))


def _get_layer_config(layer_name: str) -> LayerSpectralConfig | None:
    cal = get_calibration()
    if cal is None:
        return None

    layer_idx = _extract_layer_index(layer_name)
    return cal.get_layer(layer_idx)


@torch.compiler.disable
def is_enabled() -> bool:
    """Check if spectral rotation is active."""
    return _SPECTRAL_ENABLED


def is_truncating() -> bool:
    """Check if Phase 2 compressed-cache mode is active (old rank-based approach)."""
    return _SPECTRAL_ENABLED and _SPECTRAL_RANK is not None


def is_quantizing() -> bool:
    """Check if Phase 2 non-uniform quantization mode is active."""
    return _SPECTRAL_ENABLED and _SPECTRAL_QUANTIZE


def get_spectral_rank() -> int | None:
    """Get the configured spectral rank (None if Phase 1 only)."""
    return _SPECTRAL_RANK


def uses_compressed_cache(layer_name: str) -> bool:
    """Return True when this layer uses the old reduced-rank cache format."""
    if not is_truncating():
        return False

    lc = _get_layer_config(layer_name)
    if lc is None:
        return False
    return lc.layer_type in _COMPRESSED_LAYER_TYPES


@torch.compiler.disable
def is_phase2_quantized(layer_name: str) -> bool:
    """Return True when this layer uses Phase 2 non-uniform quantization."""
    if not _SPECTRAL_QUANTIZE:
        return False

    layer_idx = _extract_layer_index(layer_name)
    return layer_idx in _LAYER_CODEBOOKS


def get_spectral_head_size(layer_name: str) -> int | None:
    """Return the cache width override for layers needing non-standard head_dim.

    Returns padded alloc_dim for Phase 2 quantized layers (enables page_size
    unification for shared-tensor allocation), _SPECTRAL_RANK for old
    compressed-cache layers, or None for layers using default head_dim.
    """
    if uses_compressed_cache(layer_name):
        return _SPECTRAL_RANK
    if _SPECTRAL_QUANTIZE and os.environ.get("SPECTRAL_PACKED_ALLOC", "1") == "1":
        layer_idx = _extract_layer_index(layer_name)
        alloc_dim = _ALLOC_DIMS.get(layer_idx)
        if alloc_dim is not None:
            return alloc_dim
    return None


# ---------------------------------------------------------------------------
# Triton rotation GEMV wrapper
# ---------------------------------------------------------------------------

_USE_TRITON_ROTATE = os.environ.get("SPECTRAL_TRITON_ROTATE", "1") != "0"


def _triton_rotate(
    x: torch.Tensor,
    rotation: torch.Tensor,
    output: torch.Tensor | None = None,
    normalize: bool = False,
    norms_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused Triton GEMV rotation: bf16 in → f32 rotation → bf16 out.

    Args:
        x: (T, H, D_IN) input tensor (any dtype, converted to bf16/f32 in kernel).
        rotation: (H, D_IN, D_OUT) f32 rotation matrix.
        output: Optional pre-allocated (T, H, D_OUT) bf16 output buffer.
                If None, allocated internally.
        normalize: If True, L2-normalize input before rotation and output norms.
        norms_out: Optional pre-allocated (T, H) f32 norms buffer.
                   Required when normalize=True and caller wants the norms.

    Returns:
        (output, norms) where norms is (T, H) f32 if normalize=True, else None.
    """
    from vllm.v1.attention.ops.spectral_rotate_kernel import (
        spectral_rotate_kernel,
    )

    T, H, D_IN = x.shape
    D_OUT = rotation.shape[2]

    if output is None:
        output = torch.empty(T, H, D_OUT, dtype=torch.bfloat16, device=x.device)

    if normalize and norms_out is None:
        norms_out = torch.empty(T, H, dtype=torch.float32, device=x.device)

    # Ensure input is contiguous for clean stride math
    x_c = x.contiguous()
    rot_c = rotation.contiguous()
    out_c = output

    # Dummy norms pointer when not normalizing
    norms_ptr = norms_out if norms_out is not None else output

    TILE_D = 64
    grid = (T, H, (D_OUT + TILE_D - 1) // TILE_D)

    spectral_rotate_kernel[grid](
        x_c,
        rot_c,
        out_c,
        norms_ptr,
        # Input strides
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        # Rotation strides
        rot_c.stride(0), rot_c.stride(1), rot_c.stride(2),
        # Output strides
        out_c.stride(0), out_c.stride(1), out_c.stride(2),
        # Norm strides
        norms_ptr.stride(0) if norms_out is not None else 0,
        norms_ptr.stride(1) if norms_out is not None and norms_ptr.ndim > 1 else 1,
        # Constexpr
        D_IN=D_IN,
        D_OUT=D_OUT,
        NORMALIZE=normalize,
        TILE_D=TILE_D,
    )

    return output, norms_out if normalize else None


# ---------------------------------------------------------------------------
# Phase 1: Rotation functions
# ---------------------------------------------------------------------------

@torch.compiler.disable
def rotate_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate K and V tensors into spectral basis before cache storage.

    Uses Triton fused GEMV (bf16→f32 rotation→bf16) when available,
    falling back to f32 bmm otherwise. In-place via copy_().

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        Rotated key/value tensors with the same shapes as the inputs.
    """
    cal = get_calibration()
    if cal is None:
        return key, value

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return key, value

    if _USE_TRITON_ROTATE:
        k_out, _ = _triton_rotate(key, lc.k_rotation)
        v_out, _ = _triton_rotate(value, lc.v_rotation)
        key.copy_(k_out)
        value.copy_(v_out)
    else:
        k_rot = lc.k_rotation  # (H_kv, D, D) f32
        v_rot = lc.v_rotation  # (H_kv, D, D) f32
        k_f = key.transpose(0, 1).float()
        v_f = value.transpose(0, 1).float()
        key.copy_(torch.bmm(k_f, k_rot).transpose(0, 1))
        value.copy_(torch.bmm(v_f, v_rot).transpose(0, 1))
    return key, value


@torch.compiler.disable
def rotate_q(
    query: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Rotate Q tensor into spectral basis to match cached K rotation.

    For GQA: uses pre-computed expanded rotation matrix (cached).

    Args:
        query: (num_tokens, num_q_heads, head_dim)
        layer_name: vLLM layer identifier

    Returns:
        Rotated query with the same shape as the input.
    """
    cal = get_calibration()
    if cal is None:
        return query

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return query

    q_rot = lc.get_q_rot(query.shape[1])

    if _USE_TRITON_ROTATE:
        q_out, _ = _triton_rotate(query, q_rot)
        query.copy_(q_out)
    else:
        query.copy_(torch.bmm(query.transpose(0, 1).float(), q_rot).transpose(0, 1))
    return query


@torch.compiler.disable
def unrotate_output(
    output: torch.Tensor,
    layer_name: str,
    full_head_dim: int | None = None,
) -> torch.Tensor:
    """
    Rotate attention output back from spectral basis.

    Uses pre-computed f32 inverse rotation (GQA-expanded, cached).

    Args:
        output: (num_tokens, num_q_heads, head_dim)
        layer_name: vLLM layer identifier
        full_head_dim: Unused.

    Returns:
        Unrotated output with the same last dimension as the input.
    """
    cal = get_calibration()
    if cal is None:
        return output

    layer_idx = _extract_layer_index(layer_name)
    lc = cal.get_layer(layer_idx)
    if lc is None:
        return output
    del full_head_dim

    v_unrot = lc.get_v_unrot(output.shape[1])

    if _USE_TRITON_ROTATE:
        o_out, _ = _triton_rotate(output, v_unrot)
        return o_out.to(output.dtype)
    else:
        o_f = torch.bmm(output.transpose(0, 1).float(), v_unrot).transpose(0, 1)
        return o_f.to(output.dtype)


# ---------------------------------------------------------------------------
# Phase 2: Non-uniform quantization (new)
# ---------------------------------------------------------------------------

def _quantize_to_indices(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Nearest-centroid scalar quantization → uint8 indices.

    Args:
        x: (..., dim) float32 values to quantize.
        centroids: (n_levels,) float32 codebook centroids.

    Returns:
        indices: (..., dim) uint8 quantized indices.
    """
    # x: (..., dim), centroids: (n_levels,)
    diffs = x.unsqueeze(-1) - centroids  # (..., dim, n_levels)
    return diffs.abs().argmin(dim=-1).to(torch.uint8)


def _quantize_all_heads(
    data: torch.Tensor,
    sem_centroids: torch.Tensor,
    tail_centroids: torch.Tensor,
    d_effs: list[int],
) -> torch.Tensor:
    """Quantize all heads in one batched call.

    Avoids per-head Python loop by computing nearest centroid for all heads
    and all dims simultaneously, then selecting semantic vs tail result
    based on per-head d_eff.

    Args:
        data: (T, H, D) float32 — rotated, normalized input.
        sem_centroids: (H, n_sem) float32 — semantic codebook per head.
        tail_centroids: (H, n_tail) float32 — tail codebook per head.
        d_effs: list of H ints — per-head semantic/tail split.

    Returns:
        indices: (T, H, D) uint8 — quantized indices for every dim.
    """
    T, H, D = data.shape
    device = data.device

    # Semantic: (T, H, D, 1) - (1, H, 1, n_sem) → argmin over last dim
    sem_idx = (
        (data.unsqueeze(-1) - sem_centroids[None, :, None, :])
        .abs()
        .argmin(dim=-1)
    )  # (T, H, D) int64

    # Tail: same with tail centroids
    tail_idx = (
        (data.unsqueeze(-1) - tail_centroids[None, :, None, :])
        .abs()
        .argmin(dim=-1)
    )  # (T, H, D) int64

    # Per-head mask: dim < d_eff → use semantic, else tail
    d_eff_t = torch.tensor(d_effs, device=device, dtype=torch.long)  # (H,)
    is_sem = torch.arange(D, device=device)[None, :] < d_eff_t[:, None]  # (H, D)

    return torch.where(is_sem[None], sem_idx, tail_idx).to(torch.uint8)


def _dequantize_from_indices(
    indices: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Look up centroids from uint8 indices.

    Args:
        indices: (..., dim) uint8 indices.
        centroids: (n_levels,) float32 codebook centroids.

    Returns:
        values: (..., dim) float32 reconstructed values.
    """
    return centroids[indices.long()]


def _dequantize_all_heads(
    indices: torch.Tensor,
    sem_centroids: torch.Tensor,
    tail_centroids: torch.Tensor,
    d_effs: list[int],
) -> torch.Tensor:
    """Dequantize all heads in one batched call.

    Args:
        indices: (S, H, D) uint8 — quantized indices.
        sem_centroids: (H, n_sem) float32.
        tail_centroids: (H, n_tail) float32.
        d_effs: list of H ints — per-head split.

    Returns:
        values: (S, H, D) float32 — reconstructed values.
    """
    S, H, D = indices.shape
    device = indices.device
    idx_long = indices.long()

    # Semantic lookup: gather from (H, n_sem) using indices
    # sem_centroids[h, idx[s, h, d]] for d < d_eff[h]
    # Clamp indices: stale cache data in padding tokens may have byte values
    # exceeding codebook size (tail dims already clamped below).
    n_sem = sem_centroids.shape[-1]
    sem_vals = torch.gather(
        sem_centroids.unsqueeze(0).expand(S, -1, -1),  # (S, H, n_sem)
        dim=2,
        index=idx_long.clamp(max=n_sem - 1),  # (S, H, D)
    )  # (S, H, D)

    # Tail lookup — clamp indices to tail codebook range since semantic dims
    # may have indices > n_tail (those get masked out by is_sem below)
    n_tail = tail_centroids.shape[-1]
    tail_vals = torch.gather(
        tail_centroids.unsqueeze(0).expand(S, -1, -1),  # (S, H, n_tail)
        dim=2,
        index=idx_long.clamp(max=n_tail - 1),  # (S, H, D)
    )  # (S, H, D)

    # Select: semantic for d < d_eff, tail otherwise
    d_eff_t = torch.tensor(d_effs, device=device, dtype=torch.long)
    is_sem = torch.arange(D, device=device)[None, :] < d_eff_t[:, None]  # (H, D)

    return torch.where(is_sem[None], sem_vals, tail_vals)


# ---------------------------------------------------------------------------
# Fused Triton dequant: unpack + codebook lookup + norm in one kernel
# ---------------------------------------------------------------------------

_USE_TRITON_DEQUANT = os.environ.get("SPECTRAL_TRITON_DEQUANT", "1") != "0"


def _triton_dequant(
    packed_cache_u8: torch.Tensor,
    unique_blocks: torch.Tensor,
    unpack_map: tuple[torch.Tensor, ...],
    codebooks: "LayerCodebooks",
    kv: str,
    head_offset: int,
    output: torch.Tensor,
    block_size: int,
    H: int,
    D: int,
    max_blocks: int = 0,
) -> None:
    """Launch fused Triton kernel: unpack + dequant + norm rescale.

    Args:
        packed_cache_u8: (num_blocks, block_size, H, D_cache) uint8
        unique_blocks: (Nb,) int64 — block IDs to process (may contain -1 sentinels)
        unpack_map: (src, is_sem, is_high) each (H, D)
        codebooks: LayerCodebooks for this layer
        kv: "k" or "v"
        head_offset: offset into norm buffer head dimension
        output: (Nb * block_size, H, D) bfloat16 — written in place
        block_size: tokens per block
        H: number of KV heads
        D: head dimension
        max_blocks: if >0, use fixed grid size for CUDA graph safety
    """
    from vllm.v1.attention.ops.spectral_dequant_kernel import (
        spectral_dequant_kernel,
    )

    src, is_sem, is_high = unpack_map
    Nb = unique_blocks.shape[0]

    if kv == "k":
        sem_c = codebooks.k_semantic_centroids
        tail_c = codebooks.k_tail_centroids
        kv_idx = 0
    else:
        sem_c = codebooks.v_semantic_centroids
        tail_c = codebooks.v_tail_centroids
        kv_idx = 1

    N_SEM = sem_c.shape[-1]
    N_TAIL = tail_c.shape[-1]

    # Cache strides: (num_blocks, block_size, H, D_cache)
    # Note: may be non-contiguous from kv_cache.unbind(1), strides handle it.
    s = packed_cache_u8.stride()
    stride_cache_block = s[0]
    stride_cache_tok = s[1]
    stride_cache_h = s[2]

    # Output strides: treat as (Nb*block_size, H, D) contiguous
    stride_out_tok = H * D
    stride_out_h = D

    # Norm buffer strides
    has_norms = _NORM_BUFFER is not None
    if has_norms:
        ns = _NORM_BUFFER.stride()
        stride_norm_slot = ns[0]
        stride_norm_h = ns[1]
        norm_ptr = _NORM_BUFFER
    else:
        stride_norm_slot = 0
        stride_norm_h = 0
        norm_ptr = output  # dummy, won't be read

    # Convert bool maps to uint8 for Triton (Triton can't load torch.bool)
    is_sem_u8 = is_sem.to(torch.uint8) if is_sem.dtype == torch.bool else is_sem
    is_high_u8 = is_high.to(torch.uint8) if is_high.dtype == torch.bool else is_high

    D_TILE = 256
    # Use fixed grid size when max_blocks is set (CUDA graph safe).
    # Unused slots have block_id == -1 and early-exit in the kernel.
    grid_Nb = max_blocks if max_blocks > 0 else Nb
    grid = (grid_Nb, H, (D + D_TILE - 1) // D_TILE)

    spectral_dequant_kernel[grid](
        packed_cache_u8,
        unique_blocks,
        src,
        is_sem_u8,
        is_high_u8,
        sem_c,
        tail_c,
        norm_ptr,
        output,
        head_offset,
        kv_idx,
        # Cache strides
        stride_cache_block,
        stride_cache_tok,
        stride_cache_h,
        # Norm strides
        stride_norm_slot,
        stride_norm_h,
        # Output strides
        stride_out_tok,
        stride_out_h,
        # Constexpr
        block_size=block_size,
        D=D,
        D_TILE=D_TILE,
        N_SEM=N_SEM,
        N_TAIL=N_TAIL,
        HAS_NORMS=has_norms,
    )


# ---------------------------------------------------------------------------
# Fused Triton compress: quantize + pack + cache-write in one kernel
# ---------------------------------------------------------------------------

_USE_TRITON_COMPRESS = os.environ.get("SPECTRAL_TRITON_COMPRESS", "1") != "0"


def _triton_compress(
    data: torch.Tensor,
    norms: torch.Tensor,
    valid_slots: torch.Tensor,
    pack_maps: tuple[torch.Tensor, ...],
    codebooks: "LayerCodebooks",
    kv: str,
    cache_u8: torch.Tensor,
    layer_idx: int,
    head_offset: int,
) -> None:
    """Launch fused Triton kernel: quantize + pack + cache-write + norm-write.

    Args:
        data: (T, H, D) float32 — rotated, normalized input.
        norms: (T, H) float32 — precomputed L2 norms.
        valid_slots: (T,) int64 — absolute cache slot indices.
        pack_maps: (hi_src, lo_src, is_sem, has_lo, valid) each (H, packed_dim).
        codebooks: LayerCodebooks for this layer.
        kv: "k" or "v".
        cache_u8: uint8 view of KV cache slice, (num_blocks, block_size, H, D_cache).
        layer_idx: layer index (for packed_dim lookup).
        head_offset: offset into norm buffer head dimension.
    """
    from vllm.v1.attention.ops.spectral_compress_kernel import (
        spectral_compress_kernel,
    )

    T = data.shape[0]
    H = data.shape[1]
    packed_dim = _PACKED_DIMS[layer_idx]

    if kv == "k":
        sem_c = codebooks.k_semantic_centroids
        tail_c = codebooks.k_tail_centroids
        kv_idx = 0
    else:
        sem_c = codebooks.v_semantic_centroids
        tail_c = codebooks.v_tail_centroids
        kv_idx = 1

    N_SEM = sem_c.shape[-1]
    N_TAIL = tail_c.shape[-1]

    hi_src, lo_src, is_sem, has_lo, valid = pack_maps
    is_sem_u8 = is_sem.to(torch.uint8) if is_sem.dtype == torch.bool else is_sem
    has_lo_u8 = has_lo.to(torch.uint8) if has_lo.dtype == torch.bool else has_lo
    valid_u8 = valid.to(torch.uint8) if valid.dtype == torch.bool else valid

    # Data strides: (T, H, D) — may be non-contiguous from bmm+transpose
    stride_data_tok = data.stride(0)
    stride_data_h = data.stride(1)

    # Cache strides: (num_blocks, block_size, H, D_cache) — non-contiguous from unbind
    cs = cache_u8.stride()
    stride_cache_block = cs[0]
    stride_cache_tok = cs[1]
    stride_cache_h = cs[2]

    block_size = cache_u8.shape[1]

    # Norm buffer
    has_norms = _NORM_BUFFER is not None and layer_idx in _NORM_BUFFER_LAYER_OFFSETS
    if has_norms:
        norm_buf = _NORM_BUFFER
        ns = norm_buf.stride()
        stride_norm_slot = ns[0]
        stride_norm_h = ns[1]
    else:
        norm_buf = norms  # dummy, won't be written
        stride_norm_slot = 0
        stride_norm_h = 0

    # Ensure norms are contiguous (T, H) for simple indexing
    norms_c = norms.contiguous()

    P_TILE = 256
    grid = (T, H, (packed_dim + P_TILE - 1) // P_TILE)

    spectral_compress_kernel[grid](
        data,
        valid_slots,
        hi_src,
        lo_src,
        is_sem_u8,
        has_lo_u8,
        valid_u8,
        sem_c,
        tail_c,
        norms_c,
        cache_u8,
        norm_buf,
        head_offset,
        kv_idx,
        # Data strides
        stride_data_tok,
        stride_data_h,
        # Cache strides
        stride_cache_block,
        stride_cache_tok,
        stride_cache_h,
        # Norm buffer strides
        stride_norm_slot,
        stride_norm_h,
        # Constexpr
        block_size=block_size,
        packed_dim=packed_dim,
        P_TILE=P_TILE,
        N_SEM=N_SEM,
        N_TAIL=N_TAIL,
        H_KV=H,
        HAS_NORMS=has_norms,
    )


def _pack_indices(
    semantic: torch.Tensor,
    tail: torch.Tensor,
    packed_dim: int,
) -> torch.Tensor:
    """Pack semantic (full byte) + tail (nibble-packed) into packed_dim bytes.

    Args:
        semantic: (T, d_eff) uint8 — 6-bit values stored as full bytes.
        tail: (T, n_tail) uint8 — 4-bit values, two packed per byte.
        packed_dim: Target width (d_eff + ceil(n_tail / 2)).

    Returns:
        packed: (T, packed_dim) uint8.
    """
    d_eff = semantic.shape[-1]
    n_tail = tail.shape[-1]
    T = semantic.shape[0]
    packed = torch.zeros(T, packed_dim, dtype=torch.uint8, device=semantic.device)
    packed[:, :d_eff] = semantic
    # Nibble-pack tail: pairs of 4-bit values into one byte
    n_pairs = n_tail // 2
    if n_pairs > 0:
        packed[:, d_eff:d_eff + n_pairs] = (
            (tail[:, 0::2][:, :n_pairs] << 4) | tail[:, 1::2][:, :n_pairs]
        )
    if n_tail % 2 == 1:  # odd tail dim
        packed[:, d_eff + n_pairs] = tail[:, -1] << 4
    return packed


def _unpack_indices(
    packed: torch.Tensor,
    d_eff: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack packed bytes -> semantic (uint8) + tail (uint8).

    Args:
        packed: (T, packed_dim) uint8.
        d_eff: Number of semantic dims (stored as full bytes).
        head_dim: Original head dimension.

    Returns:
        semantic: (T, d_eff) uint8.
        tail: (T, head_dim - d_eff) uint8.
    """
    semantic = packed[:, :d_eff]
    n_tail = head_dim - d_eff
    n_pairs = n_tail // 2
    n_packed_tail = n_pairs + (1 if n_tail % 2 else 0)
    tail_packed = packed[:, d_eff:d_eff + n_packed_tail]
    tail = torch.empty(
        packed.shape[0], n_tail, dtype=torch.uint8, device=packed.device,
    )
    if n_pairs > 0:
        tail[:, 0::2][:, :n_pairs] = (tail_packed[:, :n_pairs] >> 4) & 0x0F
        tail[:, 1::2][:, :n_pairs] = tail_packed[:, :n_pairs] & 0x0F
    if n_tail % 2 == 1:
        tail[:, -1] = (tail_packed[:, -1] >> 4) & 0x0F
    return semantic, tail


@torch.compiler.disable
def compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    attn_layer: object,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Compress K/V via non-uniform quantization and write bit-packed to cache.

    Phase 2 global layers:
    1. Compute L2 norms, normalize
    2. Rotate into eigenbasis
    3. Split semantic/tail dims
    4. Nearest-centroid quantize each regime → uint8 indices
    5. Bit-pack: semantic as full bytes + tail nibble-packed (2 per byte)
    6. Write packed bytes via kv_cache.view(torch.uint8) (bypasses fp8 encoding)
    7. Store norms in side buffer

    Cache head_dim is packed_dim (~261 for D=512, d_eff~9) instead of full D,
    saving ~49% per global layer head.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        layer_name: vLLM layer identifier
        attn_layer: Attention layer instance (unused in direct write path)
        kv_cache: KV cache tensor, shape (num_blocks, 2, block_size, H, packed_dim)
        slot_mapping: Slot mapping for cache write
    """
    layer_idx = _extract_layer_index(layer_name)
    lc = _get_layer_config(layer_name)
    codebooks = _LAYER_CODEBOOKS.get(layer_idx)
    if lc is None or codebooks is None:
        return

    num_tokens = key.shape[0]
    if num_tokens == 0:
        return

    # Pass all tokens to the kernel — the slot validity guard in the
    # compress kernel handles slot == -1 padding (no host-device sync).
    slots = slot_mapping[:num_tokens].to(torch.long)
    key_valid = key[:num_tokens]    # (T, H, D)
    value_valid = value[:num_tokens]
    T_valid = num_tokens
    valid_slots = slots

    H = lc.num_kv_heads
    D = lc.head_dim  # full head_dim for quantization logic

    # Debug: log shapes for first call per layer
    if not hasattr(compress_kv, "_logged_layers"):
        compress_kv._logged_layers = set()
    if layer_idx not in compress_kv._logged_layers:
        compress_kv._logged_layers.add(layer_idx)
        codebooks_dbg = _LAYER_CODEBOOKS.get(layer_idx)
        logger.info(
            "compress_kv layer %d (%s): key=%s D=%d D_cache=%d packed_dim=%d "
            "kv_cache=%s kv_cache.dtype=%s k_rot=%s d_eff_k=%s d_eff_v=%s",
            layer_idx, lc.layer_type, tuple(key_valid.shape), D,
            kv_cache.shape[-1], _PACKED_DIMS[layer_idx],
            tuple(kv_cache.shape), kv_cache.dtype,
            tuple(lc.k_rotation.shape),
            codebooks_dbg.k_d_eff_int if codebooks_dbg else None,
            codebooks_dbg.v_d_eff_int if codebooks_dbg else None,
        )

    # --- Timing probe (gated by SPECTRAL_PROFILE) ---
    _profiling = os.environ.get("SPECTRAL_PROFILE", "") == "1"
    if _profiling:
        import time as _time
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()

    # 1-2. Fused normalize + rotate: bf16 → f32 norm+normalize → f32 GEMV → bf16
    # Triton kernel fuses 6 ops (float, norm, clamp, divide, bmm_K, bmm_V)
    # into 2 kernel launches (one per K, V).
    if _USE_TRITON_ROTATE:
        k_rotated_bf16, k_norms_flat = _triton_rotate(
            key_valid, lc.k_rotation, normalize=True,
        )
        v_rotated_bf16, v_norms_flat = _triton_rotate(
            value_valid, lc.v_rotation, normalize=True,
        )
        # compress kernels expect f32 data and (T, H) norms
        k_rotated = k_rotated_bf16.float()
        v_rotated = v_rotated_bf16.float()
        k_norms = k_norms_flat.unsqueeze(-1)  # (T, H, 1)
        v_norms = v_norms_flat.unsqueeze(-1)
    else:
        k_float = key_valid.float()
        v_float = value_valid.float()
        k_norms = k_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (T, H, 1)
        v_norms = v_float.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k_normed = k_float / k_norms
        v_normed = v_float / v_norms
        k_rotated = torch.bmm(k_normed.transpose(0, 1), lc.k_rotation).transpose(0, 1)
        v_rotated = torch.bmm(v_normed.transpose(0, 1), lc.v_rotation).transpose(0, 1)

    # 3-6. Quantize + pack + write cache + write norms.
    # Validate cache layout first.
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise NotImplementedError(
            f"Unsupported KV cache layout for Phase 2: {tuple(kv_cache.shape)}"
        )
    if kv_cache.element_size() != 1:
        raise RuntimeError(
            f"Phase 2 requires 1-byte KV cache dtype (fp8), got {kv_cache.dtype} "
            f"({kv_cache.element_size()} bytes). Use --kv-cache-dtype fp8_e4m3"
        )
    kv_uint8 = kv_cache.view(torch.uint8)
    key_cache_u8, value_cache_u8 = kv_uint8.unbind(1)

    k_pm = _PACK_MAPS.get((layer_idx, "k"))
    v_pm = _PACK_MAPS.get((layer_idx, "v"))
    head_offset = _NORM_BUFFER_LAYER_OFFSETS.get(layer_idx, 0)

    if _USE_TRITON_COMPRESS and k_pm is not None:
        # Fused Triton path: quantize + pack + cache-write + norm-write
        _triton_compress(
            k_rotated, k_norms.squeeze(-1), valid_slots, k_pm,
            codebooks, "k", key_cache_u8, layer_idx, head_offset,
        )
        _triton_compress(
            v_rotated, v_norms.squeeze(-1), valid_slots, v_pm,
            codebooks, "v", value_cache_u8, layer_idx, head_offset,
        )
    else:
        # Python fallback: quantize + pack + scattered write
        packed_dim = _PACKED_DIMS[layer_idx]
        D_cache = kv_cache.shape[-1]
        k_slot = torch.zeros(T_valid, H, D_cache, dtype=torch.uint8, device=key.device)
        v_slot = torch.zeros(T_valid, H, D_cache, dtype=torch.uint8, device=key.device)

        if T_valid <= 32 and k_pm is not None:
            k_indices = _quantize_all_heads(
                k_rotated, codebooks.k_semantic_centroids,
                codebooks.k_tail_centroids, codebooks.k_d_eff_int,
            )
            v_indices = _quantize_all_heads(
                v_rotated, codebooks.v_semantic_centroids,
                codebooks.v_tail_centroids, codebooks.v_d_eff_int,
            )
            k_slot[:, :, :packed_dim] = _pack_all_heads(k_indices, *k_pm)
            v_slot[:, :, :packed_dim] = _pack_all_heads(v_indices, *v_pm)
        else:
            for h in range(H):
                d_eff_k = codebooks.k_d_eff_int[h]
                d_eff_v = codebooks.v_d_eff_int[h]
                k_sem_idx = _quantize_to_indices(
                    k_rotated[:, h, :d_eff_k], codebooks.k_semantic_centroids[h],
                )
                k_tail_idx = _quantize_to_indices(
                    k_rotated[:, h, d_eff_k:], codebooks.k_tail_centroids[h],
                )
                k_slot[:, h, :packed_dim] = _pack_indices(k_sem_idx, k_tail_idx, packed_dim)
                v_sem_idx = _quantize_to_indices(
                    v_rotated[:, h, :d_eff_v], codebooks.v_semantic_centroids[h],
                )
                v_tail_idx = _quantize_to_indices(
                    v_rotated[:, h, d_eff_v:], codebooks.v_tail_centroids[h],
                )
                v_slot[:, h, :packed_dim] = _pack_indices(v_sem_idx, v_tail_idx, packed_dim)

        block_size = key_cache_u8.shape[1]
        # Python fallback: filter out slot == -1 padding
        py_valid = valid_slots >= 0
        py_slots = valid_slots[py_valid]
        block_idx = py_slots // block_size
        block_offset = py_slots % block_size
        key_cache_u8[block_idx, block_offset] = k_slot[py_valid]
        value_cache_u8[block_idx, block_offset] = v_slot[py_valid]

        if _NORM_BUFFER is not None and layer_idx in _NORM_BUFFER_LAYER_OFFSETS:
            _NORM_BUFFER[py_slots, head_offset:head_offset + H, 0] = (
                k_norms.squeeze(-1)[py_valid].to(torch.float16)
            )
            _NORM_BUFFER[py_slots, head_offset:head_offset + H, 1] = (
                v_norms.squeeze(-1)[py_valid].to(torch.float16)
            )

    if _profiling:
        torch.cuda.synchronize()
        _t1 = _time.perf_counter()
        # For Triton path, quant+write are fused — report all as "quant"
        _t2 = _t1
        if not hasattr(compress_kv, "_profile_acc"):
            compress_kv._profile_acc = {"quant": 0.0, "write": 0.0, "calls": 0}
        compress_kv._profile_acc["quant"] += _t1 - _t0
        compress_kv._profile_acc["write"] += _t2 - _t1
        compress_kv._profile_acc["calls"] += 1
        c = compress_kv._profile_acc["calls"]
        if c % 600 == 0:  # every 10 decode steps (60 layers each)
            acc = compress_kv._profile_acc
            logger.info(
                "PROFILE compress_kv: %d calls, quant=%.1fms/call write=%.1fms/call "
                "total=%.1fms/call (T=%d H=%d D=%d)",
                c, acc["quant"]/c*1000, acc["write"]/c*1000,
                (acc["quant"]+acc["write"])/c*1000, T_valid, H, D,
            )

    # 7. Round-trip verification (gated by SPECTRAL_VERIFY=1)
    # Note: verification uses k_slot/v_slot which only exist in Python fallback path.
    _triton_compress_active = _USE_TRITON_COMPRESS and k_pm is not None
    if os.environ.get("SPECTRAL_VERIFY", "") == "1" and not _triton_compress_active:
        if not hasattr(compress_kv, "_verified_real"):
            compress_kv._verified_real = set()
        # Only verify once per layer, skip warmup (zero data)
        if layer_idx not in compress_kv._verified_real and k_norms.max().item() > 0.01:
            compress_kv._verified_real.add(layer_idx)
            # Immediate round-trip: unpack + dequantize the packed data
            for h in range(H):
                d_eff_k = codebooks.k_d_eff_int[h]
                d_eff_v = codebooks.v_d_eff_int[h]
                # K round-trip
                k_sem_rt, k_tail_rt = _unpack_indices(k_slot[:, h, :packed_dim], d_eff_k, D)
                k_rt = torch.cat([
                    _dequantize_from_indices(k_sem_rt, codebooks.k_semantic_centroids[h]),
                    _dequantize_from_indices(k_tail_rt, codebooks.k_tail_centroids[h]),
                ], dim=-1)
                k_orig = k_rotated[:, h, :]
                k_cos = torch.nn.functional.cosine_similarity(
                    k_orig.float().reshape(1, -1), k_rt.float().reshape(1, -1)
                ).item()
                # V round-trip
                v_sem_rt, v_tail_rt = _unpack_indices(v_slot[:, h, :packed_dim], d_eff_v, D)
                v_rt = torch.cat([
                    _dequantize_from_indices(v_sem_rt, codebooks.v_semantic_centroids[h]),
                    _dequantize_from_indices(v_tail_rt, codebooks.v_tail_centroids[h]),
                ], dim=-1)
                v_orig = v_rotated[:, h, :]
                v_cos = torch.nn.functional.cosine_similarity(
                    v_orig.float().reshape(1, -1), v_rt.float().reshape(1, -1)
                ).item()
                if h == 0 or k_cos < 0.99 or v_cos < 0.99:
                    logger.info(
                        "VERIFY layer %d h=%d (%s) d_eff_k=%d d_eff_v=%d: "
                        "K cos=%.6f V cos=%.6f k_norm_stats=(%.4f,%.4f,%.4f) "
                        "v_norm_stats=(%.4f,%.4f,%.4f)",
                        layer_idx, h, lc.layer_type, d_eff_k, d_eff_v,
                        k_cos, v_cos,
                        k_norms[:, h].min().item(), k_norms[:, h].mean().item(),
                        k_norms[:, h].max().item(),
                        v_norms[:, h].min().item(), v_norms[:, h].mean().item(),
                        v_norms[:, h].max().item(),
                    )


@torch.compiler.disable
def spectral_attention_phase2(
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    scale: float,
) -> None:
    """Run Phase 2 attention: unpack bit-packed indices, dequantize, attend in rotated basis.

    Works for both global and local layers. Sliding window masking is handled
    by the hybrid manager's SlidingWindowSpec which limits the block table to
    window blocks, so seq_lens already reflects the window.

    Args:
        query: (num_tokens, num_q_heads, head_dim)
        output: (num_tokens, num_q_heads, head_dim) — output buffer
        layer_name: vLLM layer identifier
        kv_cache: KV cache tensor
        attn_metadata: Attention metadata with query_start_loc, seq_lens, block_table
        scale: Attention scale factor (1/sqrt(d))
    """
    if attn_metadata is None:
        output.fill_(0)
        return

    layer_idx = _extract_layer_index(layer_name)
    lc = _get_layer_config(layer_name)
    codebooks = _LAYER_CODEBOOKS.get(layer_idx)
    if lc is None or codebooks is None:
        raise RuntimeError(f"No Phase 2 config for layer {layer_name}")

    if getattr(attn_metadata, "use_cascade", False):
        raise NotImplementedError(
            "Cascade attention is not implemented for SpectralQuant Phase 2."
        )

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if query_start_loc is None or seq_lens is None or block_table is None:
        raise NotImplementedError(
            "SpectralQuant Phase 2 requires Triton-style attention metadata."
        )

    H = lc.num_kv_heads
    D = lc.head_dim  # full head_dim for dequantized output
    num_q_heads = query.shape[1]
    group_size = num_q_heads // H
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))

    packed_dim = _PACKED_DIMS[layer_idx]

    # Debug: log shapes for first call per layer
    if not hasattr(spectral_attention_phase2, "_logged_layers"):
        spectral_attention_phase2._logged_layers = set()
    if layer_idx not in spectral_attention_phase2._logged_layers:
        spectral_attention_phase2._logged_layers.add(layer_idx)
        logger.info(
            "spectral_attn layer %d (%s): query=%s H=%d D=%d group_size=%d "
            "packed_dim=%d kv_cache=%s seq_lens=%s block_table=%s",
            layer_idx, lc.layer_type, tuple(query.shape), H, D, group_size,
            packed_dim, tuple(kv_cache.shape),
            seq_lens[:3].tolist() if seq_lens is not None else None,
            tuple(block_table.shape) if block_table is not None else None,
        )

    _profiling = os.environ.get("SPECTRAL_PROFILE", "") == "1"
    if _profiling:
        import time as _time
        torch.cuda.synchronize()
        _ta0 = _time.perf_counter()

    # Determine cache layout — view as uint8 to read raw packed bytes
    if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
        kv_uint8 = kv_cache.view(torch.uint8)
        key_cache_u8, value_cache_u8 = kv_uint8.unbind(1)  # Each: (num_blocks, block_size, H, D_cache)
        block_size = key_cache_u8.shape[1]
        D_cache = key_cache_u8.shape[-1]  # alloc_dim (>= packed_dim)
    else:
        raise NotImplementedError(
            f"Unsupported KV cache layout for Phase 2: {tuple(kv_cache.shape)}"
        )

    # 1. Rotate Q into eigenbasis
    k_rot = lc.k_rotation  # (H_kv, D, D)
    if group_size == 1:
        q_rotated = torch.bmm(
            query[:num_actual_tokens].transpose(0, 1).float(), k_rot
        ).transpose(0, 1)
    else:
        expanded_rotation = k_rot.repeat_interleave(group_size, dim=0)
        q_rotated = torch.bmm(
            query[:num_actual_tokens].transpose(0, 1).float(), expanded_rotation
        ).transpose(0, 1)

    # Norm buffer offset for this layer
    head_offset = _NORM_BUFFER_LAYER_OFFSETS.get(layer_idx, 0)

    output[:num_actual_tokens].zero_()
    num_seqs = int(query_start_loc.shape[0]) - 1

    for seq_idx in range(num_seqs):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        seq_len = int(seq_lens[seq_idx].item())
        if seq_len <= 0:
            continue

        # 2. Read uint8 bytes from cache (full D bytes, packed data in first packed_dim)
        num_blocks_seq = (seq_len + block_size - 1) // block_size
        blocks = block_table[seq_idx, :num_blocks_seq].to(torch.long)

        # key_cache_u8[blocks]: (num_blocks_seq, block_size, H, D_cache)
        k_raw = key_cache_u8.index_select(0, blocks).reshape(-1, H, D_cache)[:seq_len]
        v_raw = value_cache_u8.index_select(0, blocks).reshape(-1, H, D_cache)[:seq_len]

        # 3. Vectorized unpack + batched dequantize
        k_um = _UNPACK_MAPS.get((layer_idx, "k"))
        v_um = _UNPACK_MAPS.get((layer_idx, "v"))

        if k_um is not None:
            k_full_idx = _unpack_all_heads(k_raw[:, :, :packed_dim], *k_um)
            v_full_idx = _unpack_all_heads(v_raw[:, :, :packed_dim], *v_um)
        else:
            # Fallback: per-head unpack
            k_full_idx = torch.empty(seq_len, H, D, dtype=torch.uint8, device=query.device)
            v_full_idx = torch.empty(seq_len, H, D, dtype=torch.uint8, device=query.device)
            for h in range(H):
                d_eff_k = codebooks.k_d_eff_int[h]
                d_eff_v = codebooks.v_d_eff_int[h]
                k_sem, k_tail = _unpack_indices(k_raw[:, h, :packed_dim], d_eff_k, D)
                k_full_idx[:, h, :d_eff_k] = k_sem
                k_full_idx[:, h, d_eff_k:] = k_tail
                v_sem, v_tail = _unpack_indices(v_raw[:, h, :packed_dim], d_eff_v, D)
                v_full_idx[:, h, :d_eff_v] = v_sem
                v_full_idx[:, h, d_eff_v:] = v_tail

        # Batched dequantize: all heads in one shot
        k_deq = _dequantize_all_heads(
            k_full_idx, codebooks.k_semantic_centroids,
            codebooks.k_tail_centroids, codebooks.k_d_eff_int,
        )
        v_deq = _dequantize_all_heads(
            v_full_idx, codebooks.v_semantic_centroids,
            codebooks.v_tail_centroids, codebooks.v_d_eff_int,
        )

        # 4. Rescale by norms from side buffer
        if _NORM_BUFFER is not None:
            # Reconstruct slot indices: block_id * block_size + offset (vectorized)
            seq_blocks = block_table[seq_idx, :num_blocks_seq].to(torch.long)
            offsets = torch.arange(block_size, device=query.device)
            slot_indices = (
                seq_blocks.unsqueeze(1) * block_size + offsets.unsqueeze(0)
            ).reshape(-1)[:seq_len]

            k_norms = _NORM_BUFFER[slot_indices, head_offset:head_offset + H, 0].float()  # (S, H)
            v_norms = _NORM_BUFFER[slot_indices, head_offset:head_offset + H, 1].float()
            k_deq = k_deq * k_norms.unsqueeze(-1)  # (S, H, D)
            v_deq = v_deq * v_norms.unsqueeze(-1)

        # 5-6. Attention in rotated basis (inner products preserved)
        q_seq = q_rotated[q_start:q_end]  # (q_len, num_q_heads, D)
        q_heads = q_seq.permute(1, 0, 2)  # (num_q_heads, q_len, D)

        k_heads = k_deq.repeat_interleave(group_size, dim=1).permute(1, 0, 2)  # (num_q_heads, S, D)
        v_heads = v_deq.repeat_interleave(group_size, dim=1).permute(1, 0, 2)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
        prefix_len = seq_len - q_len
        q_positions = torch.arange(q_len, device=scores.device).unsqueeze(-1)
        k_positions = torch.arange(seq_len, device=scores.device).unsqueeze(0)
        causal_mask = k_positions <= (prefix_len + q_positions)
        scores.masked_fill_(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out_heads = torch.matmul(probs, v_heads)  # (num_q_heads, q_len, D)

        # 7. Unrotate output: we computed in rotated basis, need to go back
        # out_heads is in V's rotated basis, unrotate by V_v^T
        v_unrot = lc.v_rotation.transpose(-2, -1)  # (H_kv, D, D)
        if group_size == 1:
            out_unrotated = torch.bmm(out_heads, v_unrot)  # (H, q_len, D)
        else:
            expanded_unrot = v_unrot.repeat_interleave(group_size, dim=0)
            out_unrotated = torch.bmm(out_heads, expanded_unrot)

        # 8. In-place copy to output
        out_final = out_unrotated.permute(1, 0, 2).to(output.dtype)
        output[q_start:q_end].copy_(out_final)

    if _profiling:
        torch.cuda.synchronize()
        _ta1 = _time.perf_counter()
        if not hasattr(spectral_attention_phase2, "_profile_acc"):
            spectral_attention_phase2._profile_acc = {"total": 0.0, "calls": 0}
        spectral_attention_phase2._profile_acc["total"] += _ta1 - _ta0
        spectral_attention_phase2._profile_acc["calls"] += 1
        c = spectral_attention_phase2._profile_acc["calls"]
        if c % 600 == 0:
            acc = spectral_attention_phase2._profile_acc
            logger.info(
                "PROFILE spectral_attn: %d calls, %.1fms/call (H=%d D=%d)",
                c, acc["total"]/c*1000, H, D,
            )

        # Verification: log output stats (gated by SPECTRAL_VERIFY=1)
        if os.environ.get("SPECTRAL_VERIFY", "") == "1":
            if not hasattr(spectral_attention_phase2, "_verify_count"):
                spectral_attention_phase2._verify_count = {}
            count = spectral_attention_phase2._verify_count.get(layer_idx, 0)
            spectral_attention_phase2._verify_count[layer_idx] = count + 1
            if count == 1:
                logger.info(
                    "VERIFY_ATTN layer %d (%s): k_deq abs_mean=%.6f v_deq abs_mean=%.6f "
                    "k_norms mean=%.4f v_norms mean=%.4f "
                    "scores mean=%.4f std=%.4f "
                    "out_rotated abs_mean=%.6f out_final abs_mean=%.6f "
                    "out_final std=%.6f scale=%.6f",
                    layer_idx, lc.layer_type,
                    k_deq.abs().mean().item(), v_deq.abs().mean().item(),
                    k_norms.mean().item() if _NORM_BUFFER is not None else -1,
                    v_norms.mean().item() if _NORM_BUFFER is not None else -1,
                    scores[scores > -1e30].mean().item(),
                    scores[scores > -1e30].std().item(),
                    out_heads.abs().mean().item(),
                    out_final.abs().mean().float().item(),
                    out_final.std().float().item(),
                    scale,
                )


@torch.compiler.disable
def spectral_phase2_triton_attention(
    query: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    impl: object,
    key_new: torch.Tensor | None = None,
    value_new: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> None:
    """Phase 2 attention with deferred compress and direct buffer injection.

    New flow (Opt 1+2+3):
    1. Triton rotate+norm new K/V tokens → bf16 rotated + f32 norms
    2. Dequant old blocks from cache (Triton fused kernel)
    3. Inject raw rotated K/V directly into dequant buffer at new-token positions
    4. Triton rotate Q
    5. Run unified_attention in rotated basis
    6. Triton unrotate attention output
    7. Deferred compress: write quantized K/V to cache (sync or async stream)

    This eliminates the lossy compress→dequant roundtrip for new tokens and
    moves compress off the critical path. ~9 kernels on critical path (down
    from 25).

    When key_new/value_new/slot_mapping are None, falls back to the original
    path (dequant-only, compress already done by compress_kv).
    """
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    if attn_metadata is None:
        output.fill_(0)
        return

    layer_idx = _extract_layer_index(layer_name)
    lc = _get_layer_config(layer_name)
    codebooks = _LAYER_CODEBOOKS.get(layer_idx)
    if lc is None or codebooks is None:
        raise RuntimeError(f"No Phase 2 config for layer {layer_name}")

    if getattr(attn_metadata, "use_cascade", False):
        raise NotImplementedError(
            "Cascade attention not supported with Phase 2 Triton path."
        )

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    num_actual_tokens = int(
        getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    )

    if query_start_loc is None or seq_lens is None or block_table is None:
        raise NotImplementedError(
            "Phase 2 Triton path requires Triton-style attention metadata."
        )

    H = lc.num_kv_heads
    D = lc.head_dim
    packed_dim = _PACKED_DIMS[layer_idx]
    device = query.device

    # Whether we're doing deferred compress (new tokens provided or stashed)
    if key_new is None:
        stash = pop_deferred_kv(layer_idx)
        if stash is not None:
            key_new, value_new, slot_mapping = stash
    deferred = (key_new is not None and value_new is not None
                and slot_mapping is not None)

    # Debug: log shapes for first call per layer
    if not hasattr(spectral_phase2_triton_attention, "_logged_layers"):
        spectral_phase2_triton_attention._logged_layers = set()
    if layer_idx not in spectral_phase2_triton_attention._logged_layers:
        spectral_phase2_triton_attention._logged_layers.add(layer_idx)
        logger.info(
            "spectral_triton_attn layer %d (%s): query=%s H=%d D=%d "
            "packed_dim=%d kv_cache=%s seq_lens=%s block_table=%s deferred=%s",
            layer_idx, lc.layer_type, tuple(query.shape), H, D,
            packed_dim, tuple(kv_cache.shape),
            seq_lens[:3].tolist() if seq_lens is not None else None,
            tuple(block_table.shape) if block_table is not None else None,
            deferred,
        )

    # --- Profiling ---
    _profiling = os.environ.get("SPECTRAL_PROFILE", "") == "1"
    if _profiling:
        import time as _time
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()

    # --- Cache layout ---
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise NotImplementedError(
            f"Unsupported cache layout: {tuple(kv_cache.shape)}"
        )
    kv_uint8 = kv_cache.view(torch.uint8)
    key_cache_u8, value_cache_u8 = kv_uint8.unbind(1)
    block_size = key_cache_u8.shape[1]
    D_cache = key_cache_u8.shape[-1]

    # --- Gather unique blocks (CUDA-graph safe: no torch.unique) ---
    num_seqs = query_start_loc.shape[0] - 1
    if num_seqs == 0:
        output[:num_actual_tokens].zero_()
        return

    # --- Unpack maps ---
    k_um = _UNPACK_MAPS.get((layer_idx, "k"))
    v_um = _UNPACK_MAPS.get((layer_idx, "v"))
    if k_um is None or v_um is None:
        raise RuntimeError(f"No unpack maps for layer {layer_idx}")

    # --- Validate compact buffer ---
    lt = lc.layer_type
    if lt not in _DEQUANT_KEY_BUF or lt not in _DEQUANT_VAL_BUF:
        raise RuntimeError(
            f"No dequant buffer for layer type '{lt}'. "
            "Call init_dequant_buffer() first."
        )

    # === Step 0: Rotate+norm new tokens (deferred compress path) ===
    k_rotated_new = None
    v_rotated_new = None
    k_norms_new = None
    v_norms_new = None
    T_new = 0

    if deferred:
        T_new = key_new.shape[0]
        if T_new > 0 and _USE_TRITON_ROTATE:
            # Use pre-allocated rotation buffers
            k_rot_buf = _ROTATE_BUF_K.get(lt)
            v_rot_buf = _ROTATE_BUF_V.get(lt)
            k_norm_buf = _ROTATE_NORMS_K.get(lt)
            v_norm_buf = _ROTATE_NORMS_V.get(lt)

            if k_rot_buf is not None and T_new <= k_rot_buf.shape[0]:
                k_rotated_new, k_norms_new = _triton_rotate(
                    key_new[:T_new], lc.k_rotation,
                    output=k_rot_buf[:T_new],
                    normalize=True,
                    norms_out=k_norm_buf[:T_new],
                )
                v_rotated_new, v_norms_new = _triton_rotate(
                    value_new[:T_new], lc.v_rotation,
                    output=v_rot_buf[:T_new],
                    normalize=True,
                    norms_out=v_norm_buf[:T_new],
                )
            else:
                # Fallback: allocate on the fly
                k_rotated_new, k_norms_new = _triton_rotate(
                    key_new[:T_new], lc.k_rotation, normalize=True,
                )
                v_rotated_new, v_norms_new = _triton_rotate(
                    value_new[:T_new], lc.v_rotation, normalize=True,
                )
        elif T_new > 0:
            # Non-Triton fallback for rotation
            k_f = key_new[:T_new].float()
            v_f = value_new[:T_new].float()
            k_norms_3d = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_norms_3d = v_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k_normed = k_f / k_norms_3d
            v_normed = v_f / v_norms_3d
            k_rotated_new = torch.bmm(
                k_normed.transpose(0, 1), lc.k_rotation
            ).transpose(0, 1).to(torch.bfloat16)
            v_rotated_new = torch.bmm(
                v_normed.transpose(0, 1), lc.v_rotation
            ).transpose(0, 1).to(torch.bfloat16)
            k_norms_new = k_norms_3d.squeeze(-1)
            v_norms_new = v_norms_3d.squeeze(-1)

    # A: Mark active blocks via fixed-shape scatter (replaces torch.unique)
    active_mask = _DEQUANT_ACTIVE_MASK[lc.layer_type]
    compact_idx = _DEQUANT_REMAP[lc.layer_type]
    block_list = _DEQUANT_BLOCK_LIST[lc.layer_type]

    active_mask.zero_()
    flat_bt = block_table[:num_seqs].flatten().clamp(min=0).long()
    active_mask.scatter_(0, flat_bt, 1)

    # B: Cumsum -> compact positions [0..Nb-1]
    torch.cumsum(active_mask, dim=0, out=compact_idx)
    compact_idx.sub_(1)

    # C: Build compact block list (Triton kernel, fixed grid)
    from vllm.v1.attention.ops.spectral_block_list_kernel import (
        build_block_list_kernel,
    )
    block_list.fill_(-1)  # sentinel for unused slots
    N_total_blocks = active_mask.shape[0]
    TILE = 1024
    grid_bl = ((N_total_blocks + TILE - 1) // TILE,)
    build_block_list_kernel[grid_bl](
        active_mask, compact_idx, block_list,
        N_total_blocks, TILE=TILE,
    )

    # Use fixed MAX_BLOCKS for all buffer sizing — avoids host-device sync.
    Nb = _DEQUANT_MAX_BLOCKS

    # D: Remap block_table (fixed shape)
    remapped_bt = torch.zeros_like(block_table)
    remapped_bt[:num_seqs] = compact_idx[
        block_table[:num_seqs].long()
    ].to(block_table.dtype)

    # View compact buffer as (MAX_BLOCKS, block_size, H, D) — fixed shape.
    bs_v, H_v, D_v = _DEQUANT_VIEWS[lt]
    key_buf = _DEQUANT_KEY_BUF[lt][:Nb].reshape(Nb, bs_v, H_v, D_v)
    val_buf = _DEQUANT_VAL_BUF[lt][:Nb].reshape(Nb, bs_v, H_v, D_v)

    head_offset = _NORM_BUFFER_LAYER_OFFSETS.get(layer_idx, 0)

    T_total = Nb * block_size

    # === Step 1: Dequant old blocks from cache ===
    k_dequant = key_buf.reshape(T_total, H_v, D_v)
    v_dequant = val_buf.reshape(T_total, H_v, D_v)

    if _USE_TRITON_DEQUANT:
        _triton_dequant(
            key_cache_u8, block_list, k_um, codebooks, "k",
            head_offset, k_dequant, block_size, H, D,
            max_blocks=_DEQUANT_MAX_BLOCKS,
        )
        _triton_dequant(
            value_cache_u8, block_list, v_um, codebooks, "v",
            head_offset, v_dequant, block_size, H, D,
            max_blocks=_DEQUANT_MAX_BLOCKS,
        )
    else:
        Nb_actual = int(active_mask.sum().item())
        active_block_ids = block_list[:Nb_actual]
        T_actual = Nb_actual * block_size
        k_raw = key_cache_u8[active_block_ids].reshape(-1, H, D_cache)
        v_raw = value_cache_u8[active_block_ids].reshape(-1, H, D_cache)

        k_indices = _unpack_all_heads(k_raw[:, :, :packed_dim], *k_um)
        v_indices = _unpack_all_heads(v_raw[:, :, :packed_dim], *v_um)

        if _NORM_BUFFER is not None:
            offsets_within_block = torch.arange(block_size, device=device)
            slot_indices = (
                active_block_ids.unsqueeze(1) * block_size
                + offsets_within_block.unsqueeze(0)
            ).reshape(-1)

        chunk_tokens = 32
        for start in range(0, T_actual, chunk_tokens):
            end = min(start + chunk_tokens, T_actual)
            k_deq = _dequantize_all_heads(
                k_indices[start:end],
                codebooks.k_semantic_centroids,
                codebooks.k_tail_centroids,
                codebooks.k_d_eff_int,
            )
            v_deq = _dequantize_all_heads(
                v_indices[start:end],
                codebooks.v_semantic_centroids,
                codebooks.v_tail_centroids,
                codebooks.v_d_eff_int,
            )
            if _NORM_BUFFER is not None:
                chunk_slots = slot_indices[start:end]
                k_norms = _NORM_BUFFER[
                    chunk_slots, head_offset : head_offset + H, 0
                ].float()
                v_norms = _NORM_BUFFER[
                    chunk_slots, head_offset : head_offset + H, 1
                ].float()
                k_deq = k_deq * k_norms.unsqueeze(-1)
                v_deq = v_deq * v_norms.unsqueeze(-1)
            k_dequant[start:end] = k_deq.to(torch.bfloat16)
            v_dequant[start:end] = v_deq.to(torch.bfloat16)

    # === Step 2: Inject new tokens directly into dequant buffer ===
    # Skip the lossy compress→dequant roundtrip for new tokens.
    # The raw rotated (norm-scaled) K/V is injected at the buffer positions
    # corresponding to the new tokens' cache slots.
    if deferred and T_new > 0 and k_rotated_new is not None:
        valid_slots = slot_mapping[:T_new].long()
        for t in range(T_new):
            slot = valid_slots[t]
            if slot < 0:
                continue
            block_idx = slot // block_size
            block_off = slot % block_size
            compact_pos = compact_idx[block_idx]
            buf_pos = compact_pos * block_size + block_off
            # Inject norm-scaled rotated K/V (multiply back by norm)
            k_dequant[buf_pos] = (
                k_rotated_new[t].float() * k_norms_new[t].unsqueeze(-1)
            ).to(torch.bfloat16)
            v_dequant[buf_pos] = (
                v_rotated_new[t].float() * v_norms_new[t].unsqueeze(-1)
            ).to(torch.bfloat16)

    # === Step 3: Rotate Q (Triton or bmm) ===
    num_q_heads = query.shape[1]
    group_size = num_q_heads // H
    q_slice = query[:num_actual_tokens]
    attn_dtype = key_buf.dtype

    if _USE_TRITON_ROTATE:
        q_rot_mat = lc.get_q_rot(num_q_heads)
        q_rot, _ = _triton_rotate(q_slice, q_rot_mat)
        q_rot = q_rot.to(attn_dtype)
    else:
        k_rot_mat = lc.k_rotation
        if group_size == 1:
            q_rot = torch.bmm(
                q_slice.transpose(0, 1).float(), k_rot_mat,
            ).transpose(0, 1).to(attn_dtype)
        else:
            expanded_rot = k_rot_mat.repeat_interleave(group_size, dim=0)
            q_rot = torch.bmm(
                q_slice.transpose(0, 1).float(), expanded_rot,
            ).transpose(0, 1).to(attn_dtype)

    if _profiling:
        torch.cuda.synchronize()
        _t1 = _time.perf_counter()

    # === Step 4: Triton paged attention in rotated basis ===
    attn_out = torch.empty(
        output[:num_actual_tokens].shape,
        dtype=attn_dtype,
        device=output.device,
    )
    unified_attention(
        q=q_rot,
        k=key_buf,
        v=val_buf,
        out=attn_out,
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=attn_metadata.seq_lens,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        window_size=impl.sliding_window,
        block_table=remapped_bt,
        softcap=impl.logits_soft_cap,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        alibi_slopes=impl.alibi_slopes,
        use_alibi_sqrt=impl.use_alibi_sqrt,
        sinks=impl.sinks,
        seq_threshold_3D=getattr(attn_metadata, "seq_threshold_3D", None),
        num_par_softmax_segments=getattr(
            attn_metadata, "num_par_softmax_segments", None
        ),
        softmax_segm_output=getattr(
            attn_metadata, "softmax_segm_output", None
        ),
        softmax_segm_max=getattr(attn_metadata, "softmax_segm_max", None),
        softmax_segm_expsum=getattr(
            attn_metadata, "softmax_segm_expsum", None
        ),
        mm_prefix_range=getattr(
            attn_metadata, "mm_prefix_range_tensor", None
        ),
    )

    # === Step 5: Unrotate attention output (Triton or bmm) ===
    v_unrot = lc.get_v_unrot(num_q_heads)
    if _USE_TRITON_ROTATE:
        out_unrot, _ = _triton_rotate(attn_out, v_unrot)
        output[:num_actual_tokens].copy_(out_unrot.to(output.dtype))
    else:
        if group_size == 1:
            out_unrot = torch.bmm(
                attn_out.transpose(0, 1).float(), v_unrot,
            ).transpose(0, 1).to(output.dtype)
        else:
            expanded_v_unrot = v_unrot.repeat_interleave(group_size, dim=0)
            out_unrot = torch.bmm(
                attn_out.transpose(0, 1).float(), expanded_v_unrot,
            ).transpose(0, 1).to(output.dtype)
        output[:num_actual_tokens].copy_(out_unrot)

    # === Step 6: Deferred compress (sync or async) ===
    if deferred and T_new > 0 and k_rotated_new is not None:
        valid_slots = slot_mapping[:T_new].long()

        k_pm = _PACK_MAPS.get((layer_idx, "k"))
        v_pm = _PACK_MAPS.get((layer_idx, "v"))

        def _do_compress():
            if _USE_TRITON_COMPRESS and k_pm is not None:
                _triton_compress(
                    k_rotated_new[:T_new].float(),
                    k_norms_new[:T_new],
                    valid_slots, k_pm,
                    codebooks, "k", key_cache_u8, layer_idx, head_offset,
                )
                _triton_compress(
                    v_rotated_new[:T_new].float(),
                    v_norms_new[:T_new],
                    valid_slots, v_pm,
                    codebooks, "v", value_cache_u8, layer_idx, head_offset,
                )
            else:
                # Fallback: call compress_kv directly on pre-rotated data
                _compress_rotated_kv(
                    k_rotated_new[:T_new].float(),
                    v_rotated_new[:T_new].float(),
                    k_norms_new[:T_new],
                    v_norms_new[:T_new],
                    valid_slots,
                    layer_idx, lc, codebooks,
                    key_cache_u8, value_cache_u8,
                    head_offset,
                )

        if _COMPRESS_STREAM is not None:
            # Async: record event on main stream, compress on secondary
            main_done = torch.cuda.current_stream(device).record_event()
            with torch.cuda.stream(_COMPRESS_STREAM):
                _COMPRESS_STREAM.wait_event(main_done)
                _do_compress()
        else:
            # Sync: compress on main stream
            _do_compress()

    if _profiling:
        torch.cuda.synchronize()
        _t2 = _time.perf_counter()
        if not hasattr(spectral_phase2_triton_attention, "_profile_acc"):
            spectral_phase2_triton_attention._profile_acc = {
                "dequant": 0.0,
                "triton": 0.0,
                "calls": 0,
            }
        acc = spectral_phase2_triton_attention._profile_acc
        acc["dequant"] += _t1 - _t0
        acc["triton"] += _t2 - _t1
        acc["calls"] += 1
        c = acc["calls"]
        if c % 600 == 0:
            logger.info(
                "PROFILE spectral_triton_attn: %d calls, dequant=%.1fms/call "
                "triton=%.1fms/call total=%.1fms/call (H=%d D=%d blocks=%d "
                "deferred=%s async=%s)",
                c,
                acc["dequant"] / c * 1000,
                acc["triton"] / c * 1000,
                (acc["dequant"] + acc["triton"]) / c * 1000,
                H,
                D,
                Nb,
                deferred,
                _COMPRESS_STREAM is not None,
            )


def _compress_rotated_kv(
    k_rotated: torch.Tensor,
    v_rotated: torch.Tensor,
    k_norms: torch.Tensor,
    v_norms: torch.Tensor,
    valid_slots: torch.Tensor,
    layer_idx: int,
    lc: LayerSpectralConfig,
    codebooks: LayerCodebooks,
    key_cache_u8: torch.Tensor,
    value_cache_u8: torch.Tensor,
    head_offset: int,
) -> None:
    """Python fallback for deferred compress: quantize + pack + write cache.

    Takes pre-rotated, pre-normalized data and writes packed quantized bytes
    to the KV cache. Used when Triton compress kernel is unavailable.
    """
    T = k_rotated.shape[0]
    H = lc.num_kv_heads
    D = lc.head_dim
    packed_dim = _PACKED_DIMS[layer_idx]
    D_cache = key_cache_u8.shape[-1]

    k_pm = _PACK_MAPS.get((layer_idx, "k"))
    v_pm = _PACK_MAPS.get((layer_idx, "v"))

    k_slot = torch.zeros(T, H, D_cache, dtype=torch.uint8, device=k_rotated.device)
    v_slot = torch.zeros(T, H, D_cache, dtype=torch.uint8, device=k_rotated.device)

    if T <= 32 and k_pm is not None:
        k_indices = _quantize_all_heads(
            k_rotated, codebooks.k_semantic_centroids,
            codebooks.k_tail_centroids, codebooks.k_d_eff_int,
        )
        v_indices = _quantize_all_heads(
            v_rotated, codebooks.v_semantic_centroids,
            codebooks.v_tail_centroids, codebooks.v_d_eff_int,
        )
        k_slot[:, :, :packed_dim] = _pack_all_heads(k_indices, *k_pm)
        v_slot[:, :, :packed_dim] = _pack_all_heads(v_indices, *v_pm)
    else:
        for h in range(H):
            d_eff_k = codebooks.k_d_eff_int[h]
            d_eff_v = codebooks.v_d_eff_int[h]
            k_sem_idx = _quantize_to_indices(
                k_rotated[:, h, :d_eff_k], codebooks.k_semantic_centroids[h],
            )
            k_tail_idx = _quantize_to_indices(
                k_rotated[:, h, d_eff_k:], codebooks.k_tail_centroids[h],
            )
            k_slot[:, h, :packed_dim] = _pack_indices(k_sem_idx, k_tail_idx, packed_dim)
            v_sem_idx = _quantize_to_indices(
                v_rotated[:, h, :d_eff_v], codebooks.v_semantic_centroids[h],
            )
            v_tail_idx = _quantize_to_indices(
                v_rotated[:, h, d_eff_v:], codebooks.v_tail_centroids[h],
            )
            v_slot[:, h, :packed_dim] = _pack_indices(v_sem_idx, v_tail_idx, packed_dim)

    block_size = key_cache_u8.shape[1]
    py_valid = valid_slots >= 0
    py_slots = valid_slots[py_valid]
    block_idx = py_slots // block_size
    block_offset = py_slots % block_size
    key_cache_u8[block_idx, block_offset] = k_slot[py_valid]
    value_cache_u8[block_idx, block_offset] = v_slot[py_valid]

    if _NORM_BUFFER is not None and layer_idx in _NORM_BUFFER_LAYER_OFFSETS:
        _NORM_BUFFER[py_slots, head_offset:head_offset + H, 0] = (
            k_norms[py_valid].to(torch.float16)
        )
        _NORM_BUFFER[py_slots, head_offset:head_offset + H, 1] = (
            v_norms[py_valid].to(torch.float16)
        )


# ---------------------------------------------------------------------------
# Old Phase 2: Compressed-cache functions (kept for backward compat)
# ---------------------------------------------------------------------------

@torch.compiler.disable
def store_compressed_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_name: str,
) -> None:
    """Store reduced-rank spectral K/V coordinates into the KV cache."""
    lc = _get_layer_config(layer_name)
    if lc is None or _SPECTRAL_RANK is None:
        return

    if kv_cache.dtype == torch.uint8:
        raise NotImplementedError(
            "Compressed SpectralQuant cache currently requires non-quantized "
            "KV cache storage."
        )
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise NotImplementedError(
            f"Unsupported KV cache layout for compressed SpectralQuant: "
            f"{tuple(kv_cache.shape)}"
        )

    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return

    slots = slot_mapping[:num_tokens].to(torch.long)
    valid_mask = slots >= 0
    if not bool(valid_mask.any()):
        return

    slots = slots[valid_mask]
    key = key[:num_tokens][valid_mask]
    value = value[:num_tokens][valid_mask]

    rank = min(_SPECTRAL_RANK, lc.head_dim)
    key_basis = lc.k_rotation[:, :, :rank]
    value_basis = lc.v_rotation[:, :, :rank]
    key_comp = torch.bmm(key.transpose(0, 1).float(), key_basis).transpose(0, 1)
    value_comp = torch.bmm(value.transpose(0, 1).float(), value_basis).transpose(0, 1)

    key_cache, value_cache = kv_cache.unbind(1)
    block_size = key_cache.shape[1]
    if key_cache.shape[-1] != rank or value_cache.shape[-1] != rank:
        raise RuntimeError(
            "Compressed SpectralQuant cache rank does not match allocated "
            f"KV cache shape for {layer_name}: rank={rank}, "
            f"cache={tuple(kv_cache.shape)}"
        )

    block_idx = slots // block_size
    block_offset = slots % block_size
    key_cache[block_idx, block_offset] = key_comp.to(key_cache.dtype)
    value_cache[block_idx, block_offset] = value_comp.to(value_cache.dtype)


def _reconstruct_from_spectral_cache(
    cache_rows: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Expand reduced-rank spectral rows back to full head width."""
    return torch.bmm(
        cache_rows.transpose(0, 1).float(),
        basis.transpose(-2, -1),
    ).transpose(0, 1)


@torch.compiler.disable
def compressed_attention(
    query: torch.Tensor,
    output: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    layer_name: str,
    scale: float,
) -> None:
    """Run correctness-first attention from reduced-rank cached K/V."""
    if attn_metadata is None:
        output.fill_(0)
        return
    if kv_cache.dtype == torch.uint8:
        raise NotImplementedError(
            "Compressed SpectralQuant cache currently requires non-quantized "
            "KV cache storage."
        )
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise NotImplementedError(
            f"Unsupported KV cache layout for compressed SpectralQuant: "
            f"{tuple(kv_cache.shape)}"
        )
    if getattr(attn_metadata, "use_cascade", False):
        raise NotImplementedError(
            "Cascade attention is not implemented for compressed SpectralQuant."
        )

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if query_start_loc is None or seq_lens is None or block_table is None:
        raise NotImplementedError(
            "Compressed SpectralQuant currently requires Triton-style "
            "attention metadata."
        )

    lc = _get_layer_config(layer_name)
    if lc is None or _SPECTRAL_RANK is None:
        raise RuntimeError(f"No spectral config found for layer {layer_name}")

    rank = min(_SPECTRAL_RANK, lc.head_dim)
    key_basis = lc.k_rotation[:, :, :rank]
    value_basis = lc.v_rotation[:, :, :rank]
    key_cache, value_cache = kv_cache.unbind(1)
    block_size = key_cache.shape[1]
    num_q_heads = query.shape[1]
    group_size = num_q_heads // lc.num_kv_heads
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))

    if key_cache.shape[-1] != rank or value_cache.shape[-1] != rank:
        raise RuntimeError(
            "Compressed SpectralQuant cache rank does not match allocated "
            f"KV cache shape for {layer_name}: rank={rank}, "
            f"cache={tuple(kv_cache.shape)}"
        )

    output[:num_actual_tokens].zero_()
    num_seqs = int(query_start_loc.shape[0]) - 1

    for seq_idx in range(num_seqs):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        seq_len = int(seq_lens[seq_idx].item())
        if seq_len <= 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = block_table[seq_idx, :num_blocks].to(torch.long)
        key_comp = key_cache.index_select(0, blocks).reshape(-1, lc.num_kv_heads, rank)
        value_comp = value_cache.index_select(0, blocks).reshape(
            -1, lc.num_kv_heads, rank
        )
        key_full = _reconstruct_from_spectral_cache(key_comp[:seq_len], key_basis)
        value_full = _reconstruct_from_spectral_cache(
            value_comp[:seq_len], value_basis
        )

        q_seq = query[q_start:q_end].float()
        q_heads = q_seq.permute(1, 0, 2)
        key_heads = key_full.repeat_interleave(group_size, dim=1).permute(1, 0, 2)
        value_heads = value_full.repeat_interleave(group_size, dim=1).permute(
            1, 0, 2
        )

        scores = torch.matmul(q_heads, key_heads.transpose(-2, -1)) * scale
        prefix_len = seq_len - q_len
        q_positions = torch.arange(q_len, device=scores.device).unsqueeze(-1)
        k_positions = torch.arange(seq_len, device=scores.device).unsqueeze(0)
        causal_mask = k_positions <= (prefix_len + q_positions)
        scores.masked_fill_(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out_heads = torch.matmul(probs.to(value_heads.dtype), value_heads)
        output[q_start:q_end].copy_(out_heads.permute(1, 0, 2).to(output.dtype))
