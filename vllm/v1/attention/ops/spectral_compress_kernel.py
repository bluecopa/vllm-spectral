"""Fused Triton kernel: quantize + pack + cache-write for SpectralQuant compress.

Mirror of spectral_dequant_kernel.py (read path). Replaces the Python
quantize_all_heads + pack_all_heads + scattered cache write with a single
kernel launch per K or V.

Grid: (T, H, ceil(packed_dim / P_TILE))
  - T:      tokens to compress (1 during decode, batch during prefill)
  - H:      KV heads
  - P_TILE: packed dimension tile (256)

Per program:
  1. Load pack maps (hi_src, lo_src, is_sem, has_lo, valid)
  2. Gather normalized-rotated float values at source positions
  3. Nearest-centroid quantize: argmin over N_SEM=64 or N_TAIL=16 centroids
  4. Pack nibbles: semantic full byte, tail pair (hi<<4)|lo
  5. Scattered write to KV cache via slot_mapping
  6. Write norm to side buffer (first tile only)
"""

import triton
import triton.language as tl


@triton.jit
def spectral_compress_kernel(
    # Input: rotated, normalized data
    data_ptr,               # (T, H, D) float32
    # Slot mapping
    slot_mapping_ptr,       # (T,) int64 — valid slots only
    # Pack maps (precomputed, per head)
    pack_hi_src_ptr,        # (H, packed_dim) int64
    pack_lo_src_ptr,        # (H, packed_dim) int64
    pack_is_sem_ptr,        # (H, packed_dim) uint8 (bool stored as uint8)
    pack_has_lo_ptr,        # (H, packed_dim) uint8
    pack_valid_ptr,         # (H, packed_dim) uint8
    # Codebooks
    sem_centroids_ptr,      # (H, N_SEM) float32
    tail_centroids_ptr,     # (H, N_TAIL) float32
    # Norms (precomputed L2 norms)
    norms_ptr,              # (T, H) float32
    # Cache output
    cache_ptr,              # uint8 view: (num_blocks, block_size, H, D_cache)
    # Norm buffer output
    norm_buffer_ptr,        # (max_slots, total_heads, 2) float16
    # Scalar params
    head_offset,            # int: offset into norm_buffer head dim
    kv_idx,                 # int: 0=key, 1=value
    # Data strides
    stride_data_tok,
    stride_data_h,
    # Cache strides (may be non-contiguous from unbind)
    stride_cache_block,
    stride_cache_tok,
    stride_cache_h,
    # Norm buffer strides
    stride_norm_slot,
    stride_norm_h,
    # Constexpr dimensions
    block_size: tl.constexpr,
    packed_dim: tl.constexpr,
    P_TILE: tl.constexpr,
    N_SEM: tl.constexpr,       # 64
    N_TAIL: tl.constexpr,      # 16
    H_KV: tl.constexpr,        # for norm indexing (T, H) layout
    HAS_NORMS: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    p_tile_idx = tl.program_id(2)

    # Packed position offsets for this tile
    p_start = p_tile_idx * P_TILE
    p_offs = p_start + tl.arange(0, P_TILE)
    p_mask = p_offs < packed_dim

    # Load pack maps for this head + tile
    map_base = head_idx * packed_dim + p_offs
    hi_src = tl.load(pack_hi_src_ptr + map_base, mask=p_mask, other=0).to(tl.int64)
    lo_src = tl.load(pack_lo_src_ptr + map_base, mask=p_mask, other=0).to(tl.int64)
    is_sem = tl.load(pack_is_sem_ptr + map_base, mask=p_mask, other=0) != 0
    has_lo = tl.load(pack_has_lo_ptr + map_base, mask=p_mask, other=0) != 0
    valid = tl.load(pack_valid_ptr + map_base, mask=p_mask, other=0) != 0

    # Gather float values at source dimension positions
    data_base = tok_idx * stride_data_tok + head_idx * stride_data_h
    hi_val = tl.load(data_ptr + data_base + hi_src, mask=p_mask & valid, other=0.0)
    lo_val = tl.load(data_ptr + data_base + lo_src, mask=p_mask & has_lo, other=0.0)

    # --- Semantic quantization: argmin |hi_val - centroid| over N_SEM ---
    sem_base = head_idx * N_SEM
    sem_best_idx = tl.zeros([P_TILE], dtype=tl.int32)
    sem_best_dist = tl.full([P_TILE], float('inf'), dtype=tl.float32)
    for c in range(N_SEM):
        cent = tl.load(sem_centroids_ptr + sem_base + c)
        dist = tl.abs(hi_val - cent)
        better = dist < sem_best_dist
        sem_best_dist = tl.where(better, dist, sem_best_dist)
        sem_best_idx = tl.where(better, c, sem_best_idx)

    # --- Tail quantization for hi value: argmin over N_TAIL ---
    tail_base = head_idx * N_TAIL
    tail_hi_idx = tl.zeros([P_TILE], dtype=tl.int32)
    tail_hi_dist = tl.full([P_TILE], float('inf'), dtype=tl.float32)
    for c in range(N_TAIL):
        cent = tl.load(tail_centroids_ptr + tail_base + c)
        dist = tl.abs(hi_val - cent)
        better = dist < tail_hi_dist
        tail_hi_dist = tl.where(better, dist, tail_hi_dist)
        tail_hi_idx = tl.where(better, c, tail_hi_idx)

    # --- Tail quantization for lo value: argmin over N_TAIL ---
    tail_lo_idx = tl.zeros([P_TILE], dtype=tl.int32)
    tail_lo_dist = tl.full([P_TILE], float('inf'), dtype=tl.float32)
    for c in range(N_TAIL):
        cent = tl.load(tail_centroids_ptr + tail_base + c)
        dist = tl.abs(lo_val - cent)
        better = dist < tail_lo_dist
        tail_lo_dist = tl.where(better, dist, tail_lo_dist)
        tail_lo_idx = tl.where(better, c, tail_lo_idx)

    # --- Pack: semantic → full byte, tail → (hi<<4)|lo ---
    hi_idx = tl.where(is_sem, sem_best_idx, tail_hi_idx)
    lo_idx = tl.where(has_lo, tail_lo_idx, tl.zeros([P_TILE], dtype=tl.int32))
    packed_byte = tl.where(is_sem, hi_idx, (hi_idx << 4) | lo_idx)
    packed_byte = tl.where(valid, packed_byte, tl.zeros([P_TILE], dtype=tl.int32))

    # --- Write packed bytes to cache ---
    slot = tl.load(slot_mapping_ptr + tok_idx).to(tl.int64)
    slot_valid = slot >= 0
    block_idx = slot // block_size
    block_offset = slot % block_size
    cache_off = (block_idx * stride_cache_block
                 + block_offset * stride_cache_tok
                 + head_idx * stride_cache_h
                 + p_offs)
    tl.store(cache_ptr + cache_off, packed_byte.to(tl.uint8),
             mask=p_mask & valid & slot_valid)

    # --- Write norm to side buffer (first tile only) ---
    if HAS_NORMS:
        if p_tile_idx == 0:
            if slot_valid:
                norm_val = tl.load(norms_ptr + tok_idx * H_KV + head_idx)
                norm_off = (slot * stride_norm_slot
                           + (head_offset + head_idx) * stride_norm_h
                           + kv_idx)
                tl.store(norm_buffer_ptr + norm_off, norm_val.to(tl.float16))
