"""Fused Triton kernel: unpack + dequant + norm rescale for SpectralQuant.

Replaces the Python loop of gather/shift/mask/gather/mul with a single
kernel launch per K or V. The D×D unrotation bmm stays separate (cuBLAS).

Grid: (Nb, H, ceil(D / D_TILE))
  - Nb: unique cache blocks touched this step
  - H:  KV heads
  - D_TILE: dimension tile (256)

Inner loop over block_size tokens:
  1. Scattered gather of packed bytes via precomputed src map
  2. Nibble extraction with is_sem / is_high masks
  3. Codebook lookup (L1-resident after first access)
  4. fp16 norm scalar multiply
  5. bf16 store
"""

import triton
import triton.language as tl


@triton.jit
def spectral_dequant_kernel(
    # Packed cache (read via unique_blocks indirection)
    packed_cache_ptr,       # (total_blocks, block_size, H, D_cache) uint8
    unique_blocks_ptr,      # (Nb,) int64
    # Unpack maps (precomputed, loaded once per program)
    unpack_src_ptr,         # (H, D) int64
    unpack_is_sem_ptr,      # (H, D) bool  (stored as uint8)
    unpack_is_high_ptr,     # (H, D) bool  (stored as uint8)
    # Codebooks
    sem_centroids_ptr,      # (H, N_SEM) float32
    tail_centroids_ptr,     # (H, N_TAIL) float32
    # Norm buffer
    norm_buffer_ptr,        # (max_slots, total_heads, 2) float16
    # Output
    output_ptr,             # (Nb * block_size, H, D) bfloat16
    # Scalar params
    head_offset,            # int: offset into norm_buffer head dim
    kv_idx,                 # int: 0=key, 1=value
    # Strides for packed cache: [block, token, head]
    stride_cache_block,
    stride_cache_tok,
    stride_cache_h,
    # Strides for norm buffer: [slot, head]
    stride_norm_slot,
    stride_norm_h,
    # Strides for output: [token, head]
    stride_out_tok,
    stride_out_h,
    # Constexpr dimensions
    block_size: tl.constexpr,
    D: tl.constexpr,
    D_TILE: tl.constexpr,
    N_SEM: tl.constexpr,      # 64
    N_TAIL: tl.constexpr,     # 16
    HAS_NORMS: tl.constexpr,  # whether norm_buffer_ptr is valid
):
    block_prog = tl.program_id(0)
    head_idx = tl.program_id(1)
    d_tile_idx = tl.program_id(2)

    # Resolve actual block ID via indirection
    block_id = tl.load(unique_blocks_ptr + block_prog).to(tl.int64)

    # Sentinel check: unused slots in fixed-size grid have block_id == -1
    if block_id < 0:
        return

    # Dimension offsets for this tile
    d_start = d_tile_idx * D_TILE
    d_offs = d_start + tl.arange(0, D_TILE)
    d_mask = d_offs < D

    # Load unpack maps for this head + tile (reused for all tokens in block)
    map_base = head_idx * D + d_offs
    src = tl.load(unpack_src_ptr + map_base, mask=d_mask, other=0).to(tl.int64)
    is_sem_raw = tl.load(unpack_is_sem_ptr + map_base, mask=d_mask, other=1)
    is_sem = is_sem_raw != 0
    is_high_raw = tl.load(unpack_is_high_ptr + map_base, mask=d_mask, other=1)
    is_high = is_high_raw != 0

    for t in range(block_size):
        # 1. Gather packed bytes via scattered src map
        cache_off = (block_id * stride_cache_block
                     + t * stride_cache_tok
                     + head_idx * stride_cache_h
                     + src)
        raw = tl.load(packed_cache_ptr + cache_off, mask=d_mask, other=0)
        raw_i32 = raw.to(tl.int32)

        # 2. Unpack nibbles
        # Semantic dims: index = raw byte directly
        # Tail dims: is_high → high nibble (raw >> 4) & 0xF, else low nibble raw & 0xF
        idx = tl.where(is_sem, raw_i32,
                       tl.where(is_high, (raw_i32 >> 4) & 0xF, raw_i32 & 0xF))

        # 3. Codebook lookup
        # Clamp both indices: stale cache data in padding tokens may have
        # byte values exceeding codebook sizes (semantic: 0-63, tail: 0-15).
        sem_idx = tl.minimum(idx, N_SEM - 1)
        sem_val = tl.load(sem_centroids_ptr + head_idx * N_SEM + sem_idx,
                          mask=d_mask, other=0.0)
        tail_idx = tl.minimum(idx, N_TAIL - 1)
        tail_val = tl.load(tail_centroids_ptr + head_idx * N_TAIL + tail_idx,
                           mask=d_mask, other=0.0)
        val = tl.where(is_sem, sem_val, tail_val)

        # 4. Norm rescale
        if HAS_NORMS:
            slot = block_id * block_size + t
            norm = tl.load(norm_buffer_ptr
                           + slot * stride_norm_slot
                           + (head_offset + head_idx) * stride_norm_h
                           + kv_idx).to(tl.float32)
            val = val * norm

        # 5. Store bf16 to output
        out_off = ((block_prog * block_size + t) * stride_out_tok
                   + head_idx * stride_out_h + d_offs)
        tl.store(output_ptr + out_off, val.to(tl.bfloat16), mask=d_mask)
