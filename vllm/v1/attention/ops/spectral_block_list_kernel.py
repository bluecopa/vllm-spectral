"""Triton kernel: scatter active block IDs into a compact block list.

Given an active_mask (marking which cache blocks are referenced) and
compact_idx (cumsum-1, giving each active block's position in the
compact list), this kernel scatters the active block IDs into a
fixed-size block_list.

Grid: (ceil(N / TILE),) — FIXED, independent of number of active blocks.
This makes the kernel safe for CUDA graph capture.

The block_list should be pre-filled with -1 (sentinel).  The dequant
kernel checks for block_id < 0 and early-exits on unused slots.
"""

import triton
import triton.language as tl


@triton.jit
def build_block_list_kernel(
    active_mask_ptr,    # (N,) int32 — 1 if block is referenced, 0 otherwise
    compact_idx_ptr,    # (N,) int32 — cumsum - 1 (compact position for each block)
    block_list_ptr,     # (MAX_BLOCKS,) int64 — output, pre-filled with -1
    N,                  # int: total number of cache blocks
    TILE: tl.constexpr, # tile size (1024)
):
    pid = tl.program_id(0)
    offs = pid * TILE + tl.arange(0, TILE)
    mask = offs < N

    active = tl.load(active_mask_ptr + offs, mask=mask, other=0)
    is_active = active != 0

    compact_pos = tl.load(compact_idx_ptr + offs, mask=mask & is_active, other=0)
    tl.store(block_list_ptr + compact_pos.to(tl.int64),
             offs.to(tl.int64), mask=mask & is_active)
