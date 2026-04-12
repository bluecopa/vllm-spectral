"""Fused Triton kernel: GEMV rotation with optional L2 normalization.

Replaces all torch.bmm rotation calls (+ their .float()/.copy_() wrappers)
with a single kernel that operates bf16 in → f32 rotation → f32 accum → bf16 out.
Optional NORMALIZE mode computes L2 norm, normalizes input, then rotates —
fusing 6 kernels (float + norm + clamp + divide + bmm_K + bmm_V) into 1.

Grid: (T, H, ceil(D_OUT / TILE_D))
  - T:      tokens (1-4 during decode, batch during prefill)
  - H:      heads (KV heads or Q heads for GQA-expanded rotations)
  - TILE_D: output dimension tile (64)

Each program computes TILE_D output elements for one (token, head) pair
by accumulating the full D_IN dot product in f32.

Inner loop uses tl.load for per-element scalar extraction (avoids x[i]
scalar indexing which is not supported in all Triton versions).
"""

import triton
import triton.language as tl


@triton.jit
def spectral_rotate_kernel(
    # Input: (T, H, D_IN) — bf16 or f32
    input_ptr,
    # Rotation matrix: (H, D_IN, D_OUT) — f32
    rotation_ptr,
    # Output: (T, H, D_OUT) — bf16
    output_ptr,
    # Norms output: (T, H) — f32, only written when NORMALIZE=True
    norms_ptr,
    # Strides for input: [token, head, dim]
    stride_in_t,
    stride_in_h,
    stride_in_d,
    # Strides for rotation: [head, d_in, d_out]
    stride_rot_h,
    stride_rot_din,
    stride_rot_dout,
    # Strides for output: [token, head, dim]
    stride_out_t,
    stride_out_h,
    stride_out_d,
    # Stride for norms: [token, head]
    stride_norm_t,
    stride_norm_h,
    # Dimensions
    D_IN: tl.constexpr,
    D_OUT: tl.constexpr,
    NORMALIZE: tl.constexpr,
    TILE_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Base pointer for this (token, head) input vector
    in_base = input_ptr + pid_t * stride_in_t + pid_h * stride_in_h

    # Compute L2 norm if normalizing (need full vector for reduction)
    if NORMALIZE:
        norm_offs = tl.arange(0, D_IN)
        x_full = tl.load(in_base + norm_offs * stride_in_d).to(tl.float32)
        norm_sq = tl.sum(x_full * x_full)
        norm = tl.sqrt(norm_sq + 1e-16)
        if pid_d == 0:
            tl.store(norms_ptr + pid_t * stride_norm_t + pid_h * stride_norm_h,
                     norm)

    # Tiled GEMV: out[d] = sum_i x[i] * R[h, i, d]
    d_start = pid_d * TILE_D
    d_offs = d_start + tl.arange(0, TILE_D)
    d_mask = d_offs < D_OUT

    acc = tl.zeros([TILE_D], dtype=tl.float32)
    rot_base = rotation_ptr + pid_h * stride_rot_h

    for i in range(D_IN):
        # Load single input element as scalar (no x[i] indexing)
        xi = tl.load(in_base + i * stride_in_d).to(tl.float32)
        if NORMALIZE:
            xi = xi / norm
        # Load TILE_D rotation elements for this input dim
        r = tl.load(rot_base + i * stride_rot_din
                     + d_offs * stride_rot_dout,
                     mask=d_mask, other=0.0)
        acc += xi * r

    # Store bf16 output
    out_base = pid_t * stride_out_t + pid_h * stride_out_h
    tl.store(output_ptr + out_base + d_offs * stride_out_d,
             acc.to(tl.bfloat16), mask=d_mask)
