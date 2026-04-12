from __future__ import annotations

from typing import Any

import torch
import torch.compiler


def _protected_mask(
    seq_len: int,
    protect_initial: int,
    protect_recent: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if protect_initial > 0:
        mask[: min(protect_initial, seq_len)] = True
    if protect_recent > 0:
        mask[max(seq_len - protect_recent, 0) :] = True
    return mask


def _select_keep_indices(
    strategy: str,
    budget: int,
    seq_len: int,
    protect_initial: int,
    protect_recent: int,
    device: torch.device,
    oracle_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    protected = _protected_mask(seq_len, protect_initial, protect_recent, device)
    keep_count = min(seq_len, max(int(budget), int(protected.sum().item())))
    if keep_count >= seq_len:
        return torch.arange(seq_len, dtype=torch.long, device=device)

    if strategy == "recency":
        scores = torch.arange(seq_len, dtype=torch.float32, device=device)
    elif strategy == "oracle":
        if oracle_scores is None:
            raise ValueError("oracle_scores are required for oracle pruning.")
        scores = oracle_scores.clone()
    else:
        raise ValueError(f"Unsupported global pruning strategy: {strategy}")

    scores[protected] = torch.finfo(scores.dtype).max
    keep_idx = torch.topk(scores, k=keep_count, largest=True).indices
    return torch.sort(keep_idx).values


def uses_global_pruning(
    attn_layer: Any,
    attn_metadata: object,
    kv_cache: torch.Tensor,
) -> bool:
    budget = getattr(attn_layer, "global_pruning_budget", None)
    if budget is None:
        return False
    if attn_layer.sliding_window is not None:
        return False
    if attn_metadata is None or getattr(attn_metadata, "use_cascade", False):
        return False
    if kv_cache.dtype == torch.uint8 or kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        return False

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if query_start_loc is None or seq_lens is None or block_table is None:
        return False

    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    return bool(torch.all(q_lens == 1).item())


@torch.compiler.disable
def pruned_attention(
    query: torch.Tensor,
    output: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: object,
    attn_layer: Any,
    scale: float,
) -> None:
    if kv_cache.dtype == torch.uint8:
        raise NotImplementedError(
            "The initial global-pruning prototype does not support quantized "
            "KV cache storage. Use --kv-cache-dtype auto."
        )
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise NotImplementedError(
            f"Unsupported KV cache layout for global pruning: {tuple(kv_cache.shape)}"
        )
    if getattr(attn_metadata, "use_cascade", False):
        raise NotImplementedError(
            "Cascade attention is not implemented for the global-pruning prototype."
        )

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if query_start_loc is None or seq_lens is None or block_table is None:
        raise NotImplementedError(
            "Global pruning currently requires Triton-style attention metadata."
        )

    key_cache, value_cache = kv_cache.unbind(1)
    block_size = key_cache.shape[1]
    num_q_heads = query.shape[1]
    group_size = num_q_heads // attn_layer.num_kv_heads
    budget = int(attn_layer.global_pruning_budget)
    strategy = str(attn_layer.global_pruning_strategy)
    protect_initial = int(attn_layer.global_pruning_protect_initial)
    protect_recent = int(attn_layer.global_pruning_protect_recent)
    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]))

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
        key_full = key_cache.index_select(0, blocks).reshape(
            -1, attn_layer.num_kv_heads, attn_layer.head_size
        )[:seq_len]
        value_full = value_cache.index_select(0, blocks).reshape(
            -1, attn_layer.num_kv_heads, attn_layer.head_size_v
        )[:seq_len]

        q_seq = query[q_start:q_end].float()
        q_heads = q_seq.permute(1, 0, 2)
        key_heads = (
            key_full.repeat_interleave(group_size, dim=1).permute(1, 0, 2).float()
        )
        value_heads = (
            value_full.repeat_interleave(group_size, dim=1).permute(1, 0, 2).float()
        )

        prefix_len = seq_len - q_len
        q_positions = prefix_len + torch.arange(q_len, device=q_heads.device)
        k_positions = torch.arange(seq_len, device=q_heads.device)
        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

        oracle_scores = None
        if strategy == "oracle":
            full_scores = torch.matmul(q_heads, key_heads.transpose(-2, -1)) * scale
            full_scores.masked_fill_(
                ~causal_mask.unsqueeze(0),
                torch.finfo(full_scores.dtype).min,
            )
            full_probs = torch.softmax(full_scores, dim=-1, dtype=torch.float32)
            oracle_scores = full_probs.amax(dim=(0, 1))

        keep_idx = _select_keep_indices(
            strategy=strategy,
            budget=budget,
            seq_len=seq_len,
            protect_initial=protect_initial,
            protect_recent=protect_recent,
            device=q_heads.device,
            oracle_scores=oracle_scores,
        )

        key_heads = key_heads[:, keep_idx, :]
        value_heads = value_heads[:, keep_idx, :]
        pruned_scores = torch.matmul(q_heads, key_heads.transpose(-2, -1)) * scale
        pruned_mask = keep_idx.unsqueeze(0) <= q_positions.unsqueeze(1)
        pruned_scores.masked_fill_(
            ~pruned_mask.unsqueeze(0),
            torch.finfo(pruned_scores.dtype).min,
        )
        probs = torch.softmax(pruned_scores, dim=-1, dtype=torch.float32)
        out_heads = torch.matmul(probs, value_heads)
        output[q_start:q_end].copy_(out_heads.permute(1, 0, 2).to(output.dtype))
