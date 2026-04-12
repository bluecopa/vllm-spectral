"""
Codebook Phase 2 integration for SpectralQuant.

Lloyd-Max codebook computation and pack/unpack map generation.
"""

import math
import torch

# Codebook parameters
N_SEM = 64   # 6-bit semantic codebook
N_TAIL = 16  # 4-bit tail codebook


def lloyd_max_fit(data: torch.Tensor, n_levels: int, max_iter: int = 200) -> torch.Tensor:
    """Fit Lloyd-Max quantizer to 1D data, return sorted centroids."""
    if data.numel() == 0:
        return torch.linspace(-1, 1, n_levels, device=data.device, dtype=torch.float32)
    
    data = data.float().flatten()
    # Initialize with quantiles
    quantiles = torch.linspace(0, 1, n_levels + 1, device=data.device)[:-1] + 0.5 / n_levels
    centroids = torch.quantile(data, quantiles)
    
    for _ in range(max_iter):
        # Assign to nearest centroid
        dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = data[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        
        if (new_centroids - centroids).abs().max() < 1e-6:
            break
        centroids = new_centroids
    
    return centroids.sort().values


def compute_codebooks_from_eigenvalues(
    eigenvalues: torch.Tensor,  # (H, D) eigenvalues per head
    d_eff: torch.Tensor,        # (H,) effective dimensionality per head
    n_samples: int = 10000,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Lloyd-Max codebooks from eigenvalue-derived variances.
    
    Returns:
        sem_centroids: (H, N_SEM) float32
        tail_centroids: (H, N_TAIL) float32
    """
    H, D = eigenvalues.shape
    eigenvalues = eigenvalues.to(device).float()
    d_eff = d_eff.to(device)
    
    sem_centroids = torch.zeros(H, N_SEM, device=device, dtype=torch.float32)
    tail_centroids = torch.zeros(H, N_TAIL, device=device, dtype=torch.float32)
    
    for h in range(H):
        eig = eigenvalues[h]
        d_e = max(1, int(d_eff[h].item()))
        
        # Expected sigma for L2-normalized vector
        total_var = eig.sum()
        norm_factor = math.sqrt(max(total_var.item(), 1e-6))
        
        # Sample from eigenvalue-scaled Gaussians
        sem_sigmas = (eig[:d_e].sqrt() / norm_factor).clamp(min=1e-6)
        tail_sigmas = (eig[d_e:].sqrt() / norm_factor).clamp(min=1e-6)
        
        sem_samples = torch.randn(n_samples, len(sem_sigmas), device=device) * sem_sigmas
        tail_samples = torch.randn(n_samples, max(1, len(tail_sigmas)), device=device)
        if len(tail_sigmas) > 0:
            tail_samples = tail_samples[:, :len(tail_sigmas)] * tail_sigmas
        
        # Fit Lloyd-Max
        sem_centroids[h] = lloyd_max_fit(sem_samples, N_SEM)
        tail_centroids[h] = lloyd_max_fit(tail_samples, N_TAIL)
    
    return sem_centroids, tail_centroids


def build_pack_maps(
    d_eff: torch.Tensor,  # (H,) effective dimensionality per head
    head_dim: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build pack maps for nibble-packed storage.
    
    Semantic dims (0:d_eff): store as full uint8 (6-bit codebook index)
    Tail dims (d_eff:): pack pairs as nibbles (hi<<4 | lo)
    
    Returns:
        hi_src: (H, packed_dim) source dimension for hi nibble
        lo_src: (H, packed_dim) source dimension for lo nibble  
        is_sem: (H, packed_dim) whether position is semantic (full byte)
        has_lo: (H, packed_dim) whether position has lo nibble
        valid:  (H, packed_dim) whether position is valid
        packed_dim: max packed dimension across heads
    """
    H = len(d_eff)
    d_eff = d_eff.to(device)
    
    # Compute packed dimension per head
    packed_dims = []
    for h in range(H):
        d_e = max(1, int(d_eff[h].item()))
        n_tail = head_dim - d_e
        n_tail_packed = (n_tail + 1) // 2  # pairs
        packed_dims.append(d_e + n_tail_packed)
    
    max_packed = max(packed_dims)
    
    hi_src = torch.zeros(H, max_packed, dtype=torch.int64, device=device)
    lo_src = torch.zeros(H, max_packed, dtype=torch.int64, device=device)
    is_sem = torch.zeros(H, max_packed, dtype=torch.uint8, device=device)
    has_lo = torch.zeros(H, max_packed, dtype=torch.uint8, device=device)
    valid = torch.zeros(H, max_packed, dtype=torch.uint8, device=device)
    
    for h in range(H):
        d_e = max(1, int(d_eff[h].item()))
        n_tail = head_dim - d_e
        
        pos = 0
        # Semantic positions
        for i in range(d_e):
            hi_src[h, pos] = i
            lo_src[h, pos] = i
            is_sem[h, pos] = 1
            has_lo[h, pos] = 0
            valid[h, pos] = 1
            pos += 1
        
        # Tail positions (paired)
        for i in range(0, n_tail, 2):
            hi_src[h, pos] = d_e + i
            if i + 1 < n_tail:
                lo_src[h, pos] = d_e + i + 1
                has_lo[h, pos] = 1
            else:
                lo_src[h, pos] = d_e + i
                has_lo[h, pos] = 0
            is_sem[h, pos] = 0
            valid[h, pos] = 1
            pos += 1
    
    return hi_src, lo_src, is_sem, has_lo, valid, max_packed


def build_unpack_maps(
    d_eff: torch.Tensor,  # (H,) effective dimensionality per head
    head_dim: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unpack maps for decompression.
    
    Returns:
        src: (H, D) packed byte position for each output dimension
        is_sem: (H, D) whether dimension is semantic
        is_high: (H, D) whether to use high nibble (for tail dims)
    """
    H = len(d_eff)
    d_eff = d_eff.to(device)
    D = head_dim
    
    src = torch.zeros(H, D, dtype=torch.int64, device=device)
    is_sem = torch.zeros(H, D, dtype=torch.uint8, device=device)
    is_high = torch.zeros(H, D, dtype=torch.uint8, device=device)
    
    for h in range(H):
        d_e = max(1, int(d_eff[h].item()))
        
        # Semantic dims: direct 1:1 mapping
        for i in range(d_e):
            src[h, i] = i
            is_sem[h, i] = 1
            is_high[h, i] = 1  # unused for semantic
        
        # Tail dims: packed pairs
        packed_pos = d_e
        for i in range(d_e, D, 2):
            src[h, i] = packed_pos
            is_sem[h, i] = 0
            is_high[h, i] = 1  # first of pair = high nibble
            
            if i + 1 < D:
                src[h, i + 1] = packed_pos
                is_sem[h, i + 1] = 0
                is_high[h, i + 1] = 0  # second of pair = low nibble
            
            packed_pos += 1
    
    return src, is_sem, is_high
