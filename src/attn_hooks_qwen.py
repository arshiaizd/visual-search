"""
Utilities to run Qwen-2.5-VL with attention and aggregate to 10×10 patches.
This provides pooling utilities; wiring actual hooks is model-version-specific.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image

def _avg_pool_to_grid(attn_map_hw: torch.Tensor, grid: int):
    # Always work in float32 for pooling / numpy conversion
    attn_map_hw = attn_map_hw.to(torch.float32)

    H, W = attn_map_hw.shape
    bh = H / grid
    bw = W / grid

    pooled = torch.zeros((grid, grid), dtype=torch.float32, device=attn_map_hw.device)
    for r in range(grid):
        r0 = int(round(r * bh)); r1 = int(round((r + 1) * bh))
        for c in range(grid):
            c0 = int(round(c * bw)); c1 = int(round((c + 1) * bw))
            patch = attn_map_hw[r0:r1, c0:c1]
            pooled[r, c] = patch.mean() if patch.numel() > 0 else 0.0

    # v = pooled.flatten()
    # v = v / (v.sum() + 1e-12)
    
    orig_sum = float(attn_map_hw.sum().item())
    v = pooled.flatten()
    v *= orig_sum / (float(v.sum().item()) + 1e-12)

    return v.detach().cpu().numpy()


def aggregate_attentions_to_grid(
    attn_per_layer_head: List[List[torch.Tensor]],
    token_hw: Tuple[int, int],
    grid_size: int,
) -> np.ndarray:
    """
    attn_per_layer_head: list over layers, each a list over heads, each (H, W) tensor
    token_hw:            (H, W) of the original token grid (kept for completeness)
    grid_size:           patch grid size (e.g. 10 for 10x10, 12 for 12x12)
    returns:             (L, Hh, grid_size*grid_size) numpy array
    """
    L = len(attn_per_layer_head)
    Hh = len(attn_per_layer_head[0]) if L > 0 else 0

    out = np.zeros((L, Hh, grid_size * grid_size), dtype=np.float32)
    for li in range(L):
        for hi in range(Hh):
            v = _avg_pool_to_grid(attn_per_layer_head[li][hi], grid=grid_size)
            out[li, hi] = v
    return out
