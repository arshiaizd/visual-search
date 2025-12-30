from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import torch


# =========================================================
# Spec / Policies
# =========================================================

@dataclass(frozen=True)
class StitchSpec:
    """
    Controls how we stitch isolated vision-token grids into one synthetic grid.

    Parameters
    ----------
    pad:
        Neighborhood radius around (r,c).
        pad=1 => 3x3 window (r-1..r+1, c-1..c+1), clipped to bounds.

    overwrite:
        Overlap policy.
        - True  : later objects overwrite earlier ones in overlapping cells.
        - False : first write wins (later writes are ignored where already written).

    base_idx:
        Which isolated grid to use as the background / donor for all non-object patches.
        In your experiment, base_idx=0 (use first isolated image).

    validate_rc:
        If True, assert all (r,c) are inside the grid.
        If False, clip windows and allow out-of-bound rc to effectively write nothing.
    """
    pad: int = 1
    overwrite: bool = True
    base_idx: int = 0
    validate_rc: bool = True


# =========================================================
# Indexing helpers (grid space)
# =========================================================

def pid_from_rc(r: int, c: int, W: int) -> int:
    """Row-major patch id from (r,c) for a grid with width W."""
    return r * W + c


def rc_from_pid(pid: int, W: int) -> Tuple[int, int]:
    """Row-major (r,c) from patch id for a grid with width W."""
    return pid // W, pid % W


def rc_neighborhood(r: int, c: int, H: int, W: int, pad: int) -> Tuple[slice, slice]:
    """
    Return slices for the neighborhood window around (r,c), clipped to bounds.

    Example (pad=1):
      rows slice: [r-1 .. r+1] intersect [0 .. H-1]
      cols slice: [c-1 .. c+1] intersect [0 .. W-1]
    """
    r0 = max(0, r - pad)
    r1 = min(H - 1, r + pad)
    c0 = max(0, c - pad)
    c1 = min(W - 1, c + pad)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


# =========================================================
# Shape helpers
# =========================================================

def ensure_grid3(grid: torch.Tensor) -> torch.Tensor:
    """
    Ensure grid is (H,W,D). Accepts either:
      - (H,W,D) -> unchanged
      - (1,H,W,D) -> squeeze batch
    """
    if grid.ndim == 4 and grid.shape[0] == 1:
        grid = grid[0]
    if grid.ndim != 3:
        raise ValueError(f"Expected grid shape (H,W,D) or (1,H,W,D). Got {tuple(grid.shape)}")
    return grid


def flatten_grid(grid: torch.Tensor) -> torch.Tensor:
    """
    Convert (H, W, D) -> (1, H*W, D) for feeding into model injection point.
    """
    grid = ensure_grid3(grid)
    H, W, D = grid.shape
    return grid.reshape(1, H * W, D)


# =========================================================
# Core stitching for your experiment
# =========================================================

def stitch_from_isolated_base(
    obj_grids: Sequence[torch.Tensor],
    obj_rcs: Sequence[Tuple[int, int]],
    spec: StitchSpec = StitchSpec(),
) -> torch.Tensor:
    """
    Your experiment's stitching rule:

    1) Start from the "base" grid = obj_grids[spec.base_idx]
       This supplies ALL patches that are not overwritten by object neighborhoods.
    2) For each object i, overwrite only its pad-neighborhood (r,c)±pad
       using obj_grids[i].

    Parameters
    ----------
    obj_grids:
        List of isolated vision grids, one per object image.
        Each element must be (H,W,D) or (1,H,W,D).

    obj_rcs:
        Object positions in token-grid coords, one per object.
        Must align with obj_grids order (same ordering as object ids).

    spec:
        StitchSpec controlling pad, overwrite, base_idx, validation.

    Returns
    -------
    stitched:
        Tensor (H,W,D) containing a synthetic vision grid.
    """
    if len(obj_grids) == 0:
        raise ValueError("obj_grids is empty.")
    if len(obj_grids) != len(obj_rcs):
        raise ValueError(f"obj_grids and obj_rcs length mismatch: {len(obj_grids)} vs {len(obj_rcs)}")

    if not (0 <= spec.base_idx < len(obj_grids)):
        raise ValueError(f"base_idx={spec.base_idx} out of range for {len(obj_grids)} obj_grids")

    grids3 = [ensure_grid3(g) for g in obj_grids]

    base = grids3[spec.base_idx]
    H, W, D = base.shape

    # validate shapes
    for i, g in enumerate(grids3):
        if g.shape != (H, W, D):
            raise ValueError(f"Grid {i} shape mismatch: {tuple(g.shape)} vs base {(H,W,D)}")

    out = base.clone()

    written = None
    if not spec.overwrite:
        written = torch.zeros((H, W), dtype=torch.bool, device=out.device)

    for i, ((r, c), g) in enumerate(zip(obj_rcs, grids3)):
        r = int(r)
        c = int(c)

        if spec.validate_rc:
            if not (0 <= r < H and 0 <= c < W):
                raise ValueError(f"obj[{i}] has (r,c)=({r},{c}) outside grid (H,W)=({H},{W})")

        rs, cs = rc_neighborhood(r, c, H, W, spec.pad)

        if spec.overwrite:
            out[rs, cs, :] = g[rs, cs, :]
        else:
            # first-write-wins policy
            mask = ~written[rs, cs]
            if mask.any():
                mask3 = mask.unsqueeze(-1).expand(-1, -1, D)
                out_region = out[rs, cs, :]
                g_region = g[rs, cs, :]
                out_region[mask3] = g_region[mask3]
                out[rs, cs, :] = out_region
                written[rs, cs] |= mask

    return out


def stitch_grids(
    base_grid: torch.Tensor,
    donor_grids: Sequence[torch.Tensor],
    donor_rcs: Sequence[Tuple[int, int]],
    spec: StitchSpec = StitchSpec(),
) -> torch.Tensor:
    """
    Generic stitch: start from an explicit base_grid and paste neighborhoods from donor_grids.

    This is a more general form than stitch_from_isolated_base().
    You might use this if your base grid comes from a blank image or a real image,
    but donors come from isolated images.

    Parameters
    ----------
    base_grid:
        (H,W,D) baseline grid to start from.

    donor_grids:
        List of (H,W,D) grids to paste from.

    donor_rcs:
        List of (r,c) paste centers (same length as donor_grids).

    spec:
        pad/overwrite/validation policies.
        Note: base_idx is ignored here.

    Returns
    -------
    out:
        (H,W,D) stitched grid.
    """
    base_grid = ensure_grid3(base_grid)

    if len(donor_grids) != len(donor_rcs):
        raise ValueError("donor_grids and donor_rcs must have the same length")

    H, W, D = base_grid.shape
    out = base_grid.clone()

    written = None
    if not spec.overwrite:
        written = torch.zeros((H, W), dtype=torch.bool, device=out.device)

    for i, (g, (r, c)) in enumerate(zip(donor_grids, donor_rcs)):
        g = ensure_grid3(g)
        if g.shape != (H, W, D):
            raise ValueError(f"donor grid {i} shape mismatch: {tuple(g.shape)} vs base {(H,W,D)}")

        r = int(r)
        c = int(c)

        if spec.validate_rc:
            if not (0 <= r < H and 0 <= c < W):
                raise ValueError(f"donor[{i}] has (r,c)=({r},{c}) outside grid (H,W)=({H},{W})")

        rs, cs = rc_neighborhood(r, c, H, W, spec.pad)

        if spec.overwrite:
            out[rs, cs, :] = g[rs, cs, :]
        else:
            mask = ~written[rs, cs]
            if mask.any():
                mask3 = mask.unsqueeze(-1).expand(-1, -1, D)
                out_region = out[rs, cs, :]
                g_region = g[rs, cs, :]
                out_region[mask3] = g_region[mask3]
                out[rs, cs, :] = out_region
                written[rs, cs] |= mask

    return out
