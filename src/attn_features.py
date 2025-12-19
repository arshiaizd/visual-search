import numpy as np

def _norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    s = p.sum()
    return p/(s+1e-12)

def entropy(p):
    q = _norm(p); return float(-(q*np.log(q+1e-12)).sum())

def gini(p):
    q = _norm(p); return float(1.0 - (q*q).sum())

def spread_above(p, thr=0.02):
    q = _norm(p); return float((q>=thr).mean())

def to_mask(gt_set, grid=12):
    m = np.zeros((grid,grid), dtype=np.float32)
    for (r,c) in gt_set: m[r,c]=1.0
    return m.reshape(-1)

def target_mass(p, mask):
    q = _norm(p); return float((q*mask).sum())

def contrast(p, mask):
    q = _norm(p)
    tm = (q*mask).sum()
    non = q*(1-mask)
    return float(tm - (non.max() if non.size else 0.0))

def features(vec100, gt_mask=None):
    feats = {
        "g_max": float(np.max(vec100)),
        "g_mean": float(np.mean(vec100)),
        "g_entropy": entropy(vec100),
        "g_gini": gini(vec100),
        "g_spread02": spread_above(vec100, 0.02),
    }
    if gt_mask is not None:
        feats["l_target_mass"] = target_mass(vec100, gt_mask)
        feats["l_contrast"] = contrast(vec100, gt_mask)
    return feats
