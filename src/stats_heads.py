# Compares TP vs FN feature distributions per (layer,head). Saves q-values and effect sizes.
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from src.dataset_io import read_annotations, gt_patch_set
from src.attn_features import features, to_mask

def load_npzs(p):
    for f in Path(p).glob("*.npz"):
        data = np.load(f, allow_pickle=True)
        attn = data["attn"]             # (L,H,100)
        meta = data["meta"].item()
        yield f.name, attn, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ann_idx = {Path(r["image"]).name: r for r in read_annotations(args.ann)}
    query = json.loads(args.query)

    rows = []
    for fname, attn, meta in load_npzs(args.attn_dir):
        rec = ann_idx.get(Path(meta["image"]).name)
        if rec is None: continue
        gt = gt_patch_set(rec, query)
        grid = rec["grid"]
        gt_mask = to_mask(gt, grid)

        pr = set(map(tuple, meta.get("pred_patches", [])))
        tp = gt & pr; fn = gt - pr; fp = pr - gt
        case = "TP" if len(tp)>0 else ("FN" if len(fn)>0 else ("FP" if len(fp)>0 else "TN"))

        L,H,_ = attn.shape
        for li in range(L):
            for hi in range(H):
                vec = attn[li,hi]  # (100,)
                f = features(vec, gt_mask)
                f.update({"image":meta["image"], "layer":li, "head":hi, "case":case})
                rows.append(f)

    df = pd.DataFrame(rows)
    df.to_parquet(out/"per_head_features.parquet", index=False)
    results = []
    feat_cols = [c for c in df.columns if c.startswith("g_") or c.startswith("l_")]
    for (li,hi), g in df.groupby(["layer","head"]):
        for col in feat_cols:
            a = g.loc[g.case=="TP", col].values
            b = g.loc[g.case=="FN", col].values
            if len(a)>=5 and len(b)>=5:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                m, n = len(a), len(b); U = stat
                effect = 1.0 - 2.0*U/(m*n)
                results.append({"layer":li,"head":hi,"feature":col,"p":p,"effect":effect,"n_tp":m,"n_fn":n})
    res = pd.DataFrame(results)
    if not res.empty:
        res["q"] = multipletests(res["p"].values, method="fdr_bh")[1]
        res.sort_values(["q","effect"], inplace=True)
        res.to_csv(out/"head_rankings.csv", index=False)

if __name__ == "__main__":
    main()
