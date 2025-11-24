# src/run_stats.py
"""
Generic attention-head stats across tasks.

For each (image, prompt) that has an attention cache:
  - find the corresponding row in per_case.csv
  - label it as 'correct' or 'incorrect' based on task metrics
  - compute per-head features (max, mean, entropy, gini, etc.)
  - run simple statistics comparing correct vs incorrect cases per head

This works for any task, as long as per_case.csv includes a 'correct' flag.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ---------- feature helpers --------------------------------------------------

def attn_features_for_head(vec_100: np.ndarray) -> Dict[str, float]:
    """
    Compute a few simple features from a flattened 10x10 attention map.
    vec_100: shape (100,) nonnegative, typically normalized.
    """
    v = vec_100.astype(np.float64)
    s = v.sum()
    if s <= 0:
        v = np.ones_like(v) / len(v)
        s = 1.0
    else:
        v = v / s

    # entropy
    eps = 1e-12
    ent = -(v * np.log(v + eps)).sum()

    # gini (measure of inequality / "peakedness")
    # adapted from standard Gini implementation
    sorted_v = np.sort(v)
    n = len(sorted_v)
    idx = np.arange(1, n + 1)
    gini = (2 * (idx * sorted_v).sum()) / (n * sorted_v.sum() + eps) - (n + 1) / n

    # max and mean
    return {
        "max": float(v.max()),
        "mean": float(v.mean()),
        "entropy": float(ent),
        "gini": float(gini),
    }


# ---------- stats core -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True,
                    help="Directory with *.npz attention caches (L,H,100).")
    ap.add_argument("--per_case", required=False,
                    help="Path to per_case.csv. If omitted, will look in parent of attn_dir.")
    ap.add_argument("--out", required=True,
                    help="Output directory for stats tables.")
    ap.add_argument("--group_metric", default="correct",
                    help="Column in per_case.csv used to split cases (default: 'correct').")
    ap.add_argument("--pos_label", default="True",
                    help="Value in group_metric considered 'positive' group (default: 'True').")
    ap.add_argument("--neg_label", default="False",
                    help="Value in group_metric considered 'negative' group (default: 'False').")
    args = ap.parse_args()

    attn_dir = Path(args.attn_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- locate per_case.csv ---
    per_case_path: Path
    if args.per_case is not None:
        per_case_path = Path(args.per_case)
    else:
        # assume per_case.csv is in the parent of attn_dir
        per_case_path = attn_dir.parent / "per_case.csv"

    if not per_case_path.is_file():
        raise FileNotFoundError(f"Cannot find per_case.csv at {per_case_path}")

    df = pd.read_csv(per_case_path)

    if args.group_metric not in df.columns:
        raise KeyError(f"per_case.csv does not have column '{args.group_metric}'")

    # Normalize group labels as strings for comparison
    df["_group_value"] = df[args.group_metric].astype(str)

    # We'll build a long table: one row per (case, layer, head)
    rows: List[Dict] = []

    # Index per_case by image+prompt for easy lookup.
    # NOTE: run_eval writes one row per (image, prompt).
    # Our attn npz currently only stores 'image' and 'prompt' in 'meta'.
    index = df.set_index(["image", "prompt"])

    for npz_path in sorted(attn_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        attn = data["attn"]          # (L, H, 100)
        meta = data["meta"].item()   # stored as 0-d object array

        image = meta["image"]
        prompt = meta["prompt"]

        key = (image, prompt)
        if key not in index.index:
            # could warn, but just skip silently
            continue

        row = index.loc[key]
        group_val = str(row["_group_value"])

        # Only keep if group_val matches either pos_label or neg_label
        if group_val not in {args.pos_label, args.neg_label}:
            # e.g., None or something else; skip
            continue

        group = "pos" if group_val == args.pos_label else "neg"

        L, H, _ = attn.shape
        for li in range(L):
            for hi in range(H):
                vec = attn[li, hi]  # (100,)
                feats = attn_features_for_head(vec)
                rows.append({
                    "image": image,
                    "prompt": prompt,
                    "layer": li,
                    "head": hi,
                    "group": group,
                    **feats,
                })

    if not rows:
        raise RuntimeError("No matching cases found for stats (rows list is empty).")

    df_feats = pd.DataFrame(rows)
    df_feats.to_csv(out / "head_features_long.csv", index=False)

    # --- stats per (layer, head, feature) ---
    stats_rows: List[Dict] = []
    for (li, hi), sub in df_feats.groupby(["layer", "head"]):
        for feat in ["max", "mean", "entropy", "gini"]:
            pos_vals = sub.loc[sub["group"] == "pos", feat].values
            neg_vals = sub.loc[sub["group"] == "neg", feat].values

            if len(pos_vals) < 3 or len(neg_vals) < 3:
                p_val = np.nan
                effect = np.nan
            else:
                # Mann–Whitney U test (nonparametric)
                try:
                    stat, p_val = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
                except ValueError:
                    p_val = np.nan

                effect = float(np.mean(pos_vals) - np.mean(neg_vals))

            stats_rows.append({
                "layer": li,
                "head": hi,
                "feature": feat,
                "pos_n": len(pos_vals),
                "neg_n": len(neg_vals),
                "pos_mean": float(np.mean(pos_vals)) if len(pos_vals) > 0 else np.nan,
                "neg_mean": float(np.mean(neg_vals)) if len(neg_vals) > 0 else np.nan,
                "effect_pos_minus_neg": effect,
                "p_mannwhitney": p_val,
            })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(out / "head_stats_correct_vs_incorrect.csv", index=False)

    print(f"Wrote per-head features to {out/'head_features_long.csv'}")
    print(f"Wrote per-head stats to {out/'head_stats_correct_vs_incorrect.csv'}")


if __name__ == "__main__":
    main()
