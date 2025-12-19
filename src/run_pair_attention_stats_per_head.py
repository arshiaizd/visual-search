import os
import json
import math
import csv
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import chi2


# ============================================================
# CONFIG (edit here only)
# ============================================================

# Dataset / cache config
ANN_PATH = "data/annotations.jsonl"             # paired annotations
ATTN_DIR = "reports/attn_cache"                 # .npz attention files
FAMILY = "exists_green_circle"                  # suffix in attention filenames
GRID_SIZE = 12                                  # 12x12 patch grid

# Output files
OUT_PAIR_CSV = "reports/pair_attention_stats_per_head.csv"
OUT_TEST_CSV = "reports/attention_head_tests.csv"
OUT_TOP_CSV = "reports/top_k_heads_by_metric.csv"

# Top-k reporting config
TOP_K = 10

# Metrics to rank by (column names in OUT_TEST_CSV)
METRICS = [
    "mean_diff_max",
    "neg_log10_p_max_one_sided",
    "t_max",
    "mean_diff_mean",
    "neg_log10_p_mean_one_sided",
    "t_mean",
    "neg_log10_p_fisher_one_sided",
]

# Sort direction: True = descending, False = ascending
SORT_DESC: Dict[str, bool] = {
    "mean_diff_max": True,
    "neg_log10_p_max_one_sided": True,
    "t_max": True,
    "mean_diff_mean": True,
    "neg_log10_p_mean_one_sided": True,
    "t_mean": True,
    "neg_log10_p_fisher_one_sided": True,
}


# ============================================================
# Helper functions (shared)
# ============================================================

def ensure_output_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_pairs(ann_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load annotations.jsonl and group records into pairs by pair_id.
    Expected fields in each record:
      - pair_id: int
      - has_target: bool
      - target: { "r": int, "c": int, ... }
      - image: path like "data/images/pair_00000_with.png"
    Returns:
      pairs[pid] = {"with": rec_with, "without": rec_without}
    """
    pairs: Dict[int, Dict[str, Any]] = defaultdict(dict)
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["pair_id"]
            if rec["has_target"]:
                pairs[pid]["with"] = rec
            else:
                pairs[pid]["without"] = rec
    return pairs


def neighborhood_indices_3x3(r: int, c: int, grid_size: int) -> np.ndarray:
    """
    Return flat indices of a 3x3 neighborhood centered at (r, c)
    in a grid_size x grid_size patch grid.
    Assumes (r, c) is not on the border (so r±1, c±1 are valid).
    """
    idxs: List[int] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = r + dr
            cc = c + dc
            idx = rr * grid_size + cc
            idxs.append(idx)
    return np.asarray(idxs, dtype=np.int64)


def attn_cache_path_for_image(image_path: str) -> str:
    """
    Given an image path like "data/images/pair_00000_with.png",
    return the expected attention cache path, e.g.
      "reports/attn_cache/pair_00000_with_EXISTS_FAMILY.npz"
    """
    base = os.path.splitext(os.path.basename(image_path))[0]  # "pair_00000_with"
    fname = f"{base}_{FAMILY}.npz"
    return os.path.join(ATTN_DIR, fname)


def load_attention_from_cache(image_path: str) -> np.ndarray:
    """
    Load attention for a given image from its .npz cache.
    Returns:
      attn: np.ndarray of shape (L, H, N)
    """
    path = attn_cache_path_for_image(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Attention file not found: {path}")
    data = np.load(path)
    attn = data["attn"]  # (L, H, N)
    return attn


def t_stat_onesample(diffs: np.ndarray) -> float:
    """
    Compute one-sample t-statistic for H0: mean(diffs)=0 with ddof=1.
    Returns nan if not enough samples.
    """
    n = diffs.size
    if n < 2:
        return float("nan")

    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1))

    if std == 0.0:
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return 0.0

    se = std / math.sqrt(n)
    return mean / se


def log_p_one_sided_from_t(t_stat: float, df: int, eps: float = 1e-300) -> float:
    """
    log(p_one_sided) for H1: mean_diff > 0 using Student-t:
      p_one_sided = sf(t_stat)
    Uses logsf for numerical stability, with an eps floor.
    """
    if df <= 0 or not np.isfinite(t_stat):
        # handle +/-inf explicitly
        if t_stat == float("inf"):
            return math.log(eps)   # p ~ 0
        if t_stat == float("-inf"):
            return 0.0            # p ~ 1
        return float("nan")

    logp = float(student_t.logsf(t_stat, df))  # log(p)
    return max(logp, math.log(eps))


def neglog10_from_logp(logp: float) -> float:
    """Convert natural-log p to -log10(p)."""
    if not np.isfinite(logp):
        return float("nan")
    return -logp / math.log(10.0)


def fisher_neglog10_from_logps(logp1: float, logp2: float, eps: float = 1e-300) -> float:
    """
    Fisher's method combining TWO p-values:
      X = -2 (log p1 + log p2) ~ chi2(df=4) under independence.

    We compute p_fisher via chi2.logsf for stability and return -log10(p_fisher).
    Note: p-values here are typically dependent (max vs mean), so this is best
    used as a ranking heuristic unless you calibrate for dependence.
    """
    if not (np.isfinite(logp1) and np.isfinite(logp2)):
        return float("nan")

    X = -2.0 * (logp1 + logp2)  # fisher statistic

    return X


# ============================================================
# Part 1: compute per-pair and per-head-layer stats
# ============================================================

def compute_pair_and_head_stats():
    ensure_output_dir(OUT_PAIR_CSV)
    ensure_output_dir(OUT_TEST_CSV)

    pairs = load_pairs(ANN_PATH)

    # Collect diffs across pairs
    diff_max_by_lh: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    diff_mean_by_lh: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    # Detailed per-pair, per-head-layer values
    pair_rows: List[str] = []
    pair_header = [
        "pair_id",
        "layer",
        "head",
        "max_with",
        "max_without",
        "diff_max",
        "mean_with",
        "mean_without",
        "diff_mean",
    ]
    pair_rows.append(",".join(pair_header))

    first_attn_shape = None

    for pid, recs in pairs.items():
        if "with" not in recs or "without" not in recs:
            continue  # incomplete pair

        rec_with = recs["with"]
        rec_without = recs["without"]

        img_with = rec_with["image"]
        img_without = rec_without["image"]

        tr = rec_with["target"]["r"]
        tc = rec_with["target"]["c"]

        try:
            attn_with = load_attention_from_cache(img_with)
            attn_without = load_attention_from_cache(img_without)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        if attn_with.shape != attn_without.shape:
            print(f"[WARN] Attention shape mismatch for pair {pid}: "
                  f"with {attn_with.shape}, without {attn_without.shape}")
            continue

        L, H, N = attn_with.shape

        if first_attn_shape is None:
            first_attn_shape = (L, H, N)
            print(f"First attention shape: L={L}, H={H}, N={N}")

        expected_N = GRID_SIZE * GRID_SIZE
        if N != expected_N:
            print(f"[WARN] Unexpected attention length for pair {pid}: "
                  f"N={N}, expected {expected_N} (GRID_SIZE={GRID_SIZE})")
            continue

        idxs = neighborhood_indices_3x3(tr, tc, grid_size=GRID_SIZE)

        for li in range(L):
            for hi in range(H):
                vec_with = attn_with[li, hi]      # (N,)
                vec_without = attn_without[li, hi]

                neigh_with = vec_with[idxs]       # (9,)
                neigh_without = vec_without[idxs]

                max_with = float(neigh_with.max())
                max_without = float(neigh_without.max())

                mean_with = float(neigh_with.mean())
                mean_without = float(neigh_without.mean())

                diff_max = max_with - max_without
                diff_mean = mean_with - mean_without

                diff_max_by_lh[(li, hi)].append(diff_max)
                diff_mean_by_lh[(li, hi)].append(diff_mean)

                row = [
                    str(pid),
                    str(li),
                    str(hi),
                    f"{max_with:.8f}",
                    f"{max_without:.8f}",
                    f"{diff_max:.8f}",
                    f"{mean_with:.8f}",
                    f"{mean_without:.8f}",
                    f"{diff_mean:.8f}",
                ]
                pair_rows.append(",".join(row))

    with open(OUT_PAIR_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(pair_rows))

    # Compact per-head-layer results (one-sided only + Fisher combined)
    test_rows: List[str] = []
    test_header = [
        "layer",
        "head",
        "n_pairs",
        "mean_diff_max",
        "t_max",
        "neg_log10_p_max_one_sided",
        "mean_diff_mean",
        "t_mean",
        "neg_log10_p_mean_one_sided",
        "neg_log10_p_fisher_one_sided",
    ]
    test_rows.append(",".join(test_header))

    keys_sorted = sorted(diff_max_by_lh.keys())

    for (li, hi) in keys_sorted:
        diffs_max = np.asarray(diff_max_by_lh[(li, hi)], dtype=np.float64)
        diffs_mean = np.asarray(diff_mean_by_lh[(li, hi)], dtype=np.float64)
        n = int(diffs_max.size)
        if n == 0:
            continue

        df = n - 1

        # max
        mean_diff_max = float(diffs_max.mean())
        t_max = float(t_stat_onesample(diffs_max))
        logp_max = float(log_p_one_sided_from_t(t_max, df=df))
        neglog_p_max = float(neglog10_from_logp(logp_max))

        # mean
        mean_diff_mean = float(diffs_mean.mean())
        t_mean = float(t_stat_onesample(diffs_mean))
        logp_mean = float(log_p_one_sided_from_t(t_mean, df=df))
        neglog_p_mean = float(neglog10_from_logp(logp_mean))

        # Fisher combine (heuristic if dependent)
        neglog_p_fisher = float(fisher_neglog10_from_logps(logp_max, logp_mean))

        test_row = [
            str(li),
            str(hi),
            str(n),
            f"{mean_diff_max:.8f}",
            f"{t_max:.5f}" if np.isfinite(t_max) else str(t_max),
            f"{neglog_p_max:.4f}" if np.isfinite(neglog_p_max) else str(neglog_p_max),
            f"{mean_diff_mean:.8f}",
            f"{t_mean:.5f}" if np.isfinite(t_mean) else str(t_mean),
            f"{neglog_p_mean:.4f}" if np.isfinite(neglog_p_mean) else str(neglog_p_mean),
            f"{neglog_p_fisher:.4f}" if np.isfinite(neglog_p_fisher) else str(neglog_p_fisher),
        ]
        test_rows.append(",".join(test_row))

    with open(OUT_TEST_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(test_rows))

    print(f"Per-pair stats written to: {OUT_PAIR_CSV}")
    print(f"Per-head stats written to: {OUT_TEST_CSV}")


# ============================================================
# Part 2: rank top-k heads per metric
# ============================================================

def load_test_csv(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    return rows, header


def parse_float(row: Dict[str, Any], key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except Exception:
        return float("nan")


def report_top_heads():
    rows, header = load_test_csv(OUT_TEST_CSV)
    if not rows:
        print(f"No rows found in {OUT_TEST_CSV}")
        return

    missing = [m for m in METRICS if m not in header]
    if missing:
        print(f"[WARN] These metrics are missing from {OUT_TEST_CSV}: {missing}")

    ensure_output_dir(OUT_TOP_CSV)

    out_rows: List[Dict[str, Any]] = []

    for metric in METRICS:
        if metric not in header:
            print(f"Skipping metric '{metric}' (not found in CSV header).")
            continue

        desc = SORT_DESC.get(metric, True)

        sortable: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            v = parse_float(row, metric)
            if v != v:  # NaN
                continue
            sortable.append((v, row))

        if not sortable:
            print(f"No valid numeric values for metric '{metric}', skipping.")
            continue

        sortable.sort(key=lambda x: x[0], reverse=desc)
        k = min(TOP_K, len(sortable))

        print("=" * 60)
        print(f"Top {k} heads by metric: {metric} "
              f"({'desc' if desc else 'asc'})")
        print("=" * 60)

        for rank_idx in range(k):
            value, row = sortable[rank_idx]
            layer = row.get("layer", "?")
            head = row.get("head", "?")
            n_pairs = row.get("n_pairs", "?")

            print(
                f"#{rank_idx+1:2d}  L={layer}, H={head}, "
                f"value={value:.6f}, n_pairs={n_pairs}"
            )

            out_rows.append(
                {
                    "metric": metric,
                    "rank": rank_idx + 1,
                    "layer": layer,
                    "head": head,
                    "value": f"{value:.8f}",
                    "n_pairs": n_pairs,
                }
            )
        print()

    if out_rows:
        fieldnames = ["metric", "rank", "layer", "head", "value", "n_pairs"]
        with open(OUT_TOP_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nTop-k summaries written to: {OUT_TOP_CSV}")
    else:
        print("No top-k rows to write; check if the test CSV had valid metrics.")


# ============================================================
# Entry point
# ============================================================

def main():
    print("=== Computing per-pair and per-head-layer stats (max + mean) ===")
    compute_pair_and_head_stats()
    print("\n=== Reporting top-k heads by metric ===")
    report_top_heads()


if __name__ == "__main__":
    main()
