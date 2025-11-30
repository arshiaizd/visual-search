# src/run_eval.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset_io import read_annotations
from tasks import get_task, CaseReport
from src.models import predict_adapter as PA
from src.viz import heatmap_10x10, overlay_grid, overlay_patches


# ---------------------------------------------------------
# Which (layer, head) pairs to visualize as heatmaps.
# Edit this list to select the heads you care about.
# Example: [(0,0), (0,1), (3,5)]
# ---------------------------------------------------------
HEAD_LAYER_PLOTS = [
    (19, 15), (19, 16), (19, 20), (20, 21), (19, 22), (19, 23)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--family",
        required=True,
        help="Prompt family key, e.g. en_find_green_circle",
    )
    ap.add_argument(
        "--query",
        required=True,
        help='JSON dict, e.g. {"color":"green","shape":"circle"}',
    )
    ap.add_argument("--model_id", required=True)
    ap.add_argument(
        "--task",
        default="coords",
        help="Task name, e.g. coords, exists (see src/tasks/*)",
    )
    ap.add_argument(
        "--save_attention",
        type=int,
        default=1,
        help="If 1, dump attention caches and head-specific heatmaps.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "attn_cache").mkdir(exist_ok=True)

    # --------- parse query + load data ----------
    query = json.loads(args.query)
    ann = read_annotations(args.ann)

    task = get_task(args.task)
    prompts = task.get_prompts(args.family)

    per_case: list[CaseReport] = []

    # --------- main loop: images × prompts ----------
    for rec in tqdm(ann, desc="images"):
        img_path = rec["image"]
        img_stem = Path(img_path).stem
        
        for prompt in prompts:
            # 1) run the task’s evaluation for this (image,prompt)
            report = task.score_example(
                rec=rec,
                query=query,
                image_path=img_path,
                prompt_family=args.family,
                prompt=prompt,
                model_id=args.model_id,
            )
            per_case.append(report)

            # 2) optional: save attention + preview figure

            if args.save_attention:
                cache = PA.get_attention_cache(
                    img_path, prompt, model_id=args.model_id
                )
                if cache is not None and "attn" in cache:
                    attn = cache["attn"]  # shape: (L, H, 100)
                    L, H, _ = attn.shape

                    # save raw cache (per image + family)
                    stem_full = img_stem + f"_{args.family}"
                    npz_path = out / "attn_cache" / f"{stem_full}.npz"
                    np.savez(npz_path, **cache)

                    # directory for this image's heatmaps
                    img_fig_dir = out / "figs" / img_stem
                    img_fig_dir.mkdir(parents=True, exist_ok=True)

                    # visualize only the selected (layer, head) pairs
                    for (li, hi) in HEAD_LAYER_PLOTS:
                        if li < 0 or li >= L or hi < 0 or hi >= H:
                            # skip invalid indices
                            continue

                        vec = attn[li, hi]  # (100,)
                        hm = heatmap_10x10(vec)
                        hm = overlay_grid(hm, grid=10)

                        # only coords task has patch-level tp/fn to overlay
                        if args.task == "coords":
                            tp = report.metrics.get("tp", [])
                            fn = report.metrics.get("fn", [])
                            hm = overlay_patches(
                                hm, tp, outline=(0, 255, 0, 255)
                            )
                            hm = overlay_patches(
                                hm, fn, outline=(255, 140, 0, 255)
                            )

                        # file name: L<layer>_H<head>.png inside this image's folder
                        fig_path = img_fig_dir / f"L{li}_H{hi}.png"
                        hm.save(fig_path)


    # --------- 3) overall summary.json ----------
    overall = task.aggregate(per_case)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # --------- 4) per_case.csv (no raw text) ----------
    rows = []
    for r in per_case:
        row = {
            "image": r.image,
            "prompt_family": r.prompt_family,
            "prompt": r.prompt,
        }
        row.update(r.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_case.csv", index=False)

    # --------- 5) prompt sensitivity summary (per exact prompt) ----------
    prompt_rows = []
    has_correct = "correct" in df.columns
    has_TP = "TP" in df.columns
    has_FP = "FP" in df.columns
    has_FN = "FN" in df.columns

    for prompt_text, grp in df.groupby("prompt"):
        entry = {
            "prompt": prompt_text,
            "n_examples": int(len(grp)),
        }
        if has_correct:
            entry["accuracy"] = float(grp["correct"].mean())
        if has_TP:
            entry["TP_sum"] = int(grp["TP"].sum())
            entry["TP_mean"] = float(grp["TP"].mean())
        if has_FP:
            entry["FP_sum"] = int(grp["FP"].sum())
            entry["FP_mean"] = float(grp["FP"].mean())
        if has_FN:
            entry["FN_sum"] = int(grp["FN"].sum())
            entry["FN_mean"] = float(grp["FN"].mean())

        prompt_rows.append(entry)

    df_prompts = pd.DataFrame(prompt_rows)
    df_prompts.to_csv(out / "prompt_sensitivity.csv", index=False)
    with open(out / "prompt_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(prompt_rows, f, indent=2, ensure_ascii=False)

    # --------- 6) raw model responses (for debugging) ----------
    # JSONL: one line per (image, prompt)
    responses_path = out / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as f:
        for r in per_case:
            obj = {
                "image": r.image,
                "prompt_family": r.prompt_family,
                "prompt": r.prompt,
                "raw_text": r.raw_text,
                # optional stringified gt/pred for quick inspection
                "gt": str(r.gt),
                "pred": str(r.pred),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Optional: CSV version (easier to scan)
    resp_rows = []
    for r in per_case:
        resp_rows.append(
            {
                "image": r.image,
                "prompt_family": r.prompt_family,
                "prompt": r.prompt,
                "raw_text": r.raw_text,
            }
        )
    df_resp = pd.DataFrame(resp_rows)
    df_resp.to_csv(out / "responses.csv", index=False)


if __name__ == "__main__":
    main()
