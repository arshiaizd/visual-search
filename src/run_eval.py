# src/run_eval.py
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.dataset_io import read_annotations
from tasks import get_task, CaseReport
from src.models import predict_adapter as PA
from src.viz import heatmap, overlay_grid, overlay_patches


# ---------------------------------------------------------
# Which (layer, head) pairs to visualize as heatmaps.
# Edit this list to select the heads you care about.
# Example: [(0,0), (0,1), (3,5)]
# ---------------------------------------------------------
HEAD_LAYER_PLOTS = [
    (18, 24), (19, 16), (18, 11), (18, 26), (16, 2),
    (18, 13), (21, 14), (18, 16), (18, 27), (20, 23),
    (22, 9), (17, 0),
]


def _safe_prompt_id(prompt: str, idx: int) -> str:
    """Stable short id for each prompt so outputs don't collide."""
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    return f"p{idx:02d}_{h}"


def _infer_grid(n: int) -> int:
    """Infer patch grid from attention length N (expects N == grid^2)."""
    g = int(round(n ** 0.5))
    return g if g * g == n else g  # fall back to rounded g even if imperfect


def _to_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def overlay_heatmap_on_image(
    base_img_path: str,
    hm: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a heatmap (PIL Image) onto the original image.

    - base image is loaded from disk
    - heatmap is resized to base image size
    - heatmap alpha channel is set to `alpha` (0..1)
    - result is RGBA
    """
    base = _to_rgba(Image.open(base_img_path))
    hm_rgba = _to_rgba(hm).resize(base.size, resample=Image.BILINEAR)

    # Replace heatmap alpha channel with a constant alpha (keeps colors intact)
    a = int(max(0.0, min(1.0, alpha)) * 255)
    r, g, b, _ = hm_rgba.split()
    hm_rgba = Image.merge("RGBA", (r, g, b, Image.new("L", base.size, color=a)))

    return Image.alpha_composite(base, hm_rgba)


def _to_bool_pred(x):
    """
    Best-effort conversion of a model prediction to boolean (has_target?).

    Adjust this if your `report.pred` format is different.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if x is None:
        return None

    s = str(x).strip().lower()

    if s in {"true", "yes", "y", "1", "present", "exists"}:
        return True
    if s in {"false", "no", "n", "0", "absent", "none", "not present", "does not exist"}:
        return False

    return None  # unknown / not parseable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to annotations.jsonl")
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

    # NEW: overlay options
    ap.add_argument(
        "--overlay_on_image",
        type=int,
        default=1,
        help="If 1, also save heatmaps overlaid on the original image.",
    )
    ap.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Alpha (0..1) for heatmap overlay on the original image.",
    )
    ap.add_argument(
        "--save_heatmap_only",
        type=int,
        default=1,
        help="If 1, also save the heatmap-only PNG (in addition to overlay).",
    )

    # NEW: print filter
    ap.add_argument(
        "--print_tn_only",
        type=int,
        default=1,
        help="If 1, print attention stats ONLY for TN cases (requires a binary exists-style task).",
    )

    args = ap.parse_args()

    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "attn_cache").mkdir(exist_ok=True)

    query = json.loads(args.query)
    ann = read_annotations(args.ann)

    task = get_task(args.task)
    prompts = task.get_prompts(args.family)

    per_case: list[CaseReport] = []

    # --------- main loop: images × prompts ----------
    for rec in tqdm(ann, desc="images"):
        img_path = rec["image"]
        img_stem = Path(img_path).stem

        # Fallback grid from dataset annotation if present (your generator writes this)
        grid_fallback = int(rec.get("grid", 12))

        for pi, prompt in enumerate(prompts):
            prompt_id = _safe_prompt_id(prompt, pi)

            # 1) evaluation
            report = task.score_example(
                rec=rec,
                query=query,
                image_path=img_path,
                prompt_family=args.family,
                prompt=prompt,
                model_id=args.model_id,
            )
            per_case.append(report)

            # ----- TN detection (meaningful for exists/binary tasks) -----
            # Your generator writes rec["has_target"] = True/False
            has_target_gt = bool(rec.get("has_target", False))

            # best-effort parse model pred into boolean
            pred_has_target = _to_bool_pred(getattr(report, "pred", None))

            is_tn = (has_target_gt is False) and (pred_has_target is False)

            # 2) optional: save attention + figures
            if not args.save_attention:
                continue

            cache = PA.get_attention_cache(img_path, prompt, model_id=args.model_id)
            if cache is None or "attn" not in cache:
                continue

            attn = cache["attn"]  # shape: (L, H, N)
            if getattr(attn, "ndim", None) != 3:
                print(f"[WARN] Unexpected attn shape for {img_path}: {getattr(attn, 'shape', None)}")
                continue

            L, H, N = attn.shape
            grid = _infer_grid(int(N))
            if grid * grid != int(N):
                # if it's not a perfect square, fall back to annotation grid
                grid = grid_fallback

            # save raw cache per image+family+prompt to avoid overwriting across prompts
            stem_full = f"{img_stem}_{args.family}_{prompt_id}"
            npz_path = out / "attn_cache" / f"{stem_full}.npz"
            np.savez(npz_path, **cache)

            # directory for this image+prompt's heatmaps
            img_fig_dir = out / "figs" / img_stem / prompt_id
            img_fig_dir.mkdir(parents=True, exist_ok=True)

            for (li, hi) in HEAD_LAYER_PLOTS:
                if li < 0 or li >= L or hi < 0 or hi >= H:
                    continue

                vec = np.asarray(attn[li, hi]).reshape(-1)  # (N,)

                # ---- PRINT ONLY FOR TN (optional) ----
                # if args.print_tn_only and is_tn:
                #     mx = float(np.max(vec))
                #     p99 = float(np.quantile(vec, 0.99))
                #     sm = float(np.sum(vec))
                #     print(vec.shape)
                #     print(
                #         img_stem, li, hi,
                #         "max", mx,
                #         "p99", p99,
                #         "sum", sm,
                #         "correct", report.metrics.get("correct"),
                #         "gt_has_target", has_target_gt,
                #         "pred_has_target", pred_has_target,
                #     )

                # ---- HEATMAP GENERATION ----
                # NOTE: you are normalizing here because you pass vmin/vmax.
                # Your previous code used vmin=0 and vmax=0.5 which can saturate.
                hm = heatmap(vec, vmax=0.1, vmin=0)
                hm = overlay_grid(hm, grid=grid)

                # only coords task has patch-level tp/fn to overlay
                if args.task == "coords":
                    tp = report.metrics.get("tp", [])
                    fn = report.metrics.get("fn", [])
                    hm = overlay_patches(hm, tp, grid=grid, outline=(0, 255, 0, 255))
                    hm = overlay_patches(hm, fn, grid=grid, outline=(255, 0, 0, 255))

                # Save heatmap-only
                if args.save_heatmap_only:
                    fig_path = img_fig_dir / f"L{li}_H{hi}_heatmap.png"
                    hm.save(fig_path)

                # Save overlay on original image
                if args.overlay_on_image:
                    over = overlay_heatmap_on_image(
                        base_img_path=img_path,
                        hm=hm,
                        alpha=args.overlay_alpha,
                    )
                    over_path = img_fig_dir / f"L{li}_H{hi}_overlay.png"
                    over.save(over_path)

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
        entry = {"prompt": prompt_text, "n_examples": int(len(grp))}
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
    responses_path = out / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as f:
        for r in per_case:
            obj = {
                "image": r.image,
                "prompt_family": r.prompt_family,
                "prompt": r.prompt,
                "raw_text": r.raw_text,
                "gt": str(r.gt),
                "pred": str(r.pred),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    df_resp = pd.DataFrame(
        [{"image": r.image, "prompt_family": r.prompt_family, "prompt": r.prompt, "raw_text": r.raw_text}
         for r in per_case]
    )
    df_resp.to_csv(out / "responses.csv", index=False)


if __name__ == "__main__":
    main()
