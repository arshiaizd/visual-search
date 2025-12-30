# src/run_eval.py
import argparse
import json
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.dataset_io import (
    read_annotations,
    resolve_dataset_root,
    make_blank_canvas_from_rec,
    isolated_object_paths_in_id_order,
    object_rcs_in_id_order,
)
from tasks import get_task, CaseReport
from src.models import predict_adapter as PA
from src.viz import heatmap, overlay_grid, overlay_patches


HEAD_LAYER_PLOTS = [
    (18, 24), (19, 16), (18, 11), (18, 26), (16, 2),
    (18, 13), (21, 14), (18, 16), (18, 27), (20, 23),
    (22, 9), (17, 0),
]

# If you add more stitched variants later, list them here.
# "normal" is always supported (not stitched).
KNOWN_STITCHED_VARIANTS = ["predecoder", "vitonly", "kv_offline"]


def _safe_prompt_id(prompt: str, idx: int) -> str:
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    return f"p{idx:02d}_{h}"


def _infer_grid(n: int) -> int:
    g = int(round(n ** 0.5))
    return g if g * g == n else g


def _to_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def overlay_heatmap_on_image(
    base_img_path: str,
    hm: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    base = _to_rgba(Image.open(base_img_path))
    hm_rgba = _to_rgba(hm).resize(base.size, resample=Image.BILINEAR)

    a = int(max(0.0, min(1.0, alpha)) * 255)
    r, g, b, _ = hm_rgba.split()
    hm_rgba = Image.merge("RGBA", (r, g, b, Image.new("L", base.size, color=a)))

    return Image.alpha_composite(base, hm_rgba)


def _resolve_path(dataset_root: Path, p: str | Path) -> str:
    """
    Resolve a possibly-relative path against dataset_root.
    If it's already absolute or already exists, keep it.
    """
    pth = Path(p)
    if pth.exists():
        return str(pth)
    if pth.is_absolute():
        return str(pth)
    return str(dataset_root / pth)


def _parse_variants_arg(arg: str) -> List[str]:
    v = (arg or "").strip().lower()
    if not v:
        return ["normal"]
    if v == "all":
        # normal + all known stitched
        return ["normal"] + KNOWN_STITCHED_VARIANTS[:]
    return [x.strip() for x in v.split(",") if x.strip()]


def _run_one_variant(
    *,
    variant: str,
    task,
    rec: dict,
    query: dict,
    img_path: str,
    prompt_family: str,
    prompt: str,
    model_id: str,
    pad: int,
    overwrite: bool,
    isolated_root: Path,
    isolated_naming: str,
    dataset_root: Path,
    base_index: int,
    dataset_grid: int,
) -> CaseReport:
    """
    Run one variant and return a CaseReport with metrics['mode']=variant.

    - variant == "normal": original pipeline on original image
    - else: stitched pipeline using PA.generate_text_stitched(..., qwen_variant=variant)
    """
    variant = (variant or "").lower()

    if variant == "normal":
        report = task.score_example(
            rec=rec,
            query=query,
            image_path=img_path,
            prompt_family=prompt_family,
            prompt=prompt,
            model_id=model_id,
        )
        report.metrics = dict(report.metrics)
        report.metrics["mode"] = "normal"
        return report

    # stitched variants (vitonly, predecoder, etc.)
    blank_pil = make_blank_canvas_from_rec(rec)
    canvas_pil = Image.open(img_path).convert("RGB") 

    iso_paths = list(
        isolated_object_paths_in_id_order(
            rec,
            isolated_root=isolated_root,
            naming=isolated_naming,
        )
    )
    iso_paths = [_resolve_path(dataset_root, p) for p in iso_paths]

    missing = [p for p in iso_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing isolated images:\n"
            + "\n".join(missing[:10])
            + (f"\n... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )

    iso_rcs = object_rcs_in_id_order(rec)

    if variant == "kv_offline":
        raw_text = PA.generate_one_word_kv_offline_patched(
            canvas_image=canvas_pil,
            isolated_images=iso_paths,
            isolated_rcs=iso_rcs,
            prompt=prompt,
            model_id=model_id,
            pad=pad,
            overwrite=overwrite,
            base_index=base_index,      # include only if your PA wrapper exposes it; otherwise remove
            dataset_grid=dataset_grid,
        )
    else:
        raw_text = PA.generate_text_stitched(
            blank_image=blank_pil,
            isolated_images=iso_paths,
            isolated_rcs=iso_rcs,
            prompt=prompt,
            model_id=model_id,
            pad=pad,
            max_new_tokens=128,
            overwrite=overwrite,
            base_index=base_index,
            dataset_grid=dataset_grid,
            qwen_variant=variant,
        )
    # print("\n================ MODEL OUTPUT ================")
    # # print(f"image={rec['image']}  mode={mode_label}")
    # print("prompt:", prompt)
    # print("raw_text:\n", raw_text)
    # print("================================================\n")


    gt = task.compute_gt(rec, query)
    pred = task.parse_prediction(raw_text)
    metrics = task.metrics_for_example(rec, query, gt, pred)

    report = CaseReport(
        image=rec["image"],
        prompt_family=prompt_family,
        prompt=prompt,
        gt=gt,
        pred=pred,
        metrics=metrics,
        raw_text=raw_text,
    )
    report.metrics = dict(report.metrics)
    report.metrics["mode"] = variant
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to annotations.jsonl")
    ap.add_argument("--out", required=True)

    ap.add_argument("--family", required=True, help="Prompt family key")
    ap.add_argument("--query", required=True, help='JSON dict, e.g. {"color":"green","shape":"circle"}')
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--task", default="coords", help="Task name: coords, exists, ...")

    ap.add_argument("--save_attention", type=int, default=1)
    ap.add_argument("--overlay_on_image", type=int, default=1)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--save_heatmap_only", type=int, default=1)

    # ✅ NEW: list of variants to run (no --mode anymore)
    ap.add_argument(
        "--variants",
        type=str,
        default="normal",
        help=(
            "Comma-separated variants to run. Example: 'normal,predecoder,vitonly'. "
            "Use 'all' to run normal + all known stitched variants."
        ),
    )

    ap.add_argument("--isolated_root", default=None)
    ap.add_argument("--isolated_naming", choices=["image_id", "pair_id_without"], default="image_id")
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--overwrite", type=int, default=1)

    ap.add_argument(
        "--only_without",
        type=int,
        default=0,
        help="If 1, evaluate only has_target==False examples. Default 0 evaluates both.",
    )

    ap.add_argument(
        "--save_attention_in_stitched",
        type=int,
        default=0,
        help="If 1, also save attention for stitched variants (attention is for ORIGINAL image).",
    )

    ap.add_argument("--base_index", type=int, default=0, help="Which isolated image provides the background.")
    ap.add_argument("--dataset_grid", type=int, default=12, help="Dataset grid size (e.g. 12 for 12x12).")

    args = ap.parse_args()

    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "attn_cache").mkdir(exist_ok=True)

    variants = _parse_variants_arg(args.variants)

    query = json.loads(args.query)
    ann_all = read_annotations(args.ann)

    if args.only_without:
        ann = [rec for rec in ann_all if not bool(rec.get("has_target", False))]
    else:
        ann = ann_all

    n_pos = sum(1 for rec in ann if bool(rec.get("has_target", False)))
    n_neg = len(ann) - n_pos
    if n_pos == 0:
        print("[WARN] After filtering, dataset has 0 positive (has_target=True) examples.")
        print(f"[WARN] n_examples={len(ann)} (negatives={n_neg}). Precision/Recall will be 0 by definition.")

    dataset_root = resolve_dataset_root(args.ann)
    isolated_root = Path(args.isolated_root) if args.isolated_root else (dataset_root / "isolated")

    task = get_task(args.task)
    prompts = task.get_prompts(args.family)

    per_case: list[CaseReport] = []

    for rec in tqdm(ann, desc="images"):
        img_path = _resolve_path(dataset_root, rec["image"])
        img_stem = Path(img_path).stem
        grid_fallback = int(rec.get("grid", 12))

        for pi, prompt in enumerate(prompts):
            prompt_id = _safe_prompt_id(prompt, pi)

            for variant in variants:
                report = _run_one_variant(
                    variant=variant,
                    task=task,
                    rec=rec,
                    query=query,
                    img_path=img_path,
                    prompt_family=args.family,
                    prompt=prompt,
                    model_id=args.model_id,
                    pad=args.pad,
                    overwrite=bool(args.overwrite),
                    isolated_root=isolated_root,
                    isolated_naming=args.isolated_naming,
                    dataset_root=dataset_root,
                    base_index=args.base_index,
                    dataset_grid=args.dataset_grid,
                )
                per_case.append(report)

                # ---- attention saving (always attention of ORIGINAL image) ----
                if not args.save_attention:
                    continue
                if variant != "normal" and not args.save_attention_in_stitched:
                    continue

                cache = PA.get_attention_cache(img_path, prompt, model_id=args.model_id)
                if cache is None or "attn" not in cache:
                    continue

                attn = cache["attn"]
                if getattr(attn, "ndim", None) != 3:
                    print(f"[WARN] Unexpected attn shape for {img_path}: {getattr(attn, 'shape', None)}")
                    continue

                L, H, N = attn.shape
                grid = _infer_grid(int(N))
                if grid * grid != int(N):
                    grid = grid_fallback

                variant_safe = (variant or "unknown").replace(":", "_").replace("/", "_")
                stem_full = f"{img_stem}_{args.family}_{prompt_id}_{variant_safe}"
                npz_path = out / "attn_cache" / f"{stem_full}.npz"
                np.savez(npz_path, **cache)

                img_fig_dir = out / "figs" / img_stem / prompt_id / variant_safe
                img_fig_dir.mkdir(parents=True, exist_ok=True)

                for (li, hi) in HEAD_LAYER_PLOTS:
                    if li < 0 or li >= L or hi < 0 or hi >= H:
                        continue

                    vec = np.asarray(attn[li, hi]).reshape(-1)
                    hm = heatmap(vec, vmax=0.1, vmin=0)
                    hm = overlay_grid(hm, grid=grid)

                    if args.task == "coords":
                        tp = report.metrics.get("tp", [])
                        fn = report.metrics.get("fn", [])
                        hm = overlay_patches(hm, tp, grid=grid, outline=(0, 255, 0, 255))
                        hm = overlay_patches(hm, fn, grid=grid, outline=(255, 0, 0, 255))

                    if args.save_heatmap_only:
                        hm.save(img_fig_dir / f"L{li}_H{hi}_heatmap.png")

                    if args.overlay_on_image:
                        over = overlay_heatmap_on_image(img_path, hm, alpha=args.overlay_alpha)
                        over.save(img_fig_dir / f"L{li}_H{hi}_overlay.png")

    # --------- summaries ----------
    overall = task.aggregate(per_case)

    summary_by_mode = {}
    for mode_name in sorted(set(r.metrics.get("mode", "unknown") for r in per_case)):
        sub = [r for r in per_case if r.metrics.get("mode") == mode_name]
        summary_by_mode[mode_name] = task.aggregate(sub)

    summary = dict(overall)
    summary["summary_by_mode"] = summary_by_mode
    summary["variants"] = variants
    summary["only_without"] = bool(args.only_without)
    summary["pad"] = args.pad if any(v != "normal" for v in variants) else None
    summary["base_index"] = args.base_index
    summary["dataset_grid"] = args.dataset_grid

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --------- per_case.csv ----------
    rows = []
    for r in per_case:
        row = {"image": r.image, "prompt_family": r.prompt_family, "prompt": r.prompt}
        row.update(r.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_case.csv", index=False)

    # --------- prompt sensitivity ----------
    prompt_rows = []
    has_correct = "correct" in df.columns
    has_TP = "TP" in df.columns
    has_FP = "FP" in df.columns
    has_FN = "FN" in df.columns

    group_cols = ["mode", "prompt"] if "mode" in df.columns else ["prompt"]
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, tuple):
            mode_name, prompt_text = keys
        else:
            mode_name, prompt_text = "unknown", keys

        entry = {"mode": mode_name, "prompt": prompt_text, "n_examples": int(len(grp))}
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

    # --------- raw responses ----------
    responses_path = out / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as f:
        for r in per_case:
            obj = {
                "image": r.image,
                "prompt_family": r.prompt_family,
                "prompt": r.prompt,
                "mode": r.metrics.get("mode", ""),
                "raw_text": r.raw_text,
                "gt": str(r.gt),
                "pred": str(r.pred),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    df_resp = pd.DataFrame(
        [
            {
                "image": r.image,
                "prompt_family": r.prompt_family,
                "prompt": r.prompt,
                "mode": r.metrics.get("mode", ""),
                "raw_text": r.raw_text,
            }
            for r in per_case
        ]
    )
    df_resp.to_csv(out / "responses.csv", index=False)


if __name__ == "__main__":
    main()
