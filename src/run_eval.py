import argparse, json, csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.dataset_io import read_annotations, gt_patch_set
from src.eval_metrics import confusion_per_image, macro_scores
from src.prompts import PROMPT_FAMILIES
from src.viz import heatmap_10x10, overlay_grid, overlay_patches
from src import predict_adapter_qwen as Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--save_attention", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"attn_cache").mkdir(exist_ok=True, parents=True)
    (out/"figs").mkdir(exist_ok=True, parents=True)

    ann = read_annotations(args.ann)
    prompts = PROMPT_FAMILIES[args.family]

    rows, confs = [], []
    for rec in tqdm(ann, desc="images"):
        grid = rec["grid"]; img_path = rec["image"]
        gt = gt_patch_set(rec, json.loads(args.query))
        for prompt in prompts:
            pr = Q.predict_patches(img_path, prompt, model_id=args.model_id)
            conf = confusion_per_image(gt, pr, grid)
            confs.append(conf)
            rows.append({
                "image": img_path,
                "prompt_family": args.family,
                "prompt": prompt,
                "TP": len(conf["tp"]), "FP": len(conf["fp"]), "FN": len(conf["fn"]),
                "n_gt": len(gt)
            })
            if args.save_attention:
                cache = Q.get_attention_cache(img_path, prompt, model_id=args.model_id)
                if cache is not None and "attn" in cache:
                    cache["meta"]["pred_patches"] = list(map(list, pr))
                    stem = Path(img_path).stem
                    npz_path = out/"attn_cache"/f"{stem}_{hash(prompt)%10**8}.npz"
                    np.savez(npz_path, **cache)
                    mean_over_heads = cache["attn"].mean(axis=1)  # (L,100)
                    hm = heatmap_10x10(mean_over_heads[18])
                    overlay_grid(hm, grid=10)
                    hm = overlay_patches(hm, conf["tp"], outline=(0,255,0,255))
                    hm = overlay_patches(hm, conf["fn"], outline=(255,140,0,255))
                    hm.save(out/"figs"/f"{stem}_{hash(prompt)%10**8}_L0.png")

    with open(out/"per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agg = macro_scores(confs)
    with open(out/"summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    df = pd.DataFrame(rows)
    sens = df.groupby("prompt").agg({"TP":"sum","FP":"sum","FN":"sum","n_gt":"sum"}).reset_index()
    sens["precision"] = sens["TP"]/(sens["TP"]+sens["FP"]).replace({0:0})
    sens["recall"]    = sens["TP"]/(sens["TP"]+sens["FN"]).replace({0:0})
    sens["f1"]        = (2*sens["precision"]*sens["recall"]/(sens["precision"]+sens["recall"])).fillna(0)
    sens.to_csv(out/"prompt_sensitivity.csv", index=False)

if __name__ == "__main__":
    main()
