# Visual Search on 10×10 Patches with Qwen-2.5-VL

This repo evaluates a visual search task on synthetic 280×280 images arranged
as a 10×10 grid of 28×28 patches. We use Qwen-2.5-VL for answering prompts and
dumping attention to analyze heads/layers.

## What came from where?
- All code in this project is **original** unless noted.
- The overall design (runner + attention analysis) is **inspired by**
  "MLLMs Know Where to Look" but **no code is copied**.
- If you later paste in any code from that repo, keep their MIT license header.

## Data format (already provided by you)
We expect `data/annotations.jsonl` with one JSON per line:
```json
{
  "image": "images/img_00001.png",
  "grid": 10,
  "patch": 28,
  "objects": [
    {"shape":"circle","color":"green","r":3,"c":7,"bbox":[0,0,28,28]}
  ]
}
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: cache the model ahead of time
python - <<'PY'
from transformers import AutoProcessor, AutoModelForCausalLM
m = "Qwen/Qwen2.5-VL-7B-Instruct"  # choose a size that fits your GPU
AutoProcessor.from_pretrained(m, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)
print("cached")
PY
```

## Quick run (stage 2 → stage 4)
```bash
bash scripts/run_visual_search.sh
```

This will:
1) Evaluate FP/FN/precision/recall and prompt sensitivity for a target query.
2) Save attention caches (if enabled) and per-case heatmaps (TP/FN first).
3) Run stats to rank heads/layers by TP vs FN differences (q-values + effect sizes).

## Outputs
- `reports/per_case.csv`
- `reports/summary.json`
- `reports/prompt_sensitivity.csv`
- `reports/attn_cache/*.npz`
- `reports/figs/*.png`
- `reports/stats/per_head_features.parquet`
- `reports/stats/head_rankings.csv`

## Notes
- Attention dumping relies on HF support for attentions or custom hooks.
- We aggregate raw attention to the 10×10 grid.
- If you change the grid/patch size in your dataset, adjust the analysis accordingly.
