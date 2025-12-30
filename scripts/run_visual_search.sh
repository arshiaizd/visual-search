#!/usr/bin/env bash
set -euo pipefail

if [ -d "reports" ]; then
    echo "Removing existing report directory..."
    rm -r reports
    echo "Report directory removed successfully."
else
    echo "Report directory does not exist. No action needed."
fi

# -----------------------------
# Config (overridable via env)
# -----------------------------
ANN="${ANN:-data/annotations.jsonl}"
OUT="${OUT:-reports}"
FAMILY="${FAMILY:-exists_green_circle}"
QUERY=${QUERY:-'{"color":"green","shape":"circle"}'}

export PYTHONPATH="${PYTHONPATH:-}:/home/mmd/Desktop/Arshia/models/InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/"

InternVL_ID="/home/mmd/Desktop/Arshia/models/InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79/"
Qwen_ID="/home/mmd/Desktop/Arshia/models/Qwen2.5-VL-7B-Instruct"

MODEL_ID="${MODEL_ID:-$Qwen_ID}"
TASK="${TASK:-exists}"

# Attention caches are always for ORIGINAL images (run_eval only calls get_attention_cache on img_path).
SAVE_ATTN="${SAVE_ATTN:-0}"

# -----------------------------
# Stage 2: evaluation
# -----------------------------
echo "== Stage 2: eval (variants) =="

# Which variants to run (subset allowed):
#   - normal
#   - predecoder
#   - vitonly
#   - normal,predecoder
#   - normal,predecoder,vitonly
#   - all   (runs normal + all known stitched variants)
# VARIANTS="${VARIANTS:-kv_offline,normal , predecoder}"
VARIANTS="${VARIANTS:-normal}"

python3 -m src.run_eval \
  --ann "$ANN" \
  --out "$OUT" \
  --family "$FAMILY" \
  --query "$QUERY" \
  --model_id "$MODEL_ID" \
  --task "$TASK" \
  --save_attention "$SAVE_ATTN" \
  --variants "$VARIANTS" \
  --only_without 0 \
  --pad 1 \
  --overwrite 1 \
  --base_index 0 \
  --dataset_grid 12 \
  --isolated_naming image_id \
  --isolated_root data/isolated \
  --save_attention_in_stitched 0

echo "Done. See $OUT/ (summary.json, per_case.csv, responses.jsonl)"
echo "Modes will match variants exactly: normal / predecoder / vitonly"
