#!/usr/bin/env bash
set -euo pipefail

ANN="${ANN:-data/annotations.jsonl}"
OUT="${OUT:-reports}"
FAMILY="${FAMILY:-en_find_green_circle}"
QUERY='{"color":"green","shape":"circle"}'
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
SAVE_ATTN="${SAVE_ATTN:-1}"

echo "== Stage 2: eval & prompt sensitivity =="
python -m src.run_eval   --ann "$ANN"   --out "$OUT"   --family "$FAMILY"   --query "$QUERY"   --model_id "$MODEL_ID"   --save_attention "$SAVE_ATTN"

echo "== Stage 4: per-head stats (TP vs FN) =="
python -m src.run_stats   --attn_dir "$OUT/attn_cache"   --ann "$ANN"   --query "$QUERY"   --out "$OUT/stats"

echo "Done. See $OUT/"
