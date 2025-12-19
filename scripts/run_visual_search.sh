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


MODEL_ID="$Qwen_ID"


# Task name must match src/tasks/*: coords, exists, ...
TASK="${TASK:-exists}"

# Whether to dump attention caches and figs (0 or 1)
SAVE_ATTN="${SAVE_ATTN:-1}"

# Stats grouping (used by run_stats; defaults assume 'correct' column in per_case.csv)
GROUP_METRIC="${GROUP_METRIC:-correct}"
POS_LABEL="${POS_LABEL:-True}"
NEG_LABEL="${NEG_LABEL:-False}"

# -----------------------------
# Stage 2: evaluation
# -----------------------------
echo "== Stage 2: eval & prompt sensitivity =="
python3 -m src.run_eval \
  --ann "$ANN" \
  --out "$OUT" \
  --family "$FAMILY" \
  --query "$QUERY" \
  --model_id "$MODEL_ID" \
  --task "$TASK" \
  --save_attention "$SAVE_ATTN"

# -----------------------------
# Stage 4: head-level stats
# -----------------------------
echo "== Stage 4: per-head stats (grouped by $GROUP_METRIC) =="
python3 -m src.run_stats \
  --attn_dir "$OUT/attn_cache" \
  --out "$OUT/stats" \
  --per_case "$OUT/per_case.csv" \
  --group_metric "$GROUP_METRIC" \
  --pos_label "$POS_LABEL" \
  --neg_label "$NEG_LABEL"

echo "Done. See $OUT/ and $OUT/stats/"
