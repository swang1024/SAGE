#!/bin/bash
# Run a single SAGE/mem0 ablation on the OpenAI backend.
#
# Usage:
#   run_abla_openai.sh <tech> <model> <split> [batch_size]
#     tech       : mem0 | sage
#     model      : gpt-4o-mini | gpt-4o | both   (both = run gpt-4o-mini then gpt-4o)
#     split      : smoke | 20perc | full
#     batch_size : integer (default 2). Pass 8 to reproduce the bs8 extraction baseline;
#                  output paths get a _bs8 suffix so they don't collide with bs2 results.
#
# Splits:
#   smoke  -> dataset/locomo10_smoke.json        (1% QA),  max_workers 4
#   20perc -> dataset/locomo10_smoke_20perc.json (all 10 convs, ~20% QA), max_workers 10
#   full   -> dataset/locomo10.json              (all 10 convs, ALL QA),  max_workers 10
#
# SAGE runs with compaction disabled in code (MemoryCompactor.should_compact -> False
# in mem0/memory/novelty_gate.py) and the v3 lossless+precision-aware merge prompt;
# the tau_0/tau_min/density_lambda hyperparameters are hardcoded in mem0/memory/main.py.
# Embeddings: text-embedding-3-small (1536-dim). Requires OPENAI_API_KEY (read from the
# environment, or from $REPO_ROOT/.env as a fallback).
set -euo pipefail

TECH="${1:?usage: run_abla_openai.sh <mem0|sage> <gpt-4o-mini|gpt-4o|both> <smoke|20perc|full> [batch_size]}"
MODEL_ARG="${2:?usage: run_abla_openai.sh <mem0|sage> <gpt-4o-mini|gpt-4o|both> <smoke|20perc|full> [batch_size]}"
SPLIT="${3:?usage: run_abla_openai.sh <mem0|sage> <gpt-4o-mini|gpt-4o|both> <smoke|20perc|full> [batch_size]}"
BATCH_SIZE="${4:-2}"

case "$TECH" in mem0|sage) ;; *) echo "bad technique: $TECH (want mem0|sage)" >&2; exit 2;; esac
case "$MODEL_ARG" in gpt-4o-mini|gpt-4o|both) ;; *) echo "bad model: $MODEL_ARG" >&2; exit 2;; esac
case "$SPLIT" in smoke|20perc|full) ;; *) echo "bad split: $SPLIT" >&2; exit 2;; esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ALT_PY="${ALT_PY:-python}"
cd "$REPO_ROOT"

# OPENAI_API_KEY: prefer the environment, fall back to the repo .env.
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' "$REPO_ROOT/.env" | cut -d= -f2-)
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set and not found in $REPO_ROOT/.env. Export it first." >&2
  exit 1
fi

case "$SPLIT" in
  smoke)  DATASET="$REPO_ROOT/evaluation/dataset/locomo10_smoke.json";        MAX_WORKERS=4  ;;
  20perc) DATASET="$REPO_ROOT/evaluation/dataset/locomo10_smoke_20perc.json"; MAX_WORKERS=10 ;;
  full)   DATASET="$REPO_ROOT/evaluation/dataset/locomo10.json";              MAX_WORKERS=10 ;;
esac

EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIMS=1536
SUFFIX=""
[ "$BATCH_SIZE" != "2" ] && SUFFIX="_bs${BATCH_SIZE}"

run_one() {
  local model="$1"
  local model_slug="${model//./-}"
  local tag="${TECH}_${model}"
  local out="$REPO_ROOT/results_openai_${SPLIT}${SUFFIX}/${tag}"
  local qdrant="$REPO_ROOT/data/${tag}_openai_qdrant_${SPLIT}${SUFFIX}"
  local hist="$REPO_ROOT/data/${tag}_openai_history_${SPLIT}${SUFFIX}.db"
  local stats="$out/${tag}_openai_action_stats_${SPLIT}${SUFFIX}.json"
  local collection="${TECH}_${model_slug}_openai_${SPLIT}${SUFFIX}"

  # Start from a clean store so the build is not contaminated by an earlier run.
  rm -rf "$qdrant" "$hist"
  mkdir -p "$out"

  echo "=== Running ${TECH} (openai) ${model} on ${SPLIT} (bs=${BATCH_SIZE}) -> ${out} ==="
  "$ALT_PY" evaluation/run_locomo_benchmark.py \
    --dataset_path "$DATASET" \
    --technique_type "$TECH" \
    --mem0_backend openai \
    --llm_model "$model" \
    --embedding_model "$EMBEDDING_MODEL" \
    --embedding_dims "$EMBEDDING_DIMS" \
    --infer_add \
    --batch_size "$BATCH_SIZE" \
    --max_workers "$MAX_WORKERS" \
    --top_k 30 \
    --qdrant_path "$qdrant" \
    --history_db_path "$hist" \
    --collection_name "$collection" \
    --method full \
    --output_folder "$out" \
    --action_stats_file "$stats"
  echo "=== Finished ${TECH} ${model} (${SPLIT}) -> ${out} ==="
}

if [ "$MODEL_ARG" = "both" ]; then
  run_one "gpt-4o-mini"
  run_one "gpt-4o"
else
  run_one "$MODEL_ARG"
fi

echo "All ${TECH} openai ${SPLIT} runs finished."
