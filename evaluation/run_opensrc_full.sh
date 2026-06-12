#!/bin/bash
# Run a single SAGE/mem0 experiment on the full LOCOMO dataset with an Ollama backend
# (representative open-weight Table-3 run; default model qwen2.5:3b).
#
# Usage:
#   run_opensrc_full.sh <tech> [ollama_model]
#     tech         : mem0 | sage
#     ollama_model : Ollama model tag (default qwen2.5:3b)
#
# run_locomo_benchmark.py appends a timestamp + PID to the qdrant/history paths, so
# concurrent runs do not collide. Point OLLAMA_HOST at your server (default 127.0.0.1:11434).
set -euo pipefail

TECH="${1:?usage: run_opensrc_full.sh <mem0|sage> [ollama_model]}"
OLLAMA_MODEL="${2:-qwen2.5:3b}"
case "$TECH" in mem0|sage) ;; *) echo "bad technique: $TECH (want mem0|sage)" >&2; exit 2;; esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ALT_PY="${ALT_PY:-python}"
cd "$REPO_ROOT"

# Ollama server tuning (the speed recipe used for the paper's open-weight runs). These
# are read by `ollama serve`, so they only take effect for a server launched from this
# environment; if you point OLLAMA_HOST at a separately-started server, set them there.
# Override any of them via the environment.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-true}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"

MODEL_SLUG="${OLLAMA_MODEL//:/-}"
DATASET_PATH="$REPO_ROOT/evaluation/dataset/locomo10.json"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-$REPO_ROOT/results_full}"
QDRANT_PATH="$REPO_ROOT/data/${TECH}_${MODEL_SLUG}_vmf_qdrant_persist_full"
HISTORY_DB_PATH="$REPO_ROOT/data/${TECH}_${MODEL_SLUG}_vmf_history_persist_full.db"
ACTION_STATS_FILE="$OUTPUT_FOLDER/${TECH}_${MODEL_SLUG}_vmf_action_stats_eval_full.json"

echo "Running ${TECH} experiment (${OLLAMA_MODEL}) on full locomo10..."
mkdir -p "$OUTPUT_FOLDER"

"$ALT_PY" evaluation/run_locomo_benchmark.py \
  --dataset_path "$DATASET_PATH" \
  --technique_type "$TECH" \
  --mem0_backend ollama \
  --ollama_base_url "http://${OLLAMA_HOST:-127.0.0.1:11434}" \
  --llm_model "$OLLAMA_MODEL" \
  --infer_add \
  --batch_size 8 \
  --qdrant_path "$QDRANT_PATH" \
  --max_workers 4 \
  --history_db_path "$HISTORY_DB_PATH" \
  --method full \
  --output_folder "$OUTPUT_FOLDER" \
  --action_stats_file "$ACTION_STATS_FILE"

echo "Finished ${TECH} experiment (${OLLAMA_MODEL}) on full locomo10!"
