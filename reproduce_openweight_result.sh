#!/bin/bash
# Reproduce the open-weight (Table 3) comparison end-to-end: the mem0 baseline AND
# SAGE on FULL LoCoMo, with a qwen2.5:3b backbone, then score every answer file with a
# llama3.1:8b LLM-as-judge. Run it on any machine that has a GPU and `ollama` installed.
# This is the OpenAI-free twin of reproduce_4omini_result.sh.
#
# What it does:
#   1. Ensures an ollama server is up (reuses $OLLAMA_HOST if one is already running,
#      otherwise starts one on 127.0.0.1:11434 and stops it on exit).
#   2. Pulls qwen2.5:3b, llama3.1:8b, nomic-embed-text (no-ops if already cached).
#   3. Runs the mem0 baseline and SAGE (each does add + search) -> results_full/.
#   4. Scores every answer file with the llama3.1:8b judge via evals.py
#      -> *_eval_metrics_llamajudge.json.
#
# This is a long local run (on the paper's hardware, ~3h ingest + ~1.25h search per
# system, plus judge scoring).
#
# Usage:
#   bash reproduce_openweight_result.sh
# Optional overrides (environment):
#   OLLAMA_HOST   ollama server address (default 127.0.0.1:11434). If a server is
#                 already listening there, it is reused and left running.
#   OLLAMA_MODEL  backbone tag      (default qwen2.5:3b)
#   JUDGE_MODEL   judge model tag   (default llama3.1:8b)
#   ALT_PY        python to use     (default python; e.g. point at a conda env's python)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"
export ALT_PY="${ALT_PY:-python}"
export MEM0_TELEMETRY=False
mkdir -p "$REPO_ROOT/evaluation/logs"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
JUDGE_MODEL="${JUDGE_MODEL:-llama3.1:8b}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

command -v ollama >/dev/null 2>&1 || {
  echo "ollama not found on PATH. Install it (https://ollama.com) or point OLLAMA_HOST at a running server." >&2
  exit 1
}

# --- Ensure an ollama server is up. Reuse an already-running one; otherwise start our
#     own and stop it on exit so we don't leave a stray server behind. ---
if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Reusing ollama server already listening at ${OLLAMA_HOST}"
else
  echo "Starting ollama server at ${OLLAMA_HOST}..."
  # Speed recipe used for the paper's open-weight runs (read by `ollama serve`).
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
  export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-true}"
  ollama serve > "evaluation/logs/ollama_local.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 60); do
    curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 || {
    echo "ollama server did not come up at ${OLLAMA_HOST} (see evaluation/logs/ollama_local.log)" >&2
    exit 1
  }
fi
echo "ollama up at ${OLLAMA_HOST}"

# Verify/cache the models (non-fatal if offline and already cached).
ollama pull "$OLLAMA_MODEL"  || true
ollama pull "$JUDGE_MODEL"   || true
ollama pull nomic-embed-text || true

# --- 1. Run mem0 baseline and SAGE: full LoCoMo, qwen2.5:3b backbone (add + search) ---
#     run_abla_qwen_full.sh reads OLLAMA_HOST and writes to results_full/.
bash evaluation/run_abla_qwen_full.sh mem0 "$OLLAMA_MODEL"
bash evaluation/run_abla_qwen_full.sh sage "$OLLAMA_MODEL"

# --- 2. Score every answer file with the llama3.1:8b judge (judge reads OLLAMA_HOST) ---
MODEL_SLUG="${OLLAMA_MODEL}"   # filenames use the raw tag, e.g. qwen2.5:3b
METRIC_FILES=()                # collect the metrics produced this run for step 3
for TECH in mem0 sage; do
  for INPUT in results_full/${TECH}_${MODEL_SLUG}_vmf_[0-9]*.json; do
    [ -e "$INPUT" ] || continue
    case "$INPUT" in *_eval_metrics*.json|*_search_usage.json) continue;; esac
    OUT="${INPUT%.json}_eval_metrics_llamajudge.json"
    echo "=== scoring ${TECH}: $(basename "$INPUT") with ${JUDGE_MODEL} ==="
    "$ALT_PY" evaluation/evals.py \
      --input_file "$INPUT" \
      --output_file "$OUT" \
      --judge_backend ollama \
      --judge_model "$JUDGE_MODEL" \
      --max_workers 4
    METRIC_FILES+=("$OUT")
  done
done

# --- 3. Print per-category + overall accuracy (BLEU / F1 / LLM-judge), like the
#        OpenAI reproduce_main_result.sh. Passes the exact metrics files produced above
#        (relative to results_full/) so stale runs in the folder are not mixed in. ---
if [ "${#METRIC_FILES[@]}" -gt 0 ]; then
  RUN_ARGS=()
  for f in "${METRIC_FILES[@]}"; do RUN_ARGS+=("$(basename "$f")"); done
  "$ALT_PY" evaluation/generate_scores.py results_full --runs "${RUN_ARGS[@]}"
fi

echo "=== ALL DONE: qwen2.5:3b backbone + llama3.1:8b judge, full LoCoMo (mem0 + sage) ==="
echo "Answers + action_stats + *_eval_metrics_llamajudge.json are in results_full/"
