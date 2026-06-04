#!/bin/bash
# Compute eval metrics (evals.py) for every full-dataset answer file produced by
# run_abla_qwen_full.sh for a given technique.
#
# Usage:
#   run_abla_qwen_full_eval.sh <tech> [ollama_model]
#     tech         : mem0 | sage
#     ollama_model : Ollama model tag matching the run (default qwen2.5:3b)
set -euo pipefail

TECH="${1:?usage: run_abla_qwen_full_eval.sh <mem0|sage> [ollama_model]}"
OLLAMA_MODEL="${2:-qwen2.5:3b}"
case "$TECH" in mem0|sage) ;; *) echo "bad technique: $TECH (want mem0|sage)" >&2; exit 2;; esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ALT_PY="${ALT_PY:-python}"
cd "$REPO_ROOT"

# Ollama server tuning (the speed recipe used for the paper's open-weight runs). Only
# relevant when scoring with an Ollama LLM-judge (evals.py --judge_backend ollama); read
# by `ollama serve`, so set them where the server is launched. Override via the environment.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-true}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"

OUTPUT_FOLDER="$REPO_ROOT/results_full"
echo "Running ${TECH} eval metrics for ${OLLAMA_MODEL} runs in ${OUTPUT_FOLDER}..."
mkdir -p "$OUTPUT_FOLDER"

found=0
for INPUT_FILE in "$OUTPUT_FOLDER"/${TECH}_${OLLAMA_MODEL}_vmf_[0-9]*.json; do
  [ -e "$INPUT_FILE" ] || continue
  [[ "$INPUT_FILE" == *_eval_metrics.json ]] && continue
  OUTPUT_FILE="${INPUT_FILE%.json}_eval_metrics.json"
  RUN=$(basename "$INPUT_FILE" .json)
  echo "Evaluating run ${RUN}..."
  "$ALT_PY" evaluation/evals.py \
      --input_file "$INPUT_FILE" \
      --output_file "$OUTPUT_FILE"
  echo "Done: ${OUTPUT_FILE}"
  found=1
done

if [ "$found" -eq 0 ]; then
  echo "No ${TECH} ${OLLAMA_MODEL} output files found in ${OUTPUT_FOLDER}" >&2
  exit 1
fi

echo "Finished all ${TECH} eval metrics!"
