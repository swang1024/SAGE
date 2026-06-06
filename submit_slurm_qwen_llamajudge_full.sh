#!/bin/bash
# Reproduce the open-weight (Table 3) comparison: SAGE vs the mem0 baseline on FULL
# LoCoMo with a qwen2.5:3b backbone, scored by a llama3.1:8b LLM-as-judge. Runs on a
# GPU node with a per-job ollama server (the AgenticMemory recipe). Submit from the
# repo root: `sbatch submit_slurm_qwen_llamajudge_full.sh`.
# --- Cluster-specific SBATCH directives: edit these for your scheduler ---
#SBATCH -p <PARTITION>
#SBATCH --account <ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-gpu=100G
#SBATCH -t 24:00:00
#SBATCH --job-name=sage_qwen25_3b_llamajudge_full
#SBATCH --output=evaluation/logs/qwen_llamajudge_full_%j.log
#SBATCH --error=evaluation/logs/qwen_llamajudge_full_%j.err

set -euo pipefail

# --- Conda: set CONDA_HOME / CONDA_ENV to match your install, or override at submit ---
#     e.g. `CONDA_HOME=$HOME/miniconda CONDA_ENV=myenv sbatch submit_slurm_qwen_llamajudge_full.sh`
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda}"
CONDA_ENV="${CONDA_ENV:-myenv}"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Repo root = the submit directory (run `sbatch` from the repo root, as documented above).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"
export ALT_PY="${CONDA_PREFIX}/bin/python"
export MEM0_TELEMETRY=False
mkdir -p evaluation/logs

OLLAMA_MODEL=qwen2.5:3b
JUDGE_MODEL=llama3.1:8b

# --- Start a per-job ollama server on a unique high port (avoids cross-job collisions) ---
OLLAMA_PORT=$((30000 + (SLURM_JOB_ID % 5000)))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
ollama serve > "evaluation/logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
echo "ollama up at ${OLLAMA_HOST}"

# Models are already cached under $OLLAMA_MODELS; pull verifies/no-ops (non-fatal offline).
ollama pull "$OLLAMA_MODEL"       || true
ollama pull "$JUDGE_MODEL"        || true
ollama pull nomic-embed-text      || true

# --- 1. Run mem0 baseline and SAGE: full LoCoMo, qwen2.5:3b backbone (add + search) ---
#     run_abla_qwen_full.sh reads OLLAMA_HOST and writes to results_full/.
bash evaluation/run_abla_qwen_full.sh mem0 "$OLLAMA_MODEL"
bash evaluation/run_abla_qwen_full.sh sage "$OLLAMA_MODEL"

# --- 2. Score every answer file with the llama3.1:8b judge (judge reads OLLAMA_HOST) ---
MODEL_SLUG="${OLLAMA_MODEL}"   # filenames use the raw tag, e.g. qwen2.5:3b
for TECH in mem0 sage; do
  for INPUT in results_full/${TECH}_${MODEL_SLUG}_vmf_[0-9]*.json; do
    [ -e "$INPUT" ] || continue
    case "$INPUT" in *_eval_metrics.json|*_search_usage.json) continue;; esac
    OUT="${INPUT%.json}_eval_metrics_llamajudge.json"
    echo "=== scoring ${TECH}: $(basename "$INPUT") with ${JUDGE_MODEL} ==="
    "$ALT_PY" evaluation/evals.py \
      --input_file "$INPUT" \
      --output_file "$OUT" \
      --judge_backend ollama \
      --judge_model "$JUDGE_MODEL" \
      --max_workers 4
  done
done

echo "=== ALL DONE: qwen2.5:3b backbone + llama3.1:8b judge, full LoCoMo (mem0 + sage) ==="
echo "Answers + action_stats + *_eval_metrics_llamajudge.json are in results_full/"
