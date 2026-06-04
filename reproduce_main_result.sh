#!/bin/bash
# Reproduce the main gpt-4o-mini result (SAGE vs the mem0 baseline) end-to-end, with
# no job scheduler. Runs add+search for both systems at batch_size=8 (the paper
# configuration), scores with the gpt-4o-mini LLM-as-judge, and prints per-category +
# overall accuracy.
#
# Requires OPENAI_API_KEY (in the environment or a repo-root .env). This is a long,
# paid run (~2h ingest+search per system on gpt-4o-mini). For a quick check, run a
# smaller split directly, e.g.:
#   bash evaluation/run_abla_openai.sh sage gpt-4o-mini smoke 8
#
# Usage:
#   bash reproduce_main_result.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
EVAL="$REPO_ROOT/evaluation"
export ALT_PY="${ALT_PY:-python}"

mkdir -p "$EVAL/logs"

# 1. Run the mem0 baseline and SAGE (each does add + search) at batch_size=8.
#    Outputs go to results_openai_full_bs8/{mem0,sage}_gpt-4o-mini/.
bash "$EVAL/run_abla_openai.sh" mem0 gpt-4o-mini full 8
bash "$EVAL/run_abla_openai.sh" sage gpt-4o-mini full 8

# 2. Score both runs with the gpt-4o-mini LLM-as-judge.
bash "$EVAL/score_openai.sh" bs8

# 3. Print per-category + overall accuracy (BLEU / F1 / LLM-judge).
"$ALT_PY" "$EVAL/generate_scores.py" results_openai_full_bs8 \
  --runs mem0_gpt-4o-mini/eval_metrics_judge4omini.json \
         sage_gpt-4o-mini/eval_metrics_judge4omini.json

echo
echo "Done. To regenerate the paper's LaTeX tables from these runs, see"
echo "evaluation/README.md -> 'Step-by-step ... step 5'."
