#!/bin/bash
# Score full-LOCOMO result sets with the gpt-4o-mini LLM-as-judge.
# Run after the OpenAI add/search jobs finish. Scores whichever result dirs exist.
#
# Usage:
#   score_openai.sh [variant]
#     variant : full (default) | bs8
#               full -> results_openai_full/
#               bs8  -> results_openai_full_bs8/
set -euo pipefail

VARIANT="${1:-full}"
case "$VARIANT" in
  full) RESULTS_SUBDIR="results_openai_full" ;;
  bs8)  RESULTS_SUBDIR="results_openai_full_bs8" ;;
  *) echo "bad variant: $VARIANT (want full|bs8)" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ALT_PY="${ALT_PY:-python}"
export MEM0_TELEMETRY=False
cd "$REPO_ROOT/evaluation"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' "$REPO_ROOT/.env" | cut -d= -f2-)
fi

for DIR in "$REPO_ROOT"/"$RESULTS_SUBDIR"/*/; do
  [ -d "$DIR" ] || continue
  VMF=$(ls -t "$DIR"/*_vmf_*.json 2>/dev/null | grep -v "_search_usage" | head -1) || true
  if [ -z "${VMF:-}" ]; then
    echo "skip (no vmf answers yet): $DIR"
    continue
  fi
  OUT="$DIR/eval_metrics_judge4omini.json"
  echo "=== scoring $(basename "$DIR") ==="
  "$ALT_PY" evals.py \
    --input_file "$VMF" \
    --output_file "$OUT" \
    --judge_backend openai \
    --max_workers 10
done
echo "=== ${VARIANT} scoring complete ==="
