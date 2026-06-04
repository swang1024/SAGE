#!/bin/bash
# Representative SLURM wrapper: runs SAGE on the 20% LoCoMo split, OpenAI/gpt-4o-mini.
# Submit from the repo root: `sbatch submit_slurm_sage_openai_4omini_20perc.sh`.
#
# EDIT FOR YOUR CLUSTER: set the account/partition/qos placeholders below. This job
# uses the OpenAI API backend, so it needs no GPU. Not on SLURM? Use the no-scheduler
# reproduce_main_result.sh instead.
#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=evaluation/logs/slurm_sage_openai_4omini_20perc_%j.out
#SBATCH --error=evaluation/logs/slurm_sage_openai_4omini_20perc_%j.err

set -euo pipefail

# Auto-detect the repo root from this script's location (works under sbatch --chdir too).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
export ALT_PY="${ALT_PY:-python}"

mkdir -p "$REPO_ROOT/evaluation/logs"

# Load OPENAI_API_KEY from the repo .env (python load_dotenv also reads it, but the
# run script guards on the env var being set).
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' "$REPO_ROOT/.env" | cut -d= -f2-)
fi

echo "Starting sage (openai/gpt-4o-mini) on 20perc..."
bash "$REPO_ROOT/evaluation/run_abla_openai.sh" sage gpt-4o-mini 20perc
echo "Done!"
