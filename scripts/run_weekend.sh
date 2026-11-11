#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resumable weekend study launcher.
#
# Safe to run as many times as you like: it RESUMES: finished trials are
# skipped, only unfinished ones run. `caffeinate -i` keeps the Mac awake while
# it works (closing the lid still suspends it; see RUNBOOK_WEEKEND.md for the
# detached, lid-safe form).
#
# Usage (from anywhere):
#   bash scripts/run_weekend.sh                 # all blocks
#   bash scripts/run_weekend.sh A,B_policy,C_policy   # just the error-bar runs first
#
# Recommended first pass (the error bars the supervisor asked for):
#   bash scripts/run_weekend.sh A,B_policy,C_policy
# Then, if time allows (enables the block-level paired tests):
#   bash scripts/run_weekend.sh B_traj,C_traj
# ---------------------------------------------------------------------------
set -u

# cd to 03_Prototype (this script lives in 03_Prototype/scripts/)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || { echo "cannot cd to $HERE"; exit 1; }

# activate the venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[weekend] WARNING: .venv not found — using the current python. See RUNBOOK.md step 2."
fi

mkdir -p results/seed_study
export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on CPU; avoids MPS operator gaps

BLOCKS="${1:-A,B_policy,B_traj,C_policy,C_traj}"
echo "[weekend] $(date)  blocks=$BLOCKS"
echo "[weekend] tip: run 'python -m scripts.precache' first, on a stable connection."

# keep awake while running; stream output to console and the log
caffeinate -i python -m scripts.run_seed_study --blocks "$BLOCKS" 2>&1 | tee -a results/seed_study/run.log

# refresh the statistics after whatever completed (safe even if partial)
python -m scripts.analysis.stats_report 2>&1 | tee -a results/seed_study/run.log

echo "[weekend] pass complete $(date). Re-run this script any time to resume/extend."
