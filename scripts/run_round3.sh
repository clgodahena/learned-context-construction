#!/bin/bash
# Third-review runs (2026-11-10). One command, fully resumable: re-run it any
# time and finished trials are skipped.
#
#   bash scripts/run_round3.sh            # from the project root, venv active
#
# What it does, in order:
#   1. Airlines signal diagnostic, now with three model families (minutes, CPU).
#   2. Seed-study blocks (detached, sleep-proof):
#        D_ref_delay   reference substrate, label delay 0/50/250/1000  (~30 min, CPU)
#        D_tab_delay   TabPFN, Electricity, delay 250, 9 trials        (~7 h)
#        D_tab_timing  TabPFN, Electricity, delay 0, instrumented, 5 trials (~4 h)
#      Progress: tail -f results/seed_study/run_round3.log
set -e
cd "$(dirname "$0")/.."
mkdir -p results/airlines_check results/seed_study

echo "[round3] 1/2 Airlines diagnostic (logreg / rf / hgb) ..."
python -m scripts.experiments.airlines_signal_check 2>&1 \
  | tee results/airlines_check/airlines_signal_check_3models_output.txt

echo "[round3] 2/2 launching seed-study blocks D_ref_delay,D_tab_delay,D_tab_timing (detached) ..."
python -m scripts.run_seed_study --blocks D_ref_delay,D_tab_delay,D_tab_timing --dry-run | head -3
caffeinate -i nohup python -m scripts.run_seed_study --blocks D_ref_delay,D_tab_delay,D_tab_timing \
    > results/seed_study/run_round3.log 2>&1 &
echo "[round3] launched (pid $!). Follow with:  tail -f results/seed_study/run_round3.log"
echo "[round3] When it prints 'finished this pass', run:  python -m scripts.analysis.delay_report"
