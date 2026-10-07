# Learned-policy investigation (E1–E6)

Execution scripts for the investigation plan in
`04_Experiments/Investigation_Plan_LearnedPolicy.docx`. Each script tests a
hypothesis for why the learned policy underperforms the single-axis baselines,
and writes results to `results/experiments/`.

All scripts run on the **reference classifier by default** (fast — hundreds of
instances/second) since the mechanism under study is the *selection strategy*.
Add `--tabpfn` to confirm a finding on the real substrate. They default to an
offline synthetic drift+imbalance stream (no network); add `--electricity` where
supported to use the real benchmark.

## Run order

Run **E4 first** — it decides the branch:

```bash
python -m scripts.experiments.exp_e4_ceiling            # ceiling test (the decision)
python -m scripts.experiments.exp_e1_convergence        # gap vs stream length
python -m scripts.experiments.exp_e6_stability          # weight-collapse / latency check
```

Then follow the branch E4 reports:

- **A beatable fixed target exists** → optimisation/reward problem:
  ```bash
  python -m scripts.experiments.exp_e2_optimizer        # update_rule / sigma / entropy_reg / lr
  python -m scripts.experiments.exp_e3_reward           # instance vs windowed (metric-aligned) reward
  ```
- **No static advantage** → the advantage is time-varying:
  ```bash
  python -m scripts.experiments.exp_e5_shifting         # shifting-optimal-axis regime (the positive test)
  ```

Confirm the single winning configuration/regime on TabPFN with `--tabpfn`, more
seeds and full-length streams for the results chapter.

## What each script tests

| Script | Hypothesis | Key output |
|---|---|---|
| `exp_e4_ceiling` | Does any fixed weighting beat the single-axis baselines? | verdict + `e4_ceiling_*.csv` |
| `exp_e1_convergence` | Under-convergence on short streams (H1) | gap-vs-length + `e1_convergence_*.csv` |
| `exp_e2_optimizer` | Exploration / hyper-parameters (H3, H4) | ranked configs + `e2_optimizer_*.csv` |
| `exp_e3_reward` | Reward misalignment (H2) | reward comparison + `e3_reward_*.csv` |
| `exp_e5_shifting` | Advantage lives in time-varying regimes (H5) | per-segment table + `e5_shifting.png` |
| `exp_e6_stability` | Weight collapse / latency spikes (H6) | diagnosis + `e6_stability.png` |

## Common options

`--imbalance <r>` (synthetic positive rate), `--seeds 0 1 2`, `--max-steps <n>`,
`--electricity`, `--tabpfn`. See each script's `--help`.

## Code hooks added for these experiments

- `LearnedPolicySelector(reward_mode="windowed")` — a metric-aligned reward (E3).
- `SegmentedStream` in `src/data/drift_injector.py` — compose regimes (E5).
- `run_prequential(..., record_predictions=True)` — per-instance predictions for
  exact per-segment metrics (E5/E6).
