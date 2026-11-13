# Experiments Runbook — the learned-policy investigation (E1–E6)

How to reproduce the diagnostic investigation of the learned policy. Run everything from this folder
with the venv active. Diagnostics use the fast reference classifier (minutes each); the confirmation
step uses `--tabpfn` (several hours). All outputs land in `results/experiments/`.

> **Before any `--tabpfn` run: run the probe (step P, ~3 min).** It proves TabPFN really loads and
> produces predictions *distinct* from the reference classifier in your environment, so you never
> spend hours only to discover otherwise. TabPFN inference is **deterministic at a fixed seed** —
> re-running `--tabpfn --seed 0` reproduces the same CSV/PNG byte-for-byte. That is expected and is a
> sign of a correct, reproducible pipeline, not a problem. Every E5 run also writes a `.meta.json`
> provenance sidecar and refuses to let a result identical to the reference pass silently.

---

## Sequence

Diagnostics on the reference substrate (fast, CPU):

```
1.  python -m tests.test_components                                  # degeneracy proof + components
2.  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.30
3.  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.15
4.  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.05
5.  python -m scripts.experiments.exp_e4_ceiling --electricity
6.  python -m scripts.experiments.exp_e1_convergence
7.  python -m scripts.experiments.exp_e6_stability
8.  python -m scripts.experiments.exp_e2_optimizer --imbalance 0.15
9.  python -m scripts.experiments.exp_e3_reward   --imbalance 0.15
10. python -m scripts.experiments.exp_e5_shifting                    # reference -> e5_shifting_ref.csv
```

Confirmation on the real substrate (TabPFN):

```
P.  python -m scripts.experiments.probe_tabpfn                       # pre-flight, ~3 min — must PASS
11. python -m scripts.experiments.exp_e5_shifting --tabpfn --seed 0  # several hours; deterministic
```

Wrap-up and campaign:

```
12. python -m scripts.make_report --results results/experiments      # refresh the experiments report
13. python -m scripts.run_campaign --config config/campaign_laptop.yaml   # the real run (see RUNBOOK.md §7)
14. python -m scripts.make_report --results results/campaign/laptop       # campaign report
```

> **What the campaign (step 13) adds.** It is the only run that puts the learned policy on **real
> data (Electricity) under TabPFN**, and the only in-context-versus-streaming-ensemble (ARF/SRP)
> comparison on the real substrate. Read `results/campaign/laptop/summary.csv`: the `learned_policy`
> row versus the best single-axis selector on `electricity`, and the in-context rows versus
> `river_arf` / `river_srp`. The multi-seed re-evaluation (see `RUNBOOK_SEED_STUDY.md`) is what
> establishes the headline policy result with confidence intervals.

## The E5 result (established over ten seeds)

Over **ten seeds** on the weak reference substrate the policy averages **0.825 versus the best
fixed rule's 0.843** and beats it in only **1 of 10 seeds**; a single favourable seed can show an
apparent win (0.822 versus 0.795), which the ten-seed mean does not support. On the
near-ceiling TabPFN substrate it also does not beat the best fixed rule (for example, Electricity
0.952 versus 0.959 over five seeds). So the established finding is that the learned policy does **not**
reliably beat the best fixed rule in **any** regime tested; it matches or slightly trails it and adds
run-to-run variance. Anchored at a rule it represents exactly (the recency rule) it reproduces that
rule's accuracy and variance in the settings tested (see the anchored-policy runs below); the class-ratio
rule is not representable by the policy and is used directly. (The in-context family as a whole still
substantially and consistently beats the streaming ensembles; that result is unchanged.) See `results/seed_study/` for the ten-seed data,
confidence intervals and paired tests, and `RUNBOOK_SEED_STUDY.md` to reproduce them.

## Anchored policy and Airlines signal check

```
A1. python -m scripts.experiments.exp_anchored           # reference substrate, 10 seeds, CPU, ~25 min:
                                                          #   sliding_window vs plain policy vs policy ANCHORED at the recency corner
A2. python -m scripts.run_seed_study --blocks B_synth5,B_anchor_t   # TabPFN: matched synthetic baselines + anchored policy (~12 h)
A3. python -m scripts.analysis.stats_report              # refresh CIs, seed-paired and dependence-aware window tests
S1. python -m scripts.experiments.airlines_signal_check  # static gradient-boosting ceiling for Airlines, CPU, ~3 min
```

- **A1 result:** anchored 0.844 vs sliding window 0.843 vs plain policy 0.825 (10 seeds); anchoring
  recovers the plain policy's deficit with the fixed rule's variance. Output: `results/anchored/`.
- **A2 result:** on TabPFN the temporal-anchored policy reproduces the sliding window seed for seed
  (Electricity 0.958 vs 0.958). The class-utility-anchored run (`--blocks B_anchor`) collapses, because
  that corner is degenerate (it ranks by a per-class constant, so the context is the k rarest-class
  instances); that is not the stratified class-ratio rule, which lies outside the policy's hypothesis
  space and is an external baseline. `tests/test_components.py` asserts the recency and similarity
  corners only.
- **S1 result:** a static gradient-boosted model reaches about 0.60 balanced accuracy / 0.66 AUROC even
  with leak-safe target encoding of the high-cardinality categoricals and a 120,000-row sample. Read as a
  diagnostic (see the three-model extension below), not as the dataset's signal ceiling. Output:
  `results/airlines_check/`.

## Label delay, consistent timings, and Airlines with three models

```
R3. bash scripts/run_round3.sh                           # one command, resumable; runs:
    python -m scripts.experiments.airlines_signal_check  #   S1 extended: logreg / random forest / HGB, CPU, ~5-10 min
    python -m scripts.run_seed_study --blocks D_ref_delay,D_tab_delay,D_tab_timing
                                                         #   D_ref_delay: reference, delay 0/50/250/1000, 100 fast trials
                                                         #   D_tab_delay: TabPFN Electricity, delay 250, 9 trials (~7 h)
                                                         #   D_tab_timing: TabPFN Electricity, delay 0, instrumented, 5 trials (~4 h)
R4. python -m scripts.analysis.delay_report              # tables: accuracy vs delay, delay 0 vs 250 on TabPFN,
                                                         #   per-stage timings from one procedure
```

- **Delay semantics.** `label_delay` defers the policy's feedback update to the moment the label
  is released (credited to the exploration sample captured at prediction time, `capture_update_state`),
  as well as delaying buffer entry. `tests/test_delay_timing.py` covers both behaviours and the timing
  fields; at delay 0 the results are identical to the immediate-label runs.
- **Timings.** `run_prequential(time_it=True)` records select / infer / update / step per scored instance
  with `time.perf_counter` in the same loop; `runs.csv` carries `t_<stage>_ms_{mean,p50,p95,p99,max}`
  (older rows have blanks; the file is migrated in place with a `.bak_*` copy).
- **Results:** `results/seed_study/delay_report.md` and the thesis Appendix D (Tables D.7 and D.8).

## Tuned learned-policy configuration (from the E2/E3 results)

This block is already applied to `config/campaign_laptop.yaml`:

```yaml
selector:
  name: learned_policy
  candidate_pool: 3000
  update_rule: policy_gradient
  sigma: 0.25
  entropy_reg: 0.02
  lr: 0.02
  reward_mode: instance
```

## Strengthening the results (optional, additive, fast reference classifier)

These add seeds / regimes for firmer mean ± std and a fuller RQ1b operating envelope. They do not
change or re-run anything already produced.

```
E1 more seeds:              python -m scripts.experiments.exp_e1_convergence --seed 1
                            python -m scripts.experiments.exp_e1_convergence --seed 2
E6 more regime:             python -m scripts.experiments.exp_e6_stability --imbalance 0.05 --seed 0
E2 more regimes:            python -m scripts.experiments.exp_e2_optimizer --imbalance 0.05
                            python -m scripts.experiments.exp_e2_optimizer --electricity
E3 more regime:             python -m scripts.experiments.exp_e3_reward --imbalance 0.05
E5 more seeds (reference):  python -m scripts.experiments.exp_e5_shifting --seed 1
                            python -m scripts.experiments.exp_e5_shifting --seed 2
```

## Decision flow (why this order)

```
   E4 ceiling (steps 2–5)  ──►  beatable fixed target?
        ├─ yes ──►  E2 optimiser + E3 reward (steps 8–9)   close the gap
        └─ no  ──►  E5 shifting-axis regime (step 10)       the time-varying regime
   E1 (step 6) and E6 (step 7) run alongside E4.
   Confirm on --tabpfn (step 11), then report + campaign (steps 12–14).
```

## What to read in each experiment

- **E4** — the final line: `A beatable target EXISTS` versus `No static advantage`. Result:
  Electricity is *beatable* (+0.038); the synthetic regimes are not.
- **E1** — does `gap` shrink as context length grows? Result: yes (0.054 → 0.025).
- **E6** — the `max weight>0.9` fraction (weight collapse) and the latency outliers.
- **E2** — the top-5 table; a **negative `gap`** means the policy beats the baseline. Winner:
  `policy_gradient, sigma 0.25, entropy_reg 0.02, lr 0.02` (gap −0.015).
- **E3** — smallest `gap` / best `minority_recall`. Result: the default `instance + balanced` reward
  wins; the windowed reward was worse in the original three-seed run.
- **E5** — compare `e5_shifting_ref.csv` versus `e5_shifting_tabpfn.csv`: the policy overall versus
  the best single-axis rule, per substrate. The single-seed "flip" did **not** survive the ten-seed
  re-run (see `results/seed_study/` and the section above). Also read the figure `e5_shifting_*.png`
  (the weight trajectory shifting with the regime).
- **probe** — the `VERDICT` line: ✅ PASS (TabPFN distinct from reference) versus ❌ FAIL (identical,
  i.e. not really TabPFN; do not launch a long run until it PASSes).

## Parameter reference

| Flag | Scripts | Default | Notes |
|---|---|---|---|
| `--imbalance <r>` | E1–E4, E6 | 0.15 | sweep {0.30, 0.15, 0.05} to map RQ1b |
| `--electricity` | E1–E4 | off (synthetic) | use the real benchmark |
| `--tabpfn` | all | off (reference) | confirm on the real substrate (slow) |
| `--seed <n>` | E1, E4, E5, E6 | 0 | repeat 0/1/2 and average for rigour |
| `--seeds a b c` | E2, E3 | 0 1 2 | fewer for speed, more for rigour |
| `--max-steps <n>` | E2, E3, E4, E6 | 4000 | raise for firmer estimates |
| `--lengths …` | E1 | 2000 4000 8000 16000 | raise top length to test convergence |
| `--step <s>` | E4 | 0.2 | 0.1 for a finer weighting grid |

## Guidance

- Diagnostics on `reference`; confirm on `--tabpfn` — but always run the probe (step P) first, and
  never launch a multi-hour `--tabpfn` run until it PASSes.
- Each experiment prints `[classifier] TabPFN` or `[classifier] reference` at the start, and E5
  writes a `.meta.json` provenance sidecar — check both so a run can never silently use the wrong
  model.
- A `--tabpfn --seed 0` run is deterministic: re-running reproduces byte-identical output.
  Identical-to-a-previous-TabPFN-run is fine; identical-to-*reference* is the red flag the guard
  catches.
- For anything quoted as a result, report mean ± std over ≥ 3 seeds (the multi-seed study uses 5 and
  10). Snapshot the current outputs any time with `python -m scripts.make_report`.
