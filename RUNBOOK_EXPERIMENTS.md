# Experiments Runbook — the learned-policy investigation (E1–E6)

Run everything from `03_Prototype` with the venv active. This runbook is also your
**run log**: the Master Sequence below lists every command in the exact order it
is executed, so you can always say "I've completed up to step N." Diagnostics use
the fast reference classifier (minutes); the confirmation uses `--tabpfn` (~6–12 h).
Outputs land in `results/experiments/`.

> **MANDATORY before any `--tabpfn` run: run the probe (step P, ~3 min).** It proves
> TabPFN really loads and produces predictions *distinct* from the reference in your
> venv, so you never spend 6–12 h to discover otherwise. TabPFN inference is
> **deterministic at a fixed seed** — re-running `--tabpfn --seed 0` reproduces the
> exact same CSV/PNG byte-for-byte. That is expected and is a sign of a correct,
> reproducible pipeline, *not* a clobber. Every E5 run now also writes a
> `…​.meta.json` provenance sidecar and refuses to let a result identical to the
> reference pass silently.

---

## Master sequence  (the run log)

Legend:  ✅ done · ▶ run next · ⏳ later

**Reference-classifier diagnostics — completed (do not re-run):**

```
1.  ✅  python -m tests.test_components
2.  ✅  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.30
3.  ✅  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.15
4.  ✅  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.05
5.  ✅  python -m scripts.experiments.exp_e4_ceiling --electricity
6.  ✅  python -m scripts.experiments.exp_e1_convergence
7.  ✅  python -m scripts.experiments.exp_e6_stability
8.  ✅  python -m scripts.experiments.exp_e2_optimizer --imbalance 0.15
9.  ✅  python -m scripts.experiments.exp_e3_reward   --imbalance 0.15
10. ✅  python -m scripts.experiments.exp_e5_shifting          (reference → e5_shifting_ref.csv)
```

**TabPFN confirmation — completed & CONFIRMED valid:**

```
P.  ✅  python -m scripts.experiments.probe_tabpfn            (pre-flight, ~3 min)
        → PASS: tabpfn 8.1.0 on mps, distinct from reference (0.95 vs 0.68).
11. ✅  python -m scripts.experiments.exp_e5_shifting --tabpfn --seed 0
        (~6–12 h; wrote e5_shifting_tabpfn.csv — genuine TabPFN, seed 0, deterministic)
```

**Wrap-up & campaign confirmation:**

```
12. ✅  python -m scripts.make_report --results results/experiments
        (optional: re-run to refresh after the e5_shifting.* duplicate was retired)
13. ✅  tuned learned_policy block applied to config/campaign_laptop.yaml   (done for you)

14. ▶  WIRING CHECK (~2 min)                                           ◀── START HERE
        a) in config/campaign_laptop.yaml, temporarily set:  classifier.name: reference
        b) python -m scripts.run_campaign --config config/campaign_laptop.yaml --quick
              → expect "... 8 trials", finishes in ~1–2 min, no errors
        c) set it back:                                       classifier.name: tabpfn
15. ⏳  THE REAL RUN  (~5–7 h overnight; resumable)
        python -m scripts.run_campaign --config config/campaign_laptop.yaml
        # if interrupted, continue where it left off:
        python -m scripts.run_campaign --config config/campaign_laptop.yaml --resume
16. ⏳  python -m scripts.make_report --results results/campaign/laptop     (campaign report)
        then send me results/campaign/laptop/runs.csv
```

> **Why `--quick` alone is not the wiring check.** `--quick` caps the sweep to
> 1 seed × 3000 steps but does **not** change the model — with
> `classifier.name: tabpfn` it still loads TabPFN and takes hours. The 2-minute
> check needs `classifier.name: reference` (step 14a), then switch back (14c).

> **What step 15 buys you.** It is the only run that puts the learned policy on
> **real data (Electricity)** under TabPFN — the regime where E4 found genuine
> +0.038 headroom because the selection axes actually combine — and the only
> in-context-vs-streaming-ensemble (ARF/SRP) comparison on the real substrate.
> E5 showed the policy compresses at TabPFN's ceiling on *synthetic* data; step 15
> tests whether it still wins on *real* data. Read
> `results/campaign/laptop/summary.csv`: the learned_policy row vs the best
> single-axis selector on `electricity`, and the in-context rows vs `river_arf` /
> `river_srp`.

> **The E5 headline (now confirmed on both substrates).** The learned policy's
> verdict *flips* with the substrate: on the weak reference it **wins**
> (0.822 vs best-fixed 0.795, +0.026); on strong TabPFN it does **not**
> (0.901 vs best-fixed 0.924, −0.023). Reading: TabPFN is near-ceiling, so a plain
> sliding window already captures most of the value and the policy's online
> exploration doesn't repay its cost — learned context construction helps most when
> the substrate leaves headroom. This characterises the RQ1 operating envelope; it
> is a finding, not a null result. Both `e5_shifting_ref.*` and
> `e5_shifting_tabpfn.*` are on disk for the comparison.

---

## Tuned config (from your E2/E3 results — already applied to campaign_laptop.yaml)

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

## Optional — strengthen the results (additive; fast reference classifier)

Run any of these at any time to add seeds / regimes. They do **not** change or
re-run anything already done — they only add coverage for mean ± std (rigour)
and the RQ1b operating envelope.

```
E1 more seeds:              python -m scripts.experiments.exp_e1_convergence --seed 1
                           python -m scripts.experiments.exp_e1_convergence --seed 2
E6 more regime:            python -m scripts.experiments.exp_e6_stability --imbalance 0.05 --seed 0
E2 more regimes:           python -m scripts.experiments.exp_e2_optimizer --imbalance 0.05
                           python -m scripts.experiments.exp_e2_optimizer --electricity
E3 more regime:            python -m scripts.experiments.exp_e3_reward --imbalance 0.05
E5 more seeds (reference): python -m scripts.experiments.exp_e5_shifting --seed 1
                           python -m scripts.experiments.exp_e5_shifting --seed 2
```

---

## Decision flow (why this order)

```
   E4 ceiling (steps 2–5)  ──►  beatable fixed target?
        ├─ yes ──►  E2 optimiser + E3 reward (steps 8–9)   close the gap
        └─ no  ──►  E5 shifting-axis regime (step 10)        the time-varying win
   E1 (step 6) and E6 (step 7) run alongside E4.
   Confirm the winner on --tabpfn (step 11), then report + campaign (steps 12–15).
```

## What to read in each experiment

- **E4** — the final line: `A beatable target EXISTS` vs `No static advantage`.
  Your result: Electricity is *beatable* (+0.038); synthetic regimes are not.
- **E1** — does `gap` shrink as length grows? Yours: yes (0.054 → 0.025).
- **E6** — `max weight>0.9` fraction (collapse) and latency outliers.
- **E2** — top-5 table; a **negative `gap`** means the policy beats the baseline.
  Your winner: `policy_gradient, sigma 0.25, entropy_reg 0.02, lr 0.02` (gap −0.015).
- **E3** — smallest `gap` / best `minority_recall`. Yours: default
  `instance + balanced` wins; the windowed reward is worse under 3 seeds.
- **E5** — compare `e5_shifting_ref.csv` vs `e5_shifting_tabpfn.csv`: the policy's
  overall vs the best single-axis rule, per substrate. The story is the *flip*
  (policy wins on reference, not on TabPFN) → learned selection pays off with
  headroom. Also read the figure `e5_shifting_*.png` (weight trajectory shifting
  with the regime) and the per-segment row: under TabPFN, `class_ratio` (0.885)
  beats the policy (0.78) on the `imbalanced` segment — the policy under-uses the
  class-utility axis exactly where it should dominate.
- **probe** — the `VERDICT` line: ✅ PASS (TabPFN distinct from reference) vs
  ❌ FAIL (identical → not really TabPFN; do not launch a long run until it PASSes).

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

- Diagnostics on `reference`; confirm on `--tabpfn` — but **always run the probe
  (step P) first**, and never launch a multi-hour `--tabpfn` run until it PASSes.
- Each experiment prints `[classifier] TabPFN` or `[classifier] reference` at the
  start, and E5 writes a `…​.meta.json` provenance sidecar — check both so a run can
  never silently use the wrong model.
- A `--tabpfn --seed 0` run is deterministic: re-running reproduces byte-identical
  output. Identical-to-a-previous-TabPFN-run is fine; identical-to-*reference* is the
  red flag the guard catches.
- For anything going into the thesis, report mean ± std over ≥ 3 seeds. The current
  TabPFN E5 is a single seed (seed 0); add seeds 1–2 only if you need error bars on
  the TabPFN figure specifically (~6–12 h each).
- Snapshot any time with `python -m scripts.make_report`.
