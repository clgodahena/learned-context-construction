# RUNBOOK — the multi-seed study (confidence intervals + paired tests)

How to reproduce the multi-seed evaluation behind the dissertation's central claims: more seeds for
error bars, and the per-window data needed for confidence intervals and paired tests. Everything here
is **additive** and **resumable** — it never changes or re-runs what is already on disk, and it
survives sleep, interruption, or the network dropping mid-run.

Run everything from this folder with the venv active. The completed outputs ship in
`results/seed_study/` (`runs.csv`, `stats_report.md`, `stats_cis.png`, `stats_cis.csv`,
`stats_paired.csv`, `trajectories/`), so the study can be inspected without re-running anything.

---

## What it produces

| Dimension | What the runs give |
|---|---|
| More seeds | Reference substrate 3 → **10** seeds (Block A); the TabPFN primary pair and the two external benchmarks 3 → **5** seeds for the learned policy (Blocks B_policy, C_policy) |
| Confidence intervals + paired tests | Bootstrap 95% confidence intervals and paired Wilcoxon / t-tests (`stats_report.py`) |
| Non-independence | Pairing by **seed** (primary, balanced accuracy) and by **time-window**; window tests are dependence-aware (moving-block bootstrap, disjoint-window check), because consecutive windows overlap by 75% |
| Matched synthetic baselines (2nd review) | Block **B_synth5**: fixed selectors + ensembles on synthetic seeds 1–4, so every method is compared on the SAME five seed-streams (the synthetic stream is resampled per seed; the real streams are not) |
| Anchored policy (2nd review) | Block **B_anchor_t**: the policy anchored at the temporal corner (= sliding window), 5 seeds on Electricity + synthetic. Block **B_anchor** (class-utility corner) is kept as evidence that that corner is degenerate |

On the real streams (Electricity, Airlines, Covertype) the fixed selectors and ensembles are deterministic
and the stream is replayed in its natural order, so they are run at **one seed only** — the seed changes
only the stochastic policy's exploration. On the **synthetic** stream the seed also resamples the data and
the drift onset, so each seed is a different stream and every method must be run on every seed
(block B_synth5) for the comparison to be matched.

## Step 1 — Pre-cache once, on a stable connection

```bash
python -m scripts.precache
```

This downloads and caches Electricity, Airlines, Covertype and the TabPFN weights, and accepts the
TabPFN licence interactively. **After this finishes the study needs no internet at all**, so losing
connectivity mid-run cannot interrupt it. If any cache reports `[FAIL]`, re-run on a good connection
before starting.

## Step 2 — Launch (resumable, sleep-proof)

```bash
bash scripts/run_weekend.sh A,B_policy,C_policy      # the error-bar blocks first
```

`caffeinate -i` keeps the machine awake while it runs (macOS). Closing the lid suspends the machine —
the process is paused, not killed, and continues when you reopen it; nothing is lost either way
because of Step 3's checkpointing. To be safe against closing the terminal, launch detached:

```bash
caffeinate -i nohup bash scripts/run_weekend.sh A,B_policy,C_policy \
    > results/seed_study/run.log 2>&1 &
tail -f results/seed_study/run.log                   # check progress any time
```

When time allows, add the trajectory-capture blocks (they enable the window-level paired tests on the
primary pair):

```bash
bash scripts/run_weekend.sh B_traj,C_traj
```

The second-review blocks (matched synthetic baselines and the anchored policy) are launched the same way:

```bash
caffeinate -i nohup python -m scripts.run_seed_study --blocks B_synth5,B_anchor_t \
    > results/seed_study/run_draft2.log 2>&1 &
```

(`B_anchor`, the class-utility-anchored run, is optional: it reproduces the collapse described below.)

Third review (label delay + consistent timings), one command:

```bash
bash scripts/run_round3.sh        # Airlines diagnostic (3 models) then D_ref_delay,D_tab_delay,D_tab_timing detached
python -m scripts.analysis.delay_report     # when run_round3.log says 'finished this pass'
```

`label_delay` now defers BOTH buffer entry and the policy's feedback update (credited to the exploration
sample captured at prediction time); `runs.csv` gains `label_delay` and `t_<stage>_ms_*` timing columns
(existing rows are migrated in place, backup kept as `runs.csv.bak_*`).

(The launcher script is named `run_weekend.sh` for historical reasons; it simply drives the blocks
named on its command line.)

## Step 3 — If it is interrupted, run the same command again

Every trial is checkpointed individually as
`results/seed_study/<block>__<substrate>__<stream>__<method>__seedN.done` (plus its row in `runs.csv`
and its trajectory file). On relaunch, finished trials are skipped and only the unfinished ones run.
Worst case you lose one in-progress trial (~30–50 min for a TabPFN trial, seconds to minutes for a
reference trial); completed work is never lost. Preview what is left without running anything:

```bash
python -m scripts.run_seed_study --blocks A,B_policy,B_traj,C_policy,C_traj --dry-run
```

## Step 4 — Statistics (re-runnable any time)

`run_weekend.sh` calls this at the end, but it can be run alone whenever more trials have finished:

```bash
python -m scripts.analysis.stats_report
```

It writes `stats_cis.csv`, `stats_paired.csv`, `stats_report.md` and `stats_cis.png` — the confidence
intervals, the paired tests, and the figure reported in the dissertation (Chapter 7).

## What the study establishes

- In-context selection substantially and consistently beats the streaming ensembles, with
  non-overlapping confidence intervals and dependence-aware window-level tests (Electricity window gap
  +0.060 windowed accuracy, moving-block bootstrap CI [+0.051, +0.068], disjoint-window Wilcoxon p = 0.031;
  the naive overlapping-window p ≈ 3e-5 is inflated and is not quoted).
- The learned policy does **not** reliably beat the best fixed rule in any regime: on the weak
  reference substrate over ten seeds, 0.825 versus 0.843 (winning 1 of 10 seeds); on near-ceiling TabPFN
  it trails the best fixed selector (Electricity −0.006; synthetic −0.057 on matched seed-streams, 0 of 5
  seeds, paired-t p = 0.014). Read `stats_report.md` for the full table.
- Anchored at the **temporal** corner (an exact corner of the policy; warm start at logits ±2.5, pull
  0.1, entropy off) the policy reproduced the sliding window on every seed tested (Electricity 0.958 vs
  0.958, SD 0.0022 → 0.0003; synthetic seed-for-seed), and stayed within 0.002 of it under label delays of
  50 to 1,000 on the reference substrate and 250 on Electricity/TabPFN (`delay_report.md`). This is an
  observed result for those settings, not a guarantee. Anchored at the **class-utility** corner it
  collapses (0.516 / 0.687): the class-utility score is identical for every instance of a class, so the
  top-k context is the k rarest-class instances and TabPFN's single-class fast path returns a constant.
  That corner is NOT the stratified class-ratio rule; the class-ratio rule lies outside the policy's
  hypothesis space and is an external baseline (`tests/test_components.py` checks the recency and
  similarity corners; see `anchored_tabpfn_5seed.txt`).

## Time budget (rough, at ~0.5 s/instance, 6000-step cap)

| Block | What | Trials | Rough time |
|---|---|---|---|
| A | reference, 10 seeds, 4 methods | 40 (CPU, fast) | < 1 h |
| B_policy | TabPFN, elec+synth, policy, 5 seeds | 10 | ~6–8 h |
| C_policy | TabPFN, airlines+covertype, policy, 5 seeds | 10 | ~6–8 h |
| B_traj | TabPFN, elec+synth, fixed+ensembles, seed 0 | 6 TabPFN + 6 fast | ~4 h |
| C_traj | TabPFN, airlines+covertype, fixed+ensembles, seed 0 | 6 TabPFN + 6 fast | ~4 h |
| B_synth5 | TabPFN, synthetic, fixed selectors seeds 1–4 (+ ensembles, fast) | 12 TabPFN + 12 fast | ~6 h |
| B_anchor_t | TabPFN, elec+synth, anchored policy (temporal corner), 5 seeds | 10 TabPFN | ~6 h |
| B_anchor | TabPFN, elec+synth, anchored policy (class-utility corner), 5 seeds | 10 (collapse, fast) | ~1 h |
| D_ref_delay | reference, shifting-axis, label delay 0/50/250/1000, 5 selectors, 5 seeds (third review) | 100 fast | ~30 min |
| D_tab_delay | TabPFN, Electricity, label delay 250: 3 fixed + policy x3 + anchored x3 (third review) | 9 TabPFN | ~7 h |
| D_tab_timing | TabPFN, Electricity, delay 0, instrumented per-stage timings (third review) | 5 TabPFN | ~4 h |

Run **A + B_policy + C_policy** first — that is the error-bar evidence. Add **B_traj / C_traj** to
upgrade the paired tests from seed-level to window-level on the main streams.

## Troubleshooting

- **A trial shows SKIPPED (missing dependency)** — install the matching `requirements-*.txt`
  (tabpfn / river) and re-run; it fills only the gaps.
- **TabPFN looks frozen** — it prints `[prequential] N scored (…/s…)` every 1000 instances; if the
  rate is there, it is working. Confirm MPS with
  `python -c "import torch; print(torch.backends.mps.is_available())"`.
- **`operator not implemented for MPS`** — `run_weekend.sh` already exports
  `PYTORCH_ENABLE_MPS_FALLBACK=1`; if running by hand, export it too.
- **Covertype / Airlines fetch fails mid-study** — Step 1 was skipped; run `python -m scripts.precache`
  on a connection, then resume.
