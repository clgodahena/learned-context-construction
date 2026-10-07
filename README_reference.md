# Prototype — Learned Context Construction for In-Context Tabular Models

Phase-1 **walking skeleton** for CS7P01NU. It runs an in-context tabular
classifier over an evolving data stream, choosing the context window with a
pluggable selection strategy, and reports prequential metrics suited to
imbalanced, drifting data. The spine is complete end-to-end; the novel learned
policy and the remaining baselines slot into clean interfaces (see
*Extension points*).

## Why this exists

The proposal promises a learned context-selection policy that beats single-axis
heuristics under concept drift and class imbalance. This skeleton de-risks that
promise early: it gets you to the **Month-2 baseline-reproduction checkpoint**
(TabPFN + sliding window vs. streaming ensembles) with a working harness, so if
anything about the premise is shaky you find out while the schedule still has
slack.

## Layout

```
03_Prototype/
├── src/
│   ├── data/
│   │   ├── streams.py             # Stream interface: SyntheticDriftStream + Electricity (OpenML)
│   │   └── drift_injector.py      # inject abrupt/gradual/recurring drift + imbalance onto real data
│   ├── harness/
│   │   ├── buffer.py              # bounded historical buffer (+ prediction-feedback slot)
│   │   ├── prequential.py         # test-then-train loop, label delay, online feedback, composition
│   │   ├── metrics.py             # balanced acc, AUPRC, per-class recall, prequential kappa
│   │   └── plots.py               # dashboard: metric trajectory + context composition
│   ├── selection/
│   │   ├── base.py                # ContextSelector interface  <-- the heart of the study
│   │   ├── signals.py             # temporal / feature-space / class-utility scoring signals
│   │   ├── baselines.py           # SlidingWindow, Similarity (kNN), ClassRatio, Random
│   │   └── learned_policy.py      # LearnedPolicySelector -- the central artefact (online weighting)
│   ├── classifiers/
│   │   ├── base.py                # InContextClassifier interface (frozen substrate)
│   │   ├── reference.py           # GPU-free refit-on-context substrate (for CI/validation)
│   │   └── tabpfn_clf.py          # real TabPFN v2 adapter (workstation/GPU)
│   ├── streaming/                 # DIRECT (non-in-context) baselines -- ARF / SRP family
│   │   ├── base.py                # DirectClassifier interface (predict_one / learn_one)
│   │   ├── offline_ensemble.py    # sklearn online-bagging stand-in (always available)
│   │   ├── river_models.py        # River ARF / SRP / Hoeffding adapter (workstation)
│   │   └── capymoa_models.py      # CapyMOA ARF / SRP adapter (workstation, MOA/Java)
│   └── run.py                     # config-driven entrypoint (mode: in_context | direct)
├── config/
│   ├── electricity_sliding.yaml   # the Month-2 target (TabPFN + sliding window on Electricity)
│   ├── injected_learned.yaml      # learned policy on an injected-drift imbalanced stream
│   ├── direct_arf.yaml            # ARF/SRP streaming baseline (mode: direct)
│   ├── campaign.yaml              # the full sweep: approaches x datasets x imbalance x seeds
│   └── synthetic_smoke.yaml       # offline, no-GPU pipeline check
├── scripts/
│   ├── compare_selectors.py       # one-stream head-to-head of all selectors + ensemble
│   └── run_campaign.py            # the whole campaign in one command (mean+/-std, CSVs, charts)
├── tests/
│   ├── test_smoke.py              # harness spine, end-to-end (no pytest needed)
│   ├── test_components.py         # signals, baselines, injector, policy + degeneracy proof
│   ├── test_streaming.py          # direct path + online-bagging stand-in
│   └── test_campaign.py           # campaign helpers: imbalance/jitter, trials, aggregation
├── results/                       # outputs (summary JSON, trajectory CSV, dashboard PNG)
├── requirements.txt               # core (numpy/pandas/sklearn/scipy/pyyaml/matplotlib)
├── requirements-tabpfn.txt        # tabpfn + torch (GPU)
└── requirements-streaming.txt     # capymoa + river (baselines & real datasets)
```

## Setup

> **New here? Follow `RUNBOOK.md` instead** — it's the single, ordered
> checklist from install through the Electricity data to the full campaign.
> This section is the quick reference.

```bash
cd 03_Prototype
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt                        # core — runs everything below except TabPFN
pip install -r requirements-tabpfn.txt                 # for real TabPFN runs (GPU recommended)
pip install -r requirements-streaming.txt              # for CapyMOA/River baselines & datasets (Phase 1b)
```

TabPFN downloads model weights from HuggingFace on first use and is subject to
the TabPFN licence — accept it once before batch runs. A CUDA GPU is strongly
recommended; `device: cpu` works but is slow at stream scale.

## Run it

Offline pipeline check (no GPU, no downloads — proves the machinery):

```bash
python -m tests.test_smoke           # harness spine
python -m tests.test_components       # signals, baselines, injector, learned policy + degeneracy
python -m tests.test_streaming        # direct ARF/SRP path (online-bagging stand-in)
python -m src.run --config config/synthetic_smoke.yaml
python -m src.run --config config/direct_arf.yaml     # streaming ensemble baseline (mode: direct)
python -m scripts.compare_selectors   # selectors + ensemble head-to-head on injected drift + imbalance
```

For the reported ARF/SRP numbers, install `requirements-streaming.txt` and set
`classifier.name: river_arf` (or `capymoa_arf` / `capymoa_srp`) in
`config/direct_arf.yaml`.

## Running the campaign

The whole comparison — every approach across datasets and imbalance ratios,
repeated over seeds with the drift onset resampled per trial — is one command:

```bash
python -m scripts.run_campaign --config config/campaign.yaml --quick    # fast smoke sweep
python -m scripts.run_campaign --config config/campaign.yaml            # full sweep (seeds x conditions)
python -m scripts.run_campaign --config config/campaign.yaml --resume   # continue after an interrupt
```

It writes `results/campaign/<name>/runs.csv` (one row per trial, written
incrementally so nothing is lost on interrupt), `summary.csv` (mean/std per
condition), and a balanced-accuracy bar chart per (dataset, imbalance); it also
prints a ranked table per condition. This directly produces the operating-
envelope evidence for **RQ1b** (where the advantage holds, diminishes, or
disappears across imbalance) and, via ablation configs, **RQ1a**.

To run the *real* campaign on the workstation, three edits in `campaign.yaml`:
set `classifier.name: tabpfn`, uncomment the `river_arf`/`river_srp` baselines,
and uncomment the `electricity` dataset block. A missing optional dependency
marks the affected runs `skipped` (with the reason) and the rest of the sweep
still completes.

The Month-2 baseline reproduction, on your workstation:

```bash
# 1) sanity: reference classifier (fast, CPU) — confirms data + harness on real Electricity
python -m src.run --config config/electricity_sliding.yaml   # after setting classifier.name: reference
# 2) the real thing: TabPFN + sliding window
python -m src.run --config config/electricity_sliding.yaml   # classifier.name: tabpfn
```

Each run writes `results/<experiment>_summary.json`,
`results/<experiment>_trajectory.csv`, and a dashboard PNG.

## How it maps to the proposal (Section 5)

| Proposal component | Where |
|---|---|
| Context Selection Policy | `src/selection/` — one `ContextSelector` per strategy |
| Frozen In-Context Classifier | `src/classifiers/tabpfn_clf.py` (held fixed; interchangeable) |
| Streaming Harness & Drift Injector | `src/harness/prequential.py` + `SyntheticDriftStream` (injector foundation) |
| Evaluation Dashboard | `src/harness/plots.py` + composition tracking in the loop |

The evaluation protocol (prequential test-then-train, balanced accuracy, AUPRC,
per-class recall, prequential kappa) is implemented in `src/harness/metrics.py`.

## What the prototype already shows

1. **The minority-starvation pathology.** On a synthetic imbalanced stream,
   sliding-window selection gives ~97% majority recall but only ~12% minority
   recall — the recency-starves-the-minority effect the proposal predicts.
2. **Degeneracy is exact.** `test_components.py` proves the learned policy, with
   its weight forced onto one signal, selects a context *byte-identical* to the
   sliding-window baseline (temporal) and the similarity baseline (feature). The
   "recoverable as degenerate parameterisations" claim (Objective 2) is true by
   construction, and tested.
3. **A full head-to-head runs.** `scripts.compare_selectors` (recurring drift,
   15% positive, reference classifier) puts all four single-axis baselines, the
   learned policy, and a streaming-ensemble stand-in through the same metrics.
   The in-context selectors edge out the ensemble on balanced accuracy —
   consistent with the proposal's premise — while class-ratio wins minority
   recall at the cost of temporal validity (the exact single-axis tradeoff).

**An honest de-risking finding.** On this *offline* benchmark (weak reference
classifier, short synthetic stream), the learned policy is competitive but does
**not** yet dominate the best single-axis baseline. That is expected and useful
to know now: in-context selection matters most with a strong substrate (TabPFN),
the stream is short for online learning to converge, and the update rule is
lightly tuned. The value delivered is a *correct, principled, ablatable* learner
— the real test is the TabPFN run on Electricity, and every knob needed to
investigate it (`update_rule`, `balanced_reward`, `entropy_reg`, `sigma`
annealing, `freeze`) is exposed. *All comparison numbers use the reference
substrate; reported results use TabPFN.*

## The three axes, as reusable signals

`selection/signals.py` implements the proposal's conceptual framework directly:
`temporal_relevance`, `feature_relevance`, `class_utility`. Each baseline is
"top-k by one signal"; the learned policy is "top-k by an online-weighted sum of
all three". This shared basis is what makes the degeneracy property exact.

## Extension points (the plan from here)

Phase-1b and Phase-2 scaffolding is in place, including a refined
policy-gradient update rule (Gaussian REINFORCE with reward baseline, annealed
exploration, class-balanced reward, entropy regularisation) and the ARF/SRP
streaming baselines (River + CapyMOA adapters). Remaining work:

1. **Real-data campaign.** The runner (`scripts/run_campaign.py`) is ready;
   point `campaign.yaml` at `electricity` with `classifier.name: tabpfn` and the
   real ARF/SRP baselines. This is where the learned policy's advantage should
   become visible, and where RQ1b's operating envelope gets characterised.
2. **Policy update-rule study.** The rule is isolated in `learned_policy.update`
   with `update_rule` ("policy_gradient" | "emphasis") and knobs already exposed
   — the natural place for a focused ablation (RQ1a).
3. **Latency at scale (RQ1c).** Profile per-step cost against streaming
   throughput budgets on the GPU substrate; latency is already recorded.
4. **Validate the workstation adapters.** River/CapyMOA adapters are written
   against the documented APIs but exercised only on the workstation (River needs
   its wheel; CapyMOA needs a JDK) — confirm on first real run.

## Notes

- Deterministic given seeds; no network needed except first-time dataset/model
  downloads.
- The reference classifier is a validation device, **not** an object of study —
  all reported results use TabPFN.
```
