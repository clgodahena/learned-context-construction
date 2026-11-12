# Learned Context Construction for In-Context Tabular Models under Concept Drift and Class Imbalance

Research prototype and evaluation harness for an **MSc Artificial Intelligence** dissertation
(module **CS7P01NU**, London Metropolitan University in partnership with NEXT Education Group).

**Author:** Chinthaka Lakmal Godahena (Student ID 24077469)

New to this archive? Read **`START_HERE.md`** first for the five-minute orientation.

---

## What this is

Deployed machine-learning models degrade as the data-generating process changes (concept drift),
and the usual remedy — periodic retraining — is slow, costly and reactive. Tabular foundation models
such as **TabPFN** adapt instead by conditioning on a *context* of past examples, which reframes
adaptation as the problem of **choosing which historical instances to place in that context**.

This project designs, implements and evaluates a **learned policy** that scores every candidate
instance on three axes at once — **temporal recency**, **feature-space similarity** and
**class utility** — and learns their weighting online from prediction feedback. It proves analytically
that two of the three incumbent single-axis heuristics (recency and similarity) are exact *degenerate corners*
of this policy, shows that the third (the stratified class-ratio rule) is not representable by it and is treated as an
external baseline, and it characterises
the **operating envelope**: when in-context selection beats retraining-based streaming ensembles, and
when a simple fixed rule is already enough.

## Headline findings (see `results/`)

- In-context context selection on a frozen TabPFN **substantially and consistently** beats
  retraining-based streaming ensembles, most clearly on **temporal kappa** (≈ 0.73–0.76 versus 0.38
  and −0.07 on Electricity). This is confirmed with non-overlapping confidence intervals and
  dependence-aware window-level tests (Electricity window gap +0.060 windowed accuracy, moving-block
  bootstrap CI [+0.051, +0.068], disjoint-window Wilcoxon p = 0.031 on 6 non-overlapping windows;
  a naive test on the 23 overlapping windows would give p ≈ 3e-5, which is inflated and is not used).

- The learned policy does **not** reliably beat the best fixed rule in any regime tested. Over **ten
  seeds** on the weak substrate it averages 0.825 versus the best fixed rule's 0.843, winning in only
  1 of 10 seeds (a single favourable seed can show 0.822 versus 0.795, which the mean does not support); on the
  near-ceiling TabPFN substrate it also does not beat the best fixed rule (for example, Electricity
  0.953 versus 0.959 in the campaign; on the synthetic stream, with every method matched by seed, it
  trails the best fixed rule by 0.057, losing on 5 of 5 seeds). See `results/seed_study/` for the
  seed data, confidence intervals and paired tests.

- **Anchoring is evaluated, not assumed.** Anchored at a
  rule it represents exactly (the recency rule), with the settings stated below, the policy reproduced that
  rule's accuracy and variance on every seed of the streams tested (Electricity 0.958 vs 0.958, run-to-run
  SD 0.0022 -> 0.0003; seed-for-seed on the synthetic stream; reference substrate 0.844 vs 0.843 over ten
  seeds). This is an observed result for those settings, not a guarantee. Settings: warm start of the logits
  at the recency vertex (anchor logits ±2.5), entropy term off, quadratic pull back toward the anchor logits
  of strength 0.1; learning rate, exploration and reward as the plain policy (`src/selection/anchored_policy.py`).
  Anchored at the class-utility corner it collapses (0.52 / 0.69): that corner ranks by a per-class constant
  and fills the context with rare-class instances, which is NOT the stratified class-ratio rule; the
  class-ratio rule lies outside the policy's hypothesis space (only the recency and similarity rules are
  exact corners; `tests/test_components.py`). Practical recommendation: **use the strongest fixed rule for
  the regime; where the policy represents it exactly, the anchored configuration matched its reliability
  in our runs while keeping the option to adapt; where it is the class-ratio rule, use the rule directly.**
  Evidence: `results/seed_study/anchored_tabpfn_5seed.txt`, `results/anchored/anchored_reference_e5.txt`,
  `results/seed_study/stats_report.md`.

- **Statistics are dependence-aware.** Seed-level paired tests on balanced accuracy are the primary
  evidence (seeds are independent); window-level tests use windowed accuracy with a moving-block
  bootstrap and a disjoint-window cross-check, because consecutive windows overlap by 75%. Airlines'
  near-chance result was checked with a static diagnostic (three model families, naive and leak-safe
  categorical encoding, thesis cap and a 120,000-row sample): every setting plateaus near 0.60 balanced
  accuracy. This is a diagnostic, not a proof of the dataset's signal ceiling; it says the stream carries
  little static signal under these representations (`results/airlines_check/`).

- **Late labels and end-to-end cost.** With `label_delay = d` an instance enters the buffer
  AND the policy's feedback update is applied only d steps after the prediction (credited to the exploration
  sample that produced it). Delay therefore ages every selector's context and additionally slows the policy's
  adaptation. The harness times context selection, frozen-model inference and the selector update with
  one procedure inside the same run and reports mean / p50 / p95 / p99 / max per stage. Results of the
  delay sweep (reference substrate, delays 0/50/250/1000) and the TabPFN Electricity delay-250 and timing
  runs: `results/seed_study/delay_report.md` (blocks `D_ref_delay`, `D_tab_delay`, `D_tab_timing`;
  launcher `scripts/run_round3.sh`).

## Repository structure

```
.
├── src/                       # Library: selectors, classifiers, streaming harness, data loaders
│   ├── selection/             #   single-axis selectors + the learned policy (the central artefact) + anchored_policy.py
│   ├── classifiers/           #   TabPFN substrate + fast reference (scikit-learn) stand-in
│   ├── harness/               #   prequential evaluation, buffer, metrics, plots
│   ├── streaming/             #   streaming-ensemble baselines (ARF/SRP via River, online bagging)
│   └── data/                  #   stream loaders (Electricity, Airlines, Covertype, synthetic) + drift injector
├── scripts/                   # Experiment runners, campaign driver, report + statistics tools
│   ├── experiments/           #   E1–E6 investigation scripts (+ the TabPFN probe, exp_anchored.py, airlines_signal_check.py)
│   ├── analysis/              #   stats_report.py (CIs, paired tests) · delay_report.py (label delay + per-stage timings)
│   └── run_round3.sh          #   launcher: Airlines diagnostic (3 models) + seed-study blocks D_ref_delay/D_tab_delay/D_tab_timing
├── config/                    # YAML experiment / campaign configs
├── tests/                     # Smoke, component, streaming, campaign and delay/timing tests (no GPU needed)
├── dashboard/                 # Streamlit dashboard (app.py) + screenshots
├── results/                   # Experiment outputs: trajectories, figures, report.html, seed_study/, anchored/, airlines_check/
├── data/                      # Primary Electricity dataset + fetch notes (others auto-download)
├── requirements*.txt          # core / tabpfn / streaming / dashboard dependencies
├── RUNBOOK.md                 # Step-by-step: install → verify → baseline → campaign
├── RUNBOOK_EXPERIMENTS.md     # The E1–E6 learned-policy investigation, in order
└── RUNBOOK_SEED_STUDY.md      # The multi-seed study: confidence intervals and paired tests
```

## Quickstart

Python **3.10+** is required (TabPFN does not support 3.9); 3.11 is recommended.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt            # core
pip install -r requirements-tabpfn.txt     # the real in-context substrate (torch + tabpfn)
pip install -r requirements-streaming.txt  # ARF/SRP baselines (River)

# verify the install (CPU, no downloads)
python -m tests.test_smoke && python -m tests.test_components

# a full offline pipeline run
python -m src.run --config config/synthetic_smoke.yaml
```

Full instructions are in **`RUNBOOK.md`**; the learned-policy experiment sequence (E1–E6) is in
**`RUNBOOK_EXPERIMENTS.md`**; the multi-seed study is in **`RUNBOOK_SEED_STUDY.md`**.

## Dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py          # run from this folder so paths resolve
```

The dashboard browses the `results/` folder (tables, bar charts, figures) and can launch runs from a
form, streaming the console output live. Screenshots are in `dashboard/Dashboard_Screenshots/`; see
`dashboard/README.md`.

## Data

The stream loaders fetch **Electricity** (elec2, 45,312 rows) automatically from OpenML on first use
and cache it under `~/scikit_learn_data/`, so a fresh unpack does not strictly need the committed
copy. For convenience and offline use, `data/` also ships the primary Electricity dataset; the two
external-validity benchmarks (**Airlines**, **Covertype**) download from OpenML on first use (see
`RUNBOOK_SEED_STUDY.md`, the pre-cache step). `data/README.md` has the details and offline fallbacks.

## Notes

- `results/report.html` is a generated, self-contained summary of a full run. It ships in the submission
  archive and is not committed to the repository; rebuild it at any time with `python -m scripts.make_report`.
- The **fast reference classifier** (`src/classifiers/reference.py`) is a lightweight scikit-learn
  stand-in used for cheap diagnostics and as the deliberately *weak* substrate; the real runs use
  **TabPFN**. All head-to-head numbers reported in the dissertation use TabPFN.
- Runs are deterministic given seeds; no network is needed except first-time dataset/model downloads.
- A narrated walkthrough video (the author's own voice) is provided as
  `media/24077469_Project_Walkthrough.mp4` in the repository (GitHub offers it as a download rather than
  an in-page preview because of its size) and under the same name in the submission package.

## Licence and citation

The code in this repository is released under the MIT Licence (see `LICENSE`). Third-party
components keep their own licences: TabPFN (package and pretrained weights), River, scikit-learn and
the other dependencies listed in the `requirements*.txt` files, and the benchmark datasets
(Electricity, Airlines, Covertype), which are redistributed or downloaded under their original terms.

If you use this work, please cite the dissertation:

> Godahena, C.L. (2026) *Learned Context Construction for In-Context Tabular Models under Concept
> Drift and Class Imbalance*. MSc Artificial Intelligence dissertation (CS7P01NU), London Metropolitan
> University. Code: https://github.com/clgodahena/learned-context-construction

---

*MSc dissertation artefact (CS7P01NU), London Metropolitan University, 2026.*
