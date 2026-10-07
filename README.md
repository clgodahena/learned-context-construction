# Learned Context Construction for In-Context Tabular Models under Concept Drift and Class Imbalance

Research prototype and evaluation harness for an **MSc Artificial Intelligence** dissertation
(module **CS7P01NU**, London Metropolitan University in partnership with NEXT Education Group).

**Author:** Chinthaka Lakmal Godahena (Student ID 24077469)

---

## What this is

Deployed machine-learning models degrade as the data-generating process changes (concept drift),
and the usual remedy — periodic retraining — is slow, costly and reactive. Tabular foundation models
such as **TabPFN** adapt instead by conditioning on a *context* of past examples, which reframes
adaptation as the problem of **choosing which historical instances to place in that context**.

This project designs, implements and evaluates a **learned policy** that scores every candidate
instance on three axes at once — **temporal recency**, **feature-space similarity** and
**class utility** — and learns their weighting online from prediction feedback. It proves analytically
that the incumbent single-axis heuristics are *degenerate corners* of this policy, and it characterises
the **operating envelope**: when learned context construction beats retraining-based streaming
ensembles, and when a simple fixed rule is already enough.

Headline findings (see `results/`):

- In-context context selection on a frozen TabPFN decisively beats retraining-based streaming
  ensembles, most clearly on **temporal kappa** (≈ 0.73–0.76 vs 0.38 and −0.07 on Electricity).
- The learned policy's advantage is **conditional**: it wins where the substrate leaves headroom and
  the optimal axis shifts over time (0.822 vs 0.795), but matches a mid-ranked fixed selector on a
  near-ceiling substrate (0.901 vs 0.924). Practical recommendation: **anchor the policy at the
  strongest fixed rule**.

## Walkthrough video

A narrated walkthrough (the author's own voiceover) is included at:

```
media/Combined_Walkthrough_MyVoice.mp4
```

It covers the project in three levels — a project overview, a technical deep-dive of the mechanism,
and a single-diagram view of the end-to-end pipeline.

## Repository structure

```
.
├── src/                       # Library: selectors, classifiers, streaming harness, data loaders
│   ├── selection/             #   single-axis selectors + the learned policy
│   ├── classifiers/           #   TabPFN substrate + fast reference (scikit-learn) stand-in
│   ├── harness/               #   prequential evaluation, buffer, metrics
│   ├── streaming/             #   streaming-ensemble baselines (ARF/SRP via River)
│   └── data/                  #   stream loaders (Electricity, Airlines, Covertype, synthetic)
├── scripts/                   # Experiment runners, campaign driver, report generator
│   └── experiments/           #   E1–E6 investigation scripts
├── config/                    # YAML experiment / campaign configs
├── tests/                     # Smoke, component, streaming and campaign tests
├── dashboard/                 # Streamlit dashboard (app.py) + screenshots
├── results/                   # Experiment outputs: CSV/JSON trajectories, figures, report.html
├── data/                      # Datasets (also auto-fetched from OpenML on first run)
├── media/                     # Narrated walkthrough video
├── requirements*.txt          # core / tabpfn / streaming / dashboard dependencies
├── RUNBOOK.md                 # Step-by-step: install → verify → baseline → campaign
├── RUNBOOK_EXPERIMENTS.md     # The E1–E6 learned-policy investigation, in order
└── README_reference.md        # Detailed module-by-module reference manual
```

## Quickstart

Python **3.10+** is required (TabPFN does not support 3.9); 3.11 is recommended.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt            # core
pip install -r requirements-tabpfn.txt     # the real in-context substrate (torch + tabpfn)
pip install -r requirements-streaming.txt  # ARF/SRP baselines (River)

# verify the install
python -m tests.test_smoke && python -m tests.test_components

# a full offline pipeline run
python -m src.run --config config/synthetic_smoke.yaml
```

Full instructions are in **`RUNBOOK.md`**; the learned-policy experiment sequence (E1–E6) is in
**`RUNBOOK_EXPERIMENTS.md`**.

## Dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Screenshots are in `dashboard/Dashboard_Screenshots/`.

## Data

`data/` contains the datasets used (Electricity, Covertype, plus an extra-datasets report). The
loaders also fetch Electricity automatically from OpenML on first use, so a clone does not strictly
depend on the committed copies.

## Notes

- `results/report.html` is a generated summary of a full run and can be regenerated at any time with
  `python -m scripts.make_report`.
- The **fast reference classifier** (`src/classifiers/reference.py`) is a lightweight scikit-learn
  stand-in used for cheap diagnostics and as the deliberately *weak* substrate; the real runs use
  **TabPFN**.

---

*Private academic repository — MSc dissertation artefact (CS7P01NU). All rights reserved.*
