# START HERE — Examiner guide

**Learned Context Construction for In-Context Tabular Models under Concept Drift and Class Imbalance**
MSc Artificial Intelligence dissertation · module **CS7P01NU** · London Metropolitan University
in partnership with NEXT Education Group.
**Author:** Chinthaka Lakmal Godahena (Student ID 24077469)

This archive is the complete, self-contained code artefact for the dissertation: the research
library, the experiment runners, the interactive dashboard, the reproducibility outputs, and the
guides needed to install and run everything. It unpacks and runs on a standard machine with no
special infrastructure.

---

## What is in this package

```
24077469_CS7P01NU-MScProject_CodeBase_Artefacts/
├── START_HERE.md              ← you are here
├── README.md                  Project overview, headline findings, repository map
├── RUNBOOK.md                 Install → verify → baseline reproduction → full campaign
├── RUNBOOK_EXPERIMENTS.md     Reproduce the learned-policy investigation (experiments E1–E6)
├── RUNBOOK_SEED_STUDY.md      Reproduce the multi-seed study (confidence intervals + paired tests)
├── requirements*.txt          Dependency sets: core / tabpfn / streaming / dashboard
├── src/                       The research library (selectors, classifiers, harness, data, metrics)
├── config/                    YAML configs for every run and campaign
├── scripts/                   Experiment runners, campaign driver, report + statistics tools
├── tests/                     Smoke / component / streaming / campaign tests (no GPU needed)
├── dashboard/                 Streamlit dashboard (app.py) + screenshots
├── results/                   Reference outputs, figures, report.html, and the seed-study data
└── data/                      Primary Electricity dataset + notes on fetching the others
```

A narrated walkthrough of the project (the author's own voice, three levels: overview, technical
deep-dive, and a single-diagram view) is provided **alongside this archive in the submission
package** as `24077469_Walkthrough_Combined.mp4`.

## The five-minute path (no GPU, no downloads)

This proves the machinery end to end using the fast CPU reference classifier.

```bash
# from inside this folder
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.10+ required
pip install --upgrade pip
pip install -r requirements.txt                         # core only

python -m tests.test_smoke
python -m tests.test_components        # includes the exact degeneracy checks (recency and similarity corners)
python -m tests.test_streaming
python -m tests.test_campaign

python -m src.run --config config/synthetic_smoke.yaml  # a full offline pipeline run
```

Everything above runs on CPU in minutes and needs no internet. To run the **real** in-context
substrate (TabPFN) and the streaming-ensemble baselines, and to reproduce the reported numbers,
follow **RUNBOOK.md** from the top.

## Where to look for the results

- `results/report.html` — a self-contained summary of a full run (open in any browser).
- `results/seed_study/stats_report.md` and `stats_cis.png` — the multi-seed confidence intervals
  and paired tests behind the dissertation's central claims.
- `dashboard/` — run `streamlit run dashboard/app.py` for the interactive results-and-run console
  (see `dashboard/README.md`).

## The one thing to know about the findings

In-context context selection on a frozen TabPFN **substantially and consistently** beats
retraining-based streaming ensembles. The **learned policy**, however, does **not** reliably beat
the best single-axis fixed rule in any regime tested. Anchored at a fixed rule it represents exactly
(the recency rule), it reproduces that rule's accuracy and variance; the class-ratio rule is not an exact
corner of the policy and is best used directly. The evidence, with confidence intervals, seed-paired
tests and dependence-aware window tests, is in `results/seed_study/` (and `results/anchored/`,
`results/airlines_check/`). README.md has the full statement.
