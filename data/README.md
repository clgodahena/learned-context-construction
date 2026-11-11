# Data

The stream loaders in `src/data/streams.py` fetch their datasets from **OpenML** on first use (via
scikit-learn's `fetch_openml`) and cache them under `~/scikit_learn_data/`, so a fresh unpack does not
strictly need any files here.

## What ships in this folder

- `electricity-normalized.arff` — the **primary benchmark** (Electricity / elec2, 45,312 rows), bundled
  for convenience and offline use.

## The other datasets

- **Airlines** and **Covertype** (the two external-validity benchmarks) download from OpenML on first
  use. The pre-cache step in `RUNBOOK_SEED_STUDY.md` (`python -m scripts.precache`) fetches and caches
  all of them in one go, after which the experiments need no network.
- The **synthetic** streams (`make_classification` with injected drift) need no network at all.

## Offline fallback

If the machine cannot reach OpenML:

1. Run `python -m scripts.precache` once on a connected machine to populate `~/scikit_learn_data/`, or
2. Download the datasets from OpenML as CSV and place them in `~/scikit_learn_data/openml/`
   (scikit-learn's cache layout): Electricity is https://www.openml.org/d/151, or
3. Use the synthetic streams, which require no download.
