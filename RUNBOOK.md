# RUNBOOK — running the prototype end to end

A single, do-this-in-order guide for your workstation. It covers installation,
getting the Electricity data, verifying the setup, the Month-2 baseline
reproduction, and the full experiment campaign. `README.md` is the reference
manual (what each module does); this file is the checklist.

Everything below is run from inside the `03_Prototype` folder.

> For the **learned-policy investigation (experiments E1–E6)** — which parameters
> to change and in what order — see the dedicated **`RUNBOOK_EXPERIMENTS.md`**.

---

## 0. Prerequisites

- **Python 3.10 or newer — required** (`python3 --version`). **TabPFN does not
  support Python 3.9.** 3.11 is recommended. The rest of the prototype runs on
  3.9, but the real in-context substrate will not, so use 3.10+ from the start.
  On macOS: `brew install python@3.11`, then use `python3.11` in step 2.
- **~2 GB free disk** (mostly PyTorch + TabPFN weights).
- **A CUDA GPU** is strongly recommended for TabPFN but not required (CPU works, slower).
- **Internet** on first run (to download the Electricity dataset and TabPFN weights).
- *Optional:* a **JDK 17+** only if you want the CapyMOA baselines (River needs no Java).

## 1. Get a terminal in the project

```bash
cd ~/Claude/Projects/CS7P01NU-MSc_Project/03_Prototype
```

## 2. Create a virtual environment and install the core

```bash
python3.11 -m venv .venv             # must be 3.10+; python3.11 recommended
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python --version                     # confirm 3.10+ before continuing
pip install --upgrade pip
pip install -r requirements.txt      # numpy, pandas, scikit-learn, scipy, pyyaml, matplotlib
```

> Already built a `.venv` with Python 3.9? Recreate it: `deactivate;
> rm -rf .venv; python3.11 -m venv .venv; source .venv/bin/activate`, then
> reinstall the requirements. River/other installs from a 3.9 venv don't carry
> over.

This core is enough to run the offline checks and the reference-classifier
experiments. The next two installs unlock the real substrate and baselines.

## 3. Install the real in-context model (TabPFN)

```bash
pip install -r requirements-tabpfn.txt   # tabpfn + torch
```

- On first use TabPFN downloads its weights from HuggingFace and asks you to
  accept its licence — do this once, interactively, before batch runs.
- GPU is auto-detected. To force CPU, set `device: cpu` in the config's
  `classifier:` block.

## 4. Install the streaming baselines (ARF / SRP)

```bash
pip install -r requirements-streaming.txt   # river (+ capymoa if a JDK is present)
```

- **River** is pure Python and gives you `river_arf`, `river_srp`, `river_hoeffding`.
- **CapyMOA** needs a JDK; if you don't have one, skip it and use the River
  baselines — the campaign will simply mark any `capymoa_*` runs `skipped`.

## 5. Verify the install (30 seconds, no GPU needed)

```bash
python -m tests.test_smoke
python -m tests.test_components
python -m tests.test_streaming
python -m tests.test_campaign
```

All four should end with `RESULT: all N tests passed`. Then a full offline
pipeline run:

```bash
python -m src.run --config config/synthetic_smoke.yaml
```

You should see a summary printed and three files appear under `results/`.

## 6. Get the Electricity data

**No manual download needed.** The loader uses scikit-learn's OpenML fetcher,
which downloads Electricity (elec2; 45,312 rows) on first use and caches it in
`~/scikit_learn_data/`. To fetch and cache it now:

```bash
python -c "from src.data.streams import electricity; s=electricity(); print('Electricity OK:', s.n_features, 'features,', s.n_classes, 'classes')"
```

If your network blocks OpenML, use one of these fallbacks:

- Install the streaming extras (step 4) and load via CapyMOA/River instead, **or**
- Download `electricity` from https://www.openml.org/d/151 as CSV and place it in
  `~/scikit_learn_data/openml/` (scikit-learn's cache layout), **or**
- Develop against the offline `make_classification` stream (no network) until you
  have connectivity.

## 7. The Month-2 baseline reproduction (your go/no-go)

First a fast sanity pass with the CPU reference classifier on real Electricity —
edit `config/electricity_sliding.yaml` and set `classifier.name: reference`:

```bash
python -m src.run --config config/electricity_sliding.yaml
```

Then the real thing (set `classifier.name: tabpfn` back). **Do a capped run
first** to confirm it's moving and to read the rate/ETA before committing to the
full stream:

```bash
python -m src.run --config config/electricity_sliding.yaml --max-steps 3000
```

Every run now prints a heartbeat every 200 scored instances
(`[prequential] 2000 scored (12.3/s, ..., ~30 min left)`), so it never looks
frozen. Once the rate looks acceptable, run the full stream (drop `--max-steps`).

**Speed note.** TabPFN does one forward pass *per instance*, so device and
context size matter a lot. On an Apple-Silicon Mac the run auto-selects **MPS**
(unified memory). Confirm it's active with:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Measured cost on a MacBook Pro at **k=256 is ~0.5 s per instance (~2/s)**, so
even a capped 5,000-instance run is ~25–40 min. Set `k: 256` in the config for
the checkpoint (down from 1024 — smaller context is several times faster). If
TabPFN throws an MPS "operator not implemented" error, prefix the command with
`PYTORCH_ENABLE_MPS_FALLBACK=1`, or set `device: cpu` in the config.

**What a pass looks like.** A healthy checkpoint shows TabPFN + sliding window
decisively beating the persistence baseline — e.g. balanced accuracy ~0.96 and
`kappa_temporal` comfortably positive (~0.7+) on the first few thousand
instances, stable across the run. That clears the Month-2 go/no-go. (Read
`results/electricity_sliding_summary.json`.)

## 8. The experiment campaign

The campaign runs every approach across datasets and imbalance ratios, repeated
over seeds. **Pick the config to match your hardware** — TabPFN's per-instance
cost makes the full grid a very different job on a laptop vs a GPU.

### 8a. On a laptop (Apple Silicon / MPS): `campaign_laptop.yaml`

A reduced first pass — 2 conditions × 4 selectors × 1 seed, capped streams,
k=256 — i.e. **8 TabPFN trials**. At ~0.5 s/instance that's roughly **6–7 hours**:
an overnight run, not the full grid (which would be days on a laptop).

```bash
# 1) fast wiring check (~2 min): temporarily set classifier.name: reference
python -m scripts.run_campaign --config config/campaign_laptop.yaml --quick

# 2) the real overnight run: set classifier.name: tabpfn back, then
python -m scripts.run_campaign --config config/campaign_laptop.yaml

# if interrupted, continue where it left off:
python -m scripts.run_campaign --config config/campaign_laptop.yaml --resume
```

Outputs land in `results/campaign/laptop/`. Scale it up later (more seeds,
remove `run.max_steps`, `k: 512`, more imbalance levels) as time allows.

### 8b. On a CUDA GPU (cluster / cloud): `campaign.yaml`

The full grid — all selectors × 4 conditions × 3 seeds on full-length streams —
is the rigorous version for your results chapter, and it wants a CUDA GPU. Three
edits in `config/campaign.yaml` (each flagged with a comment): set
`classifier.name: tabpfn`, uncomment `river_arf`/`river_srp`, and uncomment the
`electricity` dataset block. Then the same commands as above but with
`--config config/campaign.yaml` (outputs under `results/campaign/pilot/`).

### Outputs (either config)

- `runs.csv` — one row per trial (written incrementally; safe to interrupt).
- `summary.csv` — mean ± std per (dataset, imbalance, approach).
- `summary_<dataset>_<imbalance>.png` — a balanced-accuracy bar chart per condition.
- ranked tables printed to the console, with a live per-trial heartbeat.

`runs.csv` is the file to send me when you want help reading the results.

## 9. One-off head-to-head (optional)

For a quick single-stream comparison of all selectors + the ensemble (no sweep):

```bash
python -m scripts.compare_selectors
```

---

## Troubleshooting

- **`tabpfn` won't install / "Requires-Python >=3.10"** — your venv is Python
  3.9. TabPFN needs 3.10+. Recreate the venv with `python3.11` (see the note in
  step 2) and reinstall.
- **`ModuleNotFoundError: tabpfn / river / capymoa`** — you skipped step 3 or 4.
  Those are optional; the campaign marks the affected runs `skipped` and
  continues. Install the matching `requirements-*.txt` to enable them.
- **OpenML fetch fails / 403 / timeout** — see step 6 fallbacks. This does *not*
  affect the synthetic/`make_classification` datasets, which need no network.
- **TabPFN licence prompt in a batch job** — run one interactive TabPFN
  prediction first (step 7 with a tiny `--max-steps 200`) to accept the licence,
  then batch runs won't block.
- **Out of GPU memory** — lower `run.k` (context size) or `classifier.max_context`
  in the config, or set `device: cpu`.
- **CapyMOA import/JVM error** — you need a JDK 17+ on PATH. If you don't want
  Java, just use the River baselines and leave `capymoa_*` commented out.
- **Run looks frozen / extremely slow with TabPFN** — it's almost certainly
  running per-instance inference on CPU. Confirm it's moving via the
  `[prequential] ... scored` heartbeat. Speed it up by: making sure MPS/GPU is
  used (Apple Silicon auto-selects MPS; check the device isn't forced to `cpu`),
  lowering `run.k` (e.g. `256`), and capping with `--max-steps` while iterating.
- **`operator not implemented for MPS`** — set `PYTORCH_ENABLE_MPS_FALLBACK=1`
  before the command, or `device: cpu` in the config.
- **Runs feel slow** — add `--max-steps 5000` (campaign) or `run.max_steps` in a
  config to cap stream length while iterating; remove it for the final runs.

## Command cheat-sheet

```bash
# setup (Python 3.10+ required for TabPFN)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-tabpfn.txt -r requirements-streaming.txt

# verify
python -m tests.test_smoke && python -m tests.test_components \
  && python -m tests.test_streaming && python -m tests.test_campaign

# data + device check
python -c "from src.data.streams import electricity; electricity()"
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Month-2 reproduction (set k: 256 in the config first)
python -m src.run --config config/electricity_sliding.yaml --max-steps 5000

# campaign — laptop (overnight); GPU/cluster uses config/campaign.yaml
python -m scripts.run_campaign --config config/campaign_laptop.yaml --quick   # wiring check (reference)
python -m scripts.run_campaign --config config/campaign_laptop.yaml           # real run (tabpfn)
```
