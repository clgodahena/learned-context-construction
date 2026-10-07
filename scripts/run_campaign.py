"""Experiment campaign runner -- the whole comparison in one command.

Sweeps a set of *approaches* (in-context context selectors + direct streaming
baselines) across *datasets* and *class-imbalance ratios*, repeated over several
*seeds* with the drift onset resampled per trial, and tabulates the results with
mean +/- std -- the repeated-randomised-runs protocol the proposal specifies.

Usage
-----
    python -m scripts.run_campaign --config config/campaign.yaml
    python -m scripts.run_campaign --config config/campaign.yaml --quick     # tiny smoke sweep
    python -m scripts.run_campaign --config config/campaign.yaml --resume    # skip completed runs
    python -m scripts.run_campaign --config config/campaign.yaml --max-steps 5000

Outputs (under ``<results_dir>/<name>/``)
-----------------------------------------
* ``runs.csv``     -- one row per (dataset, imbalance, approach, seed); written
  incrementally so a crash/interrupt loses nothing (and ``--resume`` continues).
* ``summary.csv``  -- aggregated mean/std per (dataset, imbalance, approach).
* ``summary_<dataset>_<imb>.png`` -- balanced-accuracy bar charts with error bars.
* a ranked table per (dataset, imbalance) printed to the console.

Design
------
* Reuses the ``src.run`` factories, so any stream / selector / classifier the
  single runner understands is available here.
* Fresh selector and direct classifier per run (they carry state); the frozen
  in-context substrate is built once per class count and reused (so TabPFN
  weights load once, not per run).
* A missing optional dependency (tabpfn / river / capymoa) marks the affected
  runs ``skipped`` with the reason and continues -- the rest of the sweep still
  produces results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time
import traceback
from typing import Any

import numpy as np
import yaml

from src.data.drift_injector import DriftInjector
from src.harness.prequential import run_prequential, run_prequential_direct
from src.run import build_direct, build_classifier, build_selector, build_stream

METRICS = [
    "balanced_accuracy", "kappa_temporal", "kappa_m", "windowed_auprc",
    "minority_recall", "accuracy", "latency_ms_p50", "latency_ms_p95",
]
CSV_FIELDS = ["dataset", "imbalance", "approach", "family", "seed", "n", "status", "reason"] + METRICS


# ---------------------------------------------------------------------------
def _jittered_drift_points(points, jitter, seed):
    if not points or not jitter:
        return points
    rng = np.random.default_rng(1000 + seed)
    return [max(1, int(p + rng.integers(-jitter, jitter + 1))) for p in points]


def build_campaign_stream(ds_cfg: dict, imbalance, seed: int, positive_class: int):
    """Construct the stream for one trial, applying imbalance + onset jitter."""
    cfg = copy.deepcopy(ds_cfg["stream"])
    jitter = ds_cfg.get("drift_jitter", 0)

    # Vary the data realisation per seed (repeated randomised runs).
    cfg["seed"] = seed
    if "base" in cfg:
        cfg["base"]["seed"] = seed

    # Impose the target imbalance.
    if imbalance is not None:
        if cfg.get("kind") == "injected":
            cfg["target_pos_rate"] = imbalance
            cfg["positive_class"] = positive_class
        else:
            cfg = {
                "kind": "injected", "base": cfg, "drift_points": [],
                "target_pos_rate": imbalance, "positive_class": positive_class, "seed": seed,
            }

    # Resample drift onset per trial.
    if cfg.get("kind") == "injected" and cfg.get("drift_points"):
        cfg["drift_points"] = _jittered_drift_points(cfg["drift_points"], jitter, seed)

    return build_stream(cfg)


def _row_from_summary(summary: dict, positive_class: int) -> dict:
    pcr = summary.get("per_class_recall") or []
    minority = pcr[positive_class] if positive_class < len(pcr) and pcr[positive_class] is not None else float("nan")
    row = {m: summary.get(m, float("nan")) for m in METRICS}
    row["minority_recall"] = minority
    row["n"] = summary.get("n", 0)
    return row


# ---------------------------------------------------------------------------
def run_one(approach: dict, stream, run_cfg: dict, clf_cache: dict, max_steps, progress_every=0, seed=0):
    """Run a single trial; returns (metrics_row, status, reason).

    ``seed`` is threaded into the selector config so that stochastic selectors
    (the learned policy) get a *different* exploration RNG per trial. Without
    this, repeated "seeds" only vary the stream, and on a fixed real stream
    (Electricity) every seed reproduces an identical policy run. Deterministic
    selectors ignore the seed, so this is harmless for them.
    """
    pos = run_cfg.get("positive_class", 1)
    family = approach["family"]
    try:
        if family == "in_context":
            sel_cfg = {**approach["cfg"], "seed": seed}
            selector = build_selector(sel_cfg, n_classes=stream.n_classes)
            nc = stream.n_classes
            if nc not in clf_cache:
                clf_cache[nc] = build_classifier(run_cfg["classifier"], n_classes=nc)
            res = run_prequential(
                stream, selector, clf_cache[nc],
                k=run_cfg.get("k", 512), capacity=run_cfg.get("capacity"),
                warmup=run_cfg.get("warmup", 300), positive_class=pos,
                metric_window=run_cfg.get("metric_window", 1000),
                snapshot_every=run_cfg.get("snapshot_every", 500),
                max_steps=max_steps, time_it=True, progress_every=progress_every,
            )
        else:  # direct
            clf = build_direct(approach["cfg"], n_classes=stream.n_classes,
                               n_features=stream.n_features)
            res = run_prequential_direct(
                stream, clf, warmup=run_cfg.get("warmup", 300), positive_class=pos,
                metric_window=run_cfg.get("metric_window", 1000),
                snapshot_every=run_cfg.get("snapshot_every", 500),
                max_steps=max_steps, time_it=True, progress_every=progress_every,
            )
        return _row_from_summary(res.summary, pos), "ok", ""
    except ImportError as e:
        return {m: float("nan") for m in METRICS} | {"n": 0}, "skipped", f"missing dependency: {e}"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {m: float("nan") for m in METRICS} | {"n": 0}, "error", str(e)[:200]


def enumerate_approaches(cfg: dict) -> list[dict]:
    approaches = []
    for s in cfg.get("selectors", []):
        approaches.append({"name": s["name"], "family": "in_context", "cfg": s})
    for d in cfg.get("direct", []):
        approaches.append({"name": d["name"], "family": "direct", "cfg": d})
    return approaches


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["imbalance"], r["approach"], r["seed"]))
    return done


# ---------------------------------------------------------------------------
def aggregate(runs: list[dict], out_dir: str):
    """Write summary.csv (mean/std per condition) and print ranked tables."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        if r["status"] != "ok":
            continue
        key = (r["dataset"], r["imbalance"], r["approach"], r["family"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (dataset, imb, approach, family), rs in groups.items():
        row = {"dataset": dataset, "imbalance": imb, "approach": approach,
               "family": family, "n_seeds": len(rs)}
        for m in METRICS:
            vals = np.array([float(r[m]) for r in rs], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{m}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        summary_rows.append(row)

    sum_path = os.path.join(out_dir, "summary.csv")
    if summary_rows:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    # Ranked console tables + bar charts per (dataset, imbalance).
    conditions = sorted({(r["dataset"], r["imbalance"]) for r in summary_rows})
    for dataset, imb in conditions:
        block = [r for r in summary_rows if r["dataset"] == dataset and r["imbalance"] == imb]
        block.sort(key=lambda r: -r["balanced_accuracy_mean"])
        print(f"\n=== {dataset}  (imbalance={imb})  ===")
        print(f"{'approach':16} {'bal_acc':>16} {'kappa_t':>16} {'min_rec':>16} {'p50_ms':>8}")
        print("-" * 76)
        for r in block:
            print(f"{r['approach']:16} "
                  f"{r['balanced_accuracy_mean']:.3f}+/-{r['balanced_accuracy_std']:.3f}   "
                  f"{r['kappa_temporal_mean']:.3f}+/-{r['kappa_temporal_std']:.3f}   "
                  f"{r['minority_recall_mean']:.3f}+/-{r['minority_recall_std']:.3f}   "
                  f"{r['latency_ms_p50_mean']:7.2f}")
        _bar_chart(block, dataset, imb, out_dir)

    return sum_path


def _bar_chart(block, dataset, imb, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        block = sorted(block, key=lambda r: -r["balanced_accuracy_mean"])
        names = [r["approach"] for r in block]
        means = [r["balanced_accuracy_mean"] for r in block]
        errs = [r["balanced_accuracy_std"] for r in block]
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4.5))
        ax.bar(range(len(names)), means, yerr=errs, capsize=4,
               color="#4C78A8", edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("balanced accuracy (mean +/- std)")
        ax.set_title(f"{dataset}  (imbalance={imb})")
        ax.set_ylim(0.5, 1.0)
        fig.tight_layout()
        safe = f"{dataset}_{str(imb).replace('.', 'p')}"
        fig.savefig(os.path.join(out_dir, f"summary_{safe}.png"), dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"  (bar chart skipped: {e})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experiment campaign runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="override the campaign name")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep: 1 seed, 3000 steps -- for a fast smoke check")
    ap.add_argument("--resume", action="store_true", help="skip runs already in runs.csv")
    ap.add_argument("--progress", type=int, default=1000,
                    help="heartbeat every N scored instances within a trial; 0 = off")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = args.name or cfg.get("name", "campaign")
    out_dir = os.path.join(cfg.get("results_dir", "results/campaign"), name)
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.csv")

    run_cfg = dict(cfg.get("run", {}))
    run_cfg["classifier"] = cfg["classifier"]
    pos = run_cfg.get("positive_class", 1)
    seeds = [0] if args.quick else cfg.get("seeds", [0])
    max_steps = 3000 if args.quick else (args.max_steps if args.max_steps is not None
                                         else run_cfg.get("max_steps"))

    approaches = enumerate_approaches(cfg)
    datasets = cfg["datasets"]
    trials = []
    for ds in datasets:
        for imb in ds.get("imbalance", [None]):
            for ap_ in approaches:
                for seed in seeds:
                    trials.append((ds, imb, ap_, seed))

    done = load_done(runs_path) if args.resume else set()
    new_file = not os.path.exists(runs_path) or not args.resume
    mode = "w" if new_file else "a"

    print(f"[campaign] name={name}  approaches={len(approaches)}  datasets={len(datasets)}  "
          f"seeds={len(seeds)}  ->  {len(trials)} trials")
    if max_steps:
        print(f"[campaign] max_steps={max_steps}")

    runs: list[dict] = []
    # Load prior runs so aggregation covers resumed campaigns too.
    if args.resume and os.path.exists(runs_path):
        with open(runs_path) as f:
            runs.extend(list(csv.DictReader(f)))

    clf_cache: dict[int, Any] = {}
    with open(runs_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()

        t_start = time.time()
        for i, (ds, imb, ap_, seed) in enumerate(trials, 1):
            key = (ds["name"], str(imb), ap_["name"], str(seed))
            if key in done:
                continue
            t0 = time.time()
            print(f"  ({i}/{len(trials)}) {ds['name']:12} imb={str(imb):5} "
                  f"{ap_['name']:16} seed={seed}  running...", flush=True)
            try:
                stream = build_campaign_stream(ds, imb, seed, pos)
            except Exception as e:  # noqa: BLE001 -- a bad dataset must not kill the sweep
                metrics = {m: float("nan") for m in METRICS}
                metrics["n"] = 0
                status, reason = "error", f"stream build failed: {str(e)[:150]}"
            else:
                metrics, status, reason = run_one(ap_, stream, run_cfg, clf_cache, max_steps,
                                                  progress_every=args.progress, seed=seed)
            row = {"dataset": ds["name"], "imbalance": imb, "approach": ap_["name"],
                   "family": ap_["family"], "seed": seed, "status": status, "reason": reason}
            row.update({k: metrics.get(k, "") for k in ["n"] + METRICS})
            writer.writerow(row)
            f.flush()
            runs.append({k: str(v) for k, v in row.items()})
            dt = time.time() - t0
            flag = "" if status == "ok" else f"  [{status}: {reason}]"
            ba = metrics.get("balanced_accuracy", float("nan"))
            print(f"       -> bal_acc={ba:.3f}  {dt:5.1f}s{flag}", flush=True)

        print(f"\n[campaign] {len(trials)} trials in {time.time()-t_start:.1f}s")

    # Coerce numeric strings back for aggregation.
    for r in runs:
        for m in METRICS + ["n"]:
            try:
                r[m] = float(r[m])
            except (ValueError, KeyError, TypeError):
                r[m] = float("nan")
    sum_path = aggregate(runs, out_dir)
    print(f"\n[campaign] wrote {runs_path}")
    print(f"[campaign] wrote {sum_path} and per-condition bar charts in {out_dir}/")


if __name__ == "__main__":
    main()
