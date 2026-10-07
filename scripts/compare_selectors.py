"""Head-to-head comparison of all context selectors on one stream.

Runs each selection strategy through the same prequential harness with the same
frozen classifier on an injected-drift, imbalanced stream, and prints a
comparison table plus a combined metric-trajectory figure. This is the shape of
the experiment that answers RQ1a (which signals matter) and, with TabPFN in
place of the reference classifier, the Month-2..4 comparisons.

Usage:  python -m scripts.compare_selectors
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.data.drift_injector import DriftInjector, make_classification_stream
from src.harness.prequential import run_prequential, run_prequential_direct
from src.selection.baselines import (
    ClassRatioSelector,
    RandomSelector,
    SimilaritySelector,
    SlidingWindowSelector,
)
from src.selection.learned_policy import LearnedPolicySelector
from src.streaming.offline_ensemble import OnlineBaggingEnsemble

POS = 1
K = 300
CAP = 4000
WARMUP = 300


def make_stream():
    """Recurring drift + severe imbalance on an offline 'real-like' dataset."""
    base = make_classification_stream(n=12000, n_features=8, pos_rate=0.5, seed=0)
    return DriftInjector(
        base,
        drift_points=[3000, 6000, 9000],
        mechanism="feature_rotation",
        drift_kind="abrupt",
        recurring=True,          # concept 0 returns -> rewards non-destructive memory
        target_pos_rate=0.15,    # severe imbalance
        positive_class=POS,
        seed=0,
    )


def selectors(n_classes):
    return {
        "random": RandomSelector(seed=0),
        "sliding_window": SlidingWindowSelector(),
        "similarity": SimilaritySelector(candidate_pool=CAP),
        "class_ratio": ClassRatioSelector(positive_class=POS, target_pos_ratio=0.4),
        "learned_policy": LearnedPolicySelector(
            n_classes=n_classes, positive_class=POS, candidate_pool=CAP, lr=0.05
        ),
    }


def _record(rows, trajectories, name, res):
    s = res.summary
    rows.append((name, s["balanced_accuracy"], s["kappa_temporal"],
                 s.get("windowed_auprc", float("nan")),
                 s["per_class_recall"][POS], s.get("latency_ms_p50", float("nan"))))
    trajectories[name] = res.trajectory


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    trajectories = {}

    # --- in-context selectors (shared frozen reference substrate) ---
    for name, sel in selectors(2).items():
        res = run_prequential(
            make_stream(), sel, ReferenceInContextClassifier(n_classes=2),
            k=K, capacity=CAP, warmup=WARMUP, positive_class=POS,
            metric_window=1000, snapshot_every=200, time_it=True,
        )
        _record(rows, trajectories, name, res)
        s = res.summary
        extra = ""
        if isinstance(sel, LearnedPolicySelector):
            w = sel.weights
            extra = f"  final weights [temporal={w[0]:.2f} feature={w[1]:.2f} class={w[2]:.2f}]"
        print(f"[{name:16}] bal_acc={s['balanced_accuracy']:.3f} "
              f"kappa_t={s['kappa_temporal']:.3f} minority_recall={s['per_class_recall'][POS]:.3f}"
              + extra)

    # --- direct streaming baseline (non-in-context). Uses the always-available
    #     online-bagging stand-in here; swap RiverClassifier("arf") / CapyMOA on
    #     the workstation for the reported ARF/SRP numbers. ---
    res = run_prequential_direct(
        make_stream(), OnlineBaggingEnsemble(n_classes=2, seed=0),
        warmup=WARMUP, positive_class=POS, metric_window=1000, snapshot_every=200, time_it=True,
    )
    _record(rows, trajectories, "online_bagging*", res)
    s = res.summary
    print(f"[{'online_bagging*':16}] bal_acc={s['balanced_accuracy']:.3f} "
          f"kappa_t={s['kappa_temporal']:.3f} minority_recall={s['per_class_recall'][POS]:.3f}"
          "   (* stand-in for ARF/SRP; not in-context)")

    # --- table ---
    print("\n=== comparison (injected recurring drift + 15% positive) ===")
    hdr = f"{'selector':16} {'bal_acc':>8} {'kappa_t':>8} {'AUPRC':>7} {'min_rec':>8} {'p50_ms':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, bal, kap, auprc, minrec, lat in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:16} {bal:8.3f} {kap:8.3f} {auprc:7.3f} {minrec:8.3f} {lat:7.2f}")

    with open("results/comparison_summary.json", "w") as f:
        json.dump([{"selector": r[0], "balanced_accuracy": r[1], "kappa_temporal": r[2],
                    "windowed_auprc": r[3], "minority_recall": r[4], "latency_ms_p50": r[5]}
                   for r in rows], f, indent=2)

    # --- combined trajectory plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        for name, traj in trajectories.items():
            t = [r["t"] for r in traj]
            ax.plot(t, [r["balanced_accuracy"] for r in traj], label=name, linewidth=1.6)
        for dp in (3000, 6000, 9000):
            ax.axvline(dp, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("stream position (dashed = injected drift)")
        ax.set_ylabel("balanced accuracy (cumulative)")
        ax.set_title("Context selectors under recurring drift + class imbalance")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig("results/comparison_balanced_accuracy.png", dpi=120)
        print("\nwrote results/comparison_summary.json and "
              "results/comparison_balanced_accuracy.png")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
