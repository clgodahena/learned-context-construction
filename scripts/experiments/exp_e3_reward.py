"""E3 — reward design ablation.

Compares reward formulations for the learned policy: the instance-correctness
reward (with and without class-balancing) and the metric-aligned windowed reward
(the change in a rolling class-balanced accuracy). Tests whether a better-aligned
reward closes the gap to the best single-axis baseline (H2).

Usage:
  python -m scripts.experiments.exp_e3_reward
  python -m scripts.experiments.exp_e3_reward --imbalance 0.05 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from src.selection.learned_policy import LearnedPolicySelector
from . import _common as C

CONFIGS = [
    dict(label="instance + balanced", reward_mode="instance", balanced_reward=True),
    dict(label="instance + unbalanced", reward_mode="instance", balanced_reward=False),
    dict(label="windowed (metric-aligned)", reward_mode="windowed", balanced_reward=True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--electricity", action="store_true")
    ap.add_argument("--imbalance", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    C.ensure_out()
    tag = "electricity" if args.electricity else f"synth_imb{args.imbalance}"

    def stream(seed):
        return C.make_electricity() if args.electricity else C.make_synth(imbalance=args.imbalance, seed=seed)

    best_base = -1; best_name = None
    for name, sel in C.single_axis_selectors().items():
        r = C.run_sel(stream(0), sel, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
        if r.summary["balanced_accuracy"] > best_base:
            best_base, best_name = r.summary["balanced_accuracy"], name
    print(f"[E3] {tag} — best single-axis baseline: {best_name} = {best_base:.3f}\n")

    rows = []
    for cfg in CONFIGS:
        label = cfg.pop("label")
        accs, mins = [], []
        for seed in args.seeds:
            pol = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000, seed=seed, **cfg)
            r = C.run_sel(stream(seed), pol, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
            accs.append(r.summary["balanced_accuracy"])
            mins.append((r.summary.get("per_class_recall") or [0, 0])[C.POS])
        cfg["label"] = label
        m = float(np.mean(accs)); s = float(np.std(accs)); mr = float(np.mean(mins))
        rows.append(dict(reward=label, bal_acc_mean=round(m, 4), bal_acc_std=round(s, 4),
                         minority_recall=round(mr, 4), gap=round(best_base - m, 4)))
        print(f"  {label:26} bal_acc={m:.3f}+/-{s:.3f}  min_rec={mr:.3f}  gap={best_base - m:+.3f}")

    path = os.path.join(C.OUT, f"e3_reward_{tag}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    best = min(rows, key=lambda r: r["gap"])
    print(f"\n[E3] best reward: '{best['reward']}' (gap {best['gap']:+.3f}).")
    print(f"[E3] wrote {path}")


if __name__ == "__main__":
    main()
