"""E2 — exploration and optimisation grid.

Grids over the policy's optimisation knobs (update_rule, sigma, entropy_reg, lr)
across seeds and reports which configuration, if any, narrows the gap to the
best single-axis baseline. The sigma=0 arm tests whether exploration is
net-harmful on short streams (H3); the rest tests hyper-parameter sensitivity
(H4).

Usage:
  python -m scripts.experiments.exp_e2_optimizer
  python -m scripts.experiments.exp_e2_optimizer --imbalance 0.05 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np

from src.selection.learned_policy import LearnedPolicySelector
from . import _common as C


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

    # baseline reference
    best_base = -1; best_name = None
    for name, sel in C.single_axis_selectors().items():
        r = C.run_sel(stream(0), sel, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
        if r.summary["balanced_accuracy"] > best_base:
            best_base, best_name = r.summary["balanced_accuracy"], name
    print(f"[E2] {tag} — best single-axis baseline: {best_name} = {best_base:.3f}\n")

    grid = dict(update_rule=["policy_gradient", "emphasis"], sigma=[0.0, 0.1, 0.25],
                entropy_reg=[0.0, 0.02, 0.1], lr=[0.02, 0.05, 0.1])
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        accs = []
        for seed in args.seeds:
            pol = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000, seed=seed, **cfg)
            r = C.run_sel(stream(seed), pol, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
            accs.append(r.summary["balanced_accuracy"])
        m, s = float(np.mean(accs)), float(np.std(accs))
        rows.append(dict(**cfg, bal_acc_mean=round(m, 4), bal_acc_std=round(s, 4), gap=round(best_base - m, 4)))

    rows.sort(key=lambda r: -r["bal_acc_mean"])
    path = os.path.join(C.OUT, f"e2_optimizer_{tag}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("top 5 configurations:")
    for r in rows[:5]:
        print(f"  bal_acc={r['bal_acc_mean']:.3f}+/-{r['bal_acc_std']:.3f}  gap={r['gap']:+.3f}  "
              f"rule={r['update_rule']} sigma={r['sigma']} ent={r['entropy_reg']} lr={r['lr']}")
    best = rows[0]
    print(f"\n[E2] best config {'CLOSES' if best['gap'] <= 0.005 else 'does not close'} the gap "
          f"(gap={best['gap']:+.3f}).")
    print(f"[E2] wrote {path}")


if __name__ == "__main__":
    main()
