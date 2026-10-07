"""E1 — convergence versus stream length.

Runs the learned policy at increasing stream lengths and records whether the gap
to the best single-axis baseline shrinks and whether the weights stabilise. A
shrinking gap + stabilising weights supports the under-convergence hypothesis
(H1); a persistent gap rejects it.

Usage:
  python -m scripts.experiments.exp_e1_convergence
  python -m scripts.experiments.exp_e1_convergence --electricity
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from src.selection.learned_policy import LearnedPolicySelector
from . import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--electricity", action="store_true")
    ap.add_argument("--imbalance", type=float, default=0.15)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000, 8000, 16000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    C.ensure_out()
    tag = "electricity" if args.electricity else f"synth_imb{args.imbalance}"

    def stream(n_hint):
        if args.electricity:
            return C.make_electricity()
        return C.make_synth(imbalance=args.imbalance, seed=args.seed,
                            n=max(n_hint + 2000, 8000), drift_points=(n_hint // 3, 2 * n_hint // 3))

    rows = []
    print(f"[E1] {tag} — convergence vs length")
    for L in args.lengths:
        # best single-axis baseline at this length
        best_base = -1; best_name = None
        for name, sel in C.single_axis_selectors().items():
            r = C.run_sel(stream(L), sel, C.make_classifier(args.tabpfn), max_steps=L)
            ba = r.summary["balanced_accuracy"]
            if ba > best_base: best_base, best_name = ba, name
        # learned policy
        pol = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000)
        r = C.run_sel(stream(L), pol, C.make_classifier(args.tabpfn), max_steps=L)
        pol_ba = r.summary["balanced_accuracy"]; w = pol.weights
        gap = best_base - pol_ba
        # weight stability: std of the last 20% of the weight history
        wh = np.array(pol.weight_history)
        tail = wh[int(0.8 * len(wh)):] if len(wh) > 10 else wh
        wstd = float(np.mean(np.std(tail, axis=0))) if len(tail) else float("nan")
        rows.append(dict(length=L, best_baseline=best_name, best_base_bal_acc=round(best_base, 4),
                         policy_bal_acc=round(pol_ba, 4), gap=round(gap, 4),
                         w_temporal=round(w[0], 3), w_feature=round(w[1], 3), w_class=round(w[2], 3),
                         weight_stability_std=round(wstd, 4)))
        print(f"  L={L:6d}  best={best_name:14}({best_base:.3f})  policy={pol_ba:.3f}  gap={gap:+.3f}  "
              f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]  wstd={wstd:.3f}")

    path = os.path.join(C.OUT, f"e1_convergence_{tag}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    trend = rows[-1]["gap"] - rows[0]["gap"]
    print("\n[E1] gap change (last - first):", f"{trend:+.3f}",
          "-> gap shrinks with length (supports H1)" if trend < -0.005 else "-> gap does not shrink (rejects H1)")
    print(f"[E1] wrote {path}")


if __name__ == "__main__":
    main()
