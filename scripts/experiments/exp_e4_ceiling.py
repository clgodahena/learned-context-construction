"""E4 — the fixed-weighting ceiling test (run this first).

Grid-searches the best FIXED weighting of the three signals over the probability
simplex (learning switched off) and compares it against the single-axis
baselines. This decides the branch the rest of the investigation takes:

  * if the best fixed weighting BEATS every single-axis baseline -> a beatable
    target exists; the learned policy is failing to reach it (pursue E1--E3).
  * if it merely TIES the best single-axis rule -> no static advantage exists on
    this stream; the advantage (if any) is time-varying (pursue E5).

Usage:
  python -m scripts.experiments.exp_e4_ceiling                 # synthetic, imb=0.15
  python -m scripts.experiments.exp_e4_ceiling --electricity
  python -m scripts.experiments.exp_e4_ceiling --imbalance 0.05 --step 0.1
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from src.selection.learned_policy import LearnedPolicySelector
from . import _common as C

SIGNALS = ("temporal", "feature", "class_utility")


def simplex(step):
    pts = []
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    for a in grid:
        for b in grid:
            cc = round(1 - a - b, 4)
            if -1e-9 <= cc <= 1 + 1e-9:
                pts.append((float(a), float(b), max(0.0, cc)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--electricity", action="store_true")
    ap.add_argument("--imbalance", type=float, default=0.15)
    ap.add_argument("--step", type=float, default=0.2, help="simplex grid step")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    C.ensure_out()

    tag = "electricity" if args.electricity else f"synth_imb{args.imbalance}"
    def stream():
        return C.make_electricity() if args.electricity else C.make_synth(imbalance=args.imbalance, seed=args.seed)

    rows = []
    # --- single-axis baselines ---
    print(f"[E4] {tag}  — single-axis baselines")
    base = {}
    for name, sel in C.single_axis_selectors().items():
        r = C.run_sel(stream(), sel, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
        s = C.summary_row(r.summary); base[name] = s["balanced_accuracy"]
        rows.append({"kind": "baseline", "w_temporal": "", "w_feature": "", "w_class": "", "approach": name, **s})
        print(f"  {name:16} bal_acc={s['balanced_accuracy']:.3f}")
    best_base_name = max(base, key=base.get); best_base = base[best_base_name]

    # --- fixed-weighting simplex ---
    pts = simplex(args.step)
    print(f"[E4] sweeping {len(pts)} fixed weightings (freeze=on)")
    best_fixed = None
    for (wt, wf, wc) in pts:
        pol = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000,
                                    init_weights=[max(wt, 1e-6), max(wf, 1e-6), max(wc, 1e-6)],
                                    freeze=True, sigma=0.0)
        r = C.run_sel(stream(), pol, C.make_classifier(args.tabpfn), max_steps=args.max_steps)
        s = C.summary_row(r.summary)
        rows.append({"kind": "fixed", "w_temporal": wt, "w_feature": wf, "w_class": wc, "approach": "fixed", **s})
        if best_fixed is None or s["balanced_accuracy"] > best_fixed[1]:
            best_fixed = ((wt, wf, wc), s["balanced_accuracy"])

    # --- write + verdict ---
    path = os.path.join(C.OUT, f"e4_ceiling_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    (bw, bacc) = best_fixed
    print("\n=== E4 verdict ===")
    print(f"best single-axis:  {best_base_name}  bal_acc={best_base:.3f}")
    print(f"best fixed weighting: temporal={bw[0]:.2f} feature={bw[1]:.2f} class={bw[2]:.2f}  bal_acc={bacc:.3f}")
    margin = bacc - best_base
    if margin > 0.005:
        print(f"-> A beatable target EXISTS (+{margin:.3f}). Branch: pursue E1-E3 (optimisation/reward).")
    else:
        print(f"-> No static advantage ({margin:+.3f}). Branch: pursue E5 (time-varying regime).")
    print(f"[E4] wrote {path}")


if __name__ == "__main__":
    main()
