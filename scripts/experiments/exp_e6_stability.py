"""E6 — instability diagnosis.

Records the learned policy's full weight trajectory and per-step latency on the
synthetic drift + imbalance stream and inspects them for the AUPRC collapse and
latency spike seen in the first campaign. Flags whether the weights collapse onto
one axis (which would degrade probability ranking and hence AUPRC) and reports
latency outliers, distinguishing a fixable fault from a fundamental limitation.

Usage:
  python -m scripts.experiments.exp_e6_stability
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
    ap.add_argument("--imbalance", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    C.ensure_out()

    pol = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000, seed=args.seed)
    r = C.run_sel(C.make_synth(imbalance=args.imbalance, seed=args.seed), pol,
                  C.make_classifier(args.tabpfn), max_steps=args.max_steps, time_it=True)

    wh = np.array(pol.weight_history)
    lat = np.array(r.latencies_ms) if r.latencies_ms else np.array([])

    # weight collapse: max weight sustained near 1
    max_w = wh.max(axis=1) if len(wh) else np.array([])
    collapse_frac = float(np.mean(max_w > 0.9)) if len(max_w) else float("nan")
    dominant = ["temporal", "feature", "class_utility"][int(np.argmax(wh[-1]))] if len(wh) else "n/a"

    # latency outliers
    p50 = float(np.percentile(lat, 50)) if len(lat) else float("nan")
    p95 = float(np.percentile(lat, 95)) if len(lat) else float("nan")
    p999 = float(np.percentile(lat, 99.9)) if len(lat) else float("nan")
    n_out = int(np.sum(lat > 5 * p50)) if len(lat) else 0

    print("[E6] instability diagnosis")
    print(f"  summary: bal_acc={r.summary['balanced_accuracy']:.3f}  "
          f"AUPRC={r.summary.get('windowed_auprc', float('nan')):.3f}  "
          f"kappa_t={r.summary['kappa_temporal']:.3f}")
    print(f"  weights: final=[{wh[-1][0]:.2f},{wh[-1][1]:.2f},{wh[-1][2]:.2f}] dominant={dominant}  "
          f"fraction of steps with max weight>0.9 = {collapse_frac:.2f}")
    print(f"  latency ms: p50={p50:.2f} p95={p95:.2f} p99.9={p999:.2f}  steps>5x median = {n_out}")
    verdict = []
    if collapse_frac > 0.5:
        verdict.append(f"weights collapse onto '{dominant}' (likely the AUPRC degradation) -> revisit entropy_reg/reward")
    if n_out > 0:
        verdict.append(f"{n_out} latency outlier step(s) -> inspect context assembly at those steps (edge case, likely fixable)")
    if not verdict:
        verdict.append("no collapse and no latency outliers in this run")
    print("  verdict: " + "; ".join(verdict))

    # dump trajectories for inspection
    path = os.path.join(C.OUT, "e6_stability.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["step", "w_temporal", "w_feature", "w_class"])
        for i, w in enumerate(wh):
            wr.writerow([i, round(float(w[0]), 4), round(float(w[1]), 4), round(float(w[2]), 4)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6))
        for j, lbl in enumerate(["temporal", "feature", "class utility"]):
            a1.plot(wh[:, j], label=lbl, lw=1.4)
        a1.set_ylabel("weight"); a1.set_ylim(0, 1); a1.legend(fontsize=8, ncol=3); a1.set_title("E6: weight trajectory")
        if len(lat):
            a2.plot(lat, lw=0.6, color="tab:red"); a2.set_ylabel("latency (ms)"); a2.set_xlabel("scored step")
            a2.set_title("Per-step selection+inference latency")
        fig.tight_layout(); fig.savefig(os.path.join(C.OUT, "e6_stability.png"), dpi=120); plt.close(fig)
        print(f"[E6] wrote {path} and e6_stability.png")
    except Exception as e:  # noqa: BLE001
        print(f"[E6] wrote {path} (plot skipped: {e})")


if __name__ == "__main__":
    main()
