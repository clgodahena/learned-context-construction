"""E5 — the regime where the learned policy should win.

Builds a stream whose *optimal* selection axis shifts across four segments:
balanced-stationary (favours feature similarity), abruptly drifting (favours
recency), severely imbalanced (favours class utility), and recurring (favours
memory). No single fixed weighting is best throughout, so this is where a policy
that adapts its weighting online should beat any fixed rule. Reports overall and
per-segment balanced accuracy for the policy, the single-axis baselines, and a
per-segment oracle, and plots the metric trajectory + the policy's weight
trajectory against the segment boundaries.

Usage:
  python -m scripts.experiments.exp_e5_shifting
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np

from src.data.drift_injector import DriftInjector, SegmentedStream, make_classification_stream
from src.selection.learned_policy import LearnedPolicySelector
from . import _common as C

SEG = 2500


def _pred_hash(preds):
    """Stable short hash of a prediction stream (t, y_true, y_pred)."""
    h = hashlib.sha256()
    for t, yt, yp in preds:
        h.update(f"{t},{yt},{yp};".encode())
    return h.hexdigest()[:16]


def build_stream(seed=0):
    s1 = make_classification_stream(n=SEG, n_features=8, pos_rate=0.5, seed=seed)          # similarity
    s2 = DriftInjector(make_classification_stream(n=SEG, n_features=8, pos_rate=0.5, seed=seed + 1),
                       drift_points=[SEG // 2], mechanism="feature_rotation", seed=seed + 1)  # recency
    s3 = DriftInjector(make_classification_stream(n=SEG * 3, n_features=8, pos_rate=0.5, seed=seed + 2),
                       drift_points=[], target_pos_rate=0.08, positive_class=C.POS, seed=seed + 2)  # class utility
    s4 = DriftInjector(make_classification_stream(n=SEG, n_features=8, pos_rate=0.5, seed=seed + 3),
                       drift_points=[SEG // 3, 2 * SEG // 3], mechanism="feature_rotation",
                       recurring=True, seed=seed + 3)                                          # memory
    return SegmentedStream([("balanced", s1), ("drift", s2), ("imbalanced", s3), ("recurring", s4)])


def per_segment_bal_acc(preds, stream):
    buckets = {}
    for t, yt, yp in preds:
        buckets.setdefault(stream.segment_at(t), []).append((None, yt, yp))
    return {stream.segment_names[i]: C.bal_acc_from_preds(v) for i, v in sorted(buckets.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tabpfn", action="store_true")
    args = ap.parse_args()
    C.ensure_out()

    sub = "tabpfn" if args.tabpfn else "ref"
    print(f"[E5] substrate = {sub}  (seed {args.seed})")
    methods = dict(C.single_axis_selectors())
    methods["learned_policy"] = LearnedPolicySelector(n_classes=2, positive_class=C.POS, candidate_pool=3000, seed=args.seed)

    results = {}; trajectories = {}; wtraj = None; pred_hashes = {}
    ref_stream = build_stream(args.seed)
    seg_names = ref_stream.segment_names; seg_bounds = ref_stream.segment_boundaries
    print(f"[E5] segments={seg_names}  boundaries={seg_bounds}  n={ref_stream.n}")

    for name, sel in methods.items():
        r = C.run_sel(build_stream(args.seed), sel, C.make_classifier(args.tabpfn),
                      snapshot_every=150, record_predictions=True)
        overall = r.summary["balanced_accuracy"]
        per_seg = per_segment_bal_acc(r.predictions, ref_stream)
        results[name] = dict(overall=overall, **per_seg)
        trajectories[name] = r.trajectory
        pred_hashes[name] = _pred_hash(r.predictions)
        if name == "learned_policy" and isinstance(sel, LearnedPolicySelector):
            wtraj = np.array(sel.weight_history)
        print(f"  {name:16} overall={overall:.3f}  " + "  ".join(f"{k}={v:.3f}" for k, v in per_seg.items()))

    # per-segment oracle = best single-axis per segment
    axes = list(C.single_axis_selectors().keys())
    oracle = {seg: max(results[a][seg] for a in axes) for seg in seg_names}
    print("  " + f"{'oracle(per-seg)':16} " + "  ".join(f"{k}={v:.3f}" for k, v in oracle.items()))

    # write (substrate-tagged so reference and tabpfn runs never clobber each other)
    path = os.path.join(C.OUT, f"e5_shifting_{sub}.csv")
    with open(path, "w", newline="") as f:
        cols = ["approach", "overall"] + seg_names
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for name, d in results.items():
            wr.writerow({"approach": name, "overall": round(d["overall"], 4),
                         **{s: round(d.get(s, float('nan')), 4) for s in seg_names}})
        wr.writerow({"approach": "oracle_per_segment", "overall": "", **{s: round(oracle[s], 4) for s in seg_names}})

    # plot: metric trajectory + policy weights, with segment boundaries
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for name, traj in trajectories.items():
            t = [r["t"] for r in traj]
            ax1.plot(t, [r["windowed_accuracy"] for r in traj],
                     label=name, lw=2 if name == "learned_policy" else 1.2,
                     ls="-" if name == "learned_policy" else "--")
        for b in seg_bounds[1:]:
            ax1.axvline(b, color="grey", ls=":", lw=0.8); ax2.axvline(b, color="grey", ls=":", lw=0.8)
        for i, nm in enumerate(seg_names):
            mid = (seg_bounds[i] + (seg_bounds[i + 1] if i + 1 < len(seg_bounds) else ref_stream.n)) / 2
            ax1.text(mid, 1.02, nm, ha="center", fontsize=9, color="#555")
        ax1.set_ylabel("windowed accuracy"); ax1.legend(loc="lower left", fontsize=8, ncol=2)
        ax1.set_title("E5: methods across a shifting-optimal-axis stream")
        if wtraj is not None and len(wtraj):
            xs = np.linspace(0, ref_stream.n, len(wtraj))
            for j, lbl in enumerate(["temporal", "feature", "class utility"]):
                ax2.plot(xs, wtraj[:, j], label=lbl, lw=1.6)
            ax2.set_ylabel("learned policy weight"); ax2.set_ylim(0, 1)
            ax2.legend(loc="upper left", fontsize=8, ncol=3)
        ax2.set_xlabel("stream position (dotted = segment boundary)")
        fig.tight_layout(); fig.savefig(os.path.join(C.OUT, f"e5_shifting_{sub}.png"), dpi=120); plt.close(fig)
        print(f"[E5] wrote {path} and e5_shifting_{sub}.png")
    except Exception as e:  # noqa: BLE001
        print(f"[E5] wrote {path} (plot skipped: {e})")

    # --- provenance stamp (sidecar) --------------------------------------
    # Records exactly which model produced this file and a hash of every
    # method's prediction stream. Two runs with different substrates MUST have
    # different hashes; identical hashes mean the same model produced both.
    device = "reference/cpu"
    if args.tabpfn:
        try:
            from src.classifiers.tabpfn_clf import TabPFNInContextClassifier
            device = TabPFNInContextClassifier(n_classes=2, max_context=256)._resolve_device()
        except Exception:  # noqa: BLE001
            device = "tabpfn/unknown"
    meta = {"substrate": sub, "seed": args.seed, "device": device,
            "pred_hashes": pred_hashes}
    meta_path = os.path.join(C.OUT, f"e5_shifting_{sub}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[E5] provenance -> {meta_path}  (device={device}, hashes={pred_hashes})")

    # Anti-duplicate guard: if this is a TabPFN run, its predictions must NOT
    # match the reference run's. Compare against whichever reference sidecar
    # exists. Byte-identical predictions => TabPFN did not really run.
    if args.tabpfn:
        for ref_name in ("e5_shifting_ref.meta.json",):
            rp = os.path.join(C.OUT, ref_name)
            if not os.path.exists(rp):
                continue
            try:
                ref_hashes = json.load(open(rp)).get("pred_hashes", {})
            except Exception:  # noqa: BLE001
                continue
            if ref_hashes and ref_hashes == pred_hashes:
                print("\n[E5][FATAL] TabPFN prediction hashes are IDENTICAL to the "
                      f"reference run ({ref_name}).\n"
                      "           This output was NOT produced by a distinct TabPFN "
                      "model and must be discarded.\n"
                      "           Run `python -m scripts.experiments.probe_tabpfn` to "
                      "diagnose before re-running.")
                break
        else:
            print("[E5] provenance check: no reference sidecar to compare against yet "
                  "(run the reference E5 first to enable the identical-output guard).")

    pol_overall = results["learned_policy"]["overall"]
    best_fixed = max(results[a]["overall"] for a in axes)
    print(f"\n[E5] learned_policy overall={pol_overall:.3f}  vs best single-axis={best_fixed:.3f}  "
          f"({pol_overall - best_fixed:+.3f}) -> "
          + ("policy wins in the time-varying regime" if pol_overall - best_fixed > 0.005
             else "policy does not beat the best fixed rule here"))


if __name__ == "__main__":
    main()
