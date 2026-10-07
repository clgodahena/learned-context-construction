"""probe_tabpfn — a fast, definitive check that --tabpfn really uses TabPFN.

Why this exists
---------------
Twice an ~12 h `exp_e5_shifting --tabpfn` run produced output byte-identical
(CSV *and* rendered PNG) to the fast reference (GaussianNB) run. That is
impossible if TabPFN actually generated the predictions — two different models
cannot emit identical probability floats at every step. So before ever again
committing to a multi-hour TabPFN run, run THIS probe (~2–4 min): it runs a
short slice of the exact E5 stream through the reference classifier and through
TabPFN using the SAME deterministic selector, then compares them.

If the two disagree on some predictions -> TabPFN is a genuinely distinct model
in your venv (PASS): the substrate works, and any earlier identical file was a
file-level mishap, not a modelling one.

If the two are identical -> the --tabpfn path is NOT giving you a real, distinct
TabPFN in this environment (FAIL): fix that here, cheaply, instead of discovering
it after 12 h. The captured import/device/version lines below tell you why.

Usage:
  python -m scripts.experiments.probe_tabpfn                 # ~400 scored steps
  python -m scripts.experiments.probe_tabpfn --steps 600     # longer/firmer
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.harness.prequential import run_prequential
from src.selection.baselines import SlidingWindowSelector
from .exp_e5_shifting import build_stream


def _hash(preds):
    h = hashlib.sha256()
    for t, yt, yp in preds:
        h.update(f"{t},{yt},{yp};".encode())
    return h.hexdigest()[:16]


def _run(classifier, steps, seed):
    # SlidingWindowSelector is deterministic and stateless, so the ONLY thing
    # that can differ between the two runs is the classifier itself.
    return run_prequential(
        build_stream(seed), SlidingWindowSelector(), classifier,
        k=256, capacity=3000, warmup=300, positive_class=1,
        metric_window=1000, snapshot_every=200, max_steps=steps,
        record_predictions=True, progress_every=100,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700,
                    help="stream steps (warmup 300 => ~steps-300 scored TabPFN calls)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("TabPFN provenance probe — reference vs TabPFN on the E5 stream")
    print(f"  steps={args.steps}  seed={args.seed}  (~{max(0, args.steps-300)} TabPFN forward passes)")
    print("=" * 70)

    # 1) Confirm the tabpfn package is importable and see the device it picks.
    print("\n[1/3] importing tabpfn ...", flush=True)
    try:
        import tabpfn as _t
        from src.classifiers.tabpfn_clf import TabPFNInContextClassifier
        ver = getattr(_t, "__version__", "?")
        tab = TabPFNInContextClassifier(n_classes=2, max_context=256)
        print(f"      tabpfn {ver} imported; resolved device = {tab._resolve_device()}")
    except Exception as e:  # noqa: BLE001
        print(f"      FAIL: could not import/init TabPFN -> {type(e).__name__}: {e}")
        print("      This is why a --tabpfn run cannot be a real TabPFN run.")
        raise SystemExit(2)

    # 2) Reference slice (fast).
    print("\n[2/3] reference (GaussianNB) slice ...", flush=True)
    r_ref = _run(ReferenceInContextClassifier(n_classes=2, kind="gnb"), args.steps, args.seed)
    ba_ref = r_ref.summary["balanced_accuracy"]
    h_ref = _hash(r_ref.predictions)
    print(f"      bal_acc={ba_ref:.4f}  n={len(r_ref.predictions)}  pred_hash={h_ref}")

    # 3) TabPFN slice (this is the slow-ish part: ~0.5 s / step).
    print("\n[3/3] TabPFN slice (loads weights on first call) ...", flush=True)
    r_tab = _run(tab, args.steps, args.seed)
    ba_tab = r_tab.summary["balanced_accuracy"]
    h_tab = _hash(r_tab.predictions)
    print(f"      bal_acc={ba_tab:.4f}  n={len(r_tab.predictions)}  pred_hash={h_tab}")

    # Compare, aligned by t.
    ref_by_t = {t: yp for t, _, yp in r_ref.predictions}
    common = [(t, yp) for t, _, yp in r_tab.predictions if t in ref_by_t]
    agree = np.mean([ref_by_t[t] == yp for t, yp in common]) if common else float("nan")

    print("\n" + "=" * 70)
    print(f"prediction agreement (reference vs TabPFN): {agree*100:.1f}%  over {len(common)} steps")
    print(f"balanced-accuracy delta: {ba_tab - ba_ref:+.4f}")
    identical = (h_ref == h_tab) or (np.isfinite(agree) and agree >= 0.999)
    if identical:
        print("VERDICT: ❌ FAIL — TabPFN predictions are IDENTICAL to the reference.")
        print("  The --tabpfn path is NOT exercising a distinct TabPFN in this venv.")
        print("  Do NOT launch a multi-hour run until this is resolved. Check that")
        print("  `pip show tabpfn` points at the real package and that no import")
        print("  shadowing/stub is in place; re-run this probe until it PASSES.")
        raise SystemExit(1)
    print("VERDICT: ✅ PASS — TabPFN is a genuinely distinct model here.")
    print("  The substrate works. A full --tabpfn run will produce real results;")
    print("  the provenance stamp in exp_e5_shifting guards the output file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
