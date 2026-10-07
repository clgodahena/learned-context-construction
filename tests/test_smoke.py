"""Self-contained smoke tests (no pytest required).

Run with:  python -m tests.test_smoke

Validates the harness spine end-to-end on synthetic data with the reference
classifier, and checks the individual components behave. Exits non-zero on
failure so it can gate CI.
"""
from __future__ import annotations

import sys

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.data.streams import SyntheticDriftStream
from src.harness.buffer import BoundedBuffer
from src.harness.prequential import run_prequential
from src.selection.baselines import RandomSelector, SlidingWindowSelector


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_buffer_capacity():
    buf = BoundedBuffer(capacity=10)
    for inst in SyntheticDriftStream(n=50, seed=1):
        buf.add(inst)
    check(len(buf) == 10, "bounded buffer evicts to capacity")
    check(buf.X.shape == (10, 6), "buffer materialises feature matrix")


def test_sliding_window_recency():
    buf = BoundedBuffer()
    for inst in SyntheticDriftStream(n=20, seed=2):
        buf.add(inst)
    ctx = SlidingWindowSelector().select(buf, buf.X[-1], k=5)
    check(len(ctx) == 5, "sliding window returns k instances")
    check(list(ctx.indices) == [15, 16, 17, 18, 19], "sliding window takes most recent k")


def test_reference_classifier_shapes():
    clf = ReferenceInContextClassifier(n_classes=2, kind="gnb")
    X = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    y = np.array([0, 1] * 20)
    proba = clf.predict_proba(X, y, X[:3])
    check(proba.shape == (3, 2), "reference classifier returns (n_query, n_classes)")
    check(np.allclose(proba.sum(1), 1.0, atol=1e-4), "probabilities sum to 1")


def test_end_to_end_learns():
    stream = SyntheticDriftStream(n=6000, n_features=6, drift_points=[2000, 4000],
                                  pos_rate=0.3, seed=3)
    res = run_prequential(
        stream, SlidingWindowSelector(), ReferenceInContextClassifier(n_classes=2),
        k=400, capacity=2000, warmup=200, metric_window=500, snapshot_every=100,
        time_it=True,
    )
    acc = res.summary["accuracy"]
    bal = res.summary["balanced_accuracy"]
    check(res.summary["n"] > 5000, "scored most of the stream")
    check(acc > 0.6, f"end-to-end accuracy above chance (got {acc:.3f})")
    check(bal > 0.55, f"balanced accuracy above chance (got {bal:.3f})")
    check(len(res.trajectory) > 10, "metric trajectory captured")
    check(len(res.composition) > 1000, "context composition tracked")
    check("latency_ms_p50" in res.summary, "latency profiled")


def test_random_baseline_runs():
    stream = SyntheticDriftStream(n=2000, seed=4)
    res = run_prequential(
        stream, RandomSelector(seed=0), ReferenceInContextClassifier(n_classes=2),
        k=200, warmup=100, snapshot_every=100,
    )
    check(res.summary["n"] > 1500, "random-selector run completes")


def main() -> int:
    tests = [
        test_buffer_capacity,
        test_sliding_window_recency,
        test_reference_classifier_shapes,
        test_end_to_end_learns,
        test_random_baseline_runs,
    ]
    failed = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {e}")
    print()
    if failed:
        print(f"RESULT: {failed}/{len(tests)} tests FAILED")
        return 1
    print(f"RESULT: all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
