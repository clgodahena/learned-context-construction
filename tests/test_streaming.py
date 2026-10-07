"""Tests for the direct streaming-classifier path (no pytest required).

Run with:  python -m tests.test_streaming

Validates the always-available online-bagging stand-in and the direct
prequential runner. The River/CapyMOA adapters need those libraries + (for
CapyMOA) a JDK, so they are exercised on the workstation, not here.
"""
from __future__ import annotations

import sys

import numpy as np

from src.data.drift_injector import DriftInjector, make_classification_stream
from src.harness.prequential import run_prequential_direct
from src.streaming.offline_ensemble import OnlineBaggingEnsemble


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_ensemble_shapes_and_learns():
    clf = OnlineBaggingEnsemble(n_classes=2, n_models=5, seed=0)
    rng = np.random.default_rng(0)
    # Learnable pattern: label = sign of first feature.
    for _ in range(400):
        x = rng.normal(size=4)
        y = int(x[0] > 0)
        clf.learn_one(x, y)
    p = clf.predict_proba_one(np.array([2.0, 0, 0, 0]))
    check(p.shape == (2,), "ensemble returns (n_classes,) proba")
    check(abs(p.sum() - 1.0) < 1e-6, "ensemble proba sums to 1")
    check(int(np.argmax(p)) == 1, "ensemble learned a separable pattern")


def test_direct_runner_end_to_end():
    stream = DriftInjector(
        make_classification_stream(n=6000, n_features=6, pos_rate=0.5, seed=0),
        drift_points=[3000], mechanism="feature_rotation", target_pos_rate=0.2, seed=0,
    )
    res = run_prequential_direct(
        stream, OnlineBaggingEnsemble(n_classes=2, seed=0),
        warmup=200, positive_class=1, metric_window=800, snapshot_every=200, time_it=True,
    )
    check(res.summary["n"] > 2000, "direct runner scores the stream")
    check(res.summary["balanced_accuracy"] > 0.55, "direct baseline learns above chance")
    check("latency_ms_p50" in res.summary, "direct runner profiles latency")
    check(len(res.trajectory) > 5, "direct runner records a trajectory")


def main() -> int:
    tests = [test_ensemble_shapes_and_learns, test_direct_runner_end_to_end]
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
