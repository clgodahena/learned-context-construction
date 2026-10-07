"""Smoke test for the campaign runner helpers (no pytest required).

Run with:  python -m tests.test_campaign

Exercises stream construction with imbalance + onset jitter, a single in-context
and a single direct trial, and the aggregation into a summary, on a tiny sweep.
"""
from __future__ import annotations

import sys

import numpy as np

from scripts.run_campaign import aggregate, build_campaign_stream, run_one


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


DATASET = {
    "name": "synth",
    "stream": {
        "kind": "injected",
        "base": {"kind": "make_classification", "n": 4000, "n_features": 6, "pos_rate": 0.5},
        "drift_points": [2000],
        "mechanism": "feature_rotation",
    },
    "drift_jitter": 200,
}
RUN_CFG = {"k": 200, "capacity": 2000, "warmup": 200, "metric_window": 500,
           "snapshot_every": 200, "positive_class": 1,
           "classifier": {"name": "reference", "kind": "gnb"}}


def test_stream_imbalance_and_jitter():
    s1 = build_campaign_stream(DATASET, imbalance=0.1, seed=0, positive_class=1)
    ys = np.array([i.y for i in s1])
    rate = float(np.mean(ys == 1))
    check(abs(rate - 0.1) < 0.03, f"campaign stream hits target imbalance ({rate:.3f})")
    # Different seeds -> resampled drift onset (different injected drift points).
    a = build_campaign_stream(DATASET, imbalance=0.1, seed=1, positive_class=1)
    b = build_campaign_stream(DATASET, imbalance=0.1, seed=2, positive_class=1)
    check(a.drift_points != b.drift_points, "drift onset resampled across seeds")


def test_in_context_and_direct_trials_and_aggregate():
    stream = build_campaign_stream(DATASET, imbalance=0.2, seed=0, positive_class=1)
    m1, st1, _ = run_one({"name": "sliding_window", "family": "in_context",
                          "cfg": {"name": "sliding_window"}}, stream, RUN_CFG, {}, None)
    check(st1 == "ok" and m1["n"] > 1000, "in-context trial runs and reports n")

    stream = build_campaign_stream(DATASET, imbalance=0.2, seed=0, positive_class=1)
    m2, st2, _ = run_one({"name": "online_bagging", "family": "direct",
                          "cfg": {"name": "online_bagging", "n_models": 4}},
                         stream, RUN_CFG, {}, None)
    check(st2 == "ok" and m2["n"] > 1000, "direct trial runs and reports n")

    rows = []
    for seed, m in [(0, m1), (1, m1), (0, m2)]:
        rows.append({"dataset": "synth", "imbalance": "0.2",
                     "approach": ("sliding_window" if m is m1 else "online_bagging"),
                     "family": "x", "status": "ok", "seed": seed,
                     **{k: m[k] for k in m}})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = aggregate(rows, d)
        import os
        check(os.path.exists(path), "aggregate writes summary.csv")


def test_optional_dependency_handled_gracefully():
    """An optional-dependency baseline must never crash the sweep: 'skipped' if
    the library is absent, 'ok' if it is installed -- but never 'error'."""
    stream = build_campaign_stream(DATASET, imbalance=0.2, seed=0, positive_class=1)
    _, status, reason = run_one({"name": "river_arf", "family": "direct",
                                 "cfg": {"name": "river_arf"}}, stream, RUN_CFG, {}, None)
    check(status in ("skipped", "ok"),
          f"optional dep handled gracefully (status={status}, not 'error'): {reason[:60]}")


def main() -> int:
    tests = [test_stream_imbalance_and_jitter,
             test_in_context_and_direct_trials_and_aggregate,
             test_optional_dependency_handled_gracefully]
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
