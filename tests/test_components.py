"""Tests for Phase-1b / Phase-2 components (no pytest required).

Run with:  python -m tests.test_components

Covers the scoring signals, the similarity and class-ratio baselines, the drift
injector, and the learned policy -- including the headline property that the
policy recovers the sliding-window and similarity baselines exactly as
degenerate weightings.
"""
from __future__ import annotations

import sys

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.data.drift_injector import DriftInjector, make_classification_stream
from src.data.streams import SyntheticDriftStream
from src.harness.buffer import BoundedBuffer
from src.harness.prequential import run_prequential
from src.selection.baselines import ClassRatioSelector, SimilaritySelector, SlidingWindowSelector
from src.selection.learned_policy import LearnedPolicySelector


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _fill_buffer(n=300, seed=0, pos_rate=0.3, nf=6):
    buf = BoundedBuffer()
    for inst in SyntheticDriftStream(n=n, n_features=nf, pos_rate=pos_rate, seed=seed):
        buf.add(inst)
    return buf


def test_similarity_picks_nearest():
    buf = _fill_buffer()
    q = buf.X[-1]
    ctx = SimilaritySelector().select(buf, q, k=10)
    # The query itself (last row) should be among the nearest to itself.
    check(len(ctx) == 10, "similarity returns k instances")
    dists = np.linalg.norm(buf.X - q, axis=1)
    nearest = set(np.argsort(dists)[:10])
    overlap = len(nearest & set(ctx.indices.tolist()))
    check(overlap >= 9, f"similarity selects (near-)exact kNN (overlap {overlap}/10)")


def test_class_ratio_balances():
    buf = _fill_buffer(pos_rate=0.2)
    ctx = ClassRatioSelector(positive_class=1, target_pos_ratio=0.5).select(buf, buf.X[-1], k=40)
    pos_frac = float(np.mean(ctx.y == 1))
    check(len(ctx) == 40, "class-ratio returns k instances")
    check(pos_frac >= 0.45, f"class-ratio lifts minority share to ~target (got {pos_frac:.2f})")


def test_drift_injector_labels_and_concepts():
    base = make_classification_stream(n=4000, n_features=6, pos_rate=0.5, seed=0)
    inj = DriftInjector(base, drift_points=[2000], mechanism="label_flip", seed=1)
    insts = list(inj)
    check(len(insts) == 4000, "injector preserves length (no imbalance shaping)")
    check(inj.concept_at(100) == 0 and inj.concept_at(3000) == 1, "concept_at tracks drift point")
    # Compare label distribution shift around the flip on matched base rows.
    base_y = np.array([i.y for i in make_classification_stream(4000, 6, 0.5, 0)])
    inj_y = np.array([i.y for i in insts])
    before = np.mean(inj_y[:2000] == base_y[:2000])
    after = np.mean(inj_y[2000:] == base_y[2000:])
    check(before > 0.99, "labels unchanged before drift")
    check(after < 0.01, "labels flipped after drift (abrupt concept change)")


def test_drift_injector_imbalance_shaping():
    base = make_classification_stream(n=6000, n_features=6, pos_rate=0.5, seed=2)
    inj = DriftInjector(base, drift_points=[], target_pos_rate=0.1, positive_class=1, seed=2)
    ys = np.array([i.y for i in inj])
    rate = float(np.mean(ys == 1))
    check(abs(rate - 0.1) < 0.03, f"imbalance shaping hits target rate (got {rate:.3f})")


def test_policy_degeneracy_recovers_baselines():
    buf = _fill_buffer(n=500, seed=5)
    q = buf.X[-1]
    k = 20
    # sigma=0 -> deterministic selection; degeneracy is a property of the
    # scoring map, not of the explorer.
    pol = LearnedPolicySelector(n_classes=2, candidate_pool=10_000, sigma=0.0)

    pol.set_onehot("temporal")
    got = pol.select(buf, q, k).indices
    want = SlidingWindowSelector().select(buf, q, k).indices
    check(np.array_equal(got, want), "policy@temporal == sliding window (exact)")

    pol.set_onehot("feature")
    got = pol.select(buf, q, k).indices
    want = SimilaritySelector().select(buf, q, k).indices
    check(np.array_equal(got, want), "policy@feature == similarity (exact)")


def test_policy_learns_and_updates():
    pol = LearnedPolicySelector(n_classes=2, lr=0.1, candidate_pool=2000)
    w0 = pol.weights.copy()
    stream = SyntheticDriftStream(n=4000, n_features=6, drift_points=[1500], pos_rate=0.3, seed=6)
    res = run_prequential(stream, pol, ReferenceInContextClassifier(n_classes=2),
                          k=200, capacity=2500, warmup=200, snapshot_every=200)
    w1 = pol.weights
    check(res.summary["n"] > 3000, "policy run completes end-to-end")
    check(not np.allclose(w0, w1), "policy weights move under feedback")
    check(abs(w1.sum() - 1.0) < 1e-6, "weights remain a valid distribution")
    check(len(pol.weight_history) > 1000, "weight trajectory recorded")


def test_policy_freeze_holds_weights():
    pol = LearnedPolicySelector(n_classes=2, freeze=True, init_weights=[0.5, 0.3, 0.2])
    w0 = pol.weights.copy()
    stream = SyntheticDriftStream(n=1500, seed=7)
    run_prequential(stream, pol, ReferenceInContextClassifier(n_classes=2),
                    k=150, warmup=150, snapshot_every=200)
    check(np.allclose(w0, pol.weights), "frozen policy keeps weights fixed (ablation)")


def main() -> int:
    tests = [
        test_similarity_picks_nearest,
        test_class_ratio_balances,
        test_drift_injector_labels_and_concepts,
        test_drift_injector_imbalance_shaping,
        test_policy_degeneracy_recovers_baselines,
        test_policy_learns_and_updates,
        test_policy_freeze_holds_weights,
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
