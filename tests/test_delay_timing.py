"""Tests for label-delay semantics and per-stage timing.

Run with:  python -m tests.test_delay_timing
"""
from __future__ import annotations

import sys

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.data.streams import SyntheticDriftStream
from src.harness.prequential import run_prequential
from src.selection.anchored_policy import AnchoredPolicySelector
from src.selection.baselines import SlidingWindowSelector
from src.selection.learned_policy import LearnedPolicySelector


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def _run(selector, delay, n=3000, seed=11, time_it=False):
    stream = SyntheticDriftStream(n=n, n_features=6, drift_points=[1500], pos_rate=0.3, seed=seed)
    return run_prequential(stream, selector, ReferenceInContextClassifier(n_classes=2),
                           k=150, capacity=1500, warmup=200, snapshot_every=200,
                           label_delay=delay, time_it=time_it)


def test_delay_zero_unchanged():
    """With no delay the refactored loop must reproduce the previous behaviour
    exactly (same seed -> same weights, same metrics)."""
    a = LearnedPolicySelector(n_classes=2, lr=0.05, candidate_pool=2000, seed=3)
    b = LearnedPolicySelector(n_classes=2, lr=0.05, candidate_pool=2000, seed=3)
    ra = _run(a, 0)
    rb = _run(b, 0)
    check(np.allclose(a.weights, b.weights), "delay 0 is deterministic given the seed")
    check(ra.summary["balanced_accuracy"] == rb.summary["balanced_accuracy"],
          "delay 0 metrics reproduce exactly")
    check(ra.summary["label_delay"] == 0, "summary records label_delay")


def test_delay_defers_policy_update():
    """Under delay d the policy must receive its first update only after d
    further instances, and must receive (roughly) n - warmup - d updates in total."""
    d = 100
    pol = LearnedPolicySelector(n_classes=2, lr=0.05, candidate_pool=2000, seed=3)
    pol_updates = []
    orig = pol.update

    def spy(*args, **kw):
        pol_updates.append(kw.get("token") is not None)
        return orig(*args, **kw)

    pol.update = spy  # type: ignore[method-assign]
    res = _run(pol, d, n=2000)
    n_scored = res.summary["n"]
    check(len(pol_updates) > 0, "policy still receives updates under delay")
    check(all(pol_updates), "every delayed update carries the captured exploration token")
    check(n_scored - d - 5 <= len(pol_updates) <= n_scored,
          f"updates deferred by the delay: {len(pol_updates)} updates for {n_scored} scored (d={d})")
    check(res.summary["label_delay"] == d, "summary records the delay used")


def test_delay_token_credits_right_sample():
    """A token-based update must use the token's perturbation, not the latest one."""
    pol = LearnedPolicySelector(n_classes=2, lr=0.1, candidate_pool=2000, sigma=0.3, seed=5)
    buf_stream = SyntheticDriftStream(n=400, n_features=6, seed=5)
    from src.harness.buffer import BoundedBuffer
    buf = BoundedBuffer()
    for inst in buf_stream:
        buf.add(inst)
    q = buf.X[-1]
    pol.select(buf, q, 50)
    tok = pol.capture_update_state()
    pol.select(buf, q, 50)  # a newer sample is now the "last" one
    check(not np.allclose(tok["eps"], pol._last_eps), "token differs from the latest sample")
    theta0 = pol.theta.copy()
    # Fix the reward baseline so the advantage is non-zero in both updates.
    pol._baseline, pol._baseline_init = 0.0, True
    pol.update(True, margin=0.5, y_true=1, token=tok)
    moved_tok = pol.theta - theta0
    # Re-do with the latest sample for comparison from the same starting point.
    pol.theta = theta0.copy(); pol._velocity[:] = 0
    pol._baseline, pol._baseline_init = 0.0, True
    pol.update(True, margin=0.5, y_true=1)
    moved_last = pol.theta - theta0
    check(not np.allclose(moved_tok, moved_last), "token-based update follows the token's perturbation")


def test_delay_affects_fixed_selectors_too():
    """Delay ages every selector's context: the sliding window at delay d must
    never contain the d most recent instances."""
    d = 50
    seen = {}
    sel = SlidingWindowSelector()
    orig = sel.select

    def spy(buffer, query_x, k):
        ctx = orig(buffer, query_x, k)
        seen["gap"] = int(buffer.t[-1])
        return ctx

    sel.select = spy  # type: ignore[method-assign]
    stream = SyntheticDriftStream(n=1200, n_features=6, seed=8)
    last_t = None
    for inst in stream:
        last_t = inst.t
    _run(sel, d, n=1200, seed=8)
    check(last_t - seen["gap"] >= d - 1, f"newest buffered instance lags the stream by >= {d-1}")


def test_anchored_delay_and_timing():
    anc = AnchoredPolicySelector(n_classes=2, lr=0.05, candidate_pool=2000, anchor="temporal",
                                 anchor_reg=0.1, anchor_sharpness=2.5, seed=2)
    res = _run(anc, 25, time_it=True)
    s = res.summary
    for stage in ("select", "infer", "update", "step"):
        check(f"t_{stage}_ms_p50" in s and f"t_{stage}_ms_p99" in s and f"t_{stage}_ms_max" in s,
              f"timing summary present for {stage}")
    check(len(res.select_ms) == len(res.infer_ms) == len(res.step_ms) == s["n"],
          "one select/infer/step timing per scored instance")
    check(s["t_step_ms_mean"] >= s["t_select_ms_mean"] + s["t_infer_ms_mean"] - 1e-9,
          "end-to-end step time >= select + infer")
    check(s["t_update_ms_mean"] >= 0 and len(res.update_ms) > 0, "update timings recorded")


def test_fixed_selector_has_no_update_timing():
    res = _run(SlidingWindowSelector(), 0, n=1500, time_it=True)
    check(len(res.update_ms) == 0 and "t_update_ms_mean" not in res.summary,
          "fixed selector records no update stage")
    check("t_step_ms_p95" in res.summary, "fixed selector still has end-to-end timing")


def main() -> int:
    tests = [test_delay_zero_unchanged, test_delay_defers_policy_update,
             test_delay_token_credits_right_sample, test_delay_affects_fixed_selectors_too,
             test_anchored_delay_and_timing, test_fixed_selector_has_no_update_timing]
    failed = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {e}")
    print("all passed" if not failed else f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
