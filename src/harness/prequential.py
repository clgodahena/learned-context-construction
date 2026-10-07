"""Prequential (test-then-train) evaluation loop.

This is the streaming harness. For each arriving instance it:

1. selects a context from the buffer (if warmed up),
2. asks the frozen classifier to predict the query from that context,
3. scores the prediction against the true label (test),
4. adds the instance to the buffer, optionally after a label delay (train).

Label delay models the reality that ground truth arrives later than the
prediction is needed; with ``label_delay=0`` the label is available immediately
after prediction. The loop also records the *context composition* per step
(class balance, temporal span) which the dashboard visualises -- the diagnostic
that reveals *why* a strategy wins, not merely that it does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..classifiers.base import InContextClassifier
from ..data.streams import Instance, Stream
from ..selection.base import ContextSelector
from .buffer import BoundedBuffer
from .metrics import PrequentialMetrics


@dataclass
class RunResult:
    summary: dict
    trajectory: list
    composition: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)
    predictions: list = field(default_factory=list)  # (t, y_true, y_pred) when recorded


def run_prequential(
    stream: Stream,
    selector: ContextSelector,
    classifier: InContextClassifier,
    *,
    k: int = 512,
    capacity: Optional[int] = None,
    warmup: int = 100,
    label_delay: int = 0,
    positive_class: int = 1,
    metric_window: int = 1000,
    snapshot_every: int = 250,
    track_composition: bool = True,
    max_steps: Optional[int] = None,
    time_it: bool = False,
    progress_every: int = 0,
    record_predictions: bool = False,
) -> RunResult:
    """Run one prequential evaluation and return metrics + diagnostics.

    Parameters mirror the config file. ``k`` is the context size, ``capacity``
    the buffer bound (``None`` = unbounded), ``warmup`` the number of instances
    to accumulate before the first real prediction. ``progress_every`` > 0 prints
    a heartbeat (rate + ETA) every N scored instances so a slow run never looks
    frozen.
    """
    import time

    total = getattr(stream, "n", None)
    if max_steps is not None:
        total = max_steps if total is None else min(total, max_steps)
    t_start = time.perf_counter()
    scored = 0
    buffer = BoundedBuffer(capacity=capacity)
    metrics = PrequentialMetrics(
        n_classes=stream.n_classes, positive_class=positive_class, window=metric_window
    )
    pending: list[tuple[int, Instance, Optional[bool]]] = []  # delayed labels
    result = RunResult(summary={}, trajectory=[])

    rng = np.random.default_rng(0)

    for step, inst in enumerate(stream):
        if max_steps is not None and step >= max_steps:
            break

        # --- release any labels whose delay has elapsed ---
        if label_delay > 0 and pending:
            ready = [p for p in pending if inst.t - p[0] >= label_delay]
            for _, past_inst, fb in ready:
                buffer.add(past_inst, feedback=fb)
            pending = [p for p in pending if inst.t - p[0] < label_delay]

        # --- test ---
        if len(buffer) >= warmup:
            t0 = time.perf_counter() if time_it else 0.0
            ctx = selector.select(buffer, inst.x, k)
            proba = classifier.predict_proba(ctx.X, ctx.y, inst.x)[0]
            if time_it:
                result.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            y_pred = int(np.argmax(proba))
            score_pos = float(proba[positive_class]) if proba.shape[0] > positive_class else 0.0
            correct = y_pred == inst.y
            metrics.update(inst.y, y_pred, score_pos)
            if record_predictions:
                result.predictions.append((int(inst.t), int(inst.y), y_pred))

            # Feed prediction outcome back to online selectors (e.g. the learned
            # policy). Prequential evaluation reveals the label at test time, so
            # the update is available immediately regardless of buffer delay.
            update = getattr(selector, "update", None)
            if callable(update):
                others = np.delete(proba, inst.y) if proba.shape[0] > 1 else np.array([0.0])
                margin = float(proba[inst.y] - (others.max() if others.size else 0.0))
                update(bool(correct), margin=margin, y_true=int(inst.y))

            if track_composition and len(ctx) > 0:
                cy = ctx.y
                result.composition.append(
                    {
                        "t": inst.t,
                        "ctx_size": int(len(ctx)),
                        "ctx_pos_rate": float(np.mean(cy == positive_class)),
                        "ctx_temporal_span": int(
                            np.ptp(buffer.t[ctx.indices]) if len(ctx.indices) else 0
                        ),
                    }
                )

            if step % snapshot_every == 0:
                metrics.snapshot(inst.t)

            scored += 1
            if progress_every and scored % progress_every == 0:
                el = time.perf_counter() - t_start
                rate = scored / el if el > 0 else 0.0
                msg = (f"[prequential] {scored} scored ({rate:.1f}/s, {el:.0f}s"
                       f", bal_acc={metrics.balanced_accuracy:.3f}")
                if total:
                    remaining = max(0, total - step) / rate if rate > 0 else 0
                    msg += f", ~{remaining/60:.1f} min left"
                print(msg + ")", flush=True)
        else:
            # Cold start: no prediction scored, just fill the buffer.
            correct = None

        # --- train (add to buffer, respecting label delay) ---
        if label_delay > 0:
            pending.append((inst.t, inst, correct))
        else:
            buffer.add(inst, feedback=correct)

    metrics.snapshot(getattr(inst, "t", 0))
    result.summary = metrics.summary()
    result.trajectory = metrics.trajectory
    if result.latencies_ms:
        arr = np.asarray(result.latencies_ms)
        result.summary["latency_ms_p50"] = float(np.percentile(arr, 50))
        result.summary["latency_ms_p95"] = float(np.percentile(arr, 95))
    return result


def run_prequential_direct(
    stream: Stream,
    classifier,
    *,
    warmup: int = 100,
    positive_class: int = 1,
    metric_window: int = 1000,
    snapshot_every: int = 250,
    max_steps: Optional[int] = None,
    time_it: bool = False,
    progress_every: int = 0,
) -> RunResult:
    """Prequential evaluation for a *direct* streaming classifier (ARF/SRP).

    Mirrors :func:`run_prequential` but for end-to-end learners that maintain
    their own state: predict each instance, score it, then ``learn_one``. The
    same metrics are used so results are comparable with the in-context path.
    Instances before ``warmup`` are learned from but not scored, matching the
    in-context path's cold start. ``progress_every`` > 0 prints a heartbeat.
    """
    import time

    total = getattr(stream, "n", None)
    if max_steps is not None:
        total = max_steps if total is None else min(total, max_steps)
    t_start = time.perf_counter()
    scored = 0
    metrics = PrequentialMetrics(
        n_classes=stream.n_classes, positive_class=positive_class, window=metric_window
    )
    result = RunResult(summary={}, trajectory=[])
    last_t = 0

    for step, inst in enumerate(stream):
        if max_steps is not None and step >= max_steps:
            break
        last_t = inst.t
        if step >= warmup:
            t0 = time.perf_counter() if time_it else 0.0
            proba = np.asarray(classifier.predict_proba_one(inst.x), dtype=np.float64)
            if time_it:
                result.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            y_pred = int(np.argmax(proba))
            score_pos = float(proba[positive_class]) if proba.shape[0] > positive_class else 0.0
            metrics.update(inst.y, y_pred, score_pos)
            if step % snapshot_every == 0:
                metrics.snapshot(inst.t)
            scored += 1
            if progress_every and scored % progress_every == 0:
                el = time.perf_counter() - t_start
                rate = scored / el if el > 0 else 0.0
                msg = f"[prequential] {scored} scored ({rate:.1f}/s, {el:.0f}s"
                if total:
                    remaining = max(0, total - step) / rate if rate > 0 else 0
                    msg += f", ~{remaining/60:.1f} min left"
                print(msg + ")", flush=True)
        classifier.learn_one(inst.x, inst.y)

    metrics.snapshot(last_t)
    result.summary = metrics.summary()
    result.trajectory = metrics.trajectory
    if result.latencies_ms:
        arr = np.asarray(result.latencies_ms)
        result.summary["latency_ms_p50"] = float(np.percentile(arr, 50))
        result.summary["latency_ms_p95"] = float(np.percentile(arr, 95))
    return result
