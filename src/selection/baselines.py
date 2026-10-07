"""Single-axis context-selection baselines.

These are the incumbent heuristics the proposal argues are each defective along
the axes they ignore. All four are implemented:

* :class:`SlidingWindowSelector` -- recency axis (the Lourenco/Gama baseline).
* :class:`SimilaritySelector`    -- feature-space axis (LoCalPFN-style kNN).
* :class:`ClassRatioSelector`    -- class-utility axis (resampling-style).
* :class:`RandomSelector`        -- uniform sample; a sanity floor.

Each is "select top-k by one signal" (see ``selection/signals.py``). The learned
policy that arbitrates across all three signals lives in
``selection/learned_policy.py``.
"""
from __future__ import annotations

import numpy as np

from ..harness.buffer import BoundedBuffer
from . import signals
from .base import Context, take


class SlidingWindowSelector:
    """Recency baseline: the ``k`` most recently seen instances.

    This is the strategy shown to let TabPFN outperform streaming ensembles on
    non-stationary data -- the result the Month-2 checkpoint reproduces.
    """

    name = "sliding_window"

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        n = len(buffer)
        start = max(0, n - k)
        idx = np.arange(start, n, dtype=np.int64)
        return take(buffer, idx)


class RandomSelector:
    """Uniformly sampled context -- a floor for the value of *any* structure."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        n = len(buffer)
        if n <= k:
            return take(buffer, np.arange(n, dtype=np.int64))
        idx = self._rng.choice(n, size=k, replace=False)
        idx.sort()
        return take(buffer, idx)


class SimilaritySelector:
    """Feature-space axis (LoCalPFN-style): the ``k`` nearest neighbours.

    Standardises features on the pool and takes the ``k`` instances closest to
    the query in feature space. Blind to temporal validity by design -- it will
    happily retrieve neighbours from a superseded concept, the failure mode the
    proposal attributes to this axis.

    ``candidate_pool`` optionally restricts the search to the most recent N
    instances, bounding cost on long streams (``None`` = whole buffer).
    """

    name = "similarity"

    def __init__(self, candidate_pool: int | None = None):
        self.candidate_pool = candidate_pool

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        n = len(buffer)
        lo = 0 if self.candidate_pool is None else max(0, n - self.candidate_pool)
        pool_idx = np.arange(lo, n, dtype=np.int64)
        if len(pool_idx) <= k:
            return take(buffer, pool_idx)
        rel = signals.feature_relevance(buffer.X[pool_idx], query_x)
        top = np.argpartition(-rel, k)[:k]
        idx = np.sort(pool_idx[top])
        return take(buffer, idx)


class ClassRatioSelector:
    """Class-utility axis: a class-balanced (stratified) context.

    Fills the context so that a target fraction is the minority/positive class,
    drawing the most recent instances within each class. Preserves minority
    representation under severe imbalance, but -- ignoring feature and temporal
    validity beyond recency-within-class -- cannot tell a still-valid minority
    example from one rendered obsolete by drift.

    Parameters
    ----------
    positive_class : int
        The minority / positive class to protect.
    target_pos_ratio : float
        Desired fraction of the context drawn from the positive class. If the
        buffer lacks enough of either class, the remainder is filled from the
        other class (most recent first).
    """

    name = "class_ratio"

    def __init__(self, positive_class: int = 1, target_pos_ratio: float = 0.5):
        self.positive_class = positive_class
        self.target_pos_ratio = target_pos_ratio

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        n = len(buffer)
        if n <= k:
            return take(buffer, np.arange(n, dtype=np.int64))
        y = buffer.y
        pos_idx = np.where(y == self.positive_class)[0]
        neg_idx = np.where(y != self.positive_class)[0]

        want_pos = int(round(k * self.target_pos_ratio))
        want_neg = k - want_pos
        # Most recent within each class (indices are in arrival order).
        take_pos = pos_idx[-want_pos:] if want_pos > 0 else np.array([], dtype=np.int64)
        take_neg = neg_idx[-want_neg:] if want_neg > 0 else np.array([], dtype=np.int64)

        # Backfill any shortfall from the other class to reach k.
        chosen = np.concatenate([take_pos, take_neg])
        if len(chosen) < k:
            remaining = np.setdiff1d(np.arange(n, dtype=np.int64), chosen, assume_unique=False)
            backfill = remaining[-(k - len(chosen)):]
            chosen = np.concatenate([chosen, backfill])

        return take(buffer, np.sort(chosen.astype(np.int64)))
