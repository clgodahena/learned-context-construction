"""Data streams for the prototype.

A *stream* yields ``Instance`` objects one at a time, in temporal order. Two
kinds are provided:

* :class:`SyntheticDriftStream` -- fully offline, generates a stream with
  injected concept drift of known type, location and severity. Used for tests
  and as the foundation for the drift injector (proposal Section 5).
* :class:`OpenMLStream` / :func:`electricity` -- loads a real, temporally
  ordered benchmark (Electricity) via scikit-learn's OpenML fetcher, preserving
  arrival order. This is the literature-comparable stream for the Month-2
  baseline-reproduction checkpoint.

The interface is deliberately minimal so that CapyMOA / River loaders can be
added later behind the same :class:`Stream` protocol without touching the
harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

import numpy as np


@dataclass
class Instance:
    """A single stream element.

    Attributes
    ----------
    x : np.ndarray
        Feature vector, shape ``(n_features,)``.
    y : int
        Class label.
    t : int
        Arrival index (0-based), i.e. logical timestamp.
    """

    x: np.ndarray
    y: int
    t: int


class Stream(Protocol):
    """Structural type for anything the harness can consume."""

    n_features: int
    n_classes: int

    def __iter__(self) -> Iterator[Instance]:  # pragma: no cover - protocol
        ...


class SyntheticDriftStream:
    """Binary stream with a rotating linear decision boundary.

    Concept drift is injected by rotating the separating hyperplane at the
    configured drift points. This gives *ground-truth* drift locations, which
    real streams lack -- the basis for the semi-synthetic protocol in the
    proposal. Class imbalance is controllable via ``pos_rate``.

    Parameters
    ----------
    n : int
        Number of instances to emit.
    n_features : int
        Feature dimensionality.
    drift_points : sequence of int, optional
        Arrival indices at which the concept changes abruptly. ``None`` gives a
        stationary stream.
    pos_rate : float
        Target fraction of positive-class instances (controls imbalance).
    drift_kind : {"abrupt", "gradual"}
        Abrupt swaps the concept at the drift point; gradual interpolates over
        ``gradual_width`` instances.
    gradual_width : int
        Width of the transition region for gradual drift.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n: int = 20_000,
        n_features: int = 6,
        drift_points: Optional[list[int]] = None,
        pos_rate: float = 0.5,
        drift_kind: str = "abrupt",
        gradual_width: int = 500,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.n_features = n_features
        self.n_classes = 2
        self.drift_points = sorted(drift_points or [])
        self.pos_rate = pos_rate
        self.drift_kind = drift_kind
        self.gradual_width = gradual_width
        self.seed = seed

    def _concept_weights(self, concept_idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1000 * (concept_idx + 1))
        w = rng.normal(size=self.n_features)
        return w / (np.linalg.norm(w) + 1e-12)

    def _concept_index_at(self, t: int) -> int:
        return int(np.searchsorted(self.drift_points, t, side="right"))

    def __iter__(self) -> Iterator[Instance]:
        rng = np.random.default_rng(self.seed)
        # Pre-compute per-concept weight vectors.
        n_concepts = len(self.drift_points) + 1
        weights = [self._concept_weights(i) for i in range(n_concepts)]
        # Bias chosen so that the label threshold yields ~pos_rate positives.
        # (pos_rate < 0.5 -> negative bias -> positives are the minority class.)
        bias = float(np.sqrt(self.n_features)) * (self.pos_rate - 0.5) * 2.0

        for t in range(self.n):
            x = rng.normal(size=self.n_features)
            ci = self._concept_index_at(t)
            w = weights[ci]

            if self.drift_kind == "gradual" and ci > 0:
                dp = self.drift_points[ci - 1]
                frac = (t - dp) / max(1, self.gradual_width)
                if 0.0 <= frac < 1.0:
                    # Blend previous and current concept during transition.
                    w = (1 - frac) * weights[ci - 1] + frac * weights[ci]
                    w = w / (np.linalg.norm(w) + 1e-12)

            score = float(x @ w) + bias
            # Add label noise for realism.
            p = 1.0 / (1.0 + np.exp(-score))
            y = int(rng.random() < p)
            yield Instance(x=x.astype(np.float32), y=y, t=t)


class OpenMLStream:
    """Real benchmark stream loaded via ``sklearn.datasets.fetch_openml``.

    The rows are emitted in their original (temporal) order. Features are
    label-free floats; a single label column is mapped to integer classes.

    Notes
    -----
    Requires network access to OpenML on first use (results are cached by
    scikit-learn afterwards). If the fetch fails in a restricted sandbox, use
    :class:`SyntheticDriftStream` for offline validation.
    """

    def __init__(self, name: str, version: int = 1, target_positive: Optional[str] = None):
        import pandas as pd
        from sklearn.datasets import fetch_openml

        ds = fetch_openml(name=name, version=version, as_frame=True)
        frame = ds.frame
        target_col = ds.target.name
        X = frame.drop(columns=[target_col])
        y_raw = frame[target_col]

        # Encode any non-numeric feature columns numerically, preserving order.
        # Use pandas' dtype check (np.issubdtype cannot interpret CategoricalDtype,
        # which OpenML uses for columns like Electricity's 'day').
        X_num = X.copy()
        for col in X_num.columns:
            if not pd.api.types.is_numeric_dtype(X_num[col]):
                X_num[col] = X_num[col].astype("category").cat.codes
        self._X = X_num.to_numpy(dtype=np.float32)

        classes = sorted(y_raw.astype(str).unique())
        if target_positive is not None and target_positive in classes:
            # Ensure the positive class maps to 1.
            classes = [c for c in classes if c != target_positive] + [target_positive]
        self._class_to_int = {c: i for i, c in enumerate(classes)}
        self._y = y_raw.astype(str).map(self._class_to_int).to_numpy(dtype=np.int64)

        self.n_features = self._X.shape[1]
        self.n_classes = len(classes)
        self.n = self._X.shape[0]  # stream length (for progress/ETA)

    def __iter__(self) -> Iterator[Instance]:
        for t in range(self._X.shape[0]):
            yield Instance(x=self._X[t], y=int(self._y[t]), t=t)


def electricity() -> OpenMLStream:
    """The Electricity (elec2) benchmark: 45,312 rows, 8 features, binary.

    The canonical real-world concept-drift benchmark. The positive class ("UP")
    is mapped to 1.
    """
    return OpenMLStream(name="electricity", version=1, target_positive="UP")


class ArrayStream:
    """A Stream over in-memory (X, y) arrays, emitted in row (temporal) order.

    Used for real benchmarks that are not fetched through OpenML (e.g. Covertype
    via scikit-learn's built-in loader). Features are floats; labels are ints.
    """

    def __init__(self, X, y):
        self._X = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.int64)
        self.n_features = self._X.shape[1]
        self.n_classes = int(self._y.max()) + 1 if self._y.size else 0
        self.n = self._X.shape[0]

    def __iter__(self) -> Iterator[Instance]:
        for t in range(self._X.shape[0]):
            yield Instance(x=self._X[t], y=int(self._y[t]), t=t)


def airlines() -> OpenMLStream:
    """Airlines flight-delay benchmark (OpenML): ~539,383 rows, 7 features, binary.

    Naturally occurring, season-dependent drift at a scale that tests throughput.
    Large — cap the stream length via ``run.max_steps``. The positive class
    (delayed, "1") is mapped to 1.
    """
    return OpenMLStream(name="airlines", version=1, target_positive="1")


def covertype() -> ArrayStream:
    """Forest Covertype (scikit-learn ``fetch_covtype``): 581,012 rows, 54 features.

    A cartographic benchmark used in the stream literature by emitting rows in
    their natural order, subject to long-term distributional shift. The seven
    cover types are binarised as the majority class (Lodgepole Pine, type 2)
    versus the rest, yielding a roughly balanced binary target that drops into
    the binary harness. Large — cap via ``run.max_steps``. Downloads on first
    use and is cached by scikit-learn.
    """
    from sklearn.datasets import fetch_covtype

    ds = fetch_covtype()
    X = np.asarray(ds.data, dtype=np.float32)
    y = (np.asarray(ds.target) == 2).astype(np.int64)  # majority class vs rest
    return ArrayStream(X, y)
