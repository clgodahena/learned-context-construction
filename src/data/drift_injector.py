"""Drift injection onto real (or any) streams.

Real streams carry drift at unknown locations, which makes any claim about
adaptation quality depend on contestable inference about *when* drift occurred.
The proposal's answer is semi-synthetic streams: take a real dataset and inject
drift of known type, location and severity, so the ground truth exists by
construction.

:class:`DriftInjector` wraps any base :class:`~src.data.streams.Stream`,
materialises it, and re-emits it with transformations applied after each drift
point. It also performs controlled class-imbalance shaping (the injected-
imbalance protocol). Every emitted :class:`~src.data.streams.Instance` keeps its
normal fields; the injector additionally exposes ``drift_points`` and
``concept_at(t)`` so evaluation can align metrics to known drift locations.

Mechanisms
----------
* ``label_flip``      -- swap two class labels after the drift point (abrupt
  P(y|x) change; the cleanest controlled concept drift).
* ``feature_rotation``-- apply a fixed random rotation to the feature space
  after the drift point (covariate + concept change).
* ``recurring``       -- cycle a set of concepts, so earlier concepts return
  (the setting where a non-destructive, memory-based policy should shine).

Drift kinds
-----------
* ``abrupt``  -- transformation switches on exactly at the drift point.
* ``gradual`` -- probability of applying the new concept ramps up over
  ``gradual_width`` steps.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .streams import Instance, Stream


class DriftInjector:
    def __init__(
        self,
        base: Stream,
        drift_points: list[int],
        mechanism: str = "label_flip",
        drift_kind: str = "abrupt",
        gradual_width: int = 500,
        recurring: bool = False,
        target_pos_rate: Optional[float] = None,
        positive_class: int = 1,
        seed: int = 0,
    ):
        self.mechanism = mechanism
        self.drift_kind = drift_kind
        self.gradual_width = gradual_width
        self.recurring = recurring
        self.target_pos_rate = target_pos_rate
        self.positive_class = positive_class
        self.seed = seed

        # Materialise the base stream (drift injection needs the whole sequence).
        insts = list(base)
        self._X = np.stack([i.x for i in insts]).astype(np.float32)
        self._y = np.array([i.y for i in insts], dtype=np.int64)
        self.n_features = base.n_features
        self.n_classes = base.n_classes

        # Optional imbalance shaping happens first (subsample to target rate).
        if target_pos_rate is not None:
            self._apply_imbalance(target_pos_rate)

        self.n = len(self._y)
        self.drift_points = sorted([d for d in drift_points if 0 < d < self.n])
        self._prepare_concepts()

    # ------------------------------------------------------------------
    def _apply_imbalance(self, target_pos_rate: float) -> None:
        rng = np.random.default_rng(self.seed)
        pos = np.where(self._y == self.positive_class)[0]
        neg = np.where(self._y != self.positive_class)[0]
        n_neg = len(neg)
        # positives needed so that pos/(pos+neg) == target, capping by supply
        n_pos_target = int(round(target_pos_rate * n_neg / (1 - target_pos_rate)))
        n_pos_target = min(n_pos_target, len(pos))
        keep_pos = rng.choice(pos, size=n_pos_target, replace=False)
        keep = np.sort(np.concatenate([neg, keep_pos]))
        self._X = self._X[keep]
        self._y = self._y[keep]

    def _prepare_concepts(self) -> None:
        rng = np.random.default_rng(self.seed + 7)
        n_concepts = len(self.drift_points) + 1
        self._rotations = []
        self._label_maps = []
        for ci in range(n_concepts):
            eff = 0 if (self.recurring and ci % 2 == 0) else ci
            # Rotation matrix per concept (identity for concept 0 / recurring base)
            if eff == 0:
                R = np.eye(self.n_features, dtype=np.float32)
            else:
                A = rng.normal(size=(self.n_features, self.n_features))
                Q, _ = np.linalg.qr(A)
                R = Q.astype(np.float32)
            self._rotations.append(R)
            # Label map: flip the two most common classes for odd concepts
            lm = np.arange(self.n_classes)
            if eff % 2 == 1 and self.n_classes >= 2:
                lm[0], lm[1] = 1, 0
            self._label_maps.append(lm)

    def concept_at(self, t: int) -> int:
        return int(np.searchsorted(self.drift_points, t, side="right"))

    def _apply_concept(self, x, y, ci, t, rng) -> tuple[np.ndarray, int]:
        # Gradual: apply new concept with ramping probability in the transition.
        if self.drift_kind == "gradual" and ci > 0:
            dp = self.drift_points[ci - 1]
            frac = (t - dp) / max(1, self.gradual_width)
            if frac < 1.0 and rng.random() > max(0.0, frac):
                ci = ci - 1  # still (probabilistically) the previous concept
        if self.mechanism == "feature_rotation":
            x = (self._rotations[ci] @ x).astype(np.float32)
        elif self.mechanism in ("label_flip", "recurring"):
            y = int(self._label_maps[ci][y])
        return x, y

    def __iter__(self) -> Iterator[Instance]:
        rng = np.random.default_rng(self.seed + 99)
        for t in range(self.n):
            ci = self.concept_at(t)
            x, y = self._apply_concept(self._X[t].copy(), int(self._y[t]), ci, t, rng)
            yield Instance(x=x, y=y, t=t)


class SegmentedStream:
    """Concatenate several streams into one, re-indexing time continuously.

    Used by experiment E5 to build a stream whose *optimal* selection axis shifts
    over its course -- e.g. a balanced-stationary segment (favouring feature
    similarity), then an abruptly-drifting segment (favouring recency), then a
    severely-imbalanced segment (favouring class utility). No single fixed
    weighting is best throughout, so it is the regime in which an online policy
    that adapts its weighting should have a genuine advantage.

    Parameters
    ----------
    segments : list of (name, stream)
        Each stream is materialised and emitted in order. All must share
        n_features and n_classes.
    """

    def __init__(self, segments: list[tuple[str, Stream]]):
        self._insts: list[Instance] = []
        self.segment_boundaries: list[int] = []
        self.segment_names: list[str] = []
        t = 0
        first = segments[0][1]
        self.n_features = first.n_features
        self.n_classes = first.n_classes
        for name, stream in segments:
            start = t
            for inst in stream:
                self._insts.append(Instance(x=inst.x, y=inst.y, t=t))
                t += 1
            self.segment_boundaries.append(start)
            self.segment_names.append(name)
        self.n = len(self._insts)

    def segment_at(self, t: int) -> int:
        return int(np.searchsorted(self.segment_boundaries, t, side="right")) - 1

    def __iter__(self) -> Iterator[Instance]:
        yield from self._insts


def make_classification_stream(
    n: int = 12000, n_features: int = 8, pos_rate: float = 0.5, seed: int = 0
) -> Stream:
    """An offline 'real-like' dataset (sklearn) to demonstrate injection.

    Stands in for a downloaded benchmark when network is unavailable, so the
    injector can be exercised end-to-end without OpenML/CapyMOA. On the
    workstation, wrap :func:`~src.data.streams.electricity` instead.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        weights=[1 - pos_rate, pos_rate],
        class_sep=1.2,
        random_state=seed,
    )

    class _Materialised:
        def __init__(self, X, y):
            self._X = X.astype(np.float32)
            self._y = y.astype(np.int64)
            self.n_features = X.shape[1]
            self.n_classes = int(len(np.unique(y)))
            self.n = len(self._y)

        def __iter__(self):
            for t in range(len(self._y)):
                yield Instance(x=self._X[t], y=int(self._y[t]), t=t)

    return _Materialised(X, y)
