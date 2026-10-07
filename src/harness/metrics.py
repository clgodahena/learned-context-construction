"""Prequential metrics for imbalanced, drifting streams.

Headline accuracy is inadequate under severe imbalance, so (per the proposal's
evaluation protocol) reporting centres on balanced accuracy, area under the
precision--recall curve, per-class recall, and prequential kappa statistics
that correct for chance and for temporal autocorrelation.

All metrics are computed *prequentially*: each instance is tested before it is
learned from, and metrics accumulate over the stream. Windowed variants
(fading over the last ``window`` instances) track how performance evolves
across drift, which cumulative figures hide.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PrequentialMetrics:
    """Online accumulator over per-instance (y_true, y_pred, score) triples.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    positive_class : int
        Index of the minority / positive class for PR-based metrics.
    window : int
        Size of the trailing window for windowed metrics and AUPRC.
    """

    n_classes: int = 2
    positive_class: int = 1
    window: int = 1000

    # cumulative confusion counts
    _cm: np.ndarray = field(init=False)
    _n: int = field(default=0, init=False)
    _correct: int = field(default=0, init=False)
    # kappa baselines
    _prev_y: Optional[int] = field(default=None, init=False)
    _persist_correct: int = field(default=0, init=False)
    _class_counts: np.ndarray = field(init=False)
    # windowed buffers
    _win_correct: deque = field(init=False)
    _win_ytrue: deque = field(init=False)
    _win_score: deque = field(init=False)
    # trajectory (sampled)
    trajectory: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self._class_counts = np.zeros(self.n_classes, dtype=np.int64)
        self._win_correct = deque(maxlen=self.window)
        self._win_ytrue = deque(maxlen=self.window)
        self._win_score = deque(maxlen=self.window)

    def update(self, y_true: int, y_pred: int, score_pos: float) -> None:
        """Record one prediction. ``score_pos`` is P(positive class)."""
        self._cm[y_true, y_pred] += 1
        self._n += 1
        correct = int(y_true == y_pred)
        self._correct += correct
        # temporal (persistence) baseline: predict previous label
        if self._prev_y is not None:
            self._persist_correct += int(self._prev_y == y_true)
        self._prev_y = y_true
        self._class_counts[y_true] += 1
        self._win_correct.append(correct)
        self._win_ytrue.append(y_true)
        self._win_score.append(score_pos)

    # --- cumulative metrics ---------------------------------------------
    @property
    def accuracy(self) -> float:
        return self._correct / self._n if self._n else 0.0

    @property
    def per_class_recall(self) -> np.ndarray:
        recalls = np.full(self.n_classes, np.nan)
        for c in range(self.n_classes):
            support = self._cm[c].sum()
            if support:
                recalls[c] = self._cm[c, c] / support
        return recalls

    @property
    def balanced_accuracy(self) -> float:
        r = self.per_class_recall
        return float(np.nanmean(r)) if np.any(~np.isnan(r)) else 0.0

    @property
    def kappa_m(self) -> float:
        """Kappa vs a majority-class classifier (p0 - pc) / (1 - pc)."""
        if self._n == 0:
            return 0.0
        p0 = self.accuracy
        pc = self._class_counts.max() / self._n
        return (p0 - pc) / (1 - pc) if pc < 1 else 0.0

    @property
    def kappa_temporal(self) -> float:
        """Kappa vs a no-change (persistence) classifier."""
        if self._n <= 1:
            return 0.0
        p0 = self.accuracy
        pe = self._persist_correct / (self._n - 1)
        return (p0 - pe) / (1 - pe) if pe < 1 else 0.0

    # --- windowed metrics -----------------------------------------------
    @property
    def windowed_accuracy(self) -> float:
        return float(np.mean(self._win_correct)) if self._win_correct else 0.0

    @property
    def windowed_auprc(self) -> float:
        """Average precision over the trailing window (needs both classes)."""
        y = np.asarray(self._win_ytrue)
        s = np.asarray(self._win_score)
        yb = (y == self.positive_class).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            return float("nan")
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(yb, s))

    def snapshot(self, t: int) -> dict:
        """Capture the current metric values (for the trajectory / dashboard)."""
        row = {
            "t": t,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "kappa_m": self.kappa_m,
            "kappa_temporal": self.kappa_temporal,
            "windowed_accuracy": self.windowed_accuracy,
            "windowed_auprc": self.windowed_auprc,
        }
        r = self.per_class_recall
        for c in range(self.n_classes):
            row[f"recall_c{c}"] = float(r[c]) if not np.isnan(r[c]) else float("nan")
        self.trajectory.append(row)
        return row

    def summary(self) -> dict:
        r = self.per_class_recall
        return {
            "n": self._n,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "kappa_m": self.kappa_m,
            "kappa_temporal": self.kappa_temporal,
            "windowed_auprc": self.windowed_auprc,
            "per_class_recall": [None if np.isnan(v) else float(v) for v in r],
        }
