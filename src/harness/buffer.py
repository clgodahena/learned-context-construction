"""Bounded historical instance buffer.

The buffer holds the pool of past instances that a :class:`ContextSelector`
draws on to build a context window. It is deliberately richer than a plain
ring buffer: alongside each instance's features and label it records the
arrival time and a slot for *prediction feedback* (whether the frozen
classifier got that instance right when it was first seen). The learned policy
in a later phase consumes that feedback signal, so it is captured from day one
even though the sliding-window baseline ignores it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.streams import Instance


class BoundedBuffer:
    """A capacity-bounded store of past instances with FIFO eviction.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of instances retained. ``None`` means unbounded (the
        buffer grows with the stream). A finite capacity bounds memory, as the
        proposal requires for streaming deployment.
    """

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._t: list[int] = []
        self._feedback: list[Optional[bool]] = []

    def __len__(self) -> int:
        return len(self._y)

    def add(self, inst: Instance, feedback: Optional[bool] = None) -> None:
        """Append an instance, evicting the oldest if over capacity."""
        self._X.append(inst.x)
        self._y.append(inst.y)
        self._t.append(inst.t)
        self._feedback.append(feedback)
        if self.capacity is not None and len(self._y) > self.capacity:
            self._X.pop(0)
            self._y.pop(0)
            self._t.pop(0)
            self._feedback.pop(0)

    # --- vectorised views (materialised on demand) -----------------------
    @property
    def X(self) -> np.ndarray:
        return np.asarray(self._X, dtype=np.float32) if self._X else np.empty((0, 0), np.float32)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self._y, dtype=np.int64)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self._t, dtype=np.int64)

    @property
    def feedback(self) -> np.ndarray:
        """Prediction-correctness per buffered instance (NaN where unknown)."""
        return np.asarray(
            [np.nan if f is None else float(f) for f in self._feedback], dtype=np.float32
        )
