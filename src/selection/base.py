"""Context-selection interface.

A :class:`ContextSelector` is the heart of the research: given the historical
buffer and the current query, it returns the subset of past instances to place
in the frozen model's context window. Every method in the study -- the
single-axis baselines and the learned policy -- is one implementation of this
one interface. Swapping selection strategy is therefore a one-line config
change, and measured performance differences are attributable to selection
alone (the classifier is held fixed).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..harness.buffer import BoundedBuffer


class Context:
    """The selected context window handed to the classifier."""

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = indices  # positions within the buffer (for diagnostics)

    def __len__(self) -> int:
        return len(self.y)


class ContextSelector(Protocol):
    name: str

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        """Return up to ``k`` instances from ``buffer`` to condition on."""
        ...


def take(buffer: BoundedBuffer, idx: np.ndarray) -> Context:
    """Helper: materialise a :class:`Context` from buffer positions."""
    idx = np.asarray(idx, dtype=np.int64)
    return Context(X=buffer.X[idx], y=buffer.y[idx], indices=idx)
