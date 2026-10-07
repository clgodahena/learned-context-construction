"""Frozen in-context classifier interface.

The classifier is treated as an *interchangeable substrate* (proposal Section
5): it is conditioned on the selected context and asked to classify the query,
with its own parameters held fixed. Because every strategy shares this
interface, differences in measured performance are attributable to context
selection, not to the classifier.

Two implementations:

* :class:`ReferenceInContextClassifier` -- a lightweight sklearn model refit on
  the context each step. No GPU, no downloads; used to validate the harness and
  as a weak reference substrate.
* :class:`TabPFNInContextClassifier`    -- the real tabular foundation model
  (Hollmann et al., 2025), conditioned on the context via a forward pass.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class InContextClassifier(Protocol):
    name: str
    n_classes: int

    def predict_proba(
        self, context_X: np.ndarray, context_y: np.ndarray, query_X: np.ndarray
    ) -> np.ndarray:
        """Return class probabilities for ``query_X`` conditioned on the context.

        Shape ``(len(query_X), n_classes)``. ``query_X`` may be a single row
        reshaped to ``(1, n_features)``.
        """
        ...
