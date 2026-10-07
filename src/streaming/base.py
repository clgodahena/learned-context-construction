"""Direct (end-to-end) streaming-classifier interface.

The incumbent streaming methods in the proposal's related work -- Adaptive
Random Forest and Streaming Random Patches -- are *not* in-context models. They
process the raw stream directly: predict each instance, then update their own
parameters from it. They therefore do not fit the (context-selector + frozen
classifier) pattern and instead implement this separate interface, evaluated by
:func:`~src.harness.prequential.run_prequential_direct` over the same metrics so
the two families are directly comparable.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class DirectClassifier(Protocol):
    name: str
    n_classes: int

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities for one instance, shape ``(n_classes,)``."""
        ...

    def learn_one(self, x: np.ndarray, y: int) -> None:
        """Incrementally update the model from one labelled instance."""
        ...
