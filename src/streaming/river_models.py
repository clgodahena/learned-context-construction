"""River streaming-classifier adapters (pure-Python; workstation).

Wraps River's Adaptive Random Forest, Streaming Random Patches and Hoeffding
tree behind :class:`~src.streaming.base.DirectClassifier`. River is pure Python
(no Java), so this is the lightest way to get real ARF/SRP baselines. Install
with ``pip install river`` (see ``requirements-streaming.txt``).

River consumes feature *dicts* and returns probability dicts; the adapter
converts to/from the numpy rows the harness uses. Imports are lazy so the rest
of the prototype runs without River installed.
"""
from __future__ import annotations

import numpy as np


class RiverClassifier:
    """Adapter over a River classifier.

    Parameters
    ----------
    n_classes : int
        Number of classes in the stream.
    model : {"arf", "srp", "hoeffding"}
        Which River model to build.
    n_models : int
        Ensemble size for arf/srp.
    seed : int
        River random seed.
    """

    def __init__(self, n_classes: int, model: str = "arf", n_models: int = 10, seed: int = 0):
        self.n_classes = n_classes
        self.model_kind = model
        self.name = f"river_{model}"
        self._model = self._build(model, n_models, seed)

    def _build(self, model: str, n_models: int, seed: int):
        if model == "arf":
            from river import forest

            return forest.ARFClassifier(n_models=n_models, seed=seed)
        if model == "srp":
            # SRP lives under ensemble in recent River versions.
            from river import ensemble, tree

            return ensemble.SRPClassifier(
                model=tree.HoeffdingTreeClassifier(), n_models=n_models, seed=seed
            )
        if model == "hoeffding":
            from river import tree

            return tree.HoeffdingTreeClassifier()
        raise ValueError(f"unknown river model: {model}")

    @staticmethod
    def _to_dict(x: np.ndarray) -> dict:
        return {i: float(v) for i, v in enumerate(np.ravel(x))}

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba_one(self._to_dict(x))
        out = np.zeros(self.n_classes)
        for label, p in proba.items():
            if isinstance(label, (int, np.integer)) and 0 <= int(label) < self.n_classes:
                out[int(label)] = p
        s = out.sum()
        return out / s if s > 0 else np.full(self.n_classes, 1.0 / self.n_classes)

    def learn_one(self, x: np.ndarray, y: int) -> None:
        self._model.learn_one(self._to_dict(x), int(y))
