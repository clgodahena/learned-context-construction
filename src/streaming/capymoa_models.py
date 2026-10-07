"""CapyMOA streaming-classifier adapters (MOA/Java backend; workstation).

Wraps CapyMOA's Adaptive Random Forest and Streaming Random Patches -- the
mature state of the incremental-ensemble lineage and the standard baselines in
the stream-classification literature. CapyMOA runs on a Java/MOA backend, so it
is not exercised in the CPU-only CI here; this adapter is written against the
documented CapyMOA API (v0.8+) for the workstation. Install with
``pip install capymoa`` and a JDK (see ``requirements-streaming.txt``).

CapyMOA models are schema-driven and consume CapyMOA ``Instance`` objects. The
adapter builds a schema from the stream's feature/label space once, then wraps
each numpy row as a labelled instance. Imports are lazy.
"""
from __future__ import annotations

import numpy as np


class CapyMOAClassifier:
    """Adapter over a CapyMOA classifier (ARF or SRP).

    Parameters
    ----------
    n_classes, n_features : int
        Shape of the stream, used to build the CapyMOA schema.
    model : {"arf", "srp"}
        Adaptive Random Forest or Streaming Random Patches.
    ensemble_size : int
        Number of ensemble members.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        model: str = "arf",
        ensemble_size: int = 30,
        seed: int = 1,
    ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.model_kind = model
        self.name = f"capymoa_{model}"
        self.ensemble_size = ensemble_size
        self.seed = seed
        self._schema = None
        self._clf = None

    def _ensure(self):
        if self._clf is not None:
            return
        from capymoa.stream import NumpyStream

        # Build a schema from a tiny dummy array (feature/label domains only).
        dummy_X = np.zeros((self.n_classes, self.n_features), dtype=np.float32)
        dummy_y = np.arange(self.n_classes)  # ensures all class values are seen
        self._schema = NumpyStream(dummy_X, dummy_y).get_schema()

        if self.model_kind == "arf":
            from capymoa.classifier import AdaptiveRandomForestClassifier as M

            self._clf = M(self._schema, ensemble_size=self.ensemble_size, random_seed=self.seed)
        elif self.model_kind == "srp":
            from capymoa.classifier import StreamingRandomPatches as M

            self._clf = M(self._schema, ensemble_size=self.ensemble_size, random_seed=self.seed)
        else:
            raise ValueError(f"unknown capymoa model: {self.model_kind}")

    def _instance(self, x: np.ndarray, y: int | None):
        from capymoa.instance import LabeledInstance

        y_val = 0 if y is None else int(y)
        return LabeledInstance.from_array(self._schema, np.asarray(x, dtype=np.float64), y_val)

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        self._ensure()
        proba = self._clf.predict_proba(self._instance(x, None))
        out = np.asarray(proba, dtype=np.float64).ravel()
        if out.size < self.n_classes:
            out = np.pad(out, (0, self.n_classes - out.size))
        s = out.sum()
        return out / s if s > 0 else np.full(self.n_classes, 1.0 / self.n_classes)

    def learn_one(self, x: np.ndarray, y: int) -> None:
        self._ensure()
        self._clf.train(self._instance(x, y))
