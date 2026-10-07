"""Lightweight reference in-context classifier (no GPU, no downloads).

Mimics the in-context contract -- "condition on this context, classify this
query" -- by fitting a fresh, cheap scikit-learn model on the context at each
step. This is *not* the object of study; it exists so the full pipeline
(streaming, selection, metrics, dashboard) can be validated end-to-end in any
environment, and so CI runs without a GPU. Swap in
:class:`~src.classifiers.tabpfn_clf.TabPFNInContextClassifier` for real runs.
"""
from __future__ import annotations

import numpy as np


class ReferenceInContextClassifier:
    """Refit-on-context reference substrate.

    Parameters
    ----------
    n_classes : int
        Number of classes in the stream.
    kind : {"logreg", "gnb"}
        Underlying sklearn estimator. ``gnb`` (Gaussian NB) is fast and robust
        to tiny contexts; ``logreg`` is a stronger linear baseline.
    """

    name = "reference"

    def __init__(self, n_classes: int, kind: str = "gnb"):
        self.n_classes = n_classes
        self.kind = kind

    def _make(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=200)
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()

    def predict_proba(
        self, context_X: np.ndarray, context_y: np.ndarray, query_X: np.ndarray
    ) -> np.ndarray:
        query_X = np.atleast_2d(query_X)
        classes = np.arange(self.n_classes)
        # Degenerate context: fall back to a uniform / prior estimate.
        uniq = np.unique(context_y)
        if len(context_y) == 0 or len(uniq) < 2:
            proba = np.full((len(query_X), self.n_classes), 1.0 / self.n_classes, np.float32)
            if len(uniq) == 1:
                proba[:] = 0.0
                proba[:, int(uniq[0])] = 1.0
            return proba

        model = self._make()
        model.fit(context_X, context_y)
        p = model.predict_proba(query_X)
        # Align columns to the global class set (model may see a subset).
        out = np.zeros((len(query_X), self.n_classes), np.float32)
        for j, c in enumerate(model.classes_):
            out[:, int(c)] = p[:, j]
        return out
