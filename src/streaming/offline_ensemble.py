"""Pure-sklearn online-bagging ensemble -- an always-available ARF stand-in.

This is *not* Adaptive Random Forest. It is a lightweight, dependency-free
incremental ensemble used to (a) validate the direct-evaluation path in any
environment and (b) provide a non-in-context baseline when CapyMOA/River are not
installed. It applies Oza-style online bagging (Poisson-weighted ``partial_fit``)
over linear base learners, with an optional periodic reset of the weakest member
to inject plasticity under drift.

For the reported streaming baselines, use :mod:`src.streaming.river_models` or
:mod:`src.streaming.capymoa_models` (real ARF / SRP).
"""
from __future__ import annotations

import numpy as np


class OnlineBaggingEnsemble:
    name = "online_bagging"

    def __init__(
        self,
        n_classes: int,
        n_models: int = 10,
        lam: float = 6.0,
        reset_every: int | None = 2000,
        seed: int = 0,
    ):
        from sklearn.linear_model import SGDClassifier

        self.n_classes = n_classes
        self.n_models = n_models
        self.lam = lam
        self.reset_every = reset_every
        self._rng = np.random.default_rng(seed)
        self._classes = np.arange(n_classes)
        self._make = lambda: SGDClassifier(loss="log_loss", learning_rate="optimal",
                                           alpha=1e-4)
        self._models = [self._make() for _ in range(n_models)]
        self._fitted = [False] * n_models
        self._correct = np.zeros(n_models)  # rolling competence, for reset
        self._seen = 0

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        agg = np.zeros(self.n_classes)
        n_ready = 0
        for m, ok in zip(self._models, self._fitted):
            if not ok:
                continue
            p = m.predict_proba(x)[0]
            out = np.zeros(self.n_classes)
            for j, c in enumerate(m.classes_):
                out[int(c)] = p[j]
            agg += out
            n_ready += 1
        if n_ready == 0:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        return agg / n_ready

    def learn_one(self, x: np.ndarray, y: int) -> None:
        x2 = np.atleast_2d(x)
        yk = np.array([y])
        self._seen += 1
        for i, m in enumerate(self._models):
            # Online bagging: weight the instance by k ~ Poisson(lambda). Using
            # sample_weight is equivalent to presenting it k times but far cheaper.
            k = self._rng.poisson(self.lam)
            if k == 0:
                continue
            m.partial_fit(x2, yk, classes=self._classes, sample_weight=np.array([float(k)]))
            self._fitted[i] = True
            # Track competence for drift-driven reset.
            self._correct[i] = 0.99 * self._correct[i] + (1.0 if m.predict(x2)[0] == y else 0.0)

        if self.reset_every and self._seen % self.reset_every == 0:
            worst = int(np.argmin(self._correct))
            self._models[worst] = self._make()
            self._fitted[worst] = False
            self._correct[worst] = self._correct.mean()
