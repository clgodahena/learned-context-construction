"""TabPFN v2 in-context classifier adapter.

Wraps the official ``tabpfn`` package (Hollmann et al., 2025, *Nature* 637).
TabPFN performs in-context learning: ``fit`` simply stores the context, and the
transformer forward pass happens at ``predict_proba``. Adapting it to the
streaming harness is therefore a matter of calling ``fit`` on the freshly
*selected* context at each step and predicting the single query point.

The ``tabpfn`` package is imported lazily so the rest of the prototype runs
without it (e.g. in CI). Install with ``pip install tabpfn`` and run on a CUDA
GPU for the throughput the proposal assumes; ``device="cpu"`` works but is slow.

Model weights are downloaded from HuggingFace on first use and are subject to
the TabPFN licence -- accept it before batch runs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class TabPFNInContextClassifier:
    """Frozen TabPFN conditioned on the selected context.

    Parameters
    ----------
    n_classes : int
        Number of classes in the stream.
    device : {"cuda", "cpu"} or None
        Compute device. ``None`` auto-selects CUDA when available.
    max_context : int
        Hard cap on context size handed to TabPFN (protects latency and stays
        within pretraining limits). The selector's ``k`` should normally be
        <= this; contexts larger than the cap are truncated to the most recent
        rows as a safety net.
    model_version : str, optional
        Optional TabPFN model version identifier (e.g. "v2"); ``None`` uses the
        package default.
    ignore_pretraining_limits : bool
        Pass-through to TabPFN to permit contexts beyond its default size/feature
        limits.
    """

    name = "tabpfn"

    def __init__(
        self,
        n_classes: int,
        device: Optional[str] = None,
        max_context: int = 4096,
        model_version: Optional[str] = None,
        ignore_pretraining_limits: bool = True,
    ):
        self.n_classes = n_classes
        self.device = device
        self.max_context = max_context
        self.model_version = model_version
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self._clf = None  # lazy

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            # Apple Silicon unified memory (MPS) -- the intended accelerator on a
            # Mac workstation. Far faster than CPU for TabPFN inference. If TabPFN
            # hits an unimplemented MPS op, set env PYTORCH_ENABLE_MPS_FALLBACK=1
            # (falls back to CPU per-op) or force device="cpu" in the config.
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_clf(self):
        if self._clf is not None:
            return self._clf
        from tabpfn import TabPFNClassifier

        kwargs = dict(
            device=self._resolve_device(),
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        # Model-version selection differs across releases; fall back gracefully.
        if self.model_version is not None:
            try:
                self._clf = TabPFNClassifier(model=self.model_version, **kwargs)
                return self._clf
            except TypeError:
                pass
        self._clf = TabPFNClassifier(**kwargs)
        return self._clf

    def predict_proba(
        self, context_X: np.ndarray, context_y: np.ndarray, query_X: np.ndarray
    ) -> np.ndarray:
        query_X = np.atleast_2d(query_X)

        # Truncate oversized contexts to the most recent rows (latency guard).
        if len(context_y) > self.max_context:
            context_X = context_X[-self.max_context :]
            context_y = context_y[-self.max_context :]

        uniq = np.unique(context_y)
        if len(context_y) == 0 or len(uniq) < 2:
            proba = np.full((len(query_X), self.n_classes), 1.0 / self.n_classes, np.float32)
            if len(uniq) == 1:
                proba[:] = 0.0
                proba[:, int(uniq[0])] = 1.0
            return proba

        clf = self._ensure_clf()
        clf.fit(context_X, context_y)  # stores context (no gradient step)
        p = clf.predict_proba(query_X)
        out = np.zeros((len(query_X), self.n_classes), np.float32)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = p[:, j]
        return out
