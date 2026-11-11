"""Context-selection scoring signals.

The proposal's conceptual framework rests on three interacting criteria that
determine the quality of a context window. This module implements each as a
pure function that scores every candidate instance in a pool, returning a
vector of scores (higher = more desirable to include):

* :func:`temporal_relevance`  -- how recent the instance is (the recency axis).
* :func:`feature_relevance`   -- how close it is to the query in feature space.
* :func:`class_utility`       -- its marginal value given class imbalance/cost.

Keeping the signals separate and reusable is deliberate: the recency and
similarity baselines are each "select top-k by one signal", and the learned
policy is "select top-k by a learned weighting of all three". Because they
share these exact functions, the claim that those two baselines are degenerate
parameterisations of the policy is true by construction. The class-ratio
baseline is a stratified draw and is NOT a parameterisation of the policy
(the class-utility corner yields the k rarest-class instances instead).
"""
from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Z-score features; returns (X_std, mean, std). Guards zero variance."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (X - mean) / std, mean, std


def temporal_relevance(t_buf: np.ndarray, t_now: int, tau: float | None = None) -> np.ndarray:
    """Exponential recency score in ``(0, 1]`` -- 1 for the newest instance.

    ``tau`` is the decay constant (in stream steps). If ``None`` it is set to
    the mean age of the pool, giving a scale-free default. Larger ``tau`` -> a
    flatter, longer memory.
    """
    age = np.asarray(t_now - t_buf, dtype=np.float64)
    age = np.clip(age, 0, None)
    if tau is None:
        tau = float(age.mean()) if age.size and age.mean() > 0 else 1.0
    return np.exp(-age / max(tau, 1e-9))


def feature_relevance(
    X_buf: np.ndarray,
    query_x: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Proximity of each candidate to the query in standardized feature space.

    Score is ``1 / (1 + distance)`` in ``(0, 1]`` -- 1 when identical. Standard
    scaling uses pool statistics unless ``mean``/``std`` are supplied (e.g. a
    running estimate), so no single feature dominates the distance.
    """
    if X_buf.size == 0:
        return np.empty(0, dtype=np.float64)
    Xs, mu, sd = standardize(X_buf, mean, std)
    qs = (np.atleast_2d(query_x) - mu) / sd
    dist = np.linalg.norm(Xs - qs, axis=1)
    return 1.0 / (1.0 + dist)


def class_utility(
    y_buf: np.ndarray,
    n_classes: int,
    cost: np.ndarray | None = None,
) -> np.ndarray:
    """Per-instance utility from inverse class frequency (x optional cost).

    Rare classes score higher, so a context built from this signal is not
    starved of minority examples. ``cost`` optionally scales each class by its
    misclassification cost, capturing the asymmetric-cost point in the proposal.
    """
    if y_buf.size == 0:
        return np.empty(0, dtype=np.float64)
    counts = np.bincount(y_buf, minlength=n_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    inv = 1.0 / (freq + 1e-9)
    if cost is not None:
        inv = inv * np.asarray(cost, dtype=np.float64)
    weight = inv / inv.sum()
    return weight[y_buf]


def zscore_pool(scores: np.ndarray) -> np.ndarray:
    """Normalise a signal across the candidate pool (mean 0, unit std).

    Used before combining signals so the learned weights operate on comparable
    scales rather than being dominated by whichever signal has the largest raw
    magnitude.
    """
    if scores.size == 0:
        return scores
    mu = scores.mean()
    sd = scores.std()
    if sd < 1e-9:
        return np.zeros_like(scores)
    return (scores - mu) / sd
