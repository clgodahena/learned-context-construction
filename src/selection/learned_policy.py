"""Learned context-construction policy -- the central artefact.

For each prediction step the policy scores every candidate in a bounded pool by
a weighted combination of the three signals (temporal, feature-space, class
utility), selects the top-``k``, and -- after the true label arrives -- updates
the weights online using the base model's own prediction outcome. The relative
importance of the signals is learned from feedback and adapts as stream
conditions change, rather than being fixed a priori as in the baselines.

Update rule (refined)
---------------------
The weighting is parameterised by logits ``theta`` (weights = ``softmax(theta)``,
so they stay non-negative and sum to one). Learning treats the logits as a
Gaussian stochastic policy and applies a **weight-perturbation policy gradient**
(REINFORCE / evolution-strategies estimator):

    epsilon ~ N(0, sigma^2 I)
    weights_used = softmax(theta + epsilon)      # explore a nearby weighting
    ... select, predict, observe reward r ...
    advantage = r - baseline                     # EMA baseline -> variance reduction
    theta += lr * advantage * epsilon / sigma^2  # + momentum, then clip

This is a genuine gradient estimate of expected reward w.r.t. ``theta`` -- an
improvement on the earlier emphasis heuristic in three ways: built-in
exploration (the perturbation), variance reduction (the reward baseline), and a
smoother trajectory (momentum). The rule is confined to :meth:`update` /
:meth:`select` so it can be swapped without touching the harness.

Exact degeneracy is preserved for two corners: with ``sigma=0`` (or weights
collapsed onto one signal via :meth:`set_onehot`) the temporal corner is identical
to the sliding-window baseline and the feature corner to the similarity baseline
-- see ``tests/test_components.py``. The class-utility corner is NOT the
stratified class-ratio baseline (it ranks by a per-class constant and fills the
context with the rarest class), so that rule is an external baseline.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..harness.buffer import BoundedBuffer
from . import signals
from .base import Context, take

SIGNAL_NAMES = ("temporal", "feature", "class_utility")


class LearnedPolicySelector:
    """Online-weighted multi-signal context selector.

    Parameters
    ----------
    n_classes, positive_class : int
        Stream class count and the minority/positive class (for class utility).
    candidate_pool : int
        Cap on how many recent instances are scored per step (bounds latency).
    lr : float
        Learning rate for the policy-gradient update.
    sigma : float
        Initial std of the exploration perturbation on the logits. ``0`` disables
        exploration and makes selection deterministic (used for degeneracy).
    sigma_anneal : float, optional
        Exploration is annealed as ``sigma_t = max(sigma_floor, sigma /
        (1 + n_updates / sigma_anneal))`` -- explore early, exploit later.
        ``None`` keeps ``sigma`` fixed. The gradient is normalised by the
        current sigma so its magnitude stays stable as exploration shrinks.
    sigma_floor : float
        Lower bound on the annealed exploration std.
    update_rule : {"policy_gradient", "emphasis"}
        ``policy_gradient`` (default) is the Gaussian REINFORCE estimator;
        ``emphasis`` is the simpler heuristic (moves weights toward signals
        emphasised on correct steps) kept as an ablation.
    baseline_beta : float
        EMA coefficient for the reward baseline (variance reduction).
    momentum : float
        Momentum coefficient on the logit update.
    init_weights : sequence of float, optional
        Initial softmax weights over (temporal, feature, class_utility).
    tau : float, optional
        Decay constant for the temporal signal (``None`` -> adaptive).
    cost : sequence of float, optional
        Per-class misclassification cost for the class-utility signal.
    freeze : bool
        Hold weights fixed (ablation: multi-signal but not learned).
    seed : int
        RNG seed for the exploration perturbations.
    """

    name = "learned_policy"

    def __init__(
        self,
        n_classes: int = 2,
        positive_class: int = 1,
        candidate_pool: int = 4000,
        lr: float = 0.05,
        sigma: float = 0.25,
        sigma_anneal: float | None = 2000.0,
        sigma_floor: float = 0.03,
        update_rule: str = "policy_gradient",
        baseline_beta: float = 0.99,
        momentum: float = 0.5,
        entropy_reg: float = 0.02,
        balanced_reward: bool = True,
        reward_mode: str = "instance",
        reward_window: int = 200,
        init_weights=None,
        tau: float | None = None,
        cost=None,
        freeze: bool = False,
        seed: int = 0,
    ):
        self.n_classes = n_classes
        self.positive_class = positive_class
        self.candidate_pool = candidate_pool
        self.lr = lr
        self.sigma = sigma
        self.sigma_anneal = sigma_anneal
        self.sigma_floor = sigma_floor
        self.update_rule = update_rule
        self.baseline_beta = baseline_beta
        self.momentum = momentum
        self.entropy_reg = entropy_reg
        self.balanced_reward = balanced_reward
        self.reward_mode = reward_mode
        self.reward_window = reward_window
        self._n_updates = 0
        self._class_counts = np.ones(n_classes, dtype=np.float64)  # Laplace-smoothed
        # Rolling per-class correctness for the windowed (metric-aligned) reward.
        self._win = [deque(maxlen=reward_window) for _ in range(n_classes)]
        self._wba_ema = 0.0
        self._wba_init = False
        self._wba_beta = 0.99
        self.tau = tau
        self.cost = None if cost is None else np.asarray(cost, dtype=np.float64)
        self.freeze = freeze
        self._rng = np.random.default_rng(seed)

        if init_weights is None:
            self.theta = np.zeros(3, dtype=np.float64)
        else:
            w = np.asarray(init_weights, dtype=np.float64)
            self.theta = np.log(np.clip(w, 1e-6, None))

        self._baseline = 0.0
        self._baseline_init = False
        self._velocity = np.zeros(3, dtype=np.float64)
        self._last_eps: np.ndarray | None = None
        self._last_emphasis: np.ndarray | None = None
        self.weight_history: list[list[float]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    @property
    def weights(self) -> np.ndarray:
        """Current (mean) softmax weights over (temporal, feature, class_utility)."""
        return self._softmax(self.theta)

    def set_onehot(self, signal: str, sharpness: float = 20.0) -> None:
        """Collapse weights onto one signal (for degeneracy checks/ablations)."""
        i = SIGNAL_NAMES.index(signal)
        self.theta = np.full(3, -sharpness, dtype=np.float64)
        self.theta[i] = sharpness

    def _current_sigma(self) -> float:
        if self.sigma <= 0:
            return 0.0
        if self.sigma_anneal is None:
            return self.sigma
        s = self.sigma / (1.0 + self._n_updates / self.sigma_anneal)
        return max(self.sigma_floor, s)

    # ------------------------------------------------------------------
    def _signal_matrix(self, buffer: BoundedBuffer, query_x: np.ndarray, pool_idx: np.ndarray):
        X = buffer.X[pool_idx]
        y = buffer.y[pool_idx]
        t = buffer.t[pool_idx]
        t_now = int(buffer.t[-1]) + 1
        s_t = signals.zscore_pool(signals.temporal_relevance(t, t_now, self.tau))
        s_f = signals.zscore_pool(signals.feature_relevance(X, query_x))
        s_c = signals.zscore_pool(signals.class_utility(y, self.n_classes, self.cost))
        return np.vstack([s_t, s_f, s_c])  # shape (3, pool)

    def select(self, buffer: BoundedBuffer, query_x: np.ndarray, k: int) -> Context:
        n = len(buffer)
        lo = max(0, n - self.candidate_pool)
        pool_idx = np.arange(lo, n, dtype=np.int64)
        if len(pool_idx) <= k:
            self._last_eps = None
            self._last_emphasis = None
            return take(buffer, pool_idx)

        # Explore a nearby weighting (unless frozen or sigma == 0).
        sigma_t = self._current_sigma()
        if sigma_t > 0 and not self.freeze:
            eps = self._rng.normal(0.0, sigma_t, size=3)
        else:
            eps = np.zeros(3)
        self._last_eps = eps
        self._last_sigma = sigma_t
        w_used = self._softmax(self.theta + eps)

        S = self._signal_matrix(buffer, query_x, pool_idx)  # (3, pool)
        combined = w_used @ S
        top = np.argpartition(-combined, k)[:k]
        self._last_emphasis = S[:, top].mean(axis=1)  # diagnostic

        idx = np.sort(pool_idx[top])
        self.weight_history.append(self.weights.tolist())
        return take(buffer, idx)

    # ------------------------------------------------------------------
    # Delayed feedback support
    # ------------------------------------------------------------------
    def capture_update_state(self) -> dict | None:
        """Snapshot the exploration state of the most recent :meth:`select`.

        Under label delay the harness calls this right after the prediction
        and hands the token back to :meth:`update` when the label finally
        arrives, so the policy-gradient step is credited to the perturbation
        that actually produced that prediction rather than to whatever sample
        was drawn most recently. Returns ``None`` when the last selection was a
        pass-through (pool <= k) and carries no exploration sample.
        """
        if self._last_eps is None:
            return None
        return {"eps": self._last_eps.copy(),
                "sigma": float(getattr(self, "_last_sigma", self.sigma)),
                "emphasis": None if self._last_emphasis is None else self._last_emphasis.copy()}

    def _resolve_token(self, token: dict | None):
        """Return (eps, sigma, emphasis) for this update: the token's if given,
        otherwise the state left by the most recent :meth:`select`."""
        if token is None:
            return self._last_eps, getattr(self, "_last_sigma", self.sigma), self._last_emphasis
        return token["eps"], token["sigma"], token["emphasis"]

    def update(self, correct: bool, margin: float | None = None, y_true: int | None = None,
               token: dict | None = None) -> None:
        """Policy-gradient weight update from prediction feedback.

        Parameters
        ----------
        correct : bool
            Whether the frozen classifier predicted the query correctly.
        margin : float, optional
            Signed confidence margin (p_true - p_other); scales the reward so a
            confident correct call reinforces more than a marginal one.
        y_true : int, optional
            True label of the query. Used to make the reward *class-balanced*, so
            that classifying a rare-class instance correctly is worth more --
            aligning the policy's objective with balanced accuracy rather than
            raw accuracy (which imbalance would otherwise dominate).
        token : dict, optional
            Exploration state captured by :meth:`capture_update_state` at
            prediction time; required for correct credit assignment when the
            label arrives late (label delay). Without it the most recent
            selection's sample is used (immediate-feedback case).
        """
        eps, sigma_t, emph_t = self._resolve_token(token)
        if self.freeze or eps is None:
            return

        if self.reward_mode == "windowed" and y_true is not None:
            # Metric-aligned reward: the improvement in a rolling class-balanced
            # accuracy (mean of per-class recalls over the last `reward_window`
            # instances) relative to its slow EMA. Directly targets the reported
            # metric rather than single-instance correctness (E3).
            self._win[y_true].append(1.0 if correct else 0.0)
            recalls = [float(np.mean(d)) for d in self._win if len(d) > 0]
            wba = float(np.mean(recalls)) if recalls else 0.0
            if not self._wba_init:
                self._wba_ema = wba
                self._wba_init = True
            reward = (wba - self._wba_ema) * 10.0  # scale improvement to ~O(1)
            self._wba_ema = self._wba_beta * self._wba_ema + (1 - self._wba_beta) * wba
        else:
            # Instance reward in [-1, 1], optionally shaped by confidence margin.
            reward = 1.0 if correct else -1.0
            if margin is not None:
                reward *= float(np.clip(abs(margin), 0.05, 1.0))
            # Class-balanced weighting: scale by inverse frequency of the true class.
            if self.balanced_reward and y_true is not None:
                self._class_counts[y_true] += 1
                freq = self._class_counts / self._class_counts.sum()
                reward *= float((1.0 / freq[y_true]) / (1.0 / freq).sum() * self.n_classes)

        # EMA reward baseline for variance reduction.
        if not self._baseline_init:
            self._baseline = reward
            self._baseline_init = True
        else:
            self._baseline = (
                self.baseline_beta * self._baseline + (1 - self.baseline_beta) * reward
            )
        advantage = reward - self._baseline

        if self.update_rule == "emphasis" or sigma_t <= 0:
            # Heuristic: move toward signals emphasised on correct steps.
            emph = emph_t if emph_t is not None else np.zeros(3)
            grad = advantage * emph
        else:
            # Gaussian REINFORCE. Normalise by sigma (not sigma^2) so the update
            # magnitude stays ~O(lr * advantage) as exploration is annealed.
            grad = advantage * eps / sigma_t

        # Entropy regularisation: pull logits toward uniform so the policy keeps
        # arbitrating across axes rather than collapsing onto a single one.
        grad = grad - self.entropy_reg * self.theta

        self._velocity = self.momentum * self._velocity + self.lr * grad
        self.theta += self._velocity
        self.theta = np.clip(self.theta, -10.0, 10.0)
        self._n_updates += 1
