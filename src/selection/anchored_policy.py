"""Anchored learned policy — warm-started at the strongest fixed rule and held
near it by a trust-region pull, so it "begins from a known-good configuration
and departs only when feedback warrants" (Chapter 7 recommendation).

The plain policy can *represent* the recency and similarity rules exactly
(Section 4.5), but representability says nothing about where the online
learner ends up. The anchored variant (a) warm-starts the logits at the anchor
vertex (logits +/- ``anchor_sharpness``, default 2.5) and (b) replaces the
entropy-to-uniform regulariser with a pull back toward the anchor logits of
strength ``anchor_reg`` (default 0.1). What it does in practice is an empirical
question answered by the runs in results/seed_study and results/anchored, not
a guarantee: with these settings it reproduced the sliding-window rule on the
streams tested; other settings or streams may let it depart. Everything else (the Gaussian REINFORCE
estimator, the class-balanced reward, the baseline and momentum) is inherited
unchanged from LearnedPolicySelector, so the comparison is apples-to-apples.
"""
from __future__ import annotations

import numpy as np

from .learned_policy import LearnedPolicySelector, SIGNAL_NAMES


class AnchoredPolicySelector(LearnedPolicySelector):
    name = "anchored_policy"

    def __init__(self, *args, anchor: str = "class_utility", anchor_reg: float = 0.1,
                 anchor_sharpness: float = 2.5, **kwargs):
        # The anchor replaces the uniform (entropy) pull, so disable entropy.
        kwargs.setdefault("entropy_reg", 0.0)
        super().__init__(*args, **kwargs)
        if anchor not in SIGNAL_NAMES:
            raise ValueError(f"anchor must be one of {SIGNAL_NAMES}")
        self.anchor = anchor
        self.anchor_reg = float(anchor_reg)
        i = SIGNAL_NAMES.index(anchor)
        ta = np.full(3, -anchor_sharpness, dtype=np.float64)
        ta[i] = anchor_sharpness
        self.theta_anchor = ta
        self.theta = ta.copy()          # warm-start at the anchor vertex

    def update(self, correct: bool, margin: float | None = None, y_true: int | None = None,
               token: dict | None = None) -> None:
        eps, sigma_t, emph_t = self._resolve_token(token)
        if self.freeze or eps is None:
            return
        # --- reward (identical to the parent) ---
        if self.reward_mode == "windowed" and y_true is not None:
            self._win[y_true].append(1.0 if correct else 0.0)
            recalls = [float(np.mean(d)) for d in self._win if len(d) > 0]
            wba = float(np.mean(recalls)) if recalls else 0.0
            if not self._wba_init:
                self._wba_ema = wba
                self._wba_init = True
            reward = (wba - self._wba_ema) * 10.0
            self._wba_ema = self._wba_beta * self._wba_ema + (1 - self._wba_beta) * wba
        else:
            reward = 1.0 if correct else -1.0
            if margin is not None:
                reward *= float(np.clip(abs(margin), 0.05, 1.0))
            if self.balanced_reward and y_true is not None:
                self._class_counts[y_true] += 1
                freq = self._class_counts / self._class_counts.sum()
                reward *= float((1.0 / freq[y_true]) / (1.0 / freq).sum() * self.n_classes)

        if not self._baseline_init:
            self._baseline = reward
            self._baseline_init = True
        else:
            self._baseline = self.baseline_beta * self._baseline + (1 - self.baseline_beta) * reward
        advantage = reward - self._baseline

        if self.update_rule == "emphasis" or sigma_t <= 0:
            emph = emph_t if emph_t is not None else np.zeros(3)
            grad = advantage * emph
        else:
            grad = advantage * eps / sigma_t

        # --- the only change vs the parent: trust-region pull toward the anchor
        #     logits instead of the entropy pull toward uniform ---
        grad = grad - self.anchor_reg * (self.theta - self.theta_anchor)

        self._velocity = self.momentum * self._velocity + self.lr * grad
        self.theta += self._velocity
        self.theta = np.clip(self.theta, -10.0, 10.0)
        self._n_updates += 1
