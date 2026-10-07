"""Shared helpers for the learned-policy investigation scripts (E1--E6).

All experiments run on the fast reference classifier by default (hundreds of
instances/second), since the mechanism under study is the selection strategy;
confirm the winning configuration on TabPFN afterwards. Pass --tabpfn to any
script to use the real substrate.
"""
from __future__ import annotations

import os

import numpy as np

from src.classifiers.reference import ReferenceInContextClassifier
from src.data.drift_injector import DriftInjector, make_classification_stream
from src.data.streams import electricity
from src.harness.prequential import run_prequential
from src.selection.baselines import ClassRatioSelector, SimilaritySelector, SlidingWindowSelector

OUT = "results/experiments"
POS = 1


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def make_classifier(use_tabpfn: bool, n_classes: int = 2):
    if use_tabpfn:
        from src.classifiers.tabpfn_clf import TabPFNInContextClassifier
        print("[classifier] TabPFN  (real substrate; loads on first prediction)", flush=True)
        return TabPFNInContextClassifier(n_classes=n_classes, max_context=256)
    print("[classifier] reference (GaussianNB — fast, NOT TabPFN)", flush=True)
    return ReferenceInContextClassifier(n_classes=n_classes, kind="gnb")


def make_synth(imbalance=0.15, seed=0, n=8000, drift_points=(3000, 6000), recurring=True):
    """Injected feature-rotation drift + imbalance on an offline base stream."""
    base = make_classification_stream(n=n, n_features=8, pos_rate=0.5, seed=seed)
    return DriftInjector(base, drift_points=list(drift_points), mechanism="feature_rotation",
                         recurring=recurring, target_pos_rate=imbalance, positive_class=POS, seed=seed)


def make_electricity():
    return electricity()


def single_axis_selectors():
    return {
        "sliding_window": SlidingWindowSelector(),
        "similarity": SimilaritySelector(candidate_pool=3000),
        "class_ratio": ClassRatioSelector(positive_class=POS, target_pos_ratio=0.4),
    }


def run_sel(stream, selector, classifier, *, k=256, capacity=3000, warmup=300,
            max_steps=None, snapshot_every=200, time_it=False, record_predictions=False):
    return run_prequential(
        stream, selector, classifier, k=k, capacity=capacity, warmup=warmup,
        positive_class=POS, metric_window=1000, snapshot_every=snapshot_every,
        max_steps=max_steps, time_it=time_it, record_predictions=record_predictions,
    )


def bal_acc_from_preds(preds, n_classes=2):
    if not preds:
        return float("nan")
    yt = np.array([p[1] for p in preds]); yp = np.array([p[2] for p in preds])
    recs = [float((yp[yt == c] == c).mean()) for c in range(n_classes) if (yt == c).any()]
    return float(np.mean(recs)) if recs else float("nan")


def summary_row(summary):
    return dict(
        balanced_accuracy=summary.get("balanced_accuracy", float("nan")),
        kappa_temporal=summary.get("kappa_temporal", float("nan")),
        windowed_auprc=summary.get("windowed_auprc", float("nan")),
        minority_recall=(summary.get("per_class_recall") or [None, None])[POS],
        n=summary.get("n", 0),
    )
