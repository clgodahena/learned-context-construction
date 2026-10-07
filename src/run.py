"""Config-driven experiment entrypoint.

Usage
-----
    python -m src.run --config config/electricity_sliding.yaml

Reads a YAML config, builds the stream / selector / classifier via small
factories, runs one prequential evaluation, and writes a metrics summary
(JSON), the metric trajectory (CSV) and a dashboard figure (PNG) into the
configured results directory. Everything is driven by config so new
experiments -- different selector, classifier, dataset, k, imbalance -- are a
file, not a code change.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import yaml

from .classifiers.reference import ReferenceInContextClassifier
from .data import streams as S
from .data.drift_injector import DriftInjector, make_classification_stream
from .harness.prequential import run_prequential, run_prequential_direct
from .selection.baselines import (
    ClassRatioSelector,
    RandomSelector,
    SimilaritySelector,
    SlidingWindowSelector,
)
from .selection.learned_policy import LearnedPolicySelector


def build_direct(cfg: dict, n_classes: int, n_features: int) -> Any:
    """Build a direct (end-to-end) streaming classifier: ARF/SRP or the stand-in."""
    name = cfg["name"]
    if name == "online_bagging":
        from .streaming.offline_ensemble import OnlineBaggingEnsemble

        return OnlineBaggingEnsemble(n_classes=n_classes, n_models=cfg.get("n_models", 10),
                                     reset_every=cfg.get("reset_every", 2000),
                                     seed=cfg.get("seed", 0))
    if name in ("river_arf", "river_srp", "river_hoeffding"):
        from .streaming.river_models import RiverClassifier

        return RiverClassifier(n_classes=n_classes, model=name.split("_", 1)[1],
                               n_models=cfg.get("n_models", 10), seed=cfg.get("seed", 0))
    if name in ("capymoa_arf", "capymoa_srp"):
        from .streaming.capymoa_models import CapyMOAClassifier

        return CapyMOAClassifier(n_classes=n_classes, n_features=n_features,
                                 model=name.split("_", 1)[1],
                                 ensemble_size=cfg.get("ensemble_size", 30),
                                 seed=cfg.get("seed", 1))
    raise ValueError(f"unknown direct classifier: {name}")


def build_stream(cfg: dict) -> Any:
    kind = cfg["kind"]
    if kind == "electricity":
        return S.electricity()
    if kind == "openml":
        return S.OpenMLStream(
            name=cfg["name"], version=cfg.get("version", 1),
            target_positive=cfg.get("target_positive"),
        )
    if kind == "synthetic":
        return S.SyntheticDriftStream(
            n=cfg.get("n", 20000),
            n_features=cfg.get("n_features", 6),
            drift_points=cfg.get("drift_points"),
            pos_rate=cfg.get("pos_rate", 0.5),
            drift_kind=cfg.get("drift_kind", "abrupt"),
            seed=cfg.get("seed", 0),
        )
    if kind == "make_classification":
        return make_classification_stream(
            n=cfg.get("n", 12000), n_features=cfg.get("n_features", 8),
            pos_rate=cfg.get("pos_rate", 0.5), seed=cfg.get("seed", 0),
        )
    if kind == "injected":
        # Wrap a base stream with controlled drift + optional imbalance.
        base = build_stream(cfg["base"])
        return DriftInjector(
            base,
            drift_points=cfg.get("drift_points", []),
            mechanism=cfg.get("mechanism", "label_flip"),
            drift_kind=cfg.get("drift_kind", "abrupt"),
            gradual_width=cfg.get("gradual_width", 500),
            recurring=cfg.get("recurring", False),
            target_pos_rate=cfg.get("target_pos_rate"),
            positive_class=cfg.get("positive_class", 1),
            seed=cfg.get("seed", 0),
        )
    if kind == "airlines":
        return S.airlines()
    if kind == "covertype":
        return S.covertype()
    raise ValueError(f"unknown stream kind: {kind}")


def build_selector(cfg: dict, n_classes: int = 2) -> Any:
    name = cfg["name"]
    if name == "sliding_window":
        return SlidingWindowSelector()
    if name == "random":
        return RandomSelector(seed=cfg.get("seed", 0))
    if name == "similarity":
        return SimilaritySelector(candidate_pool=cfg.get("candidate_pool"))
    if name == "class_ratio":
        return ClassRatioSelector(
            positive_class=cfg.get("positive_class", 1),
            target_pos_ratio=cfg.get("target_pos_ratio", 0.5),
        )
    if name == "learned_policy":
        return LearnedPolicySelector(
            n_classes=n_classes,
            positive_class=cfg.get("positive_class", 1),
            candidate_pool=cfg.get("candidate_pool", 4000),
            lr=cfg.get("lr", 0.05),
            sigma=cfg.get("sigma", 0.25),
            update_rule=cfg.get("update_rule", "policy_gradient"),
            entropy_reg=cfg.get("entropy_reg", 0.02),
            balanced_reward=cfg.get("balanced_reward", True),
            reward_mode=cfg.get("reward_mode", "instance"),
            init_weights=cfg.get("init_weights"),
            tau=cfg.get("tau"),
            cost=cfg.get("cost"),
            freeze=cfg.get("freeze", False),
            seed=cfg.get("seed", 0),
        )
    raise ValueError(f"unknown selector: {name}")


def build_classifier(cfg: dict, n_classes: int) -> Any:
    name = cfg["name"]
    if name == "reference":
        return ReferenceInContextClassifier(n_classes=n_classes, kind=cfg.get("kind", "gnb"))
    if name == "tabpfn":
        from .classifiers.tabpfn_clf import TabPFNInContextClassifier

        return TabPFNInContextClassifier(
            n_classes=n_classes,
            device=cfg.get("device"),
            max_context=cfg.get("max_context", 4096),
            model_version=cfg.get("model_version"),
        )
    raise ValueError(f"unknown classifier: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prequential experiment runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override: cap the number of stream steps (for quick runs)")
    ap.add_argument("--progress", type=int, default=200,
                    help="print a heartbeat (rate + ETA) every N scored instances; 0 = off")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = cfg.get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)

    stream = build_stream(cfg["stream"])
    run_cfg = cfg.get("run", {})
    max_steps = args.max_steps if args.max_steps is not None else run_cfg.get("max_steps")
    mode = cfg.get("mode", "in_context")

    if mode == "direct":
        # End-to-end streaming baseline (ARF/SRP/stand-in): no context selector.
        classifier = build_direct(cfg["classifier"], n_classes=stream.n_classes,
                                  n_features=stream.n_features)
        print(f"[run] experiment={exp} stream={cfg['stream']['kind']} "
              f"mode=direct classifier={classifier.name}")
        result = run_prequential_direct(
            stream, classifier,
            warmup=run_cfg.get("warmup", 100),
            positive_class=run_cfg.get("positive_class", 1),
            metric_window=run_cfg.get("metric_window", 1000),
            snapshot_every=run_cfg.get("snapshot_every", 250),
            max_steps=max_steps,
            time_it=run_cfg.get("time_it", True),
            progress_every=args.progress,
        )
        selector = classifier  # for the label below
    else:
        selector = build_selector(cfg["selector"], n_classes=stream.n_classes)
        classifier = build_classifier(cfg["classifier"], n_classes=stream.n_classes)
        print(f"[run] experiment={exp} stream={cfg['stream']['kind']} "
              f"selector={selector.name} classifier={classifier.name}")
        result = run_prequential(
            stream, selector, classifier,
            k=run_cfg.get("k", 512),
            capacity=run_cfg.get("capacity"),
            warmup=run_cfg.get("warmup", 100),
            label_delay=run_cfg.get("label_delay", 0),
            positive_class=run_cfg.get("positive_class", 1),
            metric_window=run_cfg.get("metric_window", 1000),
            snapshot_every=run_cfg.get("snapshot_every", 250),
            max_steps=max_steps,
            time_it=run_cfg.get("time_it", True),
            progress_every=args.progress,
        )

    # --- persist outputs ---
    summary_path = os.path.join(out_dir, f"{exp}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"experiment": exp, "config": cfg, "summary": result.summary}, f, indent=2)

    traj_path = os.path.join(out_dir, f"{exp}_trajectory.csv")
    if result.trajectory:
        keys = list(result.trajectory[0].keys())
        with open(traj_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in result.trajectory:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

    fig_path = os.path.join(out_dir, f"{exp}_dashboard.png")
    try:
        from .harness.plots import plot_run

        plot_run(result.trajectory, result.composition, fig_path,
                 title=f"{exp}: {selector.name} + {classifier.name}")
    except Exception as e:  # plotting must never fail the run
        print(f"[run] plot skipped: {e}")
        fig_path = None

    print(f"[run] summary: {json.dumps(result.summary, indent=2)}")
    print(f"[run] wrote {summary_path}")
    if result.trajectory:
        print(f"[run] wrote {traj_path}")
    if fig_path:
        print(f"[run] wrote {fig_path}")


if __name__ == "__main__":
    main()
