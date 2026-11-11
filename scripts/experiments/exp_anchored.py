"""#4 — Explicit evaluation of the anchored policy on the weak REFERENCE
substrate (E5 shifting-optimal-axis stream), the fast/CPU part that needs no
TabPFN. Compares, over 10 seeds, the best fixed rule (sliding_window), the plain
learned policy (E5 config), and the anchored policy warm-started at the best
fixed vertex. Reports mean, SD and bootstrap 95% CI of balanced accuracy, plus
a seed-paired comparison of each policy against the fixed rule.
"""
from __future__ import annotations
import sys, numpy as np
sys.path.insert(0, ".")
from scripts.run_seed_study import ref_shifting_stream, POLICY_CFG_E5, POS, RUN
from src.classifiers.reference import ReferenceInContextClassifier
from src.selection.baselines import SlidingWindowSelector
from src.selection.learned_policy import LearnedPolicySelector
from src.selection.anchored_policy import AnchoredPolicySelector

SEEDS = list(range(10))
RNG = np.random.default_rng(7)

def run_one(make_selector):
    accs = []
    for s in SEEDS:
        stream = ref_shifting_stream(s)
        clf = ReferenceInContextClassifier(n_classes=stream.n_classes, kind="gnb")
        sel = make_selector(s, stream.n_classes)
        res = run(stream, sel, clf)
        accs.append(res)
    return np.array(accs)

def run(stream, selector, clf):
    from src.harness.prequential import run_prequential
    r = run_prequential(stream, selector, clf, k=RUN["k"], capacity=RUN["capacity"],
                        warmup=RUN["warmup"], positive_class=POS,
                        metric_window=RUN["metric_window"], snapshot_every=150,
                        max_steps=None, time_it=False)
    return float(r.summary["balanced_accuracy"])

def boot_ci(v, nb=10000):
    b = np.array([np.mean(RNG.choice(v, v.size, replace=True)) for _ in range(nb)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def main():
    pol_kw = {k: v for k, v in POLICY_CFG_E5.items() if k != "name"}
    makers = {
        "sliding_window (best fixed)": lambda s, nc: SlidingWindowSelector(),
        "learned_policy (plain, E5)":  lambda s, nc: LearnedPolicySelector(n_classes=nc, positive_class=POS, seed=s, **pol_kw),
        "anchored_policy (temporal)":  lambda s, nc: AnchoredPolicySelector(n_classes=nc, positive_class=POS, seed=s,
                                                                            candidate_pool=3000, anchor="temporal",
                                                                            anchor_reg=0.1, anchor_sharpness=2.5),
    }
    res = {name: run_one(mk) for name, mk in makers.items()}
    print("method                         mean      SD      95% CI            per-seed min..max")
    for name, v in res.items():
        m, lo, hi = boot_ci(v)
        print(f"{name:30s} {m:.4f}  {v.std(ddof=1):.4f}  [{lo:.4f}, {hi:.4f}]  {v.min():.3f}..{v.max():.3f}")
    fixed = res["sliding_window (best fixed)"]
    from scipy import stats
    print("\nseed-paired vs sliding_window (best fixed):")
    for name in ("learned_policy (plain, E5)", "anchored_policy (temporal)"):
        d = res[name] - fixed
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        wins = int(np.sum(d > 0))
        print(f"  {name:30s} mean diff {d.mean():+.4f}  wins {wins}/10  Wilcoxon p={w:.3f}")
    # variance comparison (the safety claim)
    print("\nSD(plain) = %.4f   SD(anchored) = %.4f" %
          (res["learned_policy (plain, E5)"].std(ddof=1), res["anchored_policy (temporal)"].std(ddof=1)))

if __name__ == "__main__":
    main()
