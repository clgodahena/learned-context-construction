"""Formal statistics for the thesis — confidence intervals + paired tests.

Consumes ONLY what scripts/run_seed_study.py already saved (runs.csv and the
per-window trajectories). No model is run here, so this is fast and can be
re-run any time as more trials complete.

What it produces (and how it answers the two supervisory reviews):

  * Bootstrap 95% confidence intervals of balanced accuracy, AUPRC, temporal
    kappa and minority recall per (stream, method), pooling every seed of a
    method across the study blocks.

  * (a) SEED-LEVEL paired tests on BALANCED ACCURACY. Seeds are independent, so
    this is the PRIMARY significance evidence. On the real streams the seed
    only changes the stochastic policy's exploration (the deterministic
    selectors and ensembles are identical on every seed, so a single run is
    paired against each policy seed); on the synthetic stream the seed also
    resamples the data and drift onset, so every method is run on every seed
    (blocks B_traj + B_synth5) and paired seed-by-seed. With five pairs the
    Wilcoxon signed-rank test has a floor of p = 0.0625, so the paired t-test,
    the bootstrap CI of the mean difference and the number of seeds won are
    the informative statistics; all are reported.

  * (b) WINDOW-LEVEL paired tests on PREQUENTIAL WINDOWED ACCURACY (plain
    accuracy over a trailing window of `metric_window` instances, snapshotted
    every `snapshot_every` instances). Consecutive windows overlap (75% with
    the default 1000/250), so the raw per-window differences are strongly
    autocorrelated and a naive Wilcoxon/bootstrap over them OVERSTATES
    significance. This script therefore reports, for every window-level
    comparison: the lag-1 autocorrelation of the differences, the effective
    sample size n*(1-rho)/(1+rho), a MOVING-BLOCK BOOTSTRAP 95% CI (block
    length 6 windows, which exceeds the overlap span of 4), and a conservative
    Wilcoxon test on DISJOINT windows (every 4th snapshot). The naive Wilcoxon
    p is still printed, labelled as such, for comparison with earlier drafts.

Outputs (under results/seed_study/):
  stats_cis.csv        per-(stream,method) mean, SD, and bootstrap 95% CI
  stats_paired.csv     every paired comparison with effect size + p-values
  stats_report.md      a readable summary to cite from Chapter 7 / Appendix D
  stats_cis.png        forest plot of balanced-accuracy CIs (thesis figure)
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

OUT = "results/seed_study"
TRAJ = os.path.join(OUT, "trajectories")
RNG = np.random.default_rng(12345)
N_BOOT = 10000
WINDOW, STEP = 1000, 250           # metric_window / snapshot_every used by run_seed_study.RUN
DISJOINT = WINDOW // STEP          # every 4th snapshot => non-overlapping windows
BLOCK_LEN = 6                      # moving-block bootstrap block length (> overlap span)

FIXED = {"sliding_window", "similarity", "class_ratio"}
ENSEMBLES = {"online_bagging", "river_arf", "river_srp"}
POLICIES = {"learned_policy", "anchored_policy_t", "anchored_policy"}


# --------------------------------------------------------------------------- #
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_runs():
    path = os.path.join(OUT, "runs.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[stats] no runs.csv at {path} — run scripts.run_seed_study first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def bootstrap_ci(vals, stat=np.mean, n_boot=N_BOOT, alpha=0.05):
    vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    boot = np.array([stat(RNG.choice(vals, size=vals.size, replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(vals)), float(lo), float(hi))


def paired_stats(diff):
    """Wilcoxon signed-rank + paired t + bootstrap CI on the mean difference (iid resampling)."""
    from scipy import stats
    diff = np.asarray([d for d in diff if not np.isnan(d)], dtype=float)
    out = {"n_pairs": int(diff.size), "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
           "wins": int(np.sum(diff > 0)) if diff.size else 0}
    m, lo, hi = bootstrap_ci(diff, np.mean)
    out["ci_lo"], out["ci_hi"] = lo, hi
    try:
        if diff.size >= 6 and np.any(diff != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
        else:
            out["wilcoxon_p"] = float("nan")   # n<6: Wilcoxon cannot reach p<0.05 two-sided
    except Exception:  # noqa: BLE001
        out["wilcoxon_p"] = float("nan")
    try:
        if diff.size >= 2 and np.std(diff) > 0:
            out["paired_t_p"] = float(stats.ttest_1samp(diff, 0.0).pvalue)
        else:
            out["paired_t_p"] = float("nan")
    except Exception:  # noqa: BLE001
        out["paired_t_p"] = float("nan")
    out["cohens_dz"] = float(np.mean(diff) / np.std(diff, ddof=1)) if diff.size >= 2 and np.std(diff, ddof=1) > 0 else float("nan")
    return out


def dependence_stats(diff, L=BLOCK_LEN, nb=N_BOOT):
    """Dependence-aware statistics for an autocorrelated per-window difference series."""
    from scipy import stats
    d = np.asarray([x for x in diff if not np.isnan(x)], dtype=float)
    out = {}
    if d.size < 3:
        return {"lag1_rho": float("nan"), "n_eff": float(d.size), "block_ci_lo": float("nan"),
                "block_ci_hi": float("nan"), "disjoint_n": 0, "disjoint_wilcoxon_p": float("nan")}
    c = d - d.mean()
    rho = float(np.sum(c[1:] * c[:-1]) / np.sum(c * c)) if np.any(c != 0) else 0.0
    out["lag1_rho"] = rho
    out["n_eff"] = float(d.size * (1 - rho) / (1 + rho)) if abs(rho) < 1 else float(d.size)
    # moving-block bootstrap of the mean difference
    L = min(L, d.size)
    nblocks = int(np.ceil(d.size / L))
    means = np.empty(nb)
    for b in range(nb):
        starts = RNG.integers(0, d.size - L + 1, size=nblocks)
        means[b] = np.concatenate([d[s:s + L] for s in starts])[:d.size].mean()
    out["block_ci_lo"], out["block_ci_hi"] = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    # conservative: disjoint windows
    dj = d[::DISJOINT]
    out["disjoint_n"] = int(dj.size)
    out["disjoint_wilcoxon_p"] = float(stats.wilcoxon(dj, alternative="two-sided").pvalue) \
        if dj.size >= 6 and np.any(dj != 0) else float("nan")
    return out


def load_trajectory(tag, col="windowed_accuracy"):
    path = os.path.join(TRAJ, tag + ".csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        return None
    return np.array([_f(r[col]) for r in rows], dtype=float)


# --------------------------------------------------------------------------- #
def confidence_intervals(runs):
    """Mean / SD / bootstrap-CI per (stream, method), pooling seeds across blocks."""
    metrics = ["balanced_accuracy", "windowed_auprc", "kappa_temporal", "minority_recall"]
    by = defaultdict(lambda: defaultdict(dict))   # (substrate,stream,method) -> metric -> {seed: val}
    blocks = defaultdict(set)
    for r in runs:
        key = (r["substrate"], r["stream"], r["method"])
        blocks[key].add(r["block"])
        for m in metrics:
            by[key][m][r["seed"]] = _f(r.get(m, "nan"))
    rows = []
    for (substrate, stream, method), md in sorted(by.items()):
        n = max(len(v) for v in md.values())
        row = {"block": "+".join(sorted(blocks[(substrate, stream, method)])), "substrate": substrate,
               "stream": stream, "method": method, "n_seeds": n}
        for m in metrics:
            vals = list(md[m].values())
            mean, lo, hi = bootstrap_ci(vals)
            good = [x for x in vals if not np.isnan(x)]
            sd = float(np.std(good, ddof=1)) if len(good) > 1 else 0.0
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_sd"] = round(sd, 4)
            row[f"{m}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    return rows


def seed_level_paired(runs):
    """(a) Pair by seed on balanced accuracy, per (substrate, stream), across blocks.

    Deterministic methods with a single seed-0 run on a fixed real stream are
    expanded to every policy seed (they are identical on every seed by
    construction). On the synthetic stream all methods must have the seeds.
    """
    idx = defaultdict(dict)   # (substrate,stream,method) -> {seed: bal_acc}
    for r in runs:
        idx[(r["substrate"], r["stream"], r["method"])][r["seed"]] = _f(r["balanced_accuracy"])
    groups = defaultdict(list)
    for (sub, stream, method) in idx:
        groups[(sub, stream)].append(method)
    REAL = {"electricity", "airlines", "covertype", "e5_shifting"}

    def series(sub, stream, method, seeds):
        d = idx[(sub, stream, method)]
        if stream in REAL and len(d) == 1 and "0" in d:          # deterministic, seed-invariant
            return np.array([d["0"]] * len(seeds))
        return np.array([d.get(s, np.nan) for s in seeds])

    results = []
    for (sub, stream), methods in sorted(groups.items()):
        pols = [m for m in ("learned_policy", "anchored_policy_t") if m in methods]
        if not pols:
            continue
        seeds = sorted(idx[(sub, stream, pols[0])])
        if len(seeds) < 3:
            continue

        def add(kind, comp, a, b):
            d = a - b
            if np.isnan(d).all():
                return
            st = paired_stats(d.tolist())
            results.append({"kind": kind, "block": sub, "stream": stream, "comparison": comp, **st})

        fixed_here = [m for m in FIXED if m in methods]
        ens_here = [m for m in ENSEMBLES if m in methods]
        for pol in pols:
            P = series(sub, stream, pol, seeds)
            for other in fixed_here:
                add("seed-paired", f"{pol} - {other}", P, series(sub, stream, other, seeds))
            if fixed_here:
                best_fixed = np.max(np.vstack([series(sub, stream, m, seeds) for m in fixed_here]), axis=0)
                add("seed-paired", f"{pol} - best fixed (per seed)", P, best_fixed)
            if ens_here:
                best_ens = np.max(np.vstack([series(sub, stream, m, seeds) for m in ens_here]), axis=0)
                add("seed-paired", f"{pol} - best ensemble (per seed)", P, best_ens)
        if "anchored_policy_t" in methods and "learned_policy" in methods:
            add("seed-paired", "anchored_policy_t - learned_policy",
                series(sub, stream, "anchored_policy_t", seeds), series(sub, stream, "learned_policy", seeds))
        if fixed_here and ens_here:
            ic = fixed_here + [p for p in pols]
            best_ic = np.max(np.vstack([series(sub, stream, m, seeds) for m in ic]), axis=0)
            best_ens = np.max(np.vstack([series(sub, stream, m, seeds) for m in ens_here]), axis=0)
            add("seed-paired", "best in-context - best ensemble (per seed)", best_ic, best_ens)
    return results


def block_level_paired(runs):
    """(b) Pair windowed-accuracy trajectories window-by-window (seed 0), with dependence-aware stats."""
    avail = defaultdict(dict)   # (substrate,stream) -> method -> (family, final_bal_acc, tag)
    for r in runs:
        if r["seed"] != "0":
            continue
        tag = f"{r['block']}__{r['substrate']}__{r['stream']}__{r['method']}__seed0"
        if os.path.exists(os.path.join(TRAJ, tag + ".csv")):
            avail[(r["substrate"], r["stream"])][r["method"]] = (r["family"], _f(r["balanced_accuracy"]), tag)
    results = []
    for (substrate, stream), methods in sorted(avail.items()):
        def best(names):
            cand = {m: v for m, v in methods.items() if m in names}
            return max(cand, key=lambda m: cand[m][1]) if cand else None

        def add(comp, ma, mb):
            a, b = load_trajectory(methods[ma][2]), load_trajectory(methods[mb][2])
            if a is None or b is None:
                return
            n = min(len(a), len(b)); d = (a[:n] - b[:n]).tolist()
            st = paired_stats(d); st.update(dependence_stats(d))
            results.append({"kind": "block-windowed", "block": substrate, "stream": stream, "comparison": comp, **st})

        bf = best(FIXED)
        if "learned_policy" in methods and bf:
            add(f"learned_policy - {bf} (best fixed)", "learned_policy", bf)
        if "anchored_policy_t" in methods and "sliding_window" in methods:
            add("anchored_policy_t - sliding_window (its anchor)", "anchored_policy_t", "sliding_window")
        bic = best(FIXED | {"learned_policy"}); ens = best(ENSEMBLES)
        if bic and ens:
            add(f"{bic} (best in-context) - {ens} (best ensemble)", bic, ens)
    return results


# --------------------------------------------------------------------------- #
def forest_plot(ci_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[stats] plot skipped: {e}")
        return
    rows = [r for r in ci_rows if not np.isnan(r["balanced_accuracy_mean"])]
    rows.sort(key=lambda r: (r["stream"], r["balanced_accuracy_mean"]))
    labels = [f"{r['stream']}/{r['method']}" for r in rows]
    means = [r["balanced_accuracy_mean"] for r in rows]
    los = [float(r["balanced_accuracy_ci"].strip("[]").split(",")[0]) for r in rows]
    his = [float(r["balanced_accuracy_ci"].strip("[]").split(",")[1]) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(rows))))
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="o", color="#4C78A8", ecolor="#999", capsize=3, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("balanced accuracy (mean, bootstrap 95% CI)")
    ax.set_title("Balanced accuracy with 95% confidence intervals")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[stats] wrote {path}")


def write_csv(rows, path, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[stats] wrote {path}")


def write_md(ci_rows, paired_rows, path):
    def p(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4g}"
    lines = ["# Formal statistics — seed study", "",
             "Generated by `scripts/analysis/stats_report.py` from the saved seed-study results.",
             "Confidence intervals are bootstrap 95% (10,000 resamples). Seed-level tests pair",
             "methods by seed on BALANCED ACCURACY (seeds are independent; this is the primary",
             "evidence). Window-level tests pair prequential WINDOWED ACCURACY window by window;",
             "because consecutive windows overlap and are autocorrelated, they are reported with the",
             "lag-1 autocorrelation, the effective sample size, a moving-block bootstrap CI and a",
             "conservative disjoint-window Wilcoxon test. The naive Wilcoxon p over the overlapping",
             "windows is shown only for comparison; it overstates significance. With five seeds the",
             "Wilcoxon signed-rank test cannot fall below p = 0.0625, so for 5-seed comparisons read",
             "the paired t-test, the CI and the number of seeds won.", "",
             "## Confidence intervals (balanced accuracy)", ""]
    lines.append("| stream | method | blocks | seeds | mean | SD | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(ci_rows, key=lambda r: (r["stream"], -r["balanced_accuracy_mean"])):
        lines.append(f"| {r['stream']} | {r['method']} | {r['block']} | {r['n_seeds']} | "
                     f"{p(r['balanced_accuracy_mean'])} | {p(r['balanced_accuracy_sd'])} | "
                     f"{r['balanced_accuracy_ci']} |")
    lines += ["", "## Seed-level paired comparisons (balanced accuracy)", "",
              "| stream | comparison | n seeds | mean diff | bootstrap 95% CI | seeds won | paired-t p | Wilcoxon p | d_z |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in [x for x in paired_rows if x["kind"] == "seed-paired"]:
        lines.append(f"| {r['stream']} | {r['comparison']} | {r['n_pairs']} | {p(r['mean_diff'])} | "
                     f"[{p(r['ci_lo'])}, {p(r['ci_hi'])}] | {r['wins']}/{r['n_pairs']} | {p(r['paired_t_p'])} | "
                     f"{p(r['wilcoxon_p'])} | {p(r['cohens_dz'])} |")
    lines += ["", "## Window-level paired comparisons (windowed accuracy, seed 0), dependence-aware", "",
              "| stream | comparison | windows | lag-1 rho | n_eff | mean diff | block-bootstrap 95% CI | disjoint n | disjoint Wilcoxon p | naive Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in [x for x in paired_rows if x["kind"] == "block-windowed"]:
        lines.append(f"| {r['stream']} | {r['comparison']} | {r['n_pairs']} | {p(r.get('lag1_rho'))} | "
                     f"{p(r.get('n_eff'))} | {p(r['mean_diff'])} | [{p(r.get('block_ci_lo'))}, {p(r.get('block_ci_hi'))}] | "
                     f"{r.get('disjoint_n','')} | {p(r.get('disjoint_wilcoxon_p'))} | {p(r['wilcoxon_p'])} |")
    lines += ["", "_A positive mean difference favours the first method in the comparison._",
              "_anchored_policy = anchored at the class-utility corner (degenerate: single-class context once the pool",
              "holds >= k minority instances; kept as evidence). anchored_policy_t = anchored at the temporal corner,",
              "an exact corner of the policy (= sliding window)._", ""]
    open(path, "w").write("\n".join(lines))
    print(f"[stats] wrote {path}")


def main():
    runs = load_runs()
    print(f"[stats] loaded {len(runs)} trial rows from runs.csv")
    ci_rows = confidence_intervals(runs)
    paired = seed_level_paired(runs) + block_level_paired(runs)

    write_csv(ci_rows, os.path.join(OUT, "stats_cis.csv"))
    pf = ["kind", "block", "stream", "comparison", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "wins",
          "wilcoxon_p", "paired_t_p", "cohens_dz", "lag1_rho", "n_eff", "block_ci_lo", "block_ci_hi",
          "disjoint_n", "disjoint_wilcoxon_p"]
    write_csv([{k: r.get(k, "") for k in pf} for r in paired], os.path.join(OUT, "stats_paired.csv"), pf)
    write_md(ci_rows, paired, os.path.join(OUT, "stats_report.md"))
    forest_plot(ci_rows, os.path.join(OUT, "stats_cis.png"))
    print("\n[stats] done. Cite stats_report.md / stats_cis.png in Chapter 7 and Appendix D.")


if __name__ == "__main__":
    main()
