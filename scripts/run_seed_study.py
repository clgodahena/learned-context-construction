"""Additional seed / statistics study — the supervisor-feedback strengthening runs.

ONE self-contained, fully resumable driver for every extra run promised in the
response to the supervisor. It does NOT modify or re-run anything already on
disk; it writes only under ``results/seed_study/``.

Why this script exists (and is not just another campaign config)
----------------------------------------------------------------
* It saves the **per-window trajectory** of every trial, which the campaign
  runner discards. Those trajectories are what the block-level paired tests in
  ``scripts/analysis/stats_report.py`` consume — so the formal statistics need
  **no new long runs** beyond this study.
* It checkpoints at the finest possible granularity: one file per
  (block, stream, method, seed). If the Mac sleeps, the network drops, or the
  process is killed mid-trial, just re-launch the SAME command — every finished
  trial is skipped and only the unfinished one repeats (worst case ~1 trial lost,
  ~30–50 min for a TabPFN trial, seconds–minutes for a reference trial).

Resumability checklist (see RUNBOOK_WEEKEND.md for the full procedure)
---------------------------------------------------------------------
1. Pre-cache data + TabPFN weights on a stable connection first:
      python -m scripts.precache
   After that the study needs NO network — moving between home and office is safe.
2. Launch detached and sleep-proof so closing the lid / terminal cannot kill it:
      caffeinate -i nohup python -m scripts.run_seed_study --blocks A,B_policy,C_policy \
          > results/seed_study/run.log 2>&1 &
3. If anything interrupts it, just run the same command again — it resumes.

Blocks (choose with --blocks; default = all)
--------------------------------------------
  A         reference (weak) substrate, shifting-axis stream, policy + 3 fixed
            selectors, seeds 0..9  — strengthens the "+0.026 policy wins on a
            weak substrate" result. FAST (CPU), ~40 trials, well under an hour.
  B_policy  TabPFN, {electricity, synth_rotate}, learned_policy, seeds 0..4
            — 5-seed error bars on the primary pair.  10 TabPFN trials.
  B_traj    TabPFN, {electricity, synth_rotate}, the 3 fixed selectors (seed 0)
            + the streaming ensembles (seed 0) — trajectories for the primary
            paired windowed tests (policy vs best-fixed; in-context vs ensemble).
            6 TabPFN trials + 6 fast ensemble trials.
  C_policy  TabPFN, {airlines, covertype}, learned_policy, seeds 0..4
            — 5-seed error bars on the two extra real benchmarks. 10 TabPFN trials.
  C_traj    (optional) TabPFN, {airlines, covertype}, fixed selectors + ensembles
            (seed 0) — trajectories if you also want paired tests there.

  --- third-review additions (2026-11-10) ---
  D_ref_delay   reference substrate, shifting-axis stream, label delay
                {0, 50, 250, 1000} x {3 fixed, learned_policy, anchored_policy_t}
                x seeds 0..4 = 100 FAST trials (~20-30 min). Breadth: how every
                selector degrades with late labels, and whether the anchored
                policy tracks the sliding window under delay.
  D_tab_delay   TabPFN, Electricity, label delay 250: 3 fixed (seed 0) +
                learned_policy seeds 0..2 + anchored_policy_t seeds 0..2 =
                9 TabPFN trials (~7 h). Depth on the primary real stream;
                compared against the existing delay-0 rows of B_traj/B_policy/
                B_anchor_t. These runs also carry the per-stage timings.
  D_tab_timing  TabPFN, Electricity, delay 0, instrumented re-run of 3 fixed
                (seed 0) + learned_policy seed 0 + anchored_policy_t seed 0 =
                5 TabPFN trials (~4 h): end-to-end select / infer / update /
                step timings from ONE procedure at the main setting (review
                point 6). Optional if time is short; the delay-250 runs carry
                the same instrumentation.

Rough cost at ~0.5 s/instance, 6000-step cap (~30–50 min per TabPFN trial):
  A ~<1 h (CPU) · B_policy ~6–8 h · B_traj ~4 h · C_policy ~6–8 h · C_traj ~4 h.
Run A + the two *_policy blocks first (the error bars the supervisor asked for);
add the *_traj blocks if time allows (they enable the block-level paired tests).

Outputs (under results/seed_study/)
-----------------------------------
  runs.csv                         one row per finished trial (append + flush)
  trajectories/<trial>.csv         per-window metric trajectory per trial
  <trial>.done                     checkpoint marker (trial is complete)
  run.log                          when launched as in step 2 above
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import traceback

import numpy as np

from src.run import build_selector, build_classifier, build_direct
from src.harness.prequential import run_prequential, run_prequential_direct

POS = 1
OUT = "results/seed_study"
TRAJ = os.path.join(OUT, "trajectories")

# Run parameters — matched to campaign_3seed / policy_seeds so the summary
# numbers reproduce the existing results (and seed 0 re-confirms them).
RUN = dict(k=256, capacity=3000, warmup=300, metric_window=1000,
           snapshot_every=250, max_steps=6000)

FIXED = ["sliding_window", "similarity", "class_ratio"]
ENSEMBLES = ["online_bagging", "river_arf", "river_srp"]

# Two policy configurations, matched to the two experiments being extended:
#  * TabPFN campaign (Blocks B/C) uses the TUNED config from campaign_3seed /
#    policy_seeds (lr 0.02) so the extra seeds extend the published campaign.
#  * Reference shifting-axis (Block A) uses E5's EXACT policy — all defaults
#    except candidate_pool/seed (notably lr 0.05) — so the extra seeds faithfully
#    extend the published E5 experiment rather than quietly changing its config.
POLICY_CFG_TUNED = dict(name="learned_policy", candidate_pool=3000,
                        update_rule="policy_gradient", sigma=0.25,
                        entropy_reg=0.02, lr=0.02, reward_mode="instance")
POLICY_CFG_E5 = dict(name="learned_policy", candidate_pool=3000)  # E5 defaults (lr 0.05)


# --------------------------------------------------------------------------- #
# Stream builders (reuse the exact definitions used elsewhere in the project)  #
# --------------------------------------------------------------------------- #
def tabpfn_streams():
    """The TabPFN streams, keyed by name, as (build_fn(seed)) callables.

    electricity / airlines / covertype are fixed real streams (seed only varies
    the stochastic policy's exploration RNG, not the data). synth_rotate matches
    policy_seeds.yaml: feature-rotation drift + 0.15 imbalance, onset jittered
    per seed.
    """
    from scripts.run_campaign import build_campaign_stream

    synth_ds = {
        "name": "synth_rotate",
        "stream": {"kind": "injected",
                   "base": {"kind": "make_classification", "n": 6000, "n_features": 8, "pos_rate": 0.5},
                   "drift_points": [2000, 4000], "mechanism": "feature_rotation", "recurring": True},
        "imbalance": [0.15], "drift_jitter": 200,
    }
    elec_ds = {"name": "electricity", "stream": {"kind": "electricity"}, "imbalance": [None], "drift_jitter": 0}
    air_ds = {"name": "airlines",
              "stream": {"kind": "openml", "name": "airlines", "version": 1, "target_positive": "1"},
              "imbalance": [None], "drift_jitter": 0}
    cov_ds = {"name": "covertype", "stream": {"kind": "covertype"}, "imbalance": [None], "drift_jitter": 0}

    return {
        "electricity": lambda seed: build_campaign_stream(elec_ds, None, seed, POS),
        "synth_rotate": lambda seed: build_campaign_stream(synth_ds, 0.15, seed, POS),
        "airlines": lambda seed: build_campaign_stream(air_ds, None, seed, POS),
        "covertype": lambda seed: build_campaign_stream(cov_ds, None, seed, POS),
    }


def ref_shifting_stream(seed):
    """Block A weak-substrate stream: the E5 shifting-optimal-axis stream."""
    from scripts.experiments.exp_e5_shifting import build_stream as e5_build
    return e5_build(seed)


# --------------------------------------------------------------------------- #
# Trial plan                                                                   #
# --------------------------------------------------------------------------- #
def build_plan(blocks):
    """Return a list of trial dicts. Each trial is independently checkpointed."""
    trials = []

    def add(block, substrate, stream, family, method, seed, delay=0):
        trials.append(dict(block=block, substrate=substrate, stream=stream,
                           family=family, method=method, seed=int(seed),
                           label_delay=int(delay)))

    if "A" in blocks:
        for seed in range(10):
            for m in FIXED + ["learned_policy"]:
                add("A", "reference", "e5_shifting", "in_context", m, seed)

    if "B_policy" in blocks:
        for stream in ("electricity", "synth_rotate"):
            for seed in range(5):
                add("B_policy", "tabpfn", stream, "in_context", "learned_policy", seed)

    if "B_traj" in blocks:
        for stream in ("electricity", "synth_rotate"):
            for m in FIXED:
                add("B_traj", "tabpfn", stream, "in_context", m, 0)
            for m in ENSEMBLES:
                add("B_traj", "tabpfn", stream, "direct", m, 0)

    # --- draft-2 feedback additions (2026-11) ------------------------------ #
    # #3 Matched synthetic baselines: the synthetic stream is resampled per
    # seed, so a 5-seed policy mean must be compared against fixed/ensemble
    # baselines on the SAME seed-streams. Seed 0 already exists in B_traj, so
    # this adds seeds 1-4 only. (Electricity/Airlines/Covertype are seed-
    # invariant, so their single-seed baselines already match all seeds.)
    if "B_synth5" in blocks:
        for seed in range(1, 5):
            for m in FIXED:
                add("B_synth5", "tabpfn", "synth_rotate", "in_context", m, seed)
            for m in ENSEMBLES:
                add("B_synth5", "tabpfn", "synth_rotate", "direct", m, seed)

    # #4 Anchored policy evaluated on the near-ceiling TabPFN substrate (the
    # regime where the plain policy's exploration variance was worst), primary
    # pair, 5 seeds — the explicit evidence that anchoring is safe in practice.
    if "B_anchor" in blocks:
        for stream in ("electricity", "synth_rotate"):
            for seed in range(5):
                add("B_anchor", "tabpfn", stream, "in_context", "anchored_policy", seed)

    # #4 (valid run) — anchored at the TEMPORAL corner (an exact corner of the
    # policy = sliding window). The B_anchor runs above (class_utility corner) are
    # KEPT as evidence that that corner is degenerate; they are not a safety result.
    if "B_anchor_t" in blocks:
        for stream in ("electricity", "synth_rotate"):
            for seed in range(5):
                add("B_anchor_t", "tabpfn", stream, "in_context", "anchored_policy_t", seed)

    # --- third-review additions (2026-11-10): label delay + timings --------- #
    if "D_ref_delay" in blocks:
        for delay in (0, 50, 250, 1000):
            for seed in range(5):
                for m in FIXED + ["learned_policy", "anchored_policy_t"]:
                    add("D_ref_delay", "reference", "e5_shifting", "in_context", m, seed, delay)

    if "D_tab_delay" in blocks:
        for m in FIXED:
            add("D_tab_delay", "tabpfn", "electricity", "in_context", m, 0, 250)
        for seed in range(3):
            add("D_tab_delay", "tabpfn", "electricity", "in_context", "learned_policy", seed, 250)
        for seed in range(3):
            add("D_tab_delay", "tabpfn", "electricity", "in_context", "anchored_policy_t", seed, 250)

    if "D_tab_timing" in blocks:
        for m in FIXED + ["learned_policy", "anchored_policy_t"]:
            add("D_tab_timing", "tabpfn", "electricity", "in_context", m, 0, 0)

    if "C_policy" in blocks:
        for stream in ("airlines", "covertype"):
            for seed in range(5):
                add("C_policy", "tabpfn", stream, "in_context", "learned_policy", seed)

    if "C_traj" in blocks:
        for stream in ("airlines", "covertype"):
            for m in FIXED:
                add("C_traj", "tabpfn", stream, "in_context", m, 0)
            for m in ENSEMBLES:
                add("C_traj", "tabpfn", stream, "direct", m, 0)

    return trials


def trial_tag(t):
    tag = f"{t['block']}__{t['substrate']}__{t['stream']}__{t['method']}__seed{t['seed']}"
    if t.get("label_delay", 0) or t["block"].startswith("D_"):
        tag += f"__delay{t.get('label_delay', 0)}"
    return tag


# --------------------------------------------------------------------------- #
# Running one trial                                                            #
# --------------------------------------------------------------------------- #
def selector_cfg(method, seed, substrate):
    if method == "learned_policy":
        base = POLICY_CFG_E5 if substrate == "reference" else POLICY_CFG_TUNED
        return {**base, "seed": seed}
    if method == "anchored_policy":
        # Supervisor point #4: the anchored policy uses the SAME tuned update
        # config as the plain policy on TabPFN, warm-started at the strongest
        # fixed vertex (class_utility = class_ratio, the best fixed rule on both
        # the Electricity and the synthetic primary streams), held near it by a
        # trust-region pull. Only the anchoring differs, so the comparison with
        # the plain policy is apples-to-apples.
        return {**POLICY_CFG_TUNED, "name": "anchored_policy", "seed": seed,
                "anchor": "class_utility", "anchor_reg": 0.1, "anchor_sharpness": 2.5}
    if method == "anchored_policy_t":
        # FINDING (2026-11-07): the class_utility corner is DEGENERATE — class_utility
        # scores every instance of a class identically, so once the pool holds >= k
        # minority instances the top-k context is single-class and TabPFN's single-
        # class fast path returns a constant prediction (bal acc ~0.5). The class-
        # ratio selector is a STRATIFIED rule and is therefore NOT an exact corner of
        # the policy (only the temporal and feature corners are; see tests/
        # test_components.py). So the valid anchored evaluation anchors at an EXACT
        # corner: the temporal/recency vertex = the published sliding-memory baseline.
        base = POLICY_CFG_E5 if substrate == "reference" else POLICY_CFG_TUNED
        return {**base, "name": "anchored_policy_t", "seed": seed,
                "anchor": "temporal", "anchor_reg": 0.1, "anchor_sharpness": 2.5}
    return {"name": method, "seed": seed,
            "candidate_pool": 3000, "target_pos_ratio": 0.4, "positive_class": POS}


def run_trial(t, clf_cache):
    stream = (ref_shifting_stream(t["seed"]) if t["stream"] == "e5_shifting"
              else tabpfn_streams()[t["stream"]](t["seed"]))
    nc = stream.n_classes

    # The reference (weak-substrate) shifting-axis stream MUST run in full to
    # match the published E5 protocol: it has four segments (~15,000 instances)
    # and the policy's advantage lives in the later imbalanced/recurring
    # segments. Capping it at 6,000 would truncate those and bias the result.
    # The reference substrate is fast (~2,000 inst/s), so the full stream is a
    # few seconds per trial. TabPFN streams keep the 6,000 cap (matches the
    # existing campaign and keeps the slow substrate tractable).
    is_ref = (t["substrate"] == "reference")
    max_steps = None if is_ref else RUN["max_steps"]
    snap = 150 if is_ref else RUN["snapshot_every"]

    if t["family"] == "in_context":
        selector = build_selector(selector_cfg(t["method"], t["seed"], t["substrate"]), n_classes=nc)
        clf_name = t["substrate"]
        ckey = (clf_name, nc)
        if ckey not in clf_cache:
            clf_cfg = {"name": "reference", "kind": "gnb"} if clf_name == "reference" \
                else {"name": "tabpfn", "device": None, "max_context": 256}
            clf_cache[ckey] = build_classifier(clf_cfg, n_classes=nc)
        res = run_prequential(stream, selector, clf_cache[ckey],
                              k=RUN["k"], capacity=RUN["capacity"], warmup=RUN["warmup"],
                              positive_class=POS, metric_window=RUN["metric_window"],
                              snapshot_every=snap, max_steps=max_steps,
                              label_delay=int(t.get("label_delay", 0)),
                              time_it=True, progress_every=1000)
    else:  # direct streaming ensemble
        clf = build_direct({"name": t["method"], "seed": t["seed"]},
                           n_classes=nc, n_features=stream.n_features)
        res = run_prequential_direct(stream, clf, warmup=RUN["warmup"], positive_class=POS,
                                     metric_window=RUN["metric_window"],
                                     snapshot_every=snap, max_steps=max_steps,
                                     time_it=True, progress_every=1000)
    return res


def write_trajectory(tag, trajectory):
    """Atomic write: .tmp then rename, so an interrupted write never looks done."""
    if not trajectory:
        return None
    path = os.path.join(TRAJ, tag + ".csv")
    tmp = path + ".tmp"
    keys = list(trajectory[0].keys())
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in trajectory:
            w.writerow([row[k] for k in keys])
    os.replace(tmp, path)
    return path


SUMMARY_FIELDS = ["balanced_accuracy", "kappa_temporal", "kappa_m",
                  "windowed_auprc", "minority_recall", "accuracy",
                  "latency_ms_p50", "latency_ms_p95", "n"]
TIMING_FIELDS = [f"t_{stage}_ms_{q}" for stage in ("select", "infer", "update", "step")
                 for q in ("mean", "p50", "p95", "p99", "max")]
KEY_FIELDS = ["block", "substrate", "stream", "family", "method", "seed", "label_delay"]
CSV_FIELDS = KEY_FIELDS + SUMMARY_FIELDS + TIMING_FIELDS


def summary_row(t, summary):
    pcr = summary.get("per_class_recall") or []
    minority = pcr[POS] if POS < len(pcr) and pcr[POS] is not None else float("nan")
    row = {k: t.get(k, 0) for k in KEY_FIELDS}
    for m in SUMMARY_FIELDS:
        row[m] = summary.get(m, float("nan"))
    row["minority_recall"] = minority
    for m in TIMING_FIELDS:
        row[m] = summary.get(m, "")
    return row


def migrate_runs_csv():
    """Bring an older runs.csv (no label_delay / timing columns) up to the
    current schema IN PLACE, keeping a backup. Old rows get label_delay=0 and
    blank timing cells; nothing else changes, so existing analyses are unaffected."""
    path = os.path.join(OUT, "runs.csv")
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        old_fields = rd.fieldnames or []
        rows = list(rd)
    if old_fields == CSV_FIELDS:
        return
    backup = path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
    os.replace(path, backup)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in CSV_FIELDS}
            if out.get("label_delay", "") in ("", None):
                out["label_delay"] = 0
            w.writerow(out)
    print(f"[seed_study] runs.csv migrated to the current schema "
          f"(+label_delay, +timing columns); backup at {backup}")


def load_done():
    done = set()
    if os.path.exists(os.path.join(OUT, "runs.csv")):
        with open(os.path.join(OUT, "runs.csv")) as f:
            for r in csv.DictReader(f):
                done.add((r["block"], r["substrate"], r["stream"], r["method"], r["seed"],
                          str(int(float(r.get("label_delay") or 0)))))
    return done


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Additional seed / statistics study")
    ap.add_argument("--blocks", default="A,B_policy,B_traj,C_policy,C_traj",
                    help="comma list from: A,B_policy,B_traj,B_synth5,B_anchor,B_anchor_t,"
                         "C_policy,C_traj,D_ref_delay,D_tab_delay,D_tab_timing")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    os.makedirs(TRAJ, exist_ok=True)
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    plan = build_plan(blocks)

    if not args.dry_run:
        migrate_runs_csv()
    done = load_done()
    remaining = [t for t in plan
                 if (t["block"], t["substrate"], t["stream"], t["method"], str(t["seed"]),
                     str(t.get("label_delay", 0))) not in done]
    print(f"[seed_study] blocks={blocks}  planned={len(plan)}  done={len(plan)-len(remaining)}  "
          f"remaining={len(remaining)}")
    for t in plan:
        mark = "done" if t not in remaining else "TODO"
        print(f"    [{mark}] {trial_tag(t)}")
    if args.dry_run:
        return

    runs_path = os.path.join(OUT, "runs.csv")
    new_file = not os.path.exists(runs_path)
    clf_cache = {}
    t_start = time.time()
    with open(runs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, t in enumerate(remaining, 1):
            tag = trial_tag(t)
            print(f"\n[{i}/{len(remaining)}] {tag}  running...", flush=True)
            t0 = time.time()
            try:
                res = run_trial(t, clf_cache)
            except ImportError as e:
                print(f"    SKIPPED (missing dependency: {e}) — install it and re-run to fill this trial.",
                      flush=True)
                continue
            except Exception as e:  # noqa: BLE001 -- a bad trial must not kill the whole study
                traceback.print_exc()
                print(f"    ERROR: {str(e)[:200]} — left unfinished; re-run to retry.", flush=True)
                continue
            write_trajectory(tag, res.trajectory)
            row = summary_row(t, res.summary)
            writer.writerow(row)
            f.flush()
            open(os.path.join(OUT, tag + ".done"), "w").write("ok\n")
            dt = time.time() - t0
            print(f"    -> bal_acc={row['balanced_accuracy']:.4f}  "
                  f"kappa_t={row['kappa_temporal']:.3f}  n={row['n']}  {dt/60:.1f} min", flush=True)

    print(f"\n[seed_study] finished this pass in {(time.time()-t_start)/60:.1f} min. "
          f"Re-run the same command any time to resume/extend. "
          f"Then: python -m scripts.analysis.stats_report")


if __name__ == "__main__":
    main()
