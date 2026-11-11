#!/usr/bin/env bash
# Status of the most recent seed-study pass: running?, progress, time remaining.
# Counts completed trials from runs.csv (authoritative) so a doubled log can't
# fool it. Auto-detects the pass from the log header, works for any block.
# Usage:  bash scripts/status.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=results/seed_study/run.log
RUNS=results/seed_study/runs.csv
[ -f "$LOG" ] || { echo "no run.log yet at $LOG"; exit 0; }

echo "==================  seed-study status  =================="
if pgrep -f "run_seed_study" >/dev/null 2>&1; then
  echo "state   : RUNNING   >  (process active right now)"
else
  echo "state   : NOT running  (finished, or paused-asleep / stopped)"
fi

python3 - "$LOG" "$RUNS" <<'PY'
import re, sys, csv, os
log = open(sys.argv[1]).read().splitlines()
RUNS = sys.argv[2]

hdrs = [i for i, l in enumerate(log) if "planned=" in l and "blocks=" in l]
if not hdrs:
    print("could not find a pass header in the log"); sys.exit()
hdr = max(hdrs)

# ordered, de-duplicated TODO list for this pass (log may be doubled)
seen, todo = set(), []
for l in log[hdr:]:
    m = re.search(r"\[TODO\]\s+(\S+)", l)
    if m and m.group(1) not in seen:
        seen.add(m.group(1)); todo.append(m.group(1))

# completed trials, from runs.csv (written once per trial -> immune to log doubling)
done_keys = set()
if os.path.exists(RUNS):
    for r in csv.DictReader(open(RUNS)):
        done_keys.add((r["block"], r["stream"], r["method"], r["seed"]))

def key(tag):                      # block__substrate__stream__method__seedN
    p = tag.split("__")
    return (p[0], p[2], p[3], p[4].replace("seed", ""))

def mins(tag):
    p = tag.split("__"); substrate, stream, method = p[1], p[2], p[3]
    if substrate == "reference": return 1
    if method in ("online_bagging", "river_arf", "river_srp"): return 2
    return 27 if stream == "synth_rotate" else 46

total = len(todo)
done = sum(1 for t in todo if key(t) in done_keys)
cur = next((t for t in todo if key(t) not in done_keys), None)

hb = [l for l in log if "prequential" in l]
m = re.search(r"~([0-9.]+) min left", hb[-1]) if hb else None
blk = re.search(r"blocks=\[([^\]]*)\]", log[hdr]).group(1)

print(f"pass    : {blk}")
print(f"done    : {done}/{total} trials complete")
if cur:
    print(f"current : {cur}")
if hb:
    print(f"latest  : {hb[-1].strip()}")
if not cur:
    print("\nremaining: 0 - pass complete. Stats: python -m scripts.analysis.stats_report")
else:
    cur_left = float(m.group(1)) if m else mins(cur)
    idx = todo.index(cur)
    rem = cur_left + sum(mins(t) for t in todo[idx+1:])
    h, mm = int(rem // 60), int(rem % 60)
    print(f"\nremaining: ~{h}h {mm}m of ACTIVE compute ({total-done} trials left)")
    print("           (wall-clock finish slips by any time the Mac is asleep/lid-closed)")
PY
echo "========================================================"
