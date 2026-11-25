"""Masquerade on every pair of real message IDs with the same period (ROAD dataset)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import argparse

import numpy as np
import matplotlib.pyplot as plt

from cids.road import parse_candump, periodic_ids
from cids.detector import NTPCIDS
from cids.metrics import detection_delay

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "data/road/ambient/ambient_dyno_drive_basic_long.log"))
ap.add_argument("--window", type=float, default=300, help="detection window after the attack (s)")
args = ap.parse_args()


def skew_ppm(ts, T):
    """Clock skew (ppm) and noise (us) from a straight-line fit (as in road_fingerprint.py)."""
    steps = np.maximum(1, np.round(np.diff(ts) / T))
    n = np.concatenate(([0], np.cumsum(steps)))
    elapsed = ts - ts[0]
    offset = n * T - elapsed
    slope, intercept = np.polyfit(elapsed, offset, 1)
    return slope * 1e6, np.std(offset - (slope * elapsed + intercept)) * 1e6


def fill_gaps(ts, T):
    """Put back an estimated time for each missing frame (as in road_fingerprint.py)."""
    k = np.round(np.diff(ts) / T).astype(int)
    out = [ts[:1]]
    for i in range(len(k)):
        if k[i] >= 2:
            out.append(ts[i] + np.arange(1, k[i]) * (ts[i + 1] - ts[i]) / k[i])
        out.append(ts[i + 1:i + 2])
    return np.concatenate(out)


# ---------------------------------------------------------------- read, clean IDs, ECU groups
print(f"Reading {args.file} ...")
rx = parse_candump(args.file)
start = min(ts[0] for ts in rx.values())
rx = {cid: ts - start for cid, ts in rx.items()}
duration = max(ts[-1] for ts in rx.values())
periods = periodic_ids(rx)
info = {}
for cid, T in periods.items():
    rx[cid] = fill_gaps(rx[cid], T)
    s, noise = skew_ppm(rx[cid], T)
    if noise < 5000:                       # keep only really periodic IDs
        info[cid] = (s, T)

# ECU groups: sort by skew; a gap of more than 3 ppm starts a new group
order = sorted(info, key=lambda c: info[c][0])
group, g = {}, 0
for i, cid in enumerate(order):
    if i > 0 and info[cid][0] - info[order[i - 1]][0] > 3:
        g += 1
    group[cid] = g
print(f"{len(info)} clean periodic IDs in {g + 1} ECU groups\n")

# ---------------------------------------------------------------- 1. each ID alone: false alarms
t_att = duration / 2
own_alarms = {}
for cid, (s, T) in info.items():
    tr = NTPCIDS(T).run(rx[cid], end_time=rx[cid][-1])
    own_alarms[cid] = sum(a and t < t_att for t, a in zip(tr.t, tr.alarm))
noisy = [c for c in info if own_alarms[c] > 0]
print(f"IDs that raise false alarms on their own traffic (first half): {len(noisy)} of {len(info)}")
print("  " + ", ".join(f"0x{c:03X} ({own_alarms[c]})" for c in sorted(noisy)) + "\n")

# ---------------------------------------------------------------- 2. every masquerade pair
results = []          # (same_ecu, |skew difference|, detected, delay)
victims = [c for c in info if own_alarms[c] == 0]           # only victims with clean traffic
print("Testing pairs (one dot per victim, this can take a few minutes) ", end="", flush=True)
for v in victims:
    print(".", end="", flush=True)
    T = info[v][1]
    for a in info:
        if a == v or info[a][1] != T:
            continue
        before = rx[v][rx[v] < t_att]
        after = rx[a][rx[a] >= t_att]
        attacked = np.concatenate((before, after - after[0] + before[-1] + T))
        tr = NTPCIDS(T).run(attacked, end_time=attacked[-1])
        d = detection_delay(tr, t_att, window=args.window)
        results.append((group[v] == group[a], abs(info[v][0] - info[a][0]), d is not None, d))
print(f"\n{len(results)} masquerade pairs tested (same period, victim without false alarms)\n")

# ---------------------------------------------------------------- 3. summary table
bins = [(0, 3, "same ECU group"), (3, 10, "3 - 10 ppm"), (10, 30, "10 - 30 ppm"),
        (30, 100, "30 - 100 ppm"), (100, 1e9, "> 100 ppm")]
print(f"{'skew difference':20s} {'pairs':>6s} {'detected':>9s} {'median delay':>13s}")
for lo, hi, label in bins:
    sel = [r for r in results if (r[0] if lo == 0 else (not r[0] and lo <= r[1] < hi))]
    if not sel:
        continue
    hits = [r[3] for r in sel if r[2]]
    med = f"{np.median(hits):8.1f} s" if hits else "        -"
    print(f"{label:20s} {len(sel):6d} {len(hits) / len(sel):9.0%} {med:>13s}")
print(f"(detected = alarm within {args.window:.0f} s of the attack)")

# ---------------------------------------------------------------- 4. plot
fig, ax = plt.subplots(figsize=(6.5, 4))
for same, colour, label in [(True, "gray", "same ECU (control)"), (False, "C0", "different ECUs")]:
    det = [(r[1], r[3]) for r in results if r[0] == same and r[2]]
    miss = [r[1] for r in results if r[0] == same and not r[2]]
    if det:
        ax.plot([x for x, _ in det], [y for _, y in det], "o", color=colour, ms=4, label=f"{label}: detected")
    if miss:
        ax.plot(miss, [args.window * 1.1] * len(miss), "x", color=colour, ms=5,
                label=f"{label}: not detected")
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("Skew difference between attacker and victim [ppm]")
ax.set_ylabel("Detection delay [s]")
ax.set_title("Masquerade on real CAN data (NTP-based CIDS)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(RESULTS / "road_all_pairs.png", dpi=150)
print("\nSaved results/road_all_pairs.png (crosses at the top = not detected)")
plt.show()
