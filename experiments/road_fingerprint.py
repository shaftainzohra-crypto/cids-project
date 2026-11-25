"""Clock-skew fingerprints and one masquerade on real CAN data (ROAD dataset)."""

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
from cids.detector import CIDS, NTPCIDS, root_cause
from cids.metrics import detection_delay

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(ROOT / "data/road/ambient/ambient_dyno_drive_basic_long.log"))
ap.add_argument("--victim", default=None, help="victim ID in hex, e.g. 0C9 (default: automatic)")
ap.add_argument("--attacker", default=None, help="attacker ID in hex (default: automatic)")
args = ap.parse_args()


def skew_ppm(ts, T):
    """Clock skew of one message ID (ppm) and the timing noise around it (us).

    Message n should arrive n*T after the first one; the offset n*T - elapsed
    grows by `skew` us every second. Counting n from the intervals keeps it
    right even when a frame is missing (an interval of about 2T counts as 2).
    """
    steps = np.maximum(1, np.round(np.diff(ts) / T))
    n = np.concatenate(([0], np.cumsum(steps)))
    elapsed = ts - ts[0]
    offset = n * T - elapsed
    slope, intercept = np.polyfit(elapsed, offset, 1)
    noise = np.std(offset - (slope * elapsed + intercept))
    return slope * 1e6, noise * 1e6


def fill_gaps(ts, T):
    """Real logs can miss a frame now and then. The NTP offset of a batch assumes
    exactly N periods between a_0 and a_N, so one missing frame looks like a huge
    clock jump. Put back an estimated time (evenly spaced) for each missing frame."""
    k = np.round(np.diff(ts) / T).astype(int)
    out = [ts[:1]]
    for i in range(len(k)):
        if k[i] >= 2:
            out.append(ts[i] + np.arange(1, k[i]) * (ts[i + 1] - ts[i]) / k[i])
        out.append(ts[i + 1:i + 2])
    return np.concatenate(out), int(np.sum(np.maximum(k - 1, 0)))


# ---------------------------------------------------------------- 1. read the file
print(f"Reading {args.file} ...")
rx = parse_candump(args.file)
start = min(ts[0] for ts in rx.values())
rx = {cid: ts - start for cid, ts in rx.items()}       # time from the start of the capture
duration = max(ts[-1] for ts in rx.values())
periods = periodic_ids(rx)
missing = 0
for cid, T in periods.items():
    rx[cid], m = fill_gaps(rx[cid], T)
    missing += m
print(f"{len(rx)} IDs, {len(periods)} periodic, {duration:.0f} s of traffic, "
      f"{missing} missing frames filled in\n")

# ---------------------------------------------------------------- 2. fingerprints
rows = []
for cid, T in periods.items():
    s, noise = skew_ppm(rx[cid], T)
    rows.append((s, cid, T, noise, len(rx[cid])))
rows.sort()
clean = [r for r in rows if r[3] < 5000]   # noise under 5 ms: really periodic
print(f"{'ID':>6s} {'period':>8s} {'messages':>9s} {'skew':>11s} {'noise':>9s}")
for s, cid, T, noise, n in rows:
    print(f"0x{cid:03X} {T * 1000:6.0f} ms {n:9d} {s:8.2f} ppm {noise:6.1f} us")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(len(clean)), [r[0] for r in clean], "o")
ax1.set_xticks(range(len(clean)))
ax1.set_xticklabels([f"{r[1]:03X}" for r in clean], rotation=90, fontsize=6)
ax1.set_xlabel("Message ID (sorted by skew)")
ax1.set_ylabel("Clock skew [ppm]")
ax1.set_title("Fingerprints in real CAN data: flat steps = one ECU", fontsize=10)
ax1.grid(alpha=0.3)

# ---------------------------------------------------------------- 3. masquerade on real data
# Victim and attacker: two IDs with the same period and very different skews
# (two different ECUs), unless chosen with --victim / --attacker.
if args.victim and args.attacker:
    victim, attacker = int(args.victim, 16), int(args.attacker, 16)
else:
    best = None
    for s1, c1, T1, *_ in clean:
        for s2, c2, T2, *_ in clean:
            if T1 == T2 and c1 != c2 and (best is None or abs(s1 - s2) > best[0]):
                best = (abs(s1 - s2), c1, c2)
    victim, attacker = best[1], best[2]
T = periods[victim]
known = {f"0x{r[1]:03X}": r[0] for r in rows if r[2] == T}
print(f"\nMasquerade: 0x{attacker:03X} (skew {known[f'0x{attacker:03X}']:.2f} ppm) takes over "
      f"0x{victim:03X} (skew {known[f'0x{victim:03X}']:.2f} ppm), period {T * 1000:.0f} ms")

# Before t_att the victim's own frames; after it, the attacker's real frames
# (its real clock and jitter), shifted into the victim's time slots.
v, a = rx[victim], rx[attacker]
t_att = duration / 2
before = v[v < t_att]
after = a[a >= t_att]
slot = before[-1] + T                       # where the victim would send next
attacked = np.concatenate((before, after - after[0] + slot))

for name, det in [("Paper's CIDS", CIDS()), ("NTP-based CIDS", NTPCIDS(T))]:
    tr = det.run(attacked, end_time=attacked[-1])
    early = sum(a and t < t_att for t, a in zip(tr.t, tr.alarm))
    delay = detection_delay(tr, t_att, window=duration)
    result = "NOT detected" if delay is None else f"detected after {delay:.1f} s"
    print(f"  {name:15s}: {result}, false alarms before the attack: {early}")
    if name == "NTP-based CIDS":
        ax2.plot(tr.t, tr.l_pos, label="L+")
        ax2.plot(tr.t, tr.l_neg, "--", label="L-")
        ax2.axhline(5, color="gray", linestyle=":", label="threshold")
        ax2.axvline(t_att, color="red", linewidth=0.8, label="attack starts")
        ax2.set_yscale("symlog", linthresh=10)
        ax2.set_ylim(0, max(max(tr.l_pos), max(tr.l_neg), 50) * 3)

s_after, _ = skew_ppm(attacked[attacked > t_att + 1], T)
print(f"  Skew after the attack: {s_after:.2f} ppm -> closest ID: {root_cause(s_after, known)}"
      f" (the attacker was 0x{attacker:03X})")

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("CUSUM")
ax2.set_title(f"NTP-based CIDS: our masquerade of 0x{victim:03X} on real data", fontsize=10)
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "road_fingerprint.png", dpi=150)
print("\nSaved results/road_fingerprint.png")
plt.show()
