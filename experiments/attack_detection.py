"""CIDS against the paper's three attacks (fabrication, suspension, masquerade)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from cids.sim import prototype_scenario, simulate
from cids.attacks import fabrication, masquerade, suspension
from cids.detector import CIDS, NTPCIDS, root_cause

T = 0.05
base = prototype_scenario(duration=600, seed=4)   # 10 minutes, no attack

# 1. Fingerprints: the skew of each ECU, measured on normal traffic
rx, busy = simulate(base, with_busy=True)
known = {"A": NTPCIDS(T).run(rx[0x11], busy[0x11]).skew[-1],
         "B": NTPCIDS(T).run(rx[0x55], busy[0x55]).skew[-1]}
print(f"Known skews: A = {known['A']:.2f} ppm, B = {known['B']:.2f} ppm\n")

# 2. The three attacks: (title, attacked scenario, attacked ID, attack time)
attacks = [
    ("Fabrication of 0x11 by B at 400 s", fabrication(base, 0x11, "B", 400), 0x11, 400),
    ("Suspension of 0x11 at 400 s", suspension(base, 0x11, 400), 0x11, 400),
    ("Masquerade of 0x55 by A at 250 s", masquerade(base, 0x55, "A", 250), 0x55, 250),
]
detectors = [("Paper's CIDS", lambda: CIDS()), ("NTP-based CIDS", lambda: NTPCIDS(T))]

fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
for row, (title, sc, cid, t_att) in enumerate(attacks):
    rx, busy = simulate(sc, with_busy=True)
    print(title)
    for col, (name, make) in enumerate(detectors):
        tr = make().run(rx[cid], busy[cid], end_time=sc.duration)
        first = next((t for t, a in zip(tr.t, tr.alarm) if a and t >= t_att), None)
        early = any(a and t < t_att for t, a in zip(tr.t, tr.alarm))
        result = "NOT detected" if first is None else f"detected after {first - t_att:.2f} s"
        print(f"  {name:15s}: {result}" + ("  (false alarm before the attack!)" if early else ""))

        ax = axes[row, col]
        ax.plot(tr.t, tr.l_pos, label="L+")
        ax.plot(tr.t, tr.l_neg, "--", label="L-")
        ax.axhline(5, color="gray", linestyle=":", label="threshold")
        ax.axvline(t_att, color="red", linewidth=0.8, label="attack starts")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(0, max(max(tr.l_pos), max(tr.l_neg), 50) * 3)
        ax.set_title(f"{name}: {title}", fontsize=9)
    if cid == 0x55:   # root cause: skew of the new traffic only
        keep = rx[cid] > t_att + 1
        after = NTPCIDS(T).run(rx[cid][keep], busy[cid][keep]).skew[-1]
        print(f"  Skew after the attack: {after:.2f} ppm -> attacker = ECU {root_cause(after, known)}")
    print()

for ax in axes[:, 0]:
    ax.set_ylabel("CUSUM")
for ax in axes[-1]:
    ax.set_xlabel("Time [s]")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "attack_detection.png", dpi=150)
print("Saved results/attack_detection.png")
plt.show()
