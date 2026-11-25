"""Does the simulated data really contain the fingerprint? Control experiment and jitter sweep."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from cids.sim import prototype_scenario, simulate
from cids.attacks import masquerade
from cids.detector import NTPCIDS

T = 0.05


def first_alarm_after(trace, t_att):
    """Time of the first alarm at or after t_att (None if there is none)."""
    return next((t for t, a in zip(trace.t, trace.alarm) if a and t >= t_att), None)


# ---------------------------------------------------------------- Check 1: control
print("CHECK 1: control experiment (A and B have the SAME skew)")
for skew_a, label in [(13.4, "different skews (normal)"), (27.2, "same skew (control)")]:
    base = prototype_scenario(duration=600, seed=4, skew_a=skew_a, skew_b=27.2)
    rx, busy = simulate(masquerade(base, 0x55, "A", 250), with_busy=True)
    tr = NTPCIDS(T).run(rx[0x55], busy[0x55], end_time=600)
    first = first_alarm_after(tr, 250)
    result = "NOT detected" if first is None else f"detected after {first - 250:.1f} s"
    print(f"  A = {skew_a:4.1f} ppm, B = 27.2 ppm, {label:26s}: {result}")

# ---------------------------------------------------------------- Check 2: jitter
print("\nCHECK 2: jitter sweep (masquerade at 250 s, 5 runs per jitter level)")
jitters_us = [2, 10, 50, 100]
medians = []
for jit in jitters_us:
    skews_a, skews_b, false_alarms, delays = [], [], 0, []
    for seed in range(5):
        base = prototype_scenario(duration=600, seed=seed)
        for ecu in base.ecus.values():
            ecu.sw_jitter = jit * 1e-6            # simulate the jitter ourselves

        rx, busy = simulate(base, with_busy=True)           # normal traffic
        tr_a = NTPCIDS(T).run(rx[0x11], busy[0x11])
        tr_b = NTPCIDS(T).run(rx[0x55], busy[0x55])
        skews_a.append(tr_a.skew[-1])
        skews_b.append(tr_b.skew[-1])
        false_alarms += any(tr_b.alarm)

        rx, busy = simulate(masquerade(base, 0x55, "A", 250), with_busy=True)
        tr = NTPCIDS(T).run(rx[0x55], busy[0x55], end_time=600)
        first = first_alarm_after(tr, 250)
        delays.append(None if first is None else first - 250)

    found = [d for d in delays if d is not None]
    medians.append(np.median(found))
    print(f"  jitter {jit:3d} us: skew A {np.mean(skews_a):6.2f} ppm, skew B {np.mean(skews_b):6.2f} ppm,"
          f" false alarms {false_alarms}/5, detected {len(found)}/5,"
          f" median delay {np.median(found):5.1f} s")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(jitters_us, medians, "o-")
ax.set_xscale("log")
ax.set_xticks(jitters_us)
ax.set_xticklabels([str(j) for j in jitters_us])
ax.set_xlabel("ECU scheduling jitter [µs]")
ax.set_ylabel("Median detection delay [s]")
ax.set_title("Masquerade detection vs jitter (NTP-based CIDS)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "fingerprint_checks.png", dpi=150)
print("\nSaved results/fingerprint_checks.png")
plt.show()
