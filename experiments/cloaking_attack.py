"""The cloaking attack: an attacker that imitates the victim's clock skew."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from cids.sim import prototype_scenario, simulate
from cids.attacks import cloaking
from cids.detector import CIDS, NTPCIDS
from cids.metrics import detection_delay, has_false_alarm

T = 0.05
T_ATT = 250
DURATION = 400
errors_us = [-50, -10, -2, -0.5, -0.1, 0, 0.1, 0.5, 2, 10, 50]   # error per period (us)
detectors = [("Paper's CIDS", lambda: CIDS()), ("NTP-based CIDS", lambda: NTPCIDS(T))]

fig, ax = plt.subplots(figsize=(6, 3.5))
print(f"{'error per period':>17s} | {'Paper':>6s} | {'NTP':>6s}")
rates = {name: [] for name, _ in detectors}
for err in errors_us:
    row = f"{err:>14g} us |"
    for name, make in detectors:
        detected = fair = 0
        for seed in range(10):
            sc = cloaking(prototype_scenario(duration=DURATION, seed=seed), 0x55, "A", T_ATT,
                          error=err * 1e-6)
            rx, busy = simulate(sc, with_busy=True)
            tr = make().run(rx[0x55], busy[0x55], end_time=DURATION)
            if has_false_alarm(tr, T_ATT):
                continue
            fair += 1
            detected += detection_delay(tr, T_ATT, window=60) is not None
        rates[name].append(detected / fair)
        row += f" {detected / fair:6.0%} |"
    print(row)

for name, style in zip(rates, ["--", "-"]):
    ax.plot(errors_us, rates[name], style, marker="o", label=name)
ax.set_xscale("symlog", linthresh=0.1)
ticks = [-50, -10, -1, -0.1, 0, 0.1, 1, 10, 50]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Cloaking error per period [µs]")
ax.set_ylabel("Detection rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Cloaking attack: A imitates B's clock", fontsize=10)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "cloaking_attack.png", dpi=150)
print("\nSaved results/cloaking_attack.png")
plt.show()
