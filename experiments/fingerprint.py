"""Clock-skew fingerprint of each ECU in the simulated prototype (paper vs NTP offset)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from cids.sim import prototype_scenario, simulate
from cids.detector import RLS, offset_ntp, offset_original

T = 0.05   # nominal period: 50 ms
N = 20     # messages per batch

rx = simulate(prototype_scenario(duration=600, seed=1))   # 10 minutes of traffic

fig, (ax_paper, ax_ntp) = plt.subplots(1, 2, figsize=(11, 4))

for cid, ecu in [(0x11, "A"), (0x13, "A"), (0x55, "B")]:
    ts = rx[cid]                                  # arrival times of this ID
    times, acc_paper, acc_ntp = [], [], []
    total_paper = total_ntp = 0.0
    rls_paper, rls_ntp = RLS(), RLS()
    mu_prev = np.mean(np.diff(ts[:N + 1]))        # first guess of the interval

    for start in range(0, len(ts) - N, N):        # one batch at a time
        a = ts[start:start + N + 1]               # a0 ... a20
        t = a[-1] - ts[0]                         # elapsed time (s)

        o, mu_prev = offset_original(a, mu_prev)  # paper's formula
        total_paper += abs(o)
        rls_paper.update(t, total_paper - rls_paper.S * t)

        total_ntp += N * offset_ntp(a, T)         # NTP formula
        rls_ntp.update(t, total_ntp - rls_ntp.S * t)

        times.append(t)
        acc_paper.append(total_paper)
        acc_ntp.append(total_ntp)

    print(f"0x{cid:02X} (ECU {ecu}):  paper skew = {rls_paper.S:6.2f} ppm   "
          f"NTP skew = {rls_ntp.S:6.2f} ppm")
    label = f"0x{cid:02X} (ECU {ecu})"
    style = "--" if cid == 0x13 else "-"
    ax_paper.plot(times, np.array(acc_paper) / 1000, style, label=label)
    ax_ntp.plot(times, np.array(acc_ntp) / 1000, style, label=label)

ax_paper.set_title("Paper's formula (Algorithm 1)")
ax_ntp.set_title("NTP formula")
for ax in (ax_paper, ax_ntp):
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Accumulated offset [ms]")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "fingerprint.png", dpi=150)
print("Saved results/fingerprint.png")
plt.show()
