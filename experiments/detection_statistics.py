"""Detection rate, delay and false positives over many runs, and the effect of N and kappa."""

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
from cids.detector import CIDS, NTPCIDS
from cids.metrics import detection_delay, false_positive_rate, has_false_alarm, summarize

T = 0.05
T_ATT = 250          # attack time (s)
DURATION = 400       # length of each attack run (s)


def attack_runs(make_detector, make_attack, cid, seeds):
    """Run one attack on many seeds; return (detection rate, median delay, runs excluded)."""
    delays, excluded = [], 0
    for seed in seeds:
        sc = make_attack(prototype_scenario(duration=DURATION, seed=seed))
        rx, busy = simulate(sc, with_busy=True)
        tr = make_detector().run(rx[cid], busy[cid], end_time=DURATION)
        if has_false_alarm(tr, T_ATT):     # alarm before the attack: not a fair run
            excluded += 1
            continue
        delays.append(detection_delay(tr, T_ATT, window=60))
    rate, median = summarize(delays)
    return rate, median, excluded


def fpr_runs(make_detector, seeds, duration=1000):
    """False-positive rate on attack-free traffic, all three IDs."""
    traces = []
    for seed in seeds:
        rx, busy = simulate(prototype_scenario(duration=duration, seed=100 + seed), with_busy=True)
        for cid in (0x11, 0x13, 0x55):
            traces.append(make_detector().run(rx[cid], busy[cid]))
    return false_positive_rate(traces)


# ---------------------------------------------------------------- Part 1
print("PART 1: both detectors, 20 runs per attack (detected = alarm within 60 s)\n")
attacks = [
    ("Fabrication", lambda sc: fabrication(sc, 0x11, "B", T_ATT), 0x11),
    ("Suspension", lambda sc: suspension(sc, 0x11, T_ATT), 0x11),
    ("Masquerade", lambda sc: masquerade(sc, 0x55, "A", T_ATT), 0x55),
    ("Masquerade, mistimed 1.04 ms", lambda sc: masquerade(sc, 0x55, "A", T_ATT, 1.04e-3), 0x55),
]
detectors = [("Paper's CIDS", lambda: CIDS()), ("NTP-based CIDS", lambda: NTPCIDS(T))]

print(f"{'Attack':30s} | {'Paper: rate':>11s} {'delay':>8s} | {'NTP: rate':>9s} {'delay':>8s}")
for name, make_attack, cid in attacks:
    row = f"{name:30s} |"
    for _, make_det in detectors:
        rate, median, excluded = attack_runs(make_det, make_attack, cid, range(20))
        delay = "-" if median is None else f"{median:.2f} s"
        row += f" {rate:10.0%}{'*' if excluded else ' '} {delay:>8s} |"
    print(row)
fprs = [fpr_runs(make_det, range(10)) for _, make_det in detectors]
print(f"{'False positives (batches)':30s} | {fprs[0]:10.1%}  {'':>8s} | {fprs[1]:8.1%}  {'':>8s} |")
print("* some runs excluded: false alarm before the attack\n")

# ---------------------------------------------------------------- Part 2
print("PART 2: NTP-based CIDS, masquerade, batch size N x CUSUM kappa (10 runs each)\n")
Ns, kappas = [10, 20, 40], [2, 3, 5, 8]
delay_table = {}
print(f"{'':8s}" + "".join(f"{'kappa=' + str(k):>22s}" for k in kappas))
for N in Ns:
    row = f"N = {N:<4d}"
    for k in kappas:
        make = lambda N=N, k=k: NTPCIDS(T, N=N, kappa=k)
        rate, median, _ = attack_runs(make, lambda sc: masquerade(sc, 0x55, "A", T_ATT), 0x55, range(10))
        fpr = fpr_runs(make, range(5), duration=600)
        delay_table[(N, k)] = median
        row += f"   {rate:4.0%} {median:5.1f}s FP {fpr:4.1%}"
    print(row)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for N in Ns:
    ax.plot(kappas, [delay_table[(N, k)] for k in kappas], "o-", label=f"N = {N}")
ax.set_xlabel("CUSUM kappa")
ax.set_ylabel("Median detection delay [s]")
ax.set_title("Masquerade: effect of N and kappa (NTP-based CIDS)", fontsize=10)
ax.set_xticks(kappas)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "detection_statistics.png", dpi=150)
print("\nSaved results/detection_statistics.png")
plt.show()
