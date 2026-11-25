"""Draw the testbed and detector-pipeline diagram (Figure 1 of the report)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, A, B, R = "#222222", "#666666", "#2a6fb0", "#d9822b", "#3b8a4f"
fig, ax = plt.subplots(figsize=(10, 3.0), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

def box(x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, color=MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

# --- testbed (left)
ax.text(1, 28.5, "Simulated testbed (cids/sim.py)", fontsize=9.5, fontweight="bold", color=INK)
box(2, 18, 13, 7, "ECU A\nskew 13.4 ppm\n0x11, 0x13", A)
box(19, 18, 13, 7, "ECU B\nskew 27.2 ppm\n0x55", B)
box(36, 18, 11, 7, "Receiver R\n(runs CIDS)", R, bold=True)
ax.plot([2, 47], [11, 11], color=INK, lw=3)
ax.text(24.5, 7.8, "CAN bus, 500 kbit/s, all messages every 50 ms", ha="center",
        fontsize=7.5, color=MUTED)
ax.text(24.5, 4.0, "jitter σ = 2 µs · arbitration by ID · 4 µs timestamps", ha="center",
        fontsize=7.5, color=MUTED)
for x, c in [(8.5, A), (25.5, B)]:
    arrow(x, 18, x, 11.4, c)
arrow(41.5, 11.4, 41.5, 18, R)

# --- detector pipeline (right)
ax.text(52, 28.5, "Detector pipeline (cids/detector.py)", fontsize=9.5, fontweight="bold",
        color=INK)
steps = [("Arrival times\nbatch of N = 20", 52, 18), ("Clock offset\nO[k] → O_acc[k]", 68, 18),
         ("RLS slope\n= clock skew S", 84, 18), ("CUSUM on error\ne = O_acc − S·t", 84, 4),
         ("Alarm +\nroot cause", 68, 4), ("Verification:\nskip queued\nbatches", 52, 4)]
for t, x, y in steps:
    box(x, y, 13.5, 7, t, R if "Alarm" in t else MUTED, fs=7.8, bold="Alarm" in t)
arrow(65.8, 21.5, 67.7, 21.5); arrow(81.8, 21.5, 83.7, 21.5)
arrow(90.75, 18, 90.75, 11.3); arrow(83.7, 7.5, 81.8, 7.5)
ax.annotate("", xy=(58.75, 17.7), xytext=(58.75, 11.3),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, ls="--"))
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
plt.savefig(RESULTS / "setup_diagram.png", bbox_inches="tight", facecolor="white")
