"""
Evaluation metrics: detection rate, detection delay, false-positive rate.
A "trace" is the cids.detector.Trace returned by CIDS.run / NTPCIDS.run.
"""
from __future__ import annotations

import numpy as np


def detection_delay(trace, t_attack: float, window: float = 60.0):
    """Seconds between the attack start and the first alarm at/after it.
    Returns None if there is no alarm within `window` seconds."""
    for t, a in zip(trace.t, trace.alarm):
        if a and t >= t_attack:
            d = t - t_attack
            return d if d <= window else None
    return None


def has_false_alarm(trace, t_attack: float) -> bool:
    """True if the trace raised an alarm before the attack started."""
    return any(a and t < t_attack for t, a in zip(trace.t, trace.alarm))


def false_positive_rate(traces) -> float:
    """Fraction of batches that raised an alarm, over attack-free traces."""
    alarms = sum(sum(tr.alarm) for tr in traces)
    batches = sum(len(tr.alarm) for tr in traces)
    return alarms / batches if batches else 0.0


def summarize(delays):
    """delays: list of detection delays (None = missed).
    Returns (detection rate, median delay of the detected runs)."""
    hit = [d for d in delays if d is not None]
    rate = len(hit) / len(delays) if delays else float("nan")
    return rate, (float(np.median(hit)) if hit else None)
