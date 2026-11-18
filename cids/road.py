"""
Loader for the ROAD dataset (Real ORNL Automotive Dynamometer).

Raw captures use the candump log format, one frame per line:
    (1600000000.123456) can0 0C9#8C00000000000000
    (timestamp)         chan ID#DATA

Note: timestamps are written with 1 us resolution but the logger guarantees
only ~100 us accuracy, and the "masquerade" captures were created in
post-processing by removing the legitimate frames.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")


def parse_candump(source) -> dict[int, np.ndarray]:
    """Parse a candump .log file (path) or an iterable of lines.
    Returns {can_id: sorted array of timestamps (s)}."""
    if isinstance(source, (str, Path)) and Path(source).exists():
        lines = Path(source).read_text().splitlines()
    else:
        lines = source
    out: dict[int, list[float]] = {}
    for line in lines:
        m = LINE.search(line)
        if m:
            out.setdefault(int(m.group(2), 16), []).append(float(m.group(1)))
    return {cid: np.sort(np.asarray(ts)) for cid, ts in out.items()}


def nominal_period(ts: np.ndarray) -> float:
    """Nominal period of a periodic message: the median interval, rounded to
    the nearest millisecond (ECUs use round periods like 10, 20, 100 ms)."""
    return round(float(np.median(np.diff(ts))) * 1000) / 1000


def periodic_ids(rx: dict[int, np.ndarray], min_count=200, max_cv=0.2):
    """IDs that look periodic: enough messages and a small coefficient of
    variation (std/mean) of their intervals. Returns {can_id: period}."""
    out = {}
    for cid, ts in rx.items():
        if len(ts) < min_count:
            continue
        d = np.diff(ts)
        if d.mean() > 0 and d.std() / d.mean() < max_cv:
            out[cid] = nominal_period(ts)
    return out
