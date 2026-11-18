"""
Replay simulated traffic on a (virtual) CAN bus with python-can.

Linux, real kernel virtual CAN:
    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0
    python tools/vcan_replay.py --interface socketcan --channel vcan0

Any OS (python-can's in-process virtual bus, no setup needed):
    python tools/vcan_replay.py --interface virtual --channel demo

The receiver timestamps frames with the host's clock, so the recorded times
include OS scheduling jitter: compare the skews CIDS estimates on the
recording with those of the simulation.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def to_frames(rx: dict[int, np.ndarray]):
    """Merge {can_id: times} into one time-ordered list of (time, can_id)."""
    frames = [(t, cid) for cid, ts in rx.items() for t in ts]
    frames.sort()
    return frames


def replay(frames, bus, speed: float = 1.0):
    """Send frames on `bus` at their (relative) times. speed=2 -> twice as fast."""
    import can
    t0 = frames[0][0]
    start = time.perf_counter()
    for t, cid in frames:
        target = start + (t - t0) / speed
        while (now := time.perf_counter()) < target:
            if target - now > 0.002:
                time.sleep(target - now - 0.001)
        bus.send(can.Message(arbitration_id=cid, data=bytes(8), is_extended_id=False))


def record(bus, duration: float) -> dict[int, np.ndarray]:
    """Receive frames for `duration` seconds; returns {can_id: times}."""
    out: dict[int, list[float]] = {}
    end = time.time() + duration
    while time.time() < end:
        msg = bus.recv(timeout=0.1)
        if msg is not None:
            out.setdefault(msg.arbitration_id, []).append(msg.timestamp)
    return {cid: np.asarray(ts) for cid, ts in out.items()}


def replay_and_record(rx, interface="virtual", channel="demo", seconds=10.0, speed=1.0):
    """Replay the first `seconds` of rx on one bus and record it on another."""
    import can
    frames = [f for f in to_frames(rx) if f[0] - min(ts[0] for ts in rx.values()) < seconds]
    tx = can.Bus(interface=interface, channel=channel)
    rxbus = can.Bus(interface=interface, channel=channel)
    result = {}
    th = threading.Thread(target=lambda: result.update(record(rxbus, seconds / speed + 1)))
    th.start()
    replay(frames, tx, speed)
    th.join()
    tx.shutdown(); rxbus.shutdown()
    return result


if __name__ == "__main__":
    from cids.sim import prototype_scenario, simulate
    from cids.detector import NTPCIDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="virtual")
    ap.add_argument("--channel", default="demo")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()

    rx = simulate(prototype_scenario(duration=a.seconds + 1, seed=1))
    rec = replay_and_record(rx, a.interface, a.channel, a.seconds)
    for cid, ts in sorted(rec.items()):
        tr = NTPCIDS(0.05).run(ts)
        sim_tr = NTPCIDS(0.05).run(rx[cid])
        print(f"0x{cid:02X}: {len(ts)} frames recorded | skew simulated "
              f"{sim_tr.skew[-1]:7.2f} ppm | skew on the bus {tr.skew[-1]:9.2f} ppm")
