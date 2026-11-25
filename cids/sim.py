"""
CAN bus simulator with skewed ECU clocks.

ECU clocks and periodic schedules: ECU.to_ref, schedule_periodic
Bus arbitration and the receiver:  frame_duration, arbitrate, receiver_timestamps

Output format (read by the detector):
    simulate(...) returns rx = {can_id: np.ndarray of reception times [s]}
    and, with with_busy=True, also busy = {can_id: np.ndarray of int}, the
    number of frames sent back-to-back right before each frame.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

BITRATE = 500_000  # bit/s, as in the paper's prototype


# ----------------------------------------------------------------- clocks
@dataclass
class ECU:
    name: str
    skew_ppm: float
    sw_jitter: float = 2e-6  # std of the software scheduling jitter (s)

    def to_ref(self, local_t):
        """Convert a time on this ECU's clock to the reference (receiver) clock.

        A clock with skew s ppm runs (1 + s*1e-6) times faster than the
        reference, so local time L corresponds to reference time L/(1+s*1e-6).
        """
        return local_t / (1.0 + self.skew_ppm * 1e-6)


def schedule_periodic(ecu: ECU, period: float, start: float, duration: float,
                      jitter: np.ndarray | None = None, t_min: float = 0.0,
                      t_max: float | None = None) -> np.ndarray:
    """Reference-clock release times of a message sent every `period` seconds
    of the ECU's *local* clock, starting at local time `start`.

    jitter: optional array of per-tick jitter (s), one value per tick.
    Returns only the times in [t_min, t_max).
    """
    t_max = duration if t_max is None else t_max
    n = int(duration * (1 + 1e-3) / period) + 2
    k = np.arange(n)
    t = ecu.to_ref(start + k * period)
    if jitter is not None:
        t = t + jitter[:n]
    return t[(t >= t_min) & (t < t_max)]


# ----------------------------------------------------------------- bus and receiver
def frame_duration(dlc: int = 8) -> float:
    """Duration of a standard CAN data frame (s): 44 overhead bits + 8*DLC
    data bits + 3 bits of interframe space, plus ~10% bit stuffing."""
    bits = int((44 + 8 * dlc + 3) * 1.1)
    return bits / BITRATE


def arbitrate(releases: list[tuple[float, int, int]]):
    """Serialise frames on the bus.

    releases: list of (release_time, can_id, dlc).
    Returns a list of (can_id, end_time, position) in bus order, where
    position = number of frames sent back-to-back right before this one.
    When several frames are pending, the lowest CAN ID wins arbitration.
    """
    releases = sorted(releases)
    out = []
    pending: list[tuple[int, float, int]] = []
    bus_free, pos, i, n = 0.0, 0, 0, len(releases)
    while i < n or pending:
        if not pending:                         # bus idle: jump to next release
            bus_free = max(bus_free, releases[i][0])
        while i < n and releases[i][0] <= bus_free:
            t, cid, dlc = releases[i]
            heapq.heappush(pending, (cid, t, dlc))
            i += 1
        cid, t, dlc = heapq.heappop(pending)    # lowest ID wins
        start = max(bus_free, t)
        end = start + frame_duration(dlc)
        pos = pos + 1 if bus_free > t else 0    # had to wait -> busy period
        out.append((cid, end, pos))
        bus_free = end
    return out


def receiver_timestamps(times: np.ndarray, rng, resolution=4e-6, noise=1e-6):
    """Timestamp quantisation of the receiver (Arduino micros(): 4 us) + noise."""
    t = times + rng.normal(0, noise, len(times))
    return np.floor(t / resolution) * resolution


# ----------------------------------------------------------------- scenario
@dataclass
class PeriodicMessage:
    can_id: int
    period: float          # nominal period (s), on the sender's clock
    sender: str
    dlc: int = 8
    start: float | None = None   # local start time; random if None
    stop: float | None = None    # reference time after which it is not sent


@dataclass
class Injection:
    """Messages sent by an attacker ECU on its own clock (see attacks.py)."""
    can_id: int
    sender: str
    local_period: float
    t_start: float
    t_stop: float = float("inf")
    dlc: int = 8
    extra_delay: float = 0.0  # added to every local period (cloaking error)


@dataclass
class Scenario:
    ecus: dict[str, ECU]
    messages: list[PeriodicMessage]
    injections: list[Injection] = field(default_factory=list)
    duration: float = 600.0
    rx_resolution: float = 4e-6
    rx_noise: float = 1e-6
    seed: int = 0


def simulate(sc: Scenario, with_busy: bool = False):
    """Run a scenario: schedule -> arbitrate -> timestamp."""
    rng = np.random.default_rng(sc.seed)
    releases = []
    # Messages of one ECU scheduled on the same tick share the tick's jitter
    # and are written to the controller 20 us apart (SPI write time).
    tick_jitter, order = {}, {}
    for m in sc.messages:
        ecu = sc.ecus[m.sender]
        start = m.start if m.start is not None else rng.uniform(0, m.period)
        stop = m.stop if m.stop is not None else sc.duration
        key = (m.sender, round(start, 9), m.period)
        n = int(sc.duration * (1 + 1e-3) / m.period) + 2
        if key not in tick_jitter:
            tick_jitter[key] = rng.normal(0, ecu.sw_jitter, n)
        pos = order.get(key, 0)
        order[key] = pos + 1
        t = schedule_periodic(ecu, m.period, start, sc.duration,
                              jitter=tick_jitter[key] + pos * 20e-6,
                              t_max=min(stop, sc.duration))
        releases += [(x, m.can_id, m.dlc) for x in t]

    for inj in sc.injections:
        ecu = sc.ecus[inj.sender]
        local0 = inj.t_start * (1.0 + ecu.skew_ppm * 1e-6)
        step = inj.local_period + inj.extra_delay
        n = max(int((min(inj.t_stop, sc.duration) - inj.t_start) / step) + 2, 0)
        t = ecu.to_ref(local0 + np.arange(n) * step) + rng.normal(0, ecu.sw_jitter, n)
        t = t[t < min(inj.t_stop, sc.duration)]
        releases += [(x, inj.can_id, inj.dlc) for x in t]

    rx, busy = {}, {}
    for cid, end, pos in arbitrate(releases):
        rx.setdefault(cid, []).append(end)
        busy.setdefault(cid, []).append(pos)
    rx = {cid: receiver_timestamps(np.asarray(ts), rng, sc.rx_resolution, sc.rx_noise)
          for cid, ts in rx.items()}
    if with_busy:
        return rx, {cid: np.asarray(b) for cid, b in busy.items()}
    return rx


def prototype_scenario(duration=600.0, seed=0, skew_a=13.4, skew_b=27.2,
                       period=0.05, stops=None) -> Scenario:
    """The paper's prototype: A sends 0x11 and 0x13, B sends 0x55, every 50 ms.
    Skews default to the values measured in the paper."""
    stops = stops or {}
    ecus = {"A": ECU("A", skew_a), "B": ECU("B", skew_b)}
    rng = np.random.default_rng(seed + 1000)
    phase_a = rng.uniform(0, period)
    msgs = [
        PeriodicMessage(0x11, period, "A", start=phase_a, stop=stops.get(0x11)),
        PeriodicMessage(0x13, period, "A", start=phase_a, stop=stops.get(0x13)),
        PeriodicMessage(0x55, period, "B", start=rng.uniform(0, period), stop=stops.get(0x55)),
    ]
    return Scenario(ecus, msgs, duration=duration, seed=seed)


def next_send_time(sc: Scenario, can_id: int, t: float) -> float:
    """Reference time of the first scheduled send of can_id at or after t."""
    m = next(m for m in sc.messages if m.can_id == can_id)
    ecu = sc.ecus[m.sender]
    k = np.ceil((t * (1 + ecu.skew_ppm * 1e-6) - m.start) / m.period)
    return float(ecu.to_ref(m.start + k * m.period))
