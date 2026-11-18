"""
Unit and integration tests of the simulator and of the detector.

    python tests/run_tests.py            # all tests
    python tests/run_tests.py clock bus  # selected groups

Each test compares the behaviour of the code with what the theory predicts
(e.g. the interval of a 100 ppm clock must be T/(1+100e-6)).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"\n         {detail}"))


# ------------------------------------------------------------------ environment
def test_environment():
    import importlib
    for mod in ("numpy", "matplotlib", "can"):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except ImportError:
            check(f"import {mod}", False, "run: pip install -r requirements.txt")


# ------------------------------------------------------------------ simulator
def test_clock():
    from cids.sim import ECU, schedule_periodic
    ecu = ECU("X", skew_ppm=100.0)
    check("to_ref(1.0) = 1/(1+100e-6)", abs(ecu.to_ref(1.0) - 1 / 1.0001) < 1e-12,
          "a clock with skew s runs (1+s*1e-6) times faster than the reference")
    t = schedule_periodic(ecu, 0.05, 0.0, 10.0)
    check("200 messages in 10 s", len(t) in (200, 201), f"got {len(t)}")
    mu = np.diff(t).mean()
    check("mean interval = T/(1+s)", abs(mu - 0.05 / 1.0001) < 1e-12,
          f"got {mu!r}, expected {0.05 / 1.0001!r}")
    rng = np.random.default_rng(0)
    tj = schedule_periodic(ecu, 0.05, 0.0, 10.0, jitter=rng.normal(0, 5e-6, 300))
    check("jitter changes the times by a few us",
          1e-6 < np.std(tj - t[:len(tj)]) < 1e-5, "add jitter[k] to the k-th release time")


def test_bus():
    from cids.sim import arbitrate, frame_duration, receiver_timestamps
    d = frame_duration(8)
    check("8-byte frame lasts ~244 us", 230e-6 < d < 260e-6, f"got {d * 1e6:.1f} us")
    out = arbitrate([(0.0, 0x55, 8), (0.0, 0x11, 8)])
    check("lowest ID wins arbitration", out[0][0] == 0x11, f"order was {[hex(o[0]) for o in out]}")
    check("loser waits one frame", abs(out[1][1] - 2 * d) < 1e-12,
          "the second frame ends at start_of_bus + 2 * frame_duration")
    check("busy position 0 then 1", [o[2] for o in out] == [0, 1],
          "position = frames sent back-to-back right before (0 if the bus was idle)")
    out = arbitrate([(0.0, 0x11, 8), (1.0, 0x55, 8)])
    check("idle bus -> no waiting", abs(out[1][1] - (1.0 + d)) < 1e-12 and out[1][2] == 0)
    ts = receiver_timestamps(np.array([0.1234567]), np.random.default_rng(0), 4e-6, 0)
    check("timestamps quantised to 4 us", abs(ts[0] / 4e-6 - round(ts[0] / 4e-6)) < 1e-6)
    from cids.sim import prototype_scenario, simulate
    rx = simulate(prototype_scenario(duration=60, seed=1))
    check("prototype: 0x11, 0x13, 0x55 all received",
          set(rx) == {0x11, 0x13, 0x55} and all(len(v) > 1150 for v in rx.values()))


def test_attacks():
    from cids.sim import prototype_scenario, simulate
    from cids.attacks import fabrication, masquerade, suspension
    base = prototype_scenario(duration=120, seed=2)
    rx = simulate(fabrication(base, 0x11, "B", 60))
    n_before = np.sum(rx[0x11] < 60); n_after = np.sum(rx[0x11] >= 60)
    check("fabrication doubles the rate of 0x11", 1.9 < n_after / n_before < 2.1,
          f"before {n_before}, after {n_after}")
    rx = simulate(suspension(base, 0x11, 60))
    check("suspension: no 0x11 after t", rx[0x11].max() < 60.1)
    rx = simulate(masquerade(base, 0x55, "A", 60))
    d = np.diff(rx[0x55])
    check("masquerade keeps the frequency", np.all(d < 0.0505) and np.all(d > 0.0495),
          "the attacker must start in the victim's next time slot")
    late = rx[0x55][rx[0x55] > 61]
    k = np.arange(len(late))
    slope = np.polyfit(k, late, 1)[0]
    check("after masquerade 0x55 has A's period", abs(slope - 0.05 / (1 + 13.4e-6)) < 2e-9,
          f"period {slope!r}; schedule the injection on the attacker's clock")


def test_replay():
    from tools.vcan_replay import replay_and_record
    from cids.sim import prototype_scenario, simulate
    rx = simulate(prototype_scenario(duration=4, seed=1))
    rec = replay_and_record(rx, "virtual", "check", seconds=3)
    n_sim = sum(np.sum(ts - ts[0] < 3) for ts in rx.values())
    n_rec = sum(len(v) for v in rec.values())
    check("frames received on the virtual bus", n_rec >= 0.95 * n_sim,
          f"sent ~{n_sim}, received {n_rec}")
    if 0x11 in rec:
        d = np.diff(rec[0x11])
        check("intervals ~50 ms on the bus", abs(np.median(d) - 0.05) < 1e-3)


def test_cloaking():
    from cids.sim import prototype_scenario, simulate
    from cids.attacks import cloaking
    rx = simulate(cloaking(prototype_scenario(duration=120, seed=2), 0x55, "A", 60))
    def period(ts):
        return np.polyfit(np.arange(len(ts)), ts, 1)[0]
    before, after = period(rx[0x55][rx[0x55] < 59]), period(rx[0x55][rx[0x55] > 61])
    check("cloaked period = victim's period (within 1 ppm)", abs(after - before) / before < 1e-9 * 1000,
          f"before {before!r}, after {after!r}")


# ------------------------------------------------------------------ detector
def _ideal(T=0.05, skew=20.0, n=2001, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.arange(n) * T / (1 + skew * 1e-6) + rng.normal(0, noise, n)


def test_offsets():
    from cids.detector import offset_ntp, offset_original
    a = _ideal()[:21]
    o = offset_ntp(a, 0.05)
    check("NTP offset of a 20 ppm clock = 0.05*20e-6 s = 1 us per period",
          abs(o - 0.05 * (1 - 1 / 1.00002) * 1e6) < 1e-6, f"got {o} us")
    o2, mu = offset_original(a, 0.05 / 1.00002)
    check("original offset ~0 when mu_prev is the true interval", abs(o2) < 1e-6,
          f"got {o2} us: this is why Algorithm 1 cannot see the skew")
    check("original returns this batch's mean interval", abs(mu - 0.05 / 1.00002) < 1e-12)


def test_rls():
    from cids.detector import RLS
    rng = np.random.default_rng(0)
    r = RLS(lam=0.9995)
    for k in range(1, 500):
        x = float(k)
        y = 3.7 * x + rng.normal(0, 1)
        r.update(x, y - r.S * x)
    check("RLS finds the slope 3.7", abs(r.S - 3.7) < 0.01, f"got {r.S}")


def test_cusum():
    from cids.detector import CUSUM
    rng = np.random.default_rng(0)
    c = CUSUM(kappa=5, threshold=5)
    e = list(rng.normal(0, 1, 200)) + list(rng.normal(12, 1, 20))
    alarms = [c.update(x) for x in e]
    check("no alarm on normal errors", not any(alarms[:200]))
    check("alarm within 2 steps of a 12-sigma shift", any(alarms[200:202]))


def test_detector():
    from cids.sim import prototype_scenario, simulate
    from cids.detector import CIDS, NTPCIDS, root_cause
    from cids.attacks import masquerade
    rx, busy = simulate(prototype_scenario(duration=300, seed=1), with_busy=True)
    s = {cid: NTPCIDS(0.05).run(rx[cid], busy[cid]).skew[-1] for cid in rx}
    check("NTP-CIDS skews 13.4 / 13.4 / 27.2 ppm",
          abs(s[0x11] - 13.4) < 0.3 and abs(s[0x13] - 13.4) < 0.3 and abs(s[0x55] - 27.2) < 0.3,
          f"got { {hex(k): round(v, 2) for k, v in s.items()} }")
    tr = NTPCIDS(0.05).run(rx[0x55], busy[0x55])
    check("no false alarm in 300 s", not any(tr.alarm))
    sc = masquerade(prototype_scenario(duration=300, seed=4), 0x55, "A", 150)
    rx, busy = simulate(sc, with_busy=True)
    tr = NTPCIDS(0.05).run(rx[0x55], busy[0x55], end_time=300)
    first = next((t for t, a in zip(tr.t, tr.alarm) if a), None)
    check("masquerade at 150 s detected within 10 s", first is not None and 150 <= first < 160,
          f"first alarm at {first}")
    keep = rx[0x55] > 151
    after = NTPCIDS(0.05).run(rx[0x55][keep], busy[0x55][keep]).skew[-1]
    check("root cause = ECU A", root_cause(after, {"A": 13.4, "B": 27.2}) == "A",
          f"skew after attack {after:.2f}")
    tr = CIDS().run(rx[0x11], busy[0x11])
    check("original CIDS runs (any result)", len(tr.skew) > 100)


def test_metrics():
    from cids.detector import Trace
    from cids.metrics import detection_delay, false_positive_rate, has_false_alarm, summarize
    tr = Trace(); tr.t = [1, 2, 3, 4, 5]; tr.alarm = [False, False, False, True, True]
    check("delay = 4 - 2.5 = 1.5 s", detection_delay(tr, 2.5) == 1.5)
    check("no false alarm before t=2.5", not has_false_alarm(tr, 2.5))
    check("FPR = 2/5", abs(false_positive_rate([tr]) - 0.4) < 1e-12)
    check("summarize", summarize([1.0, None, 3.0, 2.0]) == (0.75, 2.0))


def test_road():
    from cids.road import parse_candump, periodic_ids
    lines = [f"({1600000000 + k * 0.01:.6f}) can0 0C9#8C00000000000000" for k in range(300)]
    lines += [f"({1600000000 + k * 0.1:.6f}) can0 3E9#00" for k in range(300)]
    lines += ["garbage line"]
    rx = parse_candump(lines)
    check("parsed IDs 0x0C9 and 0x3E9", set(rx) == {0x0C9, 0x3E9})
    check("300 frames each", all(len(v) == 300 for v in rx.values()))
    p = periodic_ids(rx)
    check("periods 10 ms and 100 ms", p == {0x0C9: 0.01, 0x3E9: 0.1}, f"got {p}")


GROUPS = {"environment": test_environment, "clock": test_clock, "bus": test_bus,
          "attacks": test_attacks, "replay": test_replay, "cloaking": test_cloaking,
          "offsets": test_offsets, "rls": test_rls, "cusum": test_cusum,
          "detector": test_detector, "metrics": test_metrics, "road": test_road}

if __name__ == "__main__":
    names = sys.argv[1:] or list(GROUPS)
    for n in names:
        print(n)
        try:
            GROUPS[n]()
        except Exception as e:  # noqa: BLE001
            results.append(False)
            print(f"  [ERROR] {type(e).__name__}: {e}")
    print(f"\n{sum(results)}/{len(results)} tests passed")
