"""
Clock-based IDS (CIDS), Cho & Shin, USENIX Security 2016.

`CIDS` follows Algorithm 1 of the paper (per-message detection):
  * batches of N arrival timestamps per CAN ID
  * average clock offset O[k] relative to the extrapolation with the previous
    batch's mean interval, accumulated in absolute value
  * clock skew = slope of the accumulated offset, estimated with RLS
    (forgetting factor lambda = 0.9995)
  * CUSUM on the normalised identification error (kappa = 5, threshold 5)

`NTPCIDS` is the variant proposed by Ying et al. ("Shape of the Cloak",
IEEE TIFS 2019), which computes offsets against the *nominal* period T, keeps
their sign, and therefore estimates the true clock skew as defined by NTP.

Units: time in seconds, offsets in microseconds, so skews are in us/s = ppm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    """Per-batch history, useful for plotting."""
    t: list = field(default_factory=list)       # elapsed time at end of batch (s)
    o_acc: list = field(default_factory=list)   # accumulated offset (us)
    o_avg: list = field(default_factory=list)   # average offset of the batch (us)
    skew: list = field(default_factory=list)    # estimated skew (ppm)
    err: list = field(default_factory=list)     # identification error (us)
    l_pos: list = field(default_factory=list)
    l_neg: list = field(default_factory=list)
    alarm: list = field(default_factory=list)
    skipped: list = field(default_factory=list)  # batch discarded by verification


# ----------------------------------------------------------------- clock offset of a batch
def offset_original(a, mu_prev):
    """Average clock offset of one batch, Algorithm 1 line 23 (in us).

    a: array [a_0, a_1, ..., a_N] (a_0 = last arrival of the previous batch).
    mu_prev: mean interval of the previous batch (s).
    Returns (O[k] in us, mean interval of this batch in s).
    """
    a = np.asarray(a, dtype=float)
    N = len(a) - 1
    i = np.arange(2, N + 1)
    o = np.sum(a[2:] - (a[1] + (i - 1) * mu_prev)) / (N - 1)
    return o * 1e6, float(np.diff(a).mean())


def offset_ntp(a, T):
    """Average clock offset of one batch against the nominal period T (in us),
    Ying et al. Eq. (10): O[k] = T - (a_N - a_0)/N."""
    a = np.asarray(a, dtype=float)
    N = len(a) - 1
    return (T - (a[-1] - a[0]) / N) * 1e6


# ----------------------------------------------------------------- skew estimation
class RLS:
    """Scalar recursive least squares for y = S * x (Algorithm 1, lines 2-6)."""

    def __init__(self, lam=0.9995, delta=1.0):
        self.lam = lam
        self.P = delta
        self.S = 0.0

    def update(self, x, e):
        lam_inv = 1.0 / self.lam
        G = lam_inv * self.P * x / (1.0 + lam_inv * x * x * self.P)
        self.P = lam_inv * (self.P - G * x * self.P)
        self.S = self.S + G * e
        return self.S


# ----------------------------------------------------------------- change detection
class CUSUM:
    """Two-sided CUSUM on the identification error, as in Eq. (4)."""

    def __init__(self, kappa=5.0, threshold=5.0, warmup=20, gate=3.0, alpha=None):
        # alpha=None: cumulative mean/variance; alpha in (0,1): exponentially
        # weighted mean/variance that can follow slow drifts of the error.
        self.alpha = alpha
        self.kappa = kappa
        self.threshold = threshold
        self.warmup = warmup
        self.gate = gate
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.l_pos = 0.0
        self.l_neg = 0.0

    @property
    def std(self):
        if self.alpha is not None and self.n > self.warmup:
            return np.sqrt(self.var)
        return np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0

    def _learn(self, e):
        if self.alpha is not None and self.n >= self.warmup:
            if self.n == self.warmup:
                self.var = self.m2 / (self.n - 1)
            a = self.alpha
            d = e - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
            self.n += 1
            return
        # Welford running mean/variance
        self.n += 1
        d = e - self.mean
        self.mean += d / self.n
        self.m2 += d * (e - self.mean)

    def update(self, e):
        if self.n < self.warmup:
            self._learn(e)
            return False
        sd = max(self.std, 1e-9)
        z = (e - self.mean) / sd
        # update the target values only with "normal" errors (|z| < 3)
        if abs(z) < self.gate:
            self._learn(e)
        self.l_pos = max(0.0, self.l_pos + z - self.kappa)
        self.l_neg = max(0.0, self.l_neg - z - self.kappa)
        return self.l_pos > self.threshold or self.l_neg > self.threshold


# ----------------------------------------------------------------- detector
class _BaseCIDS:
    def __init__(self, N=20, lam=0.9995, kappa=5.0, threshold=5.0,
                 warmup=20, timeout_factor=5.0, delta=1.0, verify=True, alpha=None):
        self.N = N
        self.rls = RLS(lam, delta)
        self.cusum = CUSUM(kappa, threshold, warmup, alpha=alpha)
        self.timeout_factor = timeout_factor
        self.verify = verify
        self.trace = Trace()
        self.t0 = None
        self.o_acc = 0.0
        self.k = 0

    def _step(self, a, busy=False, decision=None):
        """Process one batch: a[0] is the last arrival of the previous batch,
        a[1..N] are the N new arrivals (seconds).

        busy=True means that the bus was busy right before one of these
        messages, so its arrival time contains a queuing delay (E[dd] != 0).
        Verification (Section 4.4): such a batch cannot be trusted, so it is
        not used to update the model nor to raise an alarm.
        """
        o_avg, incr = self._offset(a)
        if not (busy and self.verify and self.skip_accumulate):
            self.o_acc += incr
        t = a[-1] - self.t0
        e = self.o_acc - self.rls.S * t           # line 25
        if busy and self.verify:
            skew, alarm = self.rls.S, False
        else:
            skew = self.rls.update(t, e)          # line 26
            alarm = self.cusum.update(e) if self.k >= 1 else False
        tr = self.trace
        tr.t.append(a[-1] if decision is None else decision); tr.o_acc.append(self.o_acc); tr.o_avg.append(o_avg)
        tr.skew.append(skew); tr.err.append(e)
        tr.l_pos.append(self.cusum.l_pos); tr.l_neg.append(self.cusum.l_neg)
        tr.alarm.append(alarm)
        tr.skipped.append(bool(busy and self.verify))
        self.k += 1
        return alarm

    def run(self, timestamps, busy=None, end_time=None):
        """Run over a full array of arrival times of one CAN ID.

        If the message stops arriving (suspension attack), missing arrivals
        are replaced with significantly high values (Algorithm 1, l. 11-15).
        """
        ts = np.asarray(timestamps, dtype=float)
        busy = np.zeros(len(ts), int) if busy is None else np.asarray(busy, int)
        # Only a *change* in queuing delay biases the offset (E[dd] != 0). A
        # message that always waits behind the same frame (e.g. 0x13 behind
        # 0x11, sent by the same ECU at the same tick) has a constant delay,
        # so we compare each message's number of preceding back-to-back frames
        # with its usual value (the mode over the first 100 messages).
        usual = np.bincount(busy[:100]).argmax()
        busy = busy != usual
        self.t0 = ts[0]
        self._init(ts)
        idx = 1
        prev = ts[0]
        end_time = ts[-1] if end_time is None else end_time
        prev_busy = bool(busy[0])
        while True:
            batch = [prev]
            flag = prev_busy
            mu = self._expected_interval()
            decision = None
            while len(batch) <= self.N:
                nxt = ts[idx] if idx < len(ts) else np.inf
                deadline = batch[-1] + self.timeout_factor * mu
                if nxt > deadline:
                    # The message has not been received for a long time
                    # (suspension?): set the remaining timestamps to
                    # significantly high values (Algorithm 1, lines 11-15).
                    if deadline > end_time:
                        return self.trace
                    high = 10 * self.timeout_factor * mu
                    m = self.N + 1 - len(batch)
                    batch.extend(batch[-1] + high * np.arange(1, m + 1))
                    decision = deadline          # the alarm is raised now
                    break
                batch.append(nxt)
                flag = flag or bool(busy[idx])
                prev_busy = bool(busy[idx])
                idx += 1
            self._step(np.asarray(batch), flag, decision)
            prev = batch[-1] if decision is None else decision
            if decision is not None:
                prev_busy = False   # synthetic timestamps carry no queuing delay


class CIDS(_BaseCIDS):
    """Original CIDS (Algorithm 1 of the paper)."""

    # |O[k]| never cancels out, so a delayed message would leave a permanent
    # step in O_acc: an unverified batch is therefore not accumulated at all.
    skip_accumulate = True

    def _init(self, ts):
        self.mu_prev = float(np.mean(np.diff(ts[: self.N + 1])))

    def _expected_interval(self):
        return self.mu_prev

    def _offset(self, a):
        o_us, self.mu_prev = offset_original(a, self.mu_prev)   # lines 18-23
        return o_us, abs(o_us)                                  # line 24: |O[k]|


class NTPCIDS(_BaseCIDS):
    """NTP-based variant (Ying et al.): offsets against the nominal period."""

    # signed offsets telescope (a delay adds +d then -d), so keep accumulating
    skip_accumulate = False

    def __init__(self, period, **kw):
        super().__init__(**kw)
        self.T = period

    def _init(self, ts):
        pass

    def _expected_interval(self):
        return self.T

    def _offset(self, a):
        o_us = offset_ntp(a, self.T)                # Eq. (10)
        return o_us, self.N * o_us                  # Eq. (11), signed


def first_alarm(trace: Trace, t_from: float = 0.0):
    """Time of the first alarm at or after t_from (None if no alarm)."""
    for t, a in zip(trace.t, trace.alarm):
        if a and t >= t_from:
            return t
    return None


def root_cause(skew_after: float, known: dict[int, float]) -> int:
    """Return the CAN ID whose (known) skew is closest to skew_after."""
    return min(known, key=lambda cid: abs(known[cid] - skew_after))
