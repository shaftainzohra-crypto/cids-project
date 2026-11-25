"""
Attacks of the paper (Section 3.2) and the cloaking attack.

fabrication, suspension, masquerade: Cho & Shin, USENIX Security 2016
cloaking: Ying et al., "Shape of the Cloak", IEEE TIFS 2019

Each function takes a Scenario and returns a new, modified Scenario.
Attack messages are always scheduled on the *attacker's* clock.
"""
from __future__ import annotations

import copy

from .sim import Injection, Scenario, next_send_time


def _period_of(sc: Scenario, can_id: int) -> float:
    return next(m.period for m in sc.messages if m.can_id == can_id)


def _sender_of(sc: Scenario, can_id: int) -> str:
    return next(m.sender for m in sc.messages if m.can_id == can_id)


def fabrication(sc: Scenario, target_id: int, attacker: str, t_start: float) -> Scenario:
    """The attacker injects extra frames with target_id, at the same nominal
    period as the legitimate ones (so the frequency doubles)."""
    sc = copy.deepcopy(sc)
    sc.injections.append(Injection(target_id, attacker, _period_of(sc, target_id), t_start))
    return sc


def suspension(sc: Scenario, target_id: int, t_start: float) -> Scenario:
    """The (weakly compromised) sender stops transmitting target_id."""
    sc = copy.deepcopy(sc)
    for m in sc.messages:
        if m.can_id == target_id:
            m.stop = t_start
    return sc


def masquerade(sc: Scenario, target_id: int, attacker: str, t_start: float,
               mistime: float = 0.0) -> Scenario:
    """The victim stops sending target_id and the attacker sends it instead,
    at the same nominal period, starting in the victim's next time slot
    (+ `mistime` seconds, e.g. 1.04e-3 as observed in the paper)."""
    sc = copy.deepcopy(sc)
    t0 = next_send_time(sc, target_id, t_start)
    for m in sc.messages:
        if m.can_id == target_id:
            m.stop = t0 - 1e-4
    sc.injections.append(Injection(target_id, attacker, _period_of(sc, target_id),
                                   t0 + mistime))
    return sc


def cloaking(sc: Scenario, target_id: int, attacker: str, t_start: float,
             error: float = 0.0) -> Scenario:
    """Masquerade in which the attacker stretches its local period so that, on
    the receiver's clock, it equals the victim's period (+ `error` seconds per
    period). The attacker learns the victim's period by listening."""
    sc = copy.deepcopy(sc)
    T = _period_of(sc, target_id)
    s_att = sc.ecus[attacker].skew_ppm * 1e-6
    s_vic = sc.ecus[_sender_of(sc, target_id)].skew_ppm * 1e-6
    ideal = T * (1 + s_att) / (1 + s_vic)
    sc = masquerade(sc, target_id, attacker, t_start)
    sc.injections[-1].local_period = ideal
    sc.injections[-1].extra_delay = error
    return sc
