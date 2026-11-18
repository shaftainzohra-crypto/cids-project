# CIDS: clock-skew based intrusion detection for CAN

Implementation and evaluation of **CIDS** from K.-T. Cho and K. G. Shin,
*"Fingerprinting Electronic Control Units for Vehicle Intrusion Detection"*, USENIX Security 2016,
for the course *Cyber-Physical Systems and IoT Security*.

CIDS identifies the ECU that sends a periodic CAN message from the clock skew of its crystal,
estimated only from the arrival times of the message, and raises an alarm when that skew changes.
This project implements it on a simulated CAN bus that replicates the paper's prototype, compares
it with the NTP-based reformulation of Ying et al. (*"Shape of the Cloak"*, IEEE TIFS 2019), and tests
it on real CAN recordings from the ROAD dataset.

## Contents

| Path | What it contains |
|---|---|
| `cids/sim.py` | CAN bus simulator: ECUs with skewed clocks and jitter, 500 kbit/s bus with ID-based arbitration, receiver timestamps with 4 µs resolution |
| `cids/attacks.py` | Fabrication, suspension and masquerade attacks (paper, Section 3.2) and the cloaking attack (Ying et al.) |
| `cids/detector.py` | CIDS as in Algorithm 1 of the paper (`CIDS`) and the NTP-based variant (`NTPCIDS`): batch clock offset, RLS skew estimation, CUSUM detection, verification step, root-cause analysis |
| `cids/metrics.py` | Detection rate, detection delay, false-positive rate |
| `cids/road.py` | Reader for ROAD candump logs and selection of periodic message IDs |
| `tools/vcan_replay.py` | Replay of simulated traffic on a CAN interface with python-can (virtual bus or Linux `vcan0`) |
| `experiments/` | One script per experiment (see below); figures are written to `results/` |
| `tests/run_tests.py` | Unit and integration tests (39 tests) |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python tests/run_tests.py       # 39/39 tests passed
```

The ROAD dataset is not included (it is large). Download it from
<https://0xsam.com/road/> and extract it so that the recordings are in
`data/road/ambient/` and `data/road/attacks/`.

## Experiments

Run every script from the project folder. Each one prints its numbers and saves its figure in
`results/`.

| Script | Experiment | Output |
|---|---|---|
| `python experiments/fingerprint.py` | Clock skew of each message ID on 10 minutes of the simulated prototype (ECU A: 0x11, 0x13; ECU B: 0x55), with the paper's offset and with the NTP offset | `results/fingerprint.png` |
| `python experiments/attack_detection.py` | Fabrication, suspension and masquerade on one run, both detectors, with root-cause analysis | `results/attack_detection.png` |
| `python experiments/fingerprint_checks.py` | Control experiment (attacker and victim with the same skew) and jitter sweep from 2 to 100 µs | `results/fingerprint_checks.png` |
| `python experiments/detection_statistics.py` | Detection rate, median delay and false-positive rate over 20 runs per attack; effect of the batch size N and of the CUSUM parameter κ | `results/detection_statistics.png` |
| `python experiments/cloaking_attack.py` | Masquerade in which the attacker imitates the victim's clock, with an imitation error from −50 to +50 µs per period | `results/cloaking_attack.png` |
| `python tools/vcan_replay.py --interface virtual --channel demo --seconds 60` | Replay of 60 s of simulated traffic through python-can and skew estimated on the received timestamps | printed only |
| `python experiments/road_fingerprint.py` | Skew of every periodic ID in a real recording, and one masquerade built from real frames (`--victim`, `--attacker`, `--file` to choose) | `results/road_fingerprint.png` |
| `python experiments/road_all_pairs.py` | The same masquerade for every pair of clean IDs with the same period, summarised by skew difference | `results/road_all_pairs.png` |

`detection_statistics.py` and `road_all_pairs.py` take a few minutes; the others take seconds.
The replay runs in real time.

## Main results

- **Simulation.** With the NTP-based offset, CIDS recovers the simulated skews (13.40 and
  27.20 ppm), detects fabrication and suspension within the first batch, and detects a timed
  masquerade in every run in about 3 s, naming the attacking ECU. With Algorithm 1 as written in
  the paper, the estimated "skew" reflects timing noise rather than the clock (about 2.5 ppm for
  both ECUs), only 6% of timed masquerades are detected, and false positives grow over time.
- **Cloaking.** An attacker that imitates the victim's period exactly is never detected; an
  imitation error of 0.1 µs per period (2 ppm) is always detected.
- **Replay.** Replaying through python-can on a Windows PC adds host timing noise that destroys
  the fingerprint (ECU B's skew measured between −211.63 and −3.70 ppm instead of +27.24 ppm),
  so detection experiments use simulated timestamps.
- **Real data (ROAD).** The fingerprint is present: 83 clean periodic IDs form 14 groups of
  equal skew, one per ECU. Over 980 spliced masquerades, detection within 300 s is 100% for skew
  differences above 100 ppm, 88% for 30 to 100 ppm and 56% for 10 to 30 ppm, while 18% of
  same-ECU pairs and 22 of the 83 IDs on their own traffic raise false alarms.

## References

- K.-T. Cho and K. G. Shin, "Fingerprinting Electronic Control Units for Vehicle Intrusion
  Detection", USENIX Security Symposium, 2016.
- X. Ying, S. U. Sagong, A. Clark, L. Bushnell and R. Poovendran, "Shape of the Cloak: Formal
  Analysis of Clock Skew-Based Intrusion Detection System in Controller Area Networks",
  IEEE Transactions on Information Forensics and Security, 14(9), 2019.
- M. Verma, R. Bridges, J. Sosnowski, S. Hollifield and M. Iannacone, "A Comprehensive Guide to
  CAN IDS Data & Introduction of the ROAD Dataset", 2020 (ROAD dataset).
