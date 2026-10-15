"""
VAJRA Nowcasting — Verification metrics
=========================================
The standard scores used to evaluate operational nowcasts, computed at a
chosen threshold (e.g. 35 dBZ = moderate-to-heavy rain, or lightning
probability > 0.5). These are what IMD/WMO verification reports use, and
what your hackathon judges will likely ask about:

  POD (Probability of Detection)  = hits / (hits + misses)          -> did we catch real events?
  FAR (False Alarm Ratio)         = false_alarms / (hits + false_alarms) -> how often were we wrong when we alerted?
  CSI (Critical Success Index)    = hits / (hits + misses + false_alarms) -> overall skill, penalizes both errors
  HSS (Heidke Skill Score)        -> skill relative to random chance
"""

import numpy as np


def contingency_table(pred: np.ndarray, truth: np.ndarray, threshold: float):
    pred_event = pred >= threshold
    truth_event = truth >= threshold
    hits = np.sum(pred_event & truth_event)
    misses = np.sum(~pred_event & truth_event)
    false_alarms = np.sum(pred_event & ~truth_event)
    correct_negatives = np.sum(~pred_event & ~truth_event)
    return hits, misses, false_alarms, correct_negatives


def scores(pred: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    h, m, fa, cn = contingency_table(pred, truth, threshold)
    eps = 1e-9
    pod = h / (h + m + eps)
    far = fa / (h + fa + eps)
    csi = h / (h + m + fa + eps)
    expected_hits = ((h + m) * (h + fa)) / (h + m + fa + cn + eps)
    hss = (h - expected_hits) / (h + m + fa - expected_hits + eps)
    return {"POD": pod, "FAR": far, "CSI": csi, "HSS": hss,
            "hits": int(h), "misses": int(m), "false_alarms": int(fa)}


def scores_by_lead_time(pred_seq: np.ndarray, truth_seq: np.ndarray, threshold: float, step_min: int = 10):
    """pred_seq/truth_seq: (T_out, H, W). Returns a list of score dicts, one
    per lead time, so you can plot skill decay vs. forecast horizon —
    a key chart for the hackathon presentation."""
    results = []
    for t in range(pred_seq.shape[0]):
        s = scores(pred_seq[t], truth_seq[t], threshold)
        s["lead_time_min"] = (t + 1) * step_min
        results.append(s)
    return results


if __name__ == "__main__":
    # smoke test with synthetic arrays
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 50, size=(18, 128, 128))
    pred = truth + rng.normal(0, 5, size=truth.shape)
    for row in scores_by_lead_time(pred, truth, threshold=35):
        print(f"T+{row['lead_time_min']:>3} min  CSI={row['CSI']:.2f}  "
              f"POD={row['POD']:.2f}  FAR={row['FAR']:.2f}  HSS={row['HSS']:.2f}")
