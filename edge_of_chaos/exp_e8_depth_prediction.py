"""Experiment E8: how badly the one-step multiplier misjudges depth-to-align.

The layers needed to drive two inputs from cosine c0 up to a target c* is a
functional of the WHOLE correlation map,

    T(c0 -> c*) = sum_{t: c_t < c*} 1  ~  int_{c0}^{c*} dc / (C(c) - c),

whereas the one-step surrogate keeps only the multiplier chi1 = C'(1), i.e. it
replaces C(c)-c by its tangent at c=1, (1-chi1)(1-c), and predicts
1 - c_t = (1-c0) chi1^t, so t_pred = log(eps / (1-c0)) / log(chi1) with
eps = 1 - c*. This script tabulates the true depth (by iterating the exact
mean-field cosine map) against t_pred for several activations and regimes.

The map is convex near c=1, so it contracts faster than its tangent in the
transient: the surrogate systematically OVER-estimates the depth, and the error
diverges as the network approaches criticality (chi1 -> 1). At the ReLU angle-
critical point chi1 = 1 exactly, so t_pred = infinity while the true depth is
finite -- the one-number diagnosis is qualitatively wrong.

Fully deterministic (scalar iteration of the cosine map); no finite-N sim.
Writes data/exp_e8_depth_prediction.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import brentq

import eoc_common as ec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def true_depth(step, c0, cstar, tmax=200000):
    """Layers to iterate c_{t+1}=step(c_t) from c0 up to cstar."""
    c = c0
    for t in range(tmax):
        if c >= cstar:
            return t
        cn = step(c)
        if cn <= c:                 # not converging upward (chaotic / stuck)
            return None
        c = cn
    return None


def pred_depth(chi1, c0, cstar):
    """One-step surrogate depth: 1 - c_t = (1-c0) chi1^t."""
    if chi1 >= 1.0:
        return math.inf
    return math.log((1 - cstar) / (1 - c0)) / math.log(chi1)


def ordered_tanh_sw2(target_chi1, sb2=0.05):
    """sw^2 giving a prescribed ordered chi1 for tanh."""
    def f(s):
        qs, _ = ec.fixed_point_of("tanh", s, sb2, q0=1.0, max_iter=200)
        if qs is None:
            qs = 1.0
        return ec.chi1_of("tanh", s, sb2, qs) - target_chi1
    return brentq(f, 0.2, 1.75, xtol=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", type=float, default=0.2)
    args = ap.parse_args()
    c0 = args.c0

    print("# Experiment E8: true depth-to-align vs the one-step prediction "
          f"(c0={c0})")
    print("\nTrue depth iterates the exact cosine map; t_pred uses only "
          "chi1=C'(1).")
    print("\n| activation / regime | chi1 | target c* | T_true | t_pred | "
          "t_pred/T_true |")
    print("|---|---:|---:|---:|---:|---:|")

    configs = []
    # ordered tanh at a range of multipliers (closer to 1 = closer to critical)
    for chi in (0.5, 0.7, 0.85, 0.95):
        sw2 = ordered_tanh_sw2(chi)
        qs, _ = ec.fixed_point_of("tanh", sw2, 0.05, q0=1.0, max_iter=300)
        step = lambda c, s=sw2, q=qs: ec.c_of("tanh", s, 0.05, q, c)
        configs.append((f"tanh (chi1={chi})", chi, step))
    # ReLU: angle-critical, chi1 = 1 exactly (scale-free arccosine map)
    configs.append(("ReLU (angle-critical)", 1.0, lambda c: ec.khat(c)))

    rows = []
    for name, chi, step in configs:
        row = {"name": name, "chi1": chi}
        for cstar in (0.9, 0.99):
            Tt = true_depth(step, c0, cstar)
            tp = pred_depth(chi, c0, cstar)
            ratio = (tp / Tt) if (Tt and math.isfinite(tp)) else (
                math.inf if math.isinf(tp) else float("nan"))
            row[f"T_true_{cstar}"] = Tt
            row[f"t_pred_{cstar}"] = None if math.isinf(tp) else round(tp, 1)
            row[f"ratio_{cstar}"] = (None if math.isinf(ratio)
                                     else round(ratio, 2))
            rr = ("inf" if math.isinf(ratio)
                  else f"{ratio:.2f}" if not math.isnan(ratio) else "--")
            tpn = "inf" if math.isinf(tp) else f"{tp:.1f}"
            print(f"| {name} | {chi:.2f} | {cstar} | {Tt} | {tpn} | {rr} |")
        rows.append(row)

    print("\nThe surrogate over-estimates the depth (the convex map contracts "
          "faster than its tangent), and the error grows as chi1 -> 1; at the "
          "ReLU angle-critical point chi1=1 the surrogate predicts infinite "
          "depth while the true depth is finite. Only iterating the full map "
          "gives the trajectory.")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "exp_e8_depth_prediction.json"),
              "w") as fh:
        json.dump({"c0": c0, "rows": rows}, fh, indent=1)
    print("\nsaved data/exp_e8_depth_prediction.json")


if __name__ == "__main__":
    main()
