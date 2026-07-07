"""Experiment E7: the critical collapse exponent is set by the activation's
smoothness.

At the edge of chaos the correlation approaches its parallel fixed point as a
power law 1 - c_t ~ t^{-1/(p-1)}, where p is the order of the leading correction
of the cosine map at c=1. A kink in the activation (ReLU, leaky ReLU) produces a
non-analytic sqrt(1-c^2) term in the arccosine kernel, giving p=3/2 and
1 - c_t ~ t^{-2} (angle theta_t ~ t^{-1}); a smooth activation has an analytic
cosine map, generic p=2, and the slower 1 - c_t ~ t^{-1} (theta_t ~ t^{-1/2}).

This is a fully deterministic check of the mean-field cosine maps -- no finite-N
simulation -- and it is the sharpest form of the point that the edge-of-chaos
multiplier chi1 (which is 1 for all of these) does not by itself determine the
dynamics; the local geometry of the map does.

Writes data/exp_e7_activation_smoothness.json, prints a markdown table, draws
figures/fig_e7_smoothness.pdf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eoc_common as ec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")


def homogeneous_cos_orbit(name, c0, T):
    """Scale-free cosine orbit for a homogeneous activation (ReLU / leaky).
    Uses the closed-form dispatch c_of (arccosine kernel for ReLU, leaky kernel
    for leaky): Gauss-Hermite on the kink is ~1e-3 inaccurate and destroys the
    delicate t^{-2} tail, so the closed form is required here."""
    c = c0
    cs = [c0]
    for _ in range(T):
        c = ec.c_of(name, 1.0, 0.0, 1.0, c)   # scale-free; qstar arbitrary
        cs.append(min(1.0, c))
    return np.array(cs)


def critical_sw2(name, sb2, lo=1.001, hi=8.0):
    """sw^2 with chi1(sw^2, sb^2)=1 at the length fixed point, or None."""
    def f(s):
        qs, _ = ec.fixed_point_of(name, s, sb2, q0=1.0, max_iter=150)
        if qs is None:
            qs = 1.0
        return ec.chi1_of(name, s, sb2, qs) - 1.0
    try:
        if f(lo) * f(hi) > 0:
            return None
        return brentq(f, lo, hi, xtol=1e-8)
    except Exception:
        return None


def smooth_cos_orbit(name, sw2, sb2, c0, T):
    qstar, _ = ec.fixed_point_of(name, sw2, sb2, q0=1.0, max_iter=150)
    _, cs = ec.theory_orbit_of(name, sw2, sb2, qstar, c0, T)
    return cs


def fit_tail(cs, lo=200):
    e = 1.0 - np.asarray(cs)
    t = np.arange(len(e))
    m = (t >= lo) & (e > 1e-14)
    if m.sum() < 5:
        return float("nan")
    return float(np.polyfit(np.log(t[m]), np.log(e[m]), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--c0", type=float, default=0.2)
    args = ap.parse_args()
    t0 = time.time()

    print("# Experiment E7: activation smoothness sets the critical exponent")
    print(f"(deterministic cosine maps at criticality, c0={args.c0}, T={args.T})")

    kinked = [("relu", None), ("leaky_relu", None)]
    smooth = [("erf", 0.05), ("tanh", 0.05), ("sin", 0.05)]

    rows = []
    orbits = {}

    print("\n## critical collapse exponent  1-c_t ~ t^{p_fit}")
    print("| activation | class | sw^2 (crit) | chi1 | fit exponent | theory |")
    print("|---|---|---:|---:|---:|---:|")
    for name, _ in kinked:
        cs = homogeneous_cos_orbit(name, args.c0, args.T)
        p = fit_tail(cs)
        orbits[name] = cs
        rows.append({"act": name, "class": "kinked", "sw2": None,
                     "chi1": 1.0, "p_fit": p, "theory": -2.0})
        print(f"| {name} | kinked (homog.) | -- | 1.0000 | {p:.3f} | -2 |")
    for name, sb2 in smooth:
        s = critical_sw2(name, sb2)
        if s is None:
            print(f"| {name} | smooth | no critical sw^2 found | | | |")
            continue
        qs, _ = ec.fixed_point_of(name, s, sb2, q0=1.0)
        chi = ec.chi1_of(name, s, sb2, qs)
        cs = smooth_cos_orbit(name, s, sb2, args.c0, args.T)
        p = fit_tail(cs)
        orbits[name] = cs
        rows.append({"act": name, "class": "smooth", "sw2": s, "sb2": sb2,
                     "chi1": chi, "p_fit": p, "theory": -1.0})
        print(f"| {name} | smooth | {s:.4f} | {chi:.4f} | {p:.3f} | -1 |")

    print("\nKinked activations collapse as t^-2 (theta_t ~ t^-1); smooth ones "
          "collapse as t^-1 (theta_t ~ t^-1/2). All have chi1=1 -- the exponent "
          "is set by the activation's smoothness, not by the multiplier.")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(ec.rcparams())
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    t = np.arange(args.T + 1)

    a = ax[0]
    styles = {"relu": ("-", ec.WONG[0]), "leaky_relu": ("-", ec.WONG[4]),
              "erf": ("--", ec.WONG[1]), "tanh": ("--", ec.WONG[2]),
              "gelu": ("--", ec.WONG[3]), "sin": ("--", ec.WONG[5])}
    for name in orbits:
        ls, col = styles.get(name, ("-", "k"))
        e = 1.0 - orbits[name]
        a.loglog(t[1:], e[1:], ls, color=col, lw=1.2,
                 label=name.replace("_relu", "-relu"))
    a.loglog(t[10:], 4.0 * (t[10:] / 10.0) ** -2.0, "k:", lw=0.8)
    a.loglog(t[10:], 0.4 * (t[10:] / 10.0) ** -1.0, "k:", lw=0.8)
    a.text(600, 4e-4, r"$t^{-2}$", fontsize=8)
    a.text(600, 3e-2, r"$t^{-1}$", fontsize=8)
    a.set_xlabel("depth $t$"); a.set_ylabel(r"$1-c_t$")
    a.set_title("(a) kinked $\\sim t^{-2}$, smooth $\\sim t^{-1}$")
    a.legend(fontsize=6.6, ncol=2, loc="lower left")

    a = ax[1]
    for name in orbits:
        ls, col = styles.get(name, ("-", "k"))
        theta = np.arccos(np.clip(orbits[name], -1, 1))
        a.loglog(t[1:], theta[1:], ls, color=col, lw=1.2)
    a.loglog(t[10:], 9.0 * (t[10:] / 10.0) ** -1.0, "k:", lw=0.8)
    a.loglog(t[10:], 2.0 * (t[10:] / 10.0) ** -0.5, "k:", lw=0.8)
    a.text(500, 2e-2, r"$t^{-1}$", fontsize=8)
    a.text(500, 3e-1, r"$t^{-1/2}$", fontsize=8)
    a.set_xlabel("depth $t$"); a.set_ylabel(r"angle $\theta_t$")
    a.set_title("(b) angle to parallel")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "fig_e7_smoothness.pdf")
    fig.savefig(fig_path); plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "exp_e7_activation_smoothness.json"),
              "w") as fh:
        json.dump({"params": vars(args), "rows": rows}, fh, indent=1)
    print(f"\nsaved data/exp_e7_activation_smoothness.json\nwrote {fig_path}\n"
          f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
