"""Experiment E3: biases tune the edge of chaos.

For tanh, a nonzero bias variance sb^2 gives the length map a nonzero fixed
point q* and lets the multiplier chi1(sw^2, sb^2) be tuned to one. We map the
chi1 = 1 curve in the (sw^2, sb^2) plane (ordered chi1<1 below, chaotic chi1>1
above), then simulate finite-width networks at an ordered, a critical, and a
chaotic point to show: ordered -> fast (geometric) collapse to a single
direction; critical -> slow power law (1 - c_t ~ 1/t for a smooth activation,
slower than the ReLU kink's 1/t^2); chaotic -> convergence to c* < 1. We check
that the ordered decay rate matches chi1 and the depth scale xi_c = -1/log chi1.

Writes data/exp_e3_bias_criticality.json, prints markdown tables, draws
figures/fig_e3_eoc_curve.pdf.
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

NAME = "tanh"


def chi1_at(sw2, sb2):
    qstar, _ = ec.fixed_point_of(NAME, sw2, sb2, q0=1.0)
    if qstar is None:
        qstar = 1.0
    return ec.chi1_of(NAME, sw2, sb2, qstar), qstar


def solve_eoc_sw2(sb2, lo=1.01, hi=6.0):
    """sw^2 with chi1(sw^2, sb^2) = 1 (edge of chaos), for tanh."""
    f = lambda s: chi1_at(s, sb2)[0] - 1.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-8)


def sim_cos(sw2, sb2, q0, c0, T, N, reps, seed):
    phi = ec.ACTIVATIONS[NAME][0]
    cc = np.zeros(T + 1)
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        x0, y0 = ec.make_starts(rng, N, q0, c0)
        _, _, c = ec.iterate_pair(x0, y0, T, phi, sw2, sb2, rng, tied=False)
        cc += c
    return cc / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    print(f"# Experiment E3: bias-tuned edge of chaos (tanh, N={args.N}, "
          f"reps={args.reps}, seed={args.seed})")

    # (1) the chi1 = 1 curve ------------------------------------------------
    sb2_grid = np.linspace(0.0, 0.8, 11)
    print("\n## (1) edge-of-chaos curve chi1(sw^2, sb^2)=1")
    print("| sb^2 | sw^2 (EOC) | q* | chi1 check |")
    print("|---:|---:|---:|---:|")
    curve = []
    for sb2 in sb2_grid:
        s = solve_eoc_sw2(float(sb2))
        if s is None:
            continue
        chi, qs = chi1_at(s, float(sb2))
        curve.append({"sb2": float(sb2), "sw2": float(s), "qstar": qs,
                      "chi1": chi})
        if abs(sb2 - round(sb2 * 4) / 4) < 1e-9:      # print a few rows
            print(f"| {sb2:.3f} | {s:.4f} | {qs:.4f} | {chi:.6f} |")

    # (2) three regimes at sb^2 = 0.05 -------------------------------------
    sb2 = 0.05
    sw2_c = solve_eoc_sw2(sb2)
    pts = [("ordered", 0.75 * sw2_c), ("critical", sw2_c),
           ("chaotic", 1.6 * sw2_c)]
    print(f"\n## (2) three regimes at sb^2={sb2} (critical sw^2={sw2_c:.4f})")
    print("| regime | sw^2 | chi1 | q* | xi_c=-1/log chi1 |")
    print("|---|---:|---:|---:|---|")
    T = 220
    regimes = {}
    for label, sw2 in pts:
        chi, qs = chi1_at(sw2, sb2)
        xic = ec.xi_c(chi) if chi < 1 else float("inf")
        print(f"| {label} | {sw2:.4f} | {chi:.4f} | {qs:.4f} | "
              f"{xic if math.isinf(xic) else round(xic,3)} |")
        # theory orbit and finite-N sim
        _, c_th = ec.theory_orbit_of(NAME, sw2, sb2, qs, 0.2, T)
        c_sim = sim_cos(sw2, sb2, qs, 0.2, min(T, 40), args.N, args.reps,
                        args.seed)
        regimes[label] = {"sw2": sw2, "chi1": chi, "qstar": qs, "xi_c": xic,
                          "c_th": c_th.tolist(), "c_sim": c_sim.tolist()}

    # rate checks
    e_ord = 1.0 - np.array(regimes["ordered"]["c_th"])
    r_ord = float(np.median([e_ord[t+1]/e_ord[t] for t in range(5, 40)
                             if e_ord[t] > 1e-9]))
    e_crit = 1.0 - np.array(regimes["critical"]["c_th"])
    ts = np.arange(len(e_crit))
    p_crit = float(np.polyfit(np.log(ts[20:]), np.log(e_crit[20:]), 1)[0])
    c_star = float(regimes["chaotic"]["c_th"][-1])
    print(f"\nordered: geometric ratio {r_ord:.4f} vs chi1="
          f"{regimes['ordered']['chi1']:.4f}")
    print(f"critical (smooth tanh): 1-c_t ~ t^{{{p_crit:.3f}}} "
          "(theory -1: smooth activation, one power slower than ReLU's -2)")
    print(f"chaotic: c_t -> c* = {c_star:.4f} < 1 (inputs decorrelate)")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(ec.rcparams())
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)

    a = ax[0]
    cs = np.array([(c["sw2"], c["sb2"]) for c in curve])
    a.plot(cs[:, 0], cs[:, 1], "-", color=ec.WONG[6], lw=1.6)
    a.fill_betweenx(cs[:, 1], cs[:, 0], cs[:, 0].max() + 1, color=ec.WONG[0],
                    alpha=0.10)
    a.text(cs[:, 0].min() + 0.05, 0.6, "ordered\n$\\chi_1<1$", fontsize=8,
           color=ec.WONG[0])
    a.text(cs[:, 0].max() - 1.2, 0.15, "chaotic\n$\\chi_1>1$", fontsize=8,
           color=ec.WONG[1])
    for label, sw2 in pts:
        a.plot(sw2, sb2, "o", color=ec.WONG[2], ms=5, mfc="none")
    a.set_xlabel(r"$\sigma_w^2$"); a.set_ylabel(r"$\sigma_b^2$")
    a.set_title("(a) edge of chaos: $\\chi_1=1$")

    a = ax[1]
    col = {"ordered": ec.WONG[0], "critical": ec.WONG[2], "chaotic": ec.WONG[1]}
    for label in ("ordered", "critical", "chaotic"):
        e = np.abs(1.0 - np.array(regimes[label]["c_th"]))
        a.loglog(np.arange(1, T + 1), e[1:], color=col[label], lw=1.3,
                 label=label)
        cs2 = regimes[label]["c_sim"]
        a.loglog(np.arange(1, len(cs2)), np.abs(1.0 - np.array(cs2))[1:],
                 "o", color=col[label], ms=2.6, mfc="none")
    a.loglog(ts[20:], e_crit[20] * (ts[20:] / 20.0) ** -1.0, "k:", lw=0.7)
    a.set_xlabel("depth $t$"); a.set_ylabel(r"$|1-c_t|$")
    a.set_title("(b) three regimes (lines: theory, dots: $N{=}1500$)")
    a.legend(loc="lower left")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "fig_e3_eoc_curve.pdf")
    fig.savefig(fig_path); plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"params": vars(args), "eoc_curve": curve, "regimes": regimes,
               "rates": {"ordered_ratio": r_ord,
                         "ordered_chi1": regimes["ordered"]["chi1"],
                         "critical_power": p_crit, "chaotic_cstar": c_star}}
    data_path = os.path.join(DATA_DIR, "exp_e3_bias_criticality.json")
    with open(data_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {data_path}\nwrote {fig_path}\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
