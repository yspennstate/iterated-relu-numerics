"""Experiment E1: finite-width state evolution and the edge-of-chaos rates.

For the iterated map x -> phi(W x + b) with a single reused Gaussian matrix
(tied weights), we check, at finite width N:

 (a) the fixed point q* and multiplier chi1 for each activation;
 (b) the length q_t and two-input cosine c_t track the deterministic maps V, C
     of eoc_common, with the gap shrinking in N (exponential concentration);
 (c) the collapse-to-parallel rate: ordered (chi1<1) is geometric,
     1 - c_t ~ chi1^t; critical (ReLU, angle-critical) is a power law,
     1 - c_t ~ t^{-2} and theta_t = arccos(c_t) ~ 3 pi / t;
 (d) one iteration vs many: the single-multiplier surrogate
     1 - c_t ~ (1 - c_0) chi1^t is accurate only near c=1 and misses the
     transient started from a large initial angle.

Writes data/exp_e1_state_evolution.json, prints markdown tables, draws
figures/fig_e1_state_evolution.pdf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eoc_common as ec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")


def sim_pair(name, sw2, sb2, q0, c0, T, N, reps, seed):
    """Average tied-weight (q_x, c) over reps at width N."""
    phi = ec.ACTIVATIONS[name][0]
    qx = np.zeros(T + 1)
    cc = np.zeros(T + 1)
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        x0, y0 = ec.make_starts(rng, N, q0, c0)
        a, _, c = ec.iterate_pair(x0, y0, T, phi, sw2, sb2, rng, tied=True)
        qx += a
        cc += c
    return qx / reps, cc / reps


def khat_orbit(c0, T):
    """Scale-free ReLU cosine orbit c_{t+1} = Khat(c_t) (length-independent)."""
    cs = [c0]
    c = c0
    for _ in range(T):
        c = ec.khat(c)
        cs.append(c)
    return np.array(cs)


def fit_power(ts, ys):
    """OLS slope of log y vs log t over the given window."""
    t = np.asarray(ts, float)
    y = np.asarray(ys, float)
    m = (t > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan")
    lt, ly = np.log(t[m]), np.log(y[m])
    return float(np.polyfit(lt, ly, 1)[0])


def fit_geom(ys):
    """Geometric rate estimate: median of y_{t+1}/y_t over a clean window."""
    y = np.asarray(ys, float)
    r = [y[t + 1] / y[t] for t in range(len(y) - 1)
         if 1e-8 < y[t] < 0.5 and y[t + 1] > 0]
    return float(np.median(r)) if r else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=14)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    t0 = time.time()

    print(f"# Experiment E1: state evolution and edge-of-chaos rates "
          f"(T={args.T}, reps={args.reps}, seed={args.seed})")

    # (a) fixed points and multipliers -------------------------------------
    configs = [
        ("relu", 2.0, 0.0),        # angle-critical, length-critical (He)
        ("relu", 1.0, 0.0),        # ordered length (norm halves), angle-critical
        ("tanh", 1.5, 0.05),       # smooth, ordered
        ("tanh", 3.0, 0.05),       # smooth, chaotic
        ("erf", 1.0, 0.05),        # smooth sigmoidal
        ("gelu", 1.5, 0.05),
        ("swish", 1.6, 0.05),
        ("sin", 1.0, 0.10),
    ]
    print("\n## (a) fixed point q*, pre-activation qhat*, chi1, phase")
    print("| activation | sw2 | sb2 | q* | qhat* | chi1 | phase |")
    print("|---|---:|---:|---:|---:|---:|---|")
    fp = {}
    for name, sw2, sb2 in configs:
        qstar, marg = ec.fixed_point_of(name, sw2, sb2, q0=1.0)
        if qstar is None or (isinstance(qstar, float) and math.isnan(qstar)):
            qstar = 1.0
        qhat = sw2 * qstar + sb2
        chi = ec.chi1_of(name, sw2, sb2, qstar)
        phase = ("critical" if abs(chi - 1) < 0.02
                 else "ordered" if chi < 1 else "chaotic")
        fp[(name, sw2, sb2)] = dict(qstar=qstar, qhat=qhat, chi1=chi,
                                    phase=phase, marginal=bool(marg))
        note = " (angle-critical, homog.)" if name in ec.HOMOGENEOUS else ""
        print(f"| {name}{note} | {sw2} | {sb2} | {qstar:.4f} | {qhat:.4f} | "
              f"{chi:.4f} | {phase} |")

    # (b) finite-N tracking of the maps ------------------------------------
    print("\n## (b) tied-weight simulation vs theory (tanh, sw2=1.5, sb2=0.05, "
          "q0=q*, c0=0.2)")
    name, sw2, sb2 = "tanh", 1.5, 0.05
    qstar = fp[(name, sw2, sb2)]["qstar"]
    q_th, c_th = ec.theory_orbit_of(name, sw2, sb2, qstar, 0.2, args.T)
    track = {}
    print("| N | max|q_sim-q_th| | max|c_sim-c_th| |")
    print("|---:|---:|---:|")
    for N in (250, 500, 1000, 2000):
        qs, cs = sim_pair(name, sw2, sb2, qstar, 0.2, args.T, N,
                          args.reps, args.seed)
        dq = float(np.max(np.abs(qs - q_th)))
        dc = float(np.max(np.abs(cs - c_th)))
        track[N] = dict(q=qs.tolist(), c=cs.tolist(), dq=dq, dc=dc)
        print(f"| {N} | {dq:.4f} | {dc:.4f} |")

    # (c) ordered vs critical collapse rate --------------------------------
    print("\n## (c) collapse-to-parallel rate")
    # critical: ReLU angle map (homogeneous, chi1_angle = 1); long orbit
    c_relu = khat_orbit(0.2, 3000)
    e_relu = 1.0 - c_relu
    theta = np.arccos(np.clip(c_relu, -1, 1))
    t_arr = np.arange(len(c_relu))
    p_relu = fit_power(t_arr[200:], e_relu[200:])       # asymptotic window
    ratio = t_arr[1:] * theta[1:] / (3 * math.pi)
    print(f"ReLU (critical): fit 1-c_t ~ t^{{{p_relu:.3f}}} (theory -2, "
          f"asymptotic); t*theta_t/(3pi) at t=50,500,3000 = "
          f"{ratio[49]:.3f}, {ratio[499]:.3f}, {ratio[2999]:.3f} (-> 1)")
    # ordered: erf sw2=1.0 (chi1<1) geometric
    name_o, sw2_o, sb2_o = "erf", 1.0, 0.05
    qstar_o = fp[(name_o, sw2_o, sb2_o)]["qstar"]
    chi_o = fp[(name_o, sw2_o, sb2_o)]["chi1"]
    _, c_ord = ec.theory_orbit_of(name_o, sw2_o, sb2_o, qstar_o, 0.2, 60)
    e_ord = 1.0 - c_ord
    g_ord = fit_geom(e_ord)
    print(f"erf sw2=1.0 (ordered): geometric rate est {g_ord:.4f} vs "
          f"chi1={chi_o:.4f}; xi_c = {ec.xi_c(chi_o):.3f}")

    # (d) one iteration vs many --------------------------------------------
    print("\n## (d) one-iteration surrogate vs true multi-step orbit")
    # ReLU: chi1_angle = 1, so the surrogate is FLAT (predicts no alignment
    # ever) while the true orbit converges to parallel polynomially.
    c0 = 0.1
    c_relu_d = khat_orbit(c0, 200)
    t_d = np.arange(len(c_relu_d))
    c_surr_relu = 1.0 - (1.0 - c0) * (1.0 ** t_d)        # = c0, flat
    # tanh near-critical ordered (chi1=0.9386): map is strongly nonlinear
    name_d, sw2_d, sb2_d = "tanh", 1.5, 0.05
    qstar_d = fp[(name_d, sw2_d, sb2_d)]["qstar"]
    chi_d = fp[(name_d, sw2_d, sb2_d)]["chi1"]
    _, c_true_d = ec.theory_orbit_of(name_d, sw2_d, sb2_d, qstar_d, c0, 200)
    c_surr_d = 1.0 - (1.0 - c0) * chi_d ** t_d
    gap_d = np.abs(c_true_d - c_surr_d)
    dt_true = int(np.argmax(c_true_d > 0.9)) or None
    dt_surr = int(np.argmax(c_surr_d > 0.9)) or None
    print(f"ReLU (chi1_angle=1): surrogate stays flat at c0={c0}, true orbit "
          f"reaches c=0.9 at depth {int(np.argmax(c_relu_d>0.9))} "
          "(one multiplier predicts inputs NEVER align).")
    print(f"tanh sw2=1.5 (ordered, chi1={chi_d:.3f}): max|true-surrogate| "
          f"= {float(gap_d.max()):.4f} at t={int(gap_d.argmax())}; "
          f"depth to c=0.9: true={dt_true}, surrogate={dt_surr}.")
    gap = gap_d

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(ec.rcparams())
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 5.0), constrained_layout=True)

    # panel 1: finite-N tracking (cosine)
    a = ax[0, 0]
    a.plot(np.arange(args.T + 1), c_th, "k-", lw=1.4, label="theory $C$")
    for i, N in enumerate((250, 1000, 2000)):
        a.plot(np.arange(args.T + 1), track[N]["c"], marker="o",
               ms=3, mfc="none", color=ec.WONG[i], lw=0.8, label=f"$N={N}$")
    a.set_xlabel("depth $t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_title("(a) finite-width tracks the map"); a.legend(loc="lower right")

    # panel 2: tracking gap vs N
    a = ax[0, 1]
    Ns = np.array([250, 500, 1000, 2000])
    dcs = np.array([track[N]["dc"] for N in Ns])
    a.loglog(Ns, dcs, "o-", color=ec.WONG[1], mfc="none")
    a.loglog(Ns, dcs[0] * (Ns / Ns[0]) ** -0.5, "k--", lw=0.8,
             label=r"$N^{-1/2}$")
    a.set_xlabel("width $N$"); a.set_ylabel(r"$\max_t|c_{\rm sim}-c_{\rm th}|$")
    a.set_title("(b) concentration"); a.legend()

    # panel 3: ordered vs critical decay
    a = ax[1, 0]
    a.loglog(t_arr[1:], e_relu[1:], color=ec.WONG[0], lw=1.2,
             label=r"ReLU critical $\sim t^{-2}$")
    a.loglog(np.arange(1, 60), e_ord[1:60], color=ec.WONG[1], lw=1.2,
             label=r"erf ordered $\sim\chi_1^{t}$")
    a.loglog(t_arr[2:], e_relu[2] * (t_arr[2:] / 2.0) ** -2.0, "k:", lw=0.8)
    a.set_xlabel("depth $t$"); a.set_ylabel(r"$1-c_t$")
    a.set_title("(c) collapse rate"); a.legend(loc="lower left")

    # panel 4: one vs many
    a = ax[1, 1]
    a.plot(t_d, c_true_d, color=ec.WONG[2], lw=1.4,
           label=r"tanh: true orbit")
    a.plot(t_d, c_surr_d, color=ec.WONG[2], lw=1.0, ls="--",
           label=r"tanh: $1-(1-c_0)\chi_1^{t}$")
    a.plot(t_d, c_relu_d, color=ec.WONG[0], lw=1.4,
           label=r"ReLU: true orbit")
    a.plot(t_d, c_surr_relu, color=ec.WONG[0], lw=1.0, ls=":",
           label=r"ReLU: surrogate ($\chi_1{=}1$, flat)")
    a.set_xlabel("depth $t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_title("(d) one iteration misses the transient")
    a.legend(loc="lower right", fontsize=6.3)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "fig_e1_state_evolution.pdf")
    fig.savefig(fig_path)
    plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = dict(
        params=vars(args),
        fixed_points={f"{k[0]}_sw{k[1]}_sb{k[2]}": v for k, v in fp.items()},
        tracking=track,
        critical=dict(fit_power=p_relu, ratio_end=float(ratio[-1]),
                      angle_ratio_t50=float(ratio[49]),
                      angle_ratio_t500=float(ratio[499])),
        ordered=dict(geom_rate=g_ord, chi1=chi_o, xi_c=ec.xi_c(chi_o)),
        one_vs_many=dict(relu_c0=c0,
                         relu_depth_to_0p9=int(np.argmax(c_relu_d > 0.9)),
                         tanh_chi1=chi_d, tanh_max_gap=float(gap_d.max()),
                         tanh_depth_true=dt_true,
                         tanh_depth_surrogate=dt_surr),
    )
    data_path = os.path.join(DATA_DIR, "exp_e1_state_evolution.json")
    with open(data_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {data_path}\nwrote {fig_path}\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
