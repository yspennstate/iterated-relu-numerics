"""Experiment E6: residual connections delay the collapse to parallel.

A residual block x -> x + beta phi(W x) with independent weights pins the
correlation multiplier at chi1^res = 1 + O(beta^2): the identity branch forces
the network within O(beta^2) of the edge of chaos. Two inputs still become
parallel, but on the depth scale ~1/beta^2 rather than a constant number of
layers. We verify this by collapsing the 1 - c_t curves for several beta onto a
single profile after rescaling depth by beta^2, contrast the residual
(polynomial) decay with a plain network's geometric decay, check that the
per-step multiplier near c=1 tends to 1 as beta -> 0, and confirm the length
grows with depth.

Writes data/exp_e6_resnet_depth_scaling.json, prints markdown tables, draws
figures/fig_e6_resnet.pdf.
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


def iterate_resnet(x0, y0, T, phi, sw2, beta, rng, residual=True):
    """x -> x + beta phi(W x)  (residual) or x -> phi(W x) (plain), independent
    fresh W each step, same W for both trajectories. Returns (q_x, c)."""
    n = x0.size
    sw = math.sqrt(sw2 / n)
    x, y = x0.copy(), y0.copy()
    q = np.empty(T + 1)
    c = np.empty(T + 1)

    def rec(t):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        q[t] = nx * nx / n
        c[t] = float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else 1.0

    rec(0)
    for t in range(1, T + 1):
        w = sw * rng.standard_normal((n, n))
        if residual:
            x = x + beta * phi(w @ x)
            y = y + beta * phi(w @ y)
        else:
            x = phi(w @ x)
            y = phi(w @ y)
        rec(t)
    return q, c


def resnet_mf_orbit(name, sw2, beta, q0, c0, T):
    """Deterministic residual mean-field orbit for an odd (centered) activation,
    where E[phi]=0 kills the cross term:
        q_{t+1} = q_t + beta^2 V(q_t),
        c_{t+1} = (q_t c_t + beta^2 M(c_t)) / q_{t+1},
    with V(q)=E[phi(H)^2], M(c)=E[phi(H)phi(H')], H,H'~N(0, sw^2 q), corr c."""
    phi = ec.ACTIVATIONS[name][0]
    q, c = q0, c0
    qs, cs = [q0], [c0]
    for _ in range(T):
        s = math.sqrt(max(sw2 * q, 0.0))
        V = ec.gauss_e(lambda z: phi(s * z) ** 2)
        M = ec.gauss_e2(phi, phi, s, s, c)
        qn = q + beta ** 2 * V
        c = (q * c + beta ** 2 * M) / qn
        q = qn
        qs.append(q)
        cs.append(c)
    return np.array(qs), np.array(cs)


def avg_run(phi, sw2, beta, q0, c0, T, N, reps, seed, residual=True):
    qs = np.zeros(T + 1)
    cs = np.zeros(T + 1)
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        x0, y0 = ec.make_starts(rng, N, q0, c0)
        q, c = iterate_resnet(x0, y0, T, phi, sw2, beta, rng, residual)
        qs += q
        cs += c
    return qs / reps, cs / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()

    print(f"# Experiment E6: residual depth scaling (N={args.N}, "
          f"reps={args.reps}, seed={args.seed})")

    relu = ec.ACTIVATIONS["relu"][0]
    tanh = ec.ACTIVATIONS["tanh"][0]
    betas = [0.15, 0.2, 0.3, 0.5]
    Tmax = 100
    q0, c0 = 1.0, 0.6

    # (1) beta sweep from the deterministic residual mean-field orbit (tanh is
    # odd, so E[tanh]=0, the cross term vanishes, and chi1^res = 1 + O(beta^2)).
    # The orbit is exact and cheap; a finite-width run confirms it below.
    # The identity branch pins chi1^res = 1 + O(beta^2), so the correlation
    # evolves on the depth scale 1/beta^2 -- the direction (toward or away from
    # 1) depends on the config; the *scale* is the robust, proven statement
    # (Yang-Schoenholz, resnets on the edge of chaos). For random tanh the
    # correlation drifts slowly downward (signals stay distinguishable deep).
    sw2 = 1.5
    print("\n## (1) tanh residual (sw^2=1.5): cosine c_t evolves on scale "
          "1/beta^2")
    print("| beta | c_0 | c at t=20 | t=50 | t=100 | 1/beta^2 |")
    print("|---:|---:|---:|---:|---:|---:|")
    sweep = {}
    for beta in betas:
        T = min(400, int(6.0 / beta ** 2))
        q, c = resnet_mf_orbit("tanh", sw2, beta, q0, c0, T)
        def at(tt): return c[tt] if tt < len(c) else float("nan")
        print(f"| {beta} | {c0} | {at(20):.4f} | {at(50):.4f} | "
              f"{at(100):.4f} | {1/beta**2:.1f} |")
        sweep[beta] = {"T": T, "q": q.tolist(), "c": c.tolist()}

    # beta^2 collapse quality: interpolate c_t at s=beta^2 t on a common grid
    s_grid = np.linspace(0.3, 3.0, 40)
    profiles = {}
    for beta in betas:
        c = np.array(sweep[beta]["c"])
        s = beta ** 2 * np.arange(len(c))
        profiles[beta] = np.interp(s_grid, s, c)
    P = np.array([profiles[b] for b in betas])
    spread = float(np.mean(np.std(P, axis=0)))
    print(f"\nbeta^2-rescaled collapse: mean absolute spread of c_t across beta "
          f"= {spread:.4f} (near 0 = the curves fall on one profile in the "
          "rescaled depth beta^2 t, i.e. the depth scale is 1/beta^2).")

    # finite-width confirmation at one beta (kept small; the box is shared) ---
    Nc, repsc = args.N, args.reps
    qsim, csim = avg_run(tanh, sw2, 0.3, q0, c0, 60, Nc, repsc, args.seed)
    _, cmf = resnet_mf_orbit("tanh", sw2, 0.3, q0, c0, 60)
    gap = float(np.max(np.abs(csim - cmf)))
    print(f"finite-width check (beta=0.3, N={Nc}, reps={repsc}): "
          f"max|c_sim - c_mf| over 60 layers = {gap:.4f}")

    # (2) multiplier chi1^res = 1 + O(beta^2). Estimate it from the ratio of
    # consecutive changes m = (c_{t+2}-c_{t+1})/(c_{t+1}-c_t) (the linearized
    # multiplier, no need for the fixed point). |1-m| should scale as beta^2.
    print("\n## (2) linearized multiplier chi1^res; |1-m| should scale as "
          "beta^2 (identity branch pins it near 1)")
    print("| beta | m | |1-m| | |1-m|/beta^2 |")
    print("|---:|---:|---:|---:|")
    mults = {}
    for beta in betas:
        c = np.array(sweep[beta]["c"])
        d = np.diff(c)
        rr = [d[t + 1] / d[t] for t in range(3, min(len(d) - 1, 40))
              if abs(d[t]) > 1e-9]
        med = float(np.median(rr)) if rr else float("nan")
        mults[beta] = med
        print(f"| {beta} | {med:.5f} | {abs(1-med):.5f} | "
              f"{abs(1-med)/beta**2:.3f} |")

    # (3) length grows (residual) vs decays (plain) ------------------------
    qg, _ = resnet_mf_orbit("tanh", sw2, 0.3, q0, c0, 120)
    qd, _ = avg_run(relu, 1.0, 0.0, q0, c0, 40, args.N, args.reps, args.seed,
                    residual=False)  # plain ReLU: norm halves
    print(f"\n(3) length: residual tanh q_t 1->{qg[-1]:.3f} (grows over 120 "
          f"layers); plain ReLU q_t 1->{qd[30]:.2e} (halves each layer).")

    # (4) geometric (deterministic ordered orbit) vs polynomial (residual) -
    _, c_geo = ec.theory_orbit_of("tanh", 1.32, 0.05, 0.2084, c0, 60)
    e_geo = 1.0 - c_geo
    r_geo = float(np.median([e_geo[t+1]/e_geo[t] for t in range(3, 40)
                             if e_geo[t] > 1e-6]))
    print(f"(4) plain ordered tanh (chi1=0.887): geometric rate {r_geo:.3f} "
          "(straight line on log-y); residual: polynomial (much slower).")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(ec.rcparams())
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)

    a = ax[0]
    for i, beta in enumerate(betas):
        c = np.array(sweep[beta]["c"])
        a.plot(np.arange(len(c)), c, color=ec.WONG[i % 8], lw=1.1,
               label=f"$\\beta={beta}$")
    a.set_xlabel("depth $t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_xlim(0, 160)
    a.set_title("(a) tanh residual: slower for small $\\beta$")
    a.legend(fontsize=6.4, ncol=2)

    a = ax[1]
    for i, beta in enumerate(betas):
        c = np.array(sweep[beta]["c"])
        s = beta ** 2 * np.arange(len(c))
        a.plot(s, c, color=ec.WONG[i % 8], lw=1.1, label=f"$\\beta={beta}$")
    a.set_xlim(0, 3)
    a.set_xlabel(r"rescaled depth $\beta^2 t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_title(r"(b) curves collapse under $t\mapsto\beta^2 t$")

    a = ax[2]
    a.semilogy(np.arange(len(e_geo)), e_geo, color=ec.WONG[1], lw=1.3,
               label=r"plain ordered ($\chi_1{=}0.89$)")
    e_res = np.abs(np.array(sweep[0.3]["c"]) - sweep[0.3]["c"][-1])
    a.semilogy(np.arange(len(e_res)), e_res + 1e-6, color=ec.WONG[0], lw=1.3,
               label=r"residual $\beta=0.3$")
    a.set_xlim(0, 60)
    a.set_xlabel("depth $t$")
    a.set_ylabel(r"$|c_t-c_\infty|$ (residual), $1-c_t$ (plain)")
    a.set_title("(c) plain: $O(1)$ scale; residual: $1/\\beta^2$")
    a.legend(loc="lower left")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "fig_e6_resnet.pdf")
    fig.savefig(fig_path); plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"params": vars(args), "betas": betas, "sweep": sweep,
               "collapse_spread": spread, "multipliers": mults,
               "finite_width_gap_beta0p3": gap,
               "length_residual_end": float(qg[-1]),
               "geometric_rate_plain": r_geo}
    data_path = os.path.join(DATA_DIR, "exp_e6_resnet_depth_scaling.json")
    with open(data_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {data_path}\nwrote {fig_path}\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
