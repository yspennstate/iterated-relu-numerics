"""Experiment E4: signal propagation in a random 1D circular CNN.

Model (THEORY_SPEC.md, Sec. 4).  State x in R^{C x L} (C channels, L spatial
positions).  One layer with odd kernel width k, offsets Delta in
{-(k-1)/2, ..., (k-1)/2}:

    h[c',p] = sum_{c,Delta} W^Delta[c',c] x[c, (p+Delta) mod L] + b[c'],
    x_new   = phi(h),

with W^Delta_{c'c} ~ N(0, sw^2/(k C)) i.i.d., fresh at every layer
(feedforward), and per-output-channel bias b_{c'} ~ N(0, sb^2) shared across
positions.  For channel-i.i.d. translation-invariant inputs the mean-field
covariance recursion decouples per spatial lag (the receptive-field average
acts as the identity on translation-invariant kernels): with q_t the
per-position length and m_t the cross moment of any pair of channel vectors
(two images at the same position, or one image at two positions),

    qhat = sw^2 q_t + sb^2,   chat = sw^2 m_t + sb^2,
    q_{t+1} = E[phi(u)^2],    m_{t+1} = E[phi(u) phi(v)],
    (u, v) ~ N(0, [[qhat, chat], [chat, qhat]]),

exactly the fully connected maps of THEORY_SPEC Sec. 1.  The shared bias
enters diagonal and off-diagonal alike, so sb^2 > 0 is an ordering field.

Experiments (independent weights per layer throughout):

  1. Depth propagation at C = 128: per-position length q_t(p) (mean and
     spatial spread), spatially averaged two-input cosine cbar_t for two
     random images pushed through the same network, and the within-image
     patch cosine (pairs p != p'), against the quadrature mean-field orbit.
  2. Ordered/critical contrast.  For sb = 0 the ReLU cosine dynamics are the
     same for every sw (positive homogeneity gives x_t(sw) = sw^t x_t(1), so
     every cosine coincides realization by realization); the sw^2 = 1 and
     sw^2 = 2 runs share weight seeds to exhibit this identity, and both
     follow the scale-free arccosine-kernel orbit (slope 1 at c = 1).  The
     ordered/critical contrast in the angle dynamics requires a length fixed
     point: ReLU with sb^2 = 1/2 at sw^2 = 1 has q* = 1/2 and chi1 = 1/2
     (geometric collapse of 1 - c), while sw^2 = 2 has q_t growing linearly
     and cosine slope -> 1 (collapse stalls); tanh ordered vs tanh at the
     edge of chaos (chi1 = 1, solved numerically) shows the same contrast.
  3. Finite-C concentration for sw^2 = 1 ReLU: std across seed replicates of
     cbar_t, the patch cosine, and 2^t qbar_t / qbar_0 at fixed depth, for
     C in {32, 64, 128, 256}.

Writes data/exp_e4_cnn.json, prints markdown tables, and renders
figures/fig_e4_cnn.pdf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")

# Wong colorblind-safe palette (repo convention).
BLUE, VERM, GREEN, ORANGE, PURPLE, SKY, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9",
    "#666666")

PHI = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float)),
    "tanh": (np.tanh, lambda z: 1.0 / np.cosh(z) ** 2),
}

_GH = hermegauss(201)
GH_Z, GH_W = _GH[0], _GH[1] / _GH[1].sum()      # E over N(0,1), 201 nodes


# ----------------------------------------------------------------- mean field

def v_map(q, phi_name, sw2, sb2):
    """Length map q -> E[phi(sqrt(sw^2 q + sb^2) Z)^2].

    ReLU is done in closed form: Gauss-Hermite converges slowly on the kink
    (~1e-3 error at 201 nodes) and is wrong on the step derivative.  Smooth
    activations (tanh) use quadrature, where GH is spectrally accurate."""
    qhat = max(sw2 * q + sb2, 0.0)
    if phi_name == "relu":
        return qhat / 2.0
    phi = PHI[phi_name][0]
    return float(GH_W @ phi(math.sqrt(qhat) * GH_Z) ** 2)


def m_map(q, m, phi_name, sw2, sb2):
    """Cross-moment map m -> E[phi(u) phi(v)] at common length q."""
    qhat = sw2 * q + sb2
    chat = sw2 * m + sb2
    if qhat <= 0.0:
        return 0.0
    rho = min(1.0, max(-1.0, chat / qhat))
    if phi_name == "relu":                      # arccosine kernel (exact)
        return qhat * (math.sqrt(1.0 - rho * rho)
                       + rho * (math.pi - math.acos(rho))) / (2.0 * math.pi)
    phi = PHI[phi_name][0]
    s = math.sqrt(qhat)
    u = s * GH_Z[:, None]
    v = s * (rho * GH_Z[:, None] + math.sqrt(1.0 - rho * rho) * GH_Z[None, :])
    return float(GH_W @ (phi(u) * phi(v)) @ GH_W)


def mf_orbit(phi_name, sw2, sb2, depth, q0=1.0, m0=0.0):
    """Mean-field (q_t, c_t) for t = 0..depth from (q0, m0)."""
    q, m = [q0], [m0]
    for _ in range(depth):
        q.append(v_map(q[-1], phi_name, sw2, sb2))
        m.append(m_map(q[-2], m[-1], phi_name, sw2, sb2))
    c = [mm / qq if qq > 0 else float("nan") for mm, qq in zip(m, q)]
    return q, c


def q_fixed_point(phi_name, sw2, sb2, iters=4000):
    q = 1.0
    for _ in range(iters):
        qn = v_map(q, phi_name, sw2, sb2)
        if abs(qn - q) < 1e-15:
            return qn
        q = qn
    return q


def chi1(phi_name, sw2, sb2, qstar):
    """chi1 = sw^2 E[phi'(sqrt(sw^2 q* + sb^2) Z)^2] at a length fixed point.

    For homogeneous ReLU the integrand is scale free, so a degenerate
    q* = 0 is evaluated at unit scale (E[1_{Z>0}] = 1/2 for every scale).
    """
    if phi_name == "relu":
        return sw2 / 2.0                         # exact: E[1_{Z>0}] = 1/2
    dphi = PHI[phi_name][1]
    s = math.sqrt(max(sw2 * qstar + sb2, 0.0))
    return sw2 * float(GH_W @ dphi(s * GH_Z) ** 2)


def solve_tanh_eoc(sb2, lo=1.02, hi=4.0):
    """sw^2 on the tanh edge-of-chaos curve chi1(sw^2; sb^2) = 1."""

    def f(sw2):
        return chi1("tanh", sw2, sb2, q_fixed_point("tanh", sw2, sb2)) - 1.0

    sw2 = brentq(f, lo, hi, xtol=1e-12)
    qs = q_fixed_point("tanh", sw2, sb2)
    return sw2, qs, chi1("tanh", sw2, sb2, qs)


# ----------------------------------------------------------------- simulation

def conv_layer(x, w, b):
    """h[c',p] = sum_{j,c} w[j,c',c] x[c, (p + j - k//2) mod L] + b[c']."""
    k = w.shape[0]
    r = k // 2
    h = None
    for j in range(k):
        term = w[j] @ np.roll(x, r - j, axis=1)   # column p holds x[:, p+d]
        h = term if h is None else h + term
    return h + b[:, None]


def conv_layer_naive(x, w, b):
    c_dim, l_dim = x.shape
    k = w.shape[0]
    r = k // 2
    h = np.zeros((c_dim, l_dim))
    for cp in range(c_dim):
        for p in range(l_dim):
            acc = 0.0
            for j in range(k):
                for c in range(c_dim):
                    acc += w[j, cp, c] * x[c, (p + j - r) % l_dim]
            h[cp, p] = acc + b[cp]
    return h


def pair_stats(x, xp):
    """(qbar, q spatial rel. spread, cbar, patch cosine) for image pair."""
    l_dim = x.shape[1]
    q = (x * x).mean(axis=0)
    qbar = float(q.mean())
    qrel = float(q.std() / qbar) if qbar > 0 else float("nan")
    nx = np.sqrt((x * x).sum(axis=0))
    nxp = np.sqrt((xp * xp).sum(axis=0))
    cbar = float((((x * xp).sum(axis=0)) / np.maximum(nx * nxp, 1e-300)).mean())
    pat = 0.0
    for z, nz in ((x, nx), (xp, nxp)):
        zn = z / np.maximum(nz, 1e-300)
        g = zn.T @ zn
        pat += (g.sum() - np.trace(g)) / (l_dim * (l_dim - 1))
    return qbar, qrel, cbar, float(pat / 2.0)


def run_depth_all(configs, c_dim, l_dim, k, depth, reps, seed):
    """Propagate two images through the same network for every config.

    All configs share the underlying standard-Gaussian draws per rep, so
    configs differing only by (sw, sb) scaling are seed-matched; in
    particular the two zero-bias ReLU runs test the homogeneity identity.
    Returns per-rep arrays of shape (reps, depth+1) per observable.
    """
    fields = ("qbar", "qrel", "cbar", "patch")
    out = {cfg["name"]: {f: np.zeros((reps, depth + 1)) for f in fields}
           for cfg in configs}
    for rep in range(reps):
        rng = np.random.default_rng([seed, rep])
        x0 = rng.standard_normal((c_dim, l_dim))
        x0p = rng.standard_normal((c_dim, l_dim))
        gs = [rng.standard_normal((k, c_dim, c_dim)) for _ in range(depth)]
        betas = [rng.standard_normal(c_dim) for _ in range(depth)]
        for cfg in configs:
            phi = PHI[cfg["phi"]][0]
            wscale = math.sqrt(cfg["sw2"] / (k * c_dim))
            bscale = math.sqrt(cfg["sb2"])
            rec = out[cfg["name"]]
            x, xp = x0, x0p
            for f, val in zip(fields, pair_stats(x, xp)):
                rec[f][rep, 0] = val
            for t in range(depth):
                w = wscale * gs[t]
                b = bscale * betas[t]
                x = phi(conv_layer(x, w, b))
                xp = phi(conv_layer(xp, w, b))
                for f, val in zip(fields, pair_stats(x, xp)):
                    rec[f][rep, t + 1] = val
    return out


def run_concentration(c_list, l_dim, k, depth, reps, seed):
    """sw^2 = 1, sb = 0 ReLU: seed-to-seed spread of the order parameters."""
    phi = PHI["relu"][0]
    res = {}
    for c_dim in c_list:
        cbar = np.zeros((reps, depth + 1))
        patch = np.zeros((reps, depth + 1))
        lenr = np.zeros((reps, depth + 1))
        for rep in range(reps):
            rng = np.random.default_rng([seed, 91, c_dim, rep])
            x = rng.standard_normal((c_dim, l_dim))
            xp = rng.standard_normal((c_dim, l_dim))
            b = np.zeros(c_dim)
            q0, _, cbar[rep, 0], patch[rep, 0] = pair_stats(x, xp)
            lenr[rep, 0] = 1.0
            wscale = math.sqrt(1.0 / (k * c_dim))
            for t in range(depth):
                w = wscale * rng.standard_normal((k, c_dim, c_dim))
                x = phi(conv_layer(x, w, b))
                xp = phi(conv_layer(xp, w, b))
                qb, _, cb, pt = pair_stats(x, xp)
                cbar[rep, t + 1] = cb
                patch[rep, t + 1] = pt
                lenr[rep, t + 1] = 2.0 ** (t + 1) * qb / q0
        res[c_dim] = {"cbar": cbar, "patch": patch, "lenratio": lenr}
    return res


# ------------------------------------------------------------------ selftests

def selftest():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 7))
    w = rng.standard_normal((3, 3, 3))
    b = rng.standard_normal(3)
    err = np.max(np.abs(conv_layer(x, w, b) - conv_layer_naive(x, w, b)))
    assert err < 1e-12, f"conv mismatch {err:.2e}"

    # quadrature engine vs ReLU closed forms
    q, m, sw2, sb2 = 0.7, 0.3, 1.3, 0.2
    qhat, chat = sw2 * q + sb2, sw2 * m + sb2
    rho = chat / qhat
    v_ex = qhat / 2.0
    j_ex = qhat * (math.sqrt(1 - rho * rho)
                   + rho * (math.pi - math.acos(rho))) / (2 * math.pi)
    ev = abs(v_map(q, "relu", sw2, sb2) - v_ex)
    em = abs(m_map(q, m, "relu", sw2, sb2) - j_ex)
    assert ev < 1e-10 and em < 1e-8, f"GH vs closed form: {ev:.2e} {em:.2e}"

    # ReLU positive homogeneity: sb = 0 cosines identical for sw^2 = 1 vs 2
    cfgs = [{"name": "a", "phi": "relu", "sw2": 1.0, "sb2": 0.0},
            {"name": "b", "phi": "relu", "sw2": 2.0, "sb2": 0.0}]
    out = run_depth_all(cfgs, 16, 8, 3, 5, 2, 123)
    dmax = np.max(np.abs(out["a"]["cbar"] - out["b"]["cbar"]))
    assert dmax < 1e-10, f"homogeneity identity violated: {dmax:.2e}"
    print("selftest: conv exact, GH matches ReLU closed forms, "
          f"sw-homogeneity |dc| = {dmax:.1e}  -- PASS")


# --------------------------------------------------------------------- figure

def make_figure(configs, sim, mf, conc_summary, t_ref, path, png=None):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9,
        "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), layout="constrained")
    depth = len(mf["relu_sw1"][1]) - 1
    tt = np.arange(depth + 1)

    ax = axes[0]
    m1 = {f: sim["relu_sw1"][f].mean(axis=0) for f in ("cbar", "patch")}
    m2 = sim["relu_sw2"]["cbar"].mean(axis=0)
    ax.plot(tt, m1["cbar"], color=BLUE, label=r"two-input $\bar c_t$, $\sigma_w^2{=}1$")
    ax.plot(tt, m1["patch"], color=BLUE, ls="--",
            label=r"patch cosine, $\sigma_w^2{=}1$")
    ax.plot(tt[::3], m2[::3], "o", ms=3.4, mfc="none", mec=VERM, mew=0.9,
            label=r"$\sigma_w^2{=}2$ (matched seeds)")
    ax.plot(tt, mf["relu_sw1"][1], color=GREY, ls=":", lw=1.0,
            label="mean field")
    ax.set_xlabel(r"depth $t$")
    ax.set_ylabel("correlation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    series = [
        ("relu_sw1", BLUE, "o", r"ReLU $\sigma_b{=}0$ (any $\sigma_w$)"),
        ("relu_sw1_b", GREEN, "s",
         r"ReLU $\sigma_w^2{=}1,\ \sigma_b^2{=}\frac{1}{2}$ ($\chi_1{=}\frac{1}{2}$)"),
        ("relu_sw2_b", ORANGE, "^",
         r"ReLU $\sigma_w^2{=}2,\ \sigma_b^2{=}\frac{1}{2}$"),
        ("tanh_ord", PURPLE, "D", "tanh ordered"),
        ("tanh_eoc", SKY, "v", "tanh EOC ($\\chi_1{=}1$)"),
    ]
    for name, col, mk, lab in series:
        y = 1.0 - sim[name]["cbar"].mean(axis=0)
        y = np.where(y > 1e-14, y, np.nan)
        ax.semilogy(tt, y, color=col, marker=mk, ms=2.6, markevery=4,
                    label=lab)
        ymf = 1.0 - np.asarray(mf[name][1])
        ymf = np.where(ymf > 1e-14, ymf, np.nan)
        ax.semilogy(tt, ymf, color=col, ls="--", lw=0.7)
    ax.semilogy([], [], color=GREY, ls="--", lw=0.7, label="mean field")
    ax.set_xlabel(r"depth $t$")
    ax.set_ylabel(r"$1-\bar c_t$")
    ax.set_ylim(1e-10, 3)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[2]
    cs = np.array(sorted(conc_summary))
    for key, col, mk, lab in (
            ("cbar", BLUE, "o", r"$\bar c_t$"),
            ("patch", GREEN, "s", "patch cosine"),
            ("lenratio", ORANGE, "^", r"$2^{t}\bar q_t/\bar q_0$")):
        std = np.array([conc_summary[c][key + "_std"] for c in cs])
        ax.loglog(cs, std, color=col, marker=mk, ms=3.4, label=lab)
    guide = conc_summary[cs[0]]["cbar_std"] * np.sqrt(cs[0] / cs)
    ax.loglog(cs, guide, color=GREY, ls="--", lw=0.8, label=r"$\propto C^{-1/2}$")
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.minorticks_off()
    ax.set_xlabel(r"channels $C$")
    ax.set_ylabel(f"std over reps at $t={t_ref}$")
    ax.legend(loc="lower left", frameon=False)

    for ax, tag in zip(axes, "abc"):
        ax.text(0.03, 0.97, f"({tag})", transform=ax.transAxes, va="top",
                fontsize=9)
    fig.savefig(path)
    if png:
        fig.savefig(png, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=int, default=128)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--conc-depth", type=int, default=10)
    parser.add_argument("--conc-reps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--png", type=str, default=None,
                        help="optional raster copy of the figure")
    args = parser.parse_args()
    t0 = time.perf_counter()

    print("# Experiment E4: CNN signal propagation "
          f"(C={args.C}, L={args.L}, k={args.k}, depth={args.depth}, "
          f"reps={args.reps}, seed={args.seed})")
    selftest()

    sb2 = 0.5
    eoc_sb2 = 0.05
    eoc_sw2, eoc_q, eoc_chi = solve_tanh_eoc(eoc_sb2)
    configs = [
        {"name": "relu_sw1", "phi": "relu", "sw2": 1.0, "sb2": 0.0},
        {"name": "relu_sw2", "phi": "relu", "sw2": 2.0, "sb2": 0.0},
        {"name": "relu_sw1_b", "phi": "relu", "sw2": 1.0, "sb2": sb2},
        {"name": "relu_sw2_b", "phi": "relu", "sw2": 2.0, "sb2": sb2},
        {"name": "tanh_ord", "phi": "tanh", "sw2": 0.8, "sb2": eoc_sb2},
        {"name": "tanh_eoc", "phi": "tanh", "sw2": eoc_sw2, "sb2": eoc_sb2},
    ]
    notes = {
        "relu_sw1": "q_t -> 0 at rate 1/2; cosine map = Khat (scale free), "
                    "slope 1 at c=1",
        "relu_sw2": "length critical, V(q)=q; same cosine orbit as sw2=1 "
                    "(homogeneity, matched seeds)",
        "relu_sw1_b": "q* = sb^2/(2-sw^2) = 0.5, chi1 = 1/2 < 1: ordered, "
                      "geometric collapse",
        "relu_sw2_b": "no length fixed point (q_t = q_0 + t sb^2/2); cosine "
                      "slope -> 1: asymptotically critical",
        "tanh_ord": "ordered phase",
        "tanh_eoc": "edge of chaos, chi1 = 1 (solved)",
    }

    print("\n## configs (chi1 = sw^2 E[phi'(sqrt(qhat*) Z)^2] at the length "
          "fixed point; for sb=0 ReLU the cosine-map slope at c=1 is 1 for "
          "every sw^2 -- THEORY_SPEC Sec. 1)")
    print("| config | phi | sw^2 | sb^2 | q* | chi1 | note |")
    print("|---|---|---:|---:|---:|---:|---|")
    consts = {}
    for cfg in configs:
        if cfg["name"] == "relu_sw2_b":
            qs, ch = float("nan"), float("nan")
            qs_s, ch_s = "grows", "-> 1"
        else:
            qs = q_fixed_point(cfg["phi"], cfg["sw2"], cfg["sb2"])
            ch = chi1(cfg["phi"], cfg["sw2"], cfg["sb2"], qs)
            qs_s, ch_s = f"{qs:.4f}", f"{ch:.4f}"
        consts[cfg["name"]] = {"qstar": qs, "chi1": ch}
        print(f"| {cfg['name']} | {cfg['phi']} | {cfg['sw2']:.4f} | "
              f"{cfg['sb2']:.2f} | {qs_s} | {ch_s} | {notes[cfg['name']]} |")
    print(f"\ntanh EOC solve at sb^2={eoc_sb2}: sw^2={eoc_sw2:.6f}, "
          f"q*={eoc_q:.6f}, chi1={eoc_chi:.9f}")

    mf = {c["name"]: mf_orbit(c["phi"], c["sw2"], c["sb2"], args.depth)
          for c in configs}

    sim = run_depth_all(configs, args.C, args.L, args.k, args.depth,
                        args.reps, args.seed)

    t_show = [t for t in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30)
              if t <= args.depth]
    for cfg in configs:
        name = cfg["name"]
        r = sim[name]
        q_mf, c_mf = mf[name]
        print(f"\n## depth propagation: {name} "
              f"(phi={cfg['phi']}, sw^2={cfg['sw2']:.4f}, sb^2={cfg['sb2']:.2f})")
        print("| t | qbar | qbar MF | q spread/mean | cbar | cbar MF | "
              "patch | patch MF | 1-cbar |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for t in t_show:
            cb = r["cbar"][:, t].mean()
            print(f"| {t} | {r['qbar'][:, t].mean():.3g} | {q_mf[t]:.3g} | "
                  f"{r['qrel'][:, t].mean():.3f} | {cb:.4f} | {c_mf[t]:.4f} | "
                  f"{r['patch'][:, t].mean():.4f} | {c_mf[t]:.4f} | "
                  f"{1.0 - cb:.3g} |")

    ident = np.max(np.abs(sim["relu_sw1"]["cbar"] - sim["relu_sw2"]["cbar"]))
    print(f"\nReLU homogeneity identity (sb=0): max |cbar(sw^2=2) - "
          f"cbar(sw^2=1)| over all reps and depths = {ident:.3e} "
          "(floating-point rounding only; the two zero-bias runs are the "
          "same trajectory up to scale)")

    # finite-C concentration
    conc = run_concentration([32, 64, 128, 256], args.L, args.k,
                             args.conc_depth, args.conc_reps, args.seed)
    tr = args.conc_depth
    _, c_mf_relu = mf_orbit("relu", 1.0, 0.0, tr)
    conc_summary = {}
    print(f"\n## finite-C concentration (ReLU, sw^2=1, sb=0, L={args.L}, "
          f"k={args.k}, t={tr}, reps={args.conc_reps}; mean-field "
          f"cbar_{tr} = {c_mf_relu[tr]:.4f}, lenratio = 1)")
    print("| C | mean cbar | std cbar | |mean-MF| | std patch | "
          "std 2^t qbar/qbar_0 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for c_dim in sorted(conc):
        d = conc[c_dim]
        row = {
            "cbar_mean": float(d["cbar"][:, tr].mean()),
            "cbar_std": float(d["cbar"][:, tr].std(ddof=1)),
            "patch_std": float(d["patch"][:, tr].std(ddof=1)),
            "lenratio_std": float(d["lenratio"][:, tr].std(ddof=1)),
        }
        row["dev_mf"] = abs(row["cbar_mean"] - c_mf_relu[tr])
        conc_summary[c_dim] = row
        print(f"| {c_dim} | {row['cbar_mean']:.4f} | {row['cbar_std']:.2e} | "
              f"{row['dev_mf']:.2e} | {row['patch_std']:.2e} | "
              f"{row['lenratio_std']:.2e} |")
    cs = np.array(sorted(conc_summary), dtype=float)
    slopes = {}
    for key in ("cbar_std", "patch_std", "lenratio_std"):
        y = np.log([conc_summary[int(c)][key] for c in cs])
        slopes[key] = float(np.polyfit(np.log(cs), y, 1)[0])
    print("fitted log-log slope of std vs C: "
          + ", ".join(f"{k} {v:+.2f}" for k, v in slopes.items())
          + "  (CLT scale -1/2; supports the e^{-cC} concentration claim)")

    # outputs
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    payload = {
        "params": {"C": args.C, "L": args.L, "k": args.k,
                   "depth": args.depth, "reps": args.reps,
                   "conc_depth": args.conc_depth,
                   "conc_reps": args.conc_reps, "seed": args.seed},
        "tanh_eoc": {"sb2": eoc_sb2, "sw2": eoc_sw2, "qstar": eoc_q,
                     "chi1": eoc_chi},
        "configs": [
            {**cfg, "note": notes[cfg["name"]],
             "qstar": consts[cfg["name"]]["qstar"],
             "chi1": consts[cfg["name"]]["chi1"],
             "mf_q": mf[cfg["name"]][0], "mf_c": mf[cfg["name"]][1],
             **{f + "_mean": sim[cfg["name"]][f].mean(axis=0).tolist()
                for f in ("qbar", "qrel", "cbar", "patch")},
             **{f + "_std": sim[cfg["name"]][f].std(axis=0, ddof=1).tolist()
                for f in ("qbar", "cbar", "patch")}}
            for cfg in configs],
        "relu_homogeneity_max_dcbar": float(ident),
        "concentration": {"t_ref": tr, "mf_cbar": c_mf_relu[tr],
                          "by_C": {str(c): conc_summary[c]
                                   for c in sorted(conc_summary)},
                          "slopes": slopes},
    }
    json_path = os.path.join(DATA_DIR, "exp_e4_cnn.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)

    fig_path = os.path.join(FIG_DIR, "fig_e4_cnn.pdf")
    make_figure(configs, sim, mf, conc_summary, tr, fig_path, png=args.png)

    print(f"\nsaved {json_path}")
    print(f"saved {fig_path}")
    print(f"total runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
