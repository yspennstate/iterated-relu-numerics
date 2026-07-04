"""Render the three publication figures from the saved experiment data.

Reads data/experiment{A,B,C}.json (produced by the experiment_*.py scripts) and
writes vector PDFs to ../figures/:

  fig_tail_rate.pdf   empirical exterior tail rate -logE/N (real and modulus
                      count) vs matrix size N, with the I_alpha(r^2) target.
  fig_near_ray.pdf    (left) one-step gain vs self-overlap; (right) max gain
                      over near-rays vs N, against 2^{-1/2}.
  fig_finite_time.pdf norm halving, consecutive cosine vs the kernel orbit,
                      and the t*theta_t/(3 pi) -> 1 angle law.

Restrained style for a mathematics paper: serif/Computer-Modern mathtext, thin
lines, colorblind-safe (Wong) colors, no baked-in titles (captions live in the
LaTeX source).
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
SQRT_HALF = 1.0 / math.sqrt(2.0)

# Deliverable is vector PDF; a PNG copy can be added for on-screen inspection.
EXTS = [".pdf"]


def save(fig, stem, **kwargs):
    pdf_path = os.path.join(FIG_DIR, stem + ".pdf")
    for ext in EXTS:
        fig.savefig(os.path.join(FIG_DIR, stem + ext), **kwargs)
    plt.close(fig)
    return pdf_path

# Wong colorblind-safe palette
BLUE, VERM, GREEN, ORANGE, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#666666")

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name):
    with open(os.path.join(DATA_DIR, name)) as fh:
        return json.load(fh)


def fig_tail_rate(A, alpha=0.5, min_events=20):
    rows = [r for r in A["rows"] if abs(r["alpha"] - alpha) < 1e-9]
    Ns = sorted({r["N"] for r in rows})
    rvals = sorted({r["r"] for r in rows})

    def get(N, r):
        return next(x for x in rows if x["N"] == N and abs(x["r"] - r) < 1e-9)

    # 3 smallest thresholds that are reliable at the smallest N (base case)
    base = Ns[0]
    cand = [rv for rv in rvals
            if get(base, rv)["tot_real"] >= min_events
            and get(base, rv)["tot_mod"] >= min_events]
    pick = cand[:3]
    colors = [BLUE, VERM, GREEN]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for rv, c in zip(pick, colors):
        Ia = get(Ns[0], rv)["I_alpha"]
        # plot each series only over the N where that count is reliable
        Nr = [N for N in Ns if get(N, rv)["tot_real"] >= min_events]
        Nm = [N for N in Ns if get(N, rv)["tot_mod"] >= min_events]
        ax.plot(Nr, [get(N, rv)["rate_real"] for N in Nr],
                marker="o", color=c, linestyle="-")
        ax.plot(Nm, [get(N, rv)["rate_mod"] for N in Nm],
                marker="s", color=c, linestyle="--", markerfacecolor="none")
        ax.axhline(Ia, color=c, linestyle=":", linewidth=0.8)
        ax.annotate(rf"$r={rv:.2f}$", xy=(Ns[-1], Ia),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", color=c, fontsize=7.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlim(Ns[0] * 0.9, Ns[-1] * 1.45)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"empirical tail rate $-\log \mathbb{E}[\#]/N$")

    handles = [
        plt.Line2D([], [], color=GREY, marker="o", linestyle="-",
                   label="real-eigenvalue count"),
        plt.Line2D([], [], color=GREY, marker="s", linestyle="--",
                   markerfacecolor="none", label="modulus count"),
        plt.Line2D([], [], color=GREY, linestyle=":",
                   label=r"$I_\alpha(r^2)$ (target)"),
    ]
    ax.legend(handles=handles, loc="upper right")
    return save(fig, "fig_tail_rate")


def fig_near_ray(B):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 2.9),
                                   constrained_layout=True)

    sc = B["scatter"]
    gain = sc["gain"]
    selfov = sc["selfov"]
    tag = sc["tag"]
    groups = [
        ([0], "random", BLUE, "."),
        ([1, 2], "trajectory / near-ray", VERM, "."),
        ([3], "submatrix eigvec", GREEN, "."),
    ]
    for tags, label, color, marker in groups:
        xs = [s for s, t in zip(selfov, tag) if t in tags]
        ys = [g for g, t in zip(gain, tag) if t in tags]
        if xs:
            axL.scatter(xs, ys, s=3, c=color, marker=marker, alpha=0.35,
                        linewidths=0, label=label, rasterized=True)
    axL.axhline(SQRT_HALF, color="black", linestyle="--", linewidth=0.8)
    axL.annotate(r"$2^{-1/2}$", xy=(-0.02, SQRT_HALF), xytext=(2, 3),
                 textcoords="offset points", fontsize=7.5, va="bottom")
    axL.set_xlabel(r"self-overlap $\cos(u, F(u))$")
    axL.set_ylabel(r"one-step gain $g(u)=\|\mathrm{ReLU}(Wu)\|$")
    axL.set_xlim(-0.05, 1.02)
    axL.legend(loc="lower left", markerscale=2.5, handletextpad=0.2)

    per_N = B["per_N"]
    Ns = sorted(int(k) for k in per_N)

    def nearray_max(N, delta):
        e = next(x for x in per_N[str(N)]["nearray"]
                 if abs(x["delta"] - delta) < 1e-9)
        return e["max"] if e["n"] else None

    deltas = [d for d in (0.10, 0.20)
              if all(nearray_max(N, d) is not None for N in Ns)]
    dcolors = {0.10: VERM, 0.20: BLUE, 0.30: GREEN}
    for d in deltas:
        ys = [nearray_max(N, d) for N in Ns]
        axR.plot(Ns, ys, marker="o", color=dcolors[d],
                 label=rf"near-rays $\delta={d:.2f}$")
    overall = [per_N[str(N)]["overall_max"] for N in Ns]
    axR.plot(Ns, overall, marker="^", color=GREY, linestyle=":",
             label="max over all directions")
    axR.axhline(SQRT_HALF, color="black", linestyle="--", linewidth=0.8)
    axR.annotate(r"$2^{-1/2}$", xy=(Ns[0], SQRT_HALF), xytext=(2, 3),
                 textcoords="offset points", fontsize=7.5, va="bottom")
    axR.set_xscale("log", base=2)
    axR.set_xticks(Ns)
    axR.set_xticklabels([str(N) for N in Ns])
    axR.set_xlabel(r"$N$")
    axR.set_ylabel(r"max gain")
    axR.legend(loc="upper right")
    return save(fig, "fig_near_ray", dpi=300)


def fig_finite_time(C):
    diag = C["diag"]
    orbit = C["orbit"]
    t = [d["t"] for d in diag]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.2),
                                  constrained_layout=True)

    a.plot(t, [d["norm2"] for d in diag], marker="o", color=BLUE,
           label=r"$2^{t}\|x_t\|^2$")
    a.plot(t, [d["support"] for d in diag], marker="s", color=VERM,
           markerfacecolor="none", label=r"$|S_t|/N$")
    a.axhline(1.0, color=GREY, linestyle=":", linewidth=0.7)
    a.axhline(0.5, color=GREY, linestyle=":", linewidth=0.7)
    a.set_xlabel(r"$t$")
    a.set_ylim(0.35, 1.1)
    a.legend(loc="center right")

    b.plot(t, [d["cos"] for d in diag], marker="o", linestyle="none",
           color=BLUE, label=r"$\cos(x_{t-1},x_t)$")
    b.plot(t, [d["kernel"] for d in diag], color=VERM,
           label="kernel orbit")
    b.set_xlabel(r"$t$")
    b.set_ylabel("consecutive cosine")
    b.legend(loc="lower right")

    to = [o["t"] for o in orbit]
    ratio = [o["ratio"] for o in orbit]
    c.plot(to, ratio, marker="o", color=GREEN)
    c.axhline(1.0, color=GREY, linestyle=":", linewidth=0.7)
    c.set_xscale("log")
    c.set_xlabel(r"$t$")
    c.set_ylabel(r"$t\,\theta_t / (3\pi)$")
    c.set_ylim(0.4, 1.05)
    return save(fig, "fig_finite_time")


def main():
    import sys
    if "--png" in sys.argv:
        EXTS.append(".png")
    os.makedirs(FIG_DIR, exist_ok=True)
    outs = []
    outs.append(fig_tail_rate(load("experimentA.json")))
    outs.append(fig_near_ray(load("experimentB.json")))
    outs.append(fig_finite_time(load("experimentC.json")))
    for o in outs:
        size = os.path.getsize(o) if os.path.exists(o) else 0
        print(f"wrote {o}  ({size} bytes)")


if __name__ == "__main__":
    main()
