"""Experiment E5: token rank collapse in stacked self-attention (ViT case).

n tokens are the rows of X in R^{n x d}.  One multi-head self-attention layer,
with fresh weights at every layer (feedforward ViT regime), is

    A_h     = softmax( (X WQ_h)(X WK_h)^T / sqrt(d_head) ),   A_h 1 = 1,
    Attn(X) = [ A_1 X WV_1 | ... | A_H X WV_H ] WO,
    WQ, WK, WV ~ N(0, 1/d) i.i.d.,   WO ~ N(0, 1/d) i.i.d.

Four block types are stacked to depth L:

    pure      X <- Attn(X)
    skip      X <- X + Attn(X)
    skip+MLP  X <- X + Attn(X);  X <- X + MLP(X)
              (MLP = Linear-GELU-Linear, width 4d, N(0,1/fan_in), zero bias)
    pre-LN    X <- X + Attn(LN(X));  X <- X + MLP(LN(X))
              (LN = per-token mean/variance normalization, no learned affine)

Per depth we record (all scale-invariant):

    cos    mean pairwise cosine between token rows      (-> 1: tokens parallel)
    sr     stable rank ||X||_F^2 / ||X||_2^2            (-> 1: rank collapse)
    R      ||(I - 11^T/n) X||_F / ||X||_F               (-> 0: tokens equal)
    delta  per-layer Dobrushin coefficient of the attention matrix,
           delta(A) = (1/2) max_{p,q} sum_j |A_pj - A_qj|  (max over heads).

Since every A_h is row-stochastic, the oscillation of each column contracts by
delta(A_h) per layer; stacking gives the Dobrushin bound

    R(X^t) <= prod_{s<t} delta(A^s) * R(X^0)

for the pure stack, which we verify rep by rep (in the normalized form above
and in the unnormalized form ||(I-P)X^t|| / ||(I-P)X^0|| <= prod delta).  A
single-head value-free stack X <- A X — the exact object of the Dobrushin
theorem, where the per-column oscillation bound is provable with no matrix
constants — runs alongside as a control, checked in the oscillation seminorm.

Rate: softmax is shift-invariant within each row, so the component of the
logit variation that is constant across keys cancels; the token spread enters
A only quadratically, delta(A) ~ R^2, hence R <- delta * R ~ R^3 per layer.
This is the doubly-exponential collapse of Dong-Cordonnier-Loukas 2021 (rate
3^t).  We fit the exponent p in R_{t+1} ~ C R_t^p and the slope of
log delta_t vs log R_t (predicted 2), and report the depth window in which
the fit is meaningful before R hits the float64 resolution ~1e-16.

Writes data/exp_e5_attention_rank_collapse.json, prints markdown tables, and
draws figures/fig_e5_attention.pdf.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from scipy.special import erf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")

CONFIGS = ["pure", "skip", "skip_mlp", "preln"]          # the four main stacks
CONTROL = "pure_valuefree"                               # X <- A X (theorem form)
LABELS = {"pure": "pure attention", "skip": "+ skip",
          "skip_mlp": "+ skip + MLP", "preln": "pre-LN block",
          CONTROL: "pure, value-free (X <- AX)"}

EPS64 = np.finfo(np.float64).eps          # 2.22e-16, resolution floor
NOISE_FLOOR = 1e-13                       # below this, values are roundoff
TABLE_TS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40]


# ---------------------------------------------------------------- primitives

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def dobrushin(a):
    """delta(A) = (1/2) max_{p,q} ||A_p. - A_q.||_1  for row-stochastic A."""
    diff = np.abs(a[:, None, :] - a[None, :, :]).sum(axis=2)
    return 0.5 * float(diff.max())


def attention(x, rng, n_heads):
    """Multi-head self-attention with fresh N(0,1/d) weights.

    Returns (Attn(x), max over heads of delta(A_h))."""
    n, d = x.shape
    dh = d // n_heads
    sig = 1.0 / math.sqrt(d)
    heads, dmax = [], 0.0
    for _ in range(n_heads):
        wq = sig * rng.standard_normal((d, dh))
        wk = sig * rng.standard_normal((d, dh))
        wv = sig * rng.standard_normal((d, dh))
        a = softmax_rows((x @ wq) @ (x @ wk).T / math.sqrt(dh))
        dmax = max(dmax, dobrushin(a))
        heads.append(a @ (x @ wv))
    wo = sig * rng.standard_normal((d, d))
    return np.concatenate(heads, axis=1) @ wo, dmax


def attention_valuefree(x, rng):
    """Single head, no value/output matrices: the exact Dobrushin object."""
    n, d = x.shape
    sig = 1.0 / math.sqrt(d)
    wq = sig * rng.standard_normal((d, d))
    wk = sig * rng.standard_normal((d, d))
    a = softmax_rows((x @ wq) @ (x @ wk).T / math.sqrt(d))
    return a @ x, dobrushin(a)


def mlp(x, rng):
    n, d = x.shape
    w1 = rng.standard_normal((d, 4 * d)) / math.sqrt(d)
    w2 = rng.standard_normal((4 * d, d)) / math.sqrt(4 * d)
    return gelu(x @ w1) @ w2


# ------------------------------------------------------------------- metrics

def token_cosine(x):
    """Mean pairwise cosine between token rows (off-diagonal average)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norms, 1e-300)
    g = xn @ xn.T
    n = x.shape[0]
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def stable_rank(x):
    s = np.linalg.svd(x, compute_uv=False)
    return float((s ** 2).sum() / s[0] ** 2)


def residuals(x):
    """(normalized R, unnormalized ||(I-P)X||_F, oscillation seminorm)."""
    xc = x - x.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(xc))
    osc = float((x.max(axis=0) - x.min(axis=0)).max())
    return num / float(np.linalg.norm(x)), num, osc


def measure(x):
    r, rn, osc = residuals(x)
    return {"cos": token_cosine(x), "sr": stable_rank(x), "R": r,
            "Rnum": rn, "osc": osc, "normF": float(np.linalg.norm(x))}


# ---------------------------------------------------------------- simulation

def run_config(cfg, x0, rng, depth, n_heads):
    x = x0.copy()
    rows = [measure(x)]
    deltas = []
    for _ in range(depth):
        if cfg == "pure":
            out, dl = attention(x, rng, n_heads)
            x = out
        elif cfg == "skip":
            out, dl = attention(x, rng, n_heads)
            x = x + out
        elif cfg == "skip_mlp":
            out, dl = attention(x, rng, n_heads)
            x = x + out
            x = x + mlp(x, rng)
        elif cfg == "preln":
            out, dl = attention(layer_norm(x), rng, n_heads)
            x = x + out
            x = x + mlp(layer_norm(x), rng)
        elif cfg == CONTROL:
            x, dl = attention_valuefree(x, rng)
        else:
            raise ValueError(cfg)
        deltas.append(dl)
        rows.append(measure(x))
    return rows, deltas


def gmean(a, axis=0, floor=1e-320):
    return np.exp(np.log(np.clip(a, floor, None)).mean(axis=axis))


def ols_slope(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    xm, ym = xs.mean(), ys.mean()
    sxx = ((xs - xm) ** 2).sum()
    if sxx == 0.0:
        return float("nan"), float("nan")
    b = float(((xs - xm) * (ys - ym)).sum() / sxx)
    return b, float(ym - b * xm)


def check_bound(res_by_rep, prod_by_rep):
    """Per rep and depth, is res_t <= prod_t * res_0 (above the noise floor)?"""
    viol, checked, worst = 0, 0, 0.0
    for res, prod in zip(res_by_rep, prod_by_rep):
        for t in range(1, len(res)):
            bound = prod[t - 1] * res[0]
            if bound < NOISE_FLOOR and res[t] < NOISE_FLOOR:
                continue                    # both at roundoff level: untestable
            checked += 1
            ratio = res[t] / max(bound, 1e-320)
            worst = max(worst, ratio)
            if ratio > 1.0 + 1e-9:
                viol += 1
    return {"checked": checked, "violations": viol, "worst_ratio": worst}


def fit_exponent(seq_by_rep, lo=1e-12, hi=0.5):
    """Pooled OLS of log y_{t+1} vs log y_t over the informative window."""
    xs, ys = [], []
    for seq in seq_by_rep:
        for t in range(len(seq) - 1):
            if lo < seq[t] < hi and seq[t + 1] > NOISE_FLOOR:
                xs.append(math.log(seq[t]))
                ys.append(math.log(seq[t + 1]))
    if len(xs) < 3:
        return {"p": float("nan"), "logC": float("nan"), "npairs": len(xs)}
    p, c = ols_slope(xs, ys)
    return {"p": p, "logC": c, "npairs": len(xs)}


# ------------------------------------------------------------------ printing

def print_table(cfg, agg, depth):
    print(f"\n## {LABELS[cfg]}   ({cfg})")
    print("| t | cos (mean) | 1-cos (gm) | sr (mean) | R (gm) | "
          "delta_t (mean) | prod delta (gm) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for t in TABLE_TS:
        if t > depth:
            continue
        d_s = f"{agg['delta_mean'][t - 1]:.4g}" if t >= 1 else "--"
        p_s = f"{agg['prod_gm'][t - 1]:.4g}" if t >= 1 else "--"
        print(f"| {t} | {agg['cos_mean'][t]:.6f} | {agg['omc_gm'][t]:.4g} | "
              f"{agg['sr_mean'][t]:.3f} | {agg['R_gm'][t]:.4g} | {d_s} | {p_s} |")


# ---------------------------------------------------------------------- plot

WONG = {"pure": "#0072B2", "skip": "#D55E00",
        "skip_mlp": "#009E73", "preln": "#E69F00"}
MARKS = {"pure": "o", "skip": "s", "skip_mlp": "^", "preln": "D"}

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
    "lines.markersize": 3.4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def make_figure(results, depth, path):
    ts = np.arange(depth + 1)
    clip = 1e-17                          # display floor (float64 resolution)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.9), constrained_layout=True)
    (ax_c, ax_sr), (ax_r, ax_d) = axes

    def eps_line(ax):
        ax.axhline(EPS64, color="#666666", linestyle=":", linewidth=0.7)

    for cfg in CONFIGS:
        agg = results[cfg]
        kw = dict(color=WONG[cfg], marker=MARKS[cfg], markevery=3,
                  markerfacecolor="none", label=LABELS[cfg])
        ax_c.semilogy(ts, np.clip(agg["omc_gm"], clip, None), **kw)
        ax_sr.plot(ts, agg["sr_mean"], **kw)
        ax_r.semilogy(ts, np.clip(agg["R_gm"], clip, None), **kw)
        ax_d.semilogy(ts[1:], np.clip(agg["delta_gm"], clip, None), **kw)

    prod = np.clip(results["pure"]["prod_gm"], 1e-320, None)
    ax_r.semilogy(np.arange(1, depth + 1), prod, color="black",
                  linestyle="--", linewidth=0.9,
                  label=r"$\prod_s \delta(A^s)$ (pure)")

    ax_c.set_ylabel(r"$1 - \overline{\cos}(x_p, x_q)$")
    ax_c.set_ylim(clip / 3, 8)
    eps_line(ax_c)
    ax_c.annotate(r"float64 $\varepsilon$", xy=(depth, EPS64),
                  xytext=(-2, 4), textcoords="offset points",
                  ha="right", fontsize=7, color="#666666")

    ax_sr.set_ylabel(r"stable rank $\|X\|_F^2 / \|X\|_2^2$")
    ax_sr.axhline(1.0, color="#666666", linestyle=":", linewidth=0.7)
    ax_sr.set_ylim(0, 27)

    ax_r.set_ylabel(r"$R(X^t) = \|(I - \frac{1}{n}11^{\top})X^t\|_F/\|X^t\|_F$")
    ax_r.set_ylim(clip / 3, 8)
    eps_line(ax_r)
    ax_r.legend(loc="lower left", handlelength=1.7)

    ax_d.set_ylabel(r"$\delta(A^t) = \max_h \delta(A_h^t)$")
    ax_d.set_ylim(clip / 3, 8)
    ax_d.axhline(1.0, color="#666666", linestyle=":", linewidth=0.7)

    for ax in (ax_c, ax_sr, ax_r, ax_d):
        ax.set_xlabel(r"depth $t$")
        ax.set_xlim(0, depth)

    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64, help="tokens")
    parser.add_argument("--d", type=int, default=64, help="embedding dim")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    t0 = time.time()

    print("# Experiment E5: self-attention rank collapse "
          f"(n={args.n}, d={args.d}, heads={args.heads}, depth={args.depth}, "
          f"reps={args.reps}, seed={args.seed})")

    all_cfgs = CONFIGS + [CONTROL]
    raw = {cfg: {"cos": [], "sr": [], "R": [], "Rnum": [], "osc": [],
                 "normF": [], "delta": [], "prod": []} for cfg in all_cfgs}

    for rep in range(args.reps):
        x0 = np.random.default_rng([args.seed, 0, rep]).standard_normal(
            (args.n, args.d))
        for ci, cfg in enumerate(all_cfgs):
            rng = np.random.default_rng([args.seed, 1 + ci, rep])
            rows, deltas = run_config(cfg, x0, rng, args.depth, args.heads)
            for key in ("cos", "sr", "R", "Rnum", "osc", "normF"):
                raw[cfg][key].append([r[key] for r in rows])
            raw[cfg]["delta"].append(deltas)
            raw[cfg]["prod"].append(np.cumprod(deltas).tolist())

    results = {}
    for cfg in all_cfgs:
        r = {k: np.array(v) for k, v in raw[cfg].items()}
        results[cfg] = {
            "cos_mean": r["cos"].mean(axis=0).tolist(),
            "omc_gm": gmean(1.0 - r["cos"]).tolist(),
            "sr_mean": r["sr"].mean(axis=0).tolist(),
            "R_gm": gmean(r["R"]).tolist(),
            "normF_gm": gmean(r["normF"]).tolist(),
            "delta_mean": r["delta"].mean(axis=0).tolist(),
            "delta_gm": gmean(r["delta"]).tolist(),
            "prod_gm": gmean(r["prod"]).tolist(),
        }
        print_table(cfg, results[cfg], args.depth)

    # ---- Dobrushin bound checks (pure stacks) -----------------------------
    checks = {}
    for cfg in ("pure", CONTROL):
        rr = raw[cfg]
        checks[cfg] = {
            "normalized_R": check_bound(rr["R"], rr["prod"]),
            "unnormalized": check_bound(
                [np.array(x) / x[0] for x in map(np.array, rr["Rnum"])],
                rr["prod"]),
            "oscillation": check_bound(
                [np.array(x) / x[0] for x in map(np.array, rr["osc"])],
                rr["prod"]),
        }

    # ---- collapse-rate fits (pure) ----------------------------------------
    fits = {
        "R_exponent": fit_exponent(raw["pure"]["R"]),
        "omc_exponent": fit_exponent(
            [(1.0 - np.array(c)).tolist() for c in raw["pure"]["cos"]]),
        "R_exponent_valuefree": fit_exponent(raw[CONTROL]["R"]),
    }
    # delta(A) ~ R^2 (softmax shift invariance): slope of log delta vs log R
    xs, ys = [], []
    for rvec, dvec in zip(raw["pure"]["R"], raw["pure"]["delta"]):
        for t in range(len(dvec)):
            if 1e-7 < rvec[t] < 0.5 and dvec[t] > NOISE_FLOOR:
                xs.append(math.log(rvec[t]))
                ys.append(math.log(dvec[t]))
    sl, ic = ols_slope(xs, ys) if len(xs) >= 3 else (float("nan"),) * 2
    fits["delta_vs_R_slope"] = {"slope": sl, "intercept": ic, "npairs": len(xs)}

    # per-step exponents log R_{t+1} / log R_t (pure, first rep, window)
    steps = []
    rv = raw["pure"]["R"][0]
    for t in range(len(rv) - 1):
        if rv[t] < 0.5 and rv[t + 1] > NOISE_FLOOR:
            steps.append({"t": t, "R_t": rv[t], "R_t1": rv[t + 1],
                          "log_ratio": math.log(rv[t + 1]) / math.log(rv[t])})
    fits["per_step_pure_rep0"] = steps

    # depth at which each config passes collapse milestones (mean cosine)
    milestones = {}
    for cfg in all_cfgs:
        omc = np.array(results[cfg]["omc_gm"])
        milestones[cfg] = {
            thr: (int(np.argmax(omc < float(thr)))
                  if (omc < float(thr)).any() else None)
            for thr in ("1e-3", "1e-6", "1e-12")}

    print("\n## Dobrushin bound  R(X^t) <= prod_s delta(A^s) * R(X^0)")
    print("| stack | residual form | checked | violations | worst R/bound |")
    print("|---|---|---:|---:|---:|")
    for cfg in ("pure", CONTROL):
        for form in ("normalized_R", "unnormalized", "oscillation"):
            c = checks[cfg][form]
            print(f"| {cfg} | {form} | {c['checked']} | {c['violations']} | "
                  f"{c['worst_ratio']:.3g} |")

    print("\n## collapse-rate fits (pure attention, window "
          f"{NOISE_FLOOR:g} < y < 0.5)")
    print(f"R_{{t+1}} ~ C R_t^p:            p = {fits['R_exponent']['p']:.3f}  "
          f"(n={fits['R_exponent']['npairs']}; value-free control p = "
          f"{fits['R_exponent_valuefree']['p']:.3f})")
    print(f"(1-cos)_{{t+1}} ~ C (1-cos)^p:  p = {fits['omc_exponent']['p']:.3f}"
          f"  (n={fits['omc_exponent']['npairs']})")
    print(f"log delta(A^t) vs log R_t:     slope = "
          f"{fits['delta_vs_R_slope']['slope']:.3f}  (predicted 2; n="
          f"{fits['delta_vs_R_slope']['npairs']})")
    print("per-step exponents log R_{t+1}/log R_t (rep 0): "
          + ", ".join(f"t={s['t']}: {s['log_ratio']:.2f}" for s in steps))

    print("\n## depth to reach 1 - cos < threshold (gm over reps; "
          "None = not reached by depth "
          f"{args.depth}; float64 floor ~ {EPS64:.2e})")
    print("| stack | 1e-3 | 1e-6 | 1e-12 |")
    print("|---|---:|---:|---:|")
    for cfg in all_cfgs:
        m = milestones[cfg]
        print(f"| {cfg} | {m['1e-3']} | {m['1e-6']} | {m['1e-12']} |")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    payload = {
        "params": vars(args),
        "results": results,
        "dobrushin_checks": checks,
        "fits": fits,
        "milestones": milestones,
        "per_rep_pure": {"R": raw["pure"]["R"], "delta": raw["pure"]["delta"],
                         "prod": raw["pure"]["prod"]},
        "per_rep_valuefree": {"R": raw[CONTROL]["R"],
                              "osc": raw[CONTROL]["osc"],
                              "delta": raw[CONTROL]["delta"],
                              "prod": raw[CONTROL]["prod"]},
    }
    data_path = os.path.join(DATA_DIR, "exp_e5_attention_rank_collapse.json")
    with open(data_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {data_path}")

    fig_path = os.path.join(FIG_DIR, "fig_e5_attention.pdf")
    make_figure(results, args.depth, fig_path)
    print(f"wrote {fig_path}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
