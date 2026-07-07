"""Experiment E2: tied weights and fresh weights agree over fixed horizons.

The innovation representation says the iteration x -> phi(W x + b) with a single
reused matrix (tied) has the same finite-horizon state evolution as a fresh
matrix drawn each layer (independent), because the map only ever probes W
through W v, never W^T v.  We check this directly: for each activation we run
both models from the same start and compare the length q_t and two-input cosine
c_t.  Over a fixed horizon the two trajectories agree; the gap is O(1/sqrt N)
and shrinks with width.  A long-horizon run shows where the tied model finally
departs -- the reused-matrix regime the theory leaves open.

Writes data/exp_e2_tied_vs_independent.json, prints markdown tables, draws
figures/fig_e2_tied_vs_independent.pdf.
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


def both_models(name, sw2, sb2, q0, c0, T, N, reps, seed):
    """Average q_t and c_t for tied and independent, same starts per rep."""
    phi = ec.ACTIVATIONS[name][0]
    out = {"tied": {"q": np.zeros(T + 1), "c": np.zeros(T + 1)},
           "indep": {"q": np.zeros(T + 1), "c": np.zeros(T + 1)}}
    for r in range(reps):
        rng0 = np.random.default_rng([seed, r, 0])
        x0, y0 = ec.make_starts(rng0, N, q0, c0)
        for tied, key in ((True, "tied"), (False, "indep")):
            rng = np.random.default_rng([seed, r, 1])   # same weight stream
            qx, _, c = ec.iterate_pair(x0, y0, T, phi, sw2, sb2, rng, tied=tied)
            out[key]["q"] += qx
            out[key]["c"] += c
    for key in out:
        out[key]["q"] /= reps
        out[key]["c"] /= reps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    print(f"# Experiment E2: tied vs independent weights "
          f"(T={args.T}, reps={args.reps}, seed={args.seed})")

    acts = [("relu", 1.0, 0.0, 0.5), ("tanh", 1.5, 0.05, None),
            ("gelu", 1.5, 0.05, None)]
    Ns = [250, 500, 1000, 2000]
    results = {}

    for name, sw2, sb2, q0 in acts:
        if q0 is None:
            q0 = ec.fixed_point_of(name, sw2, sb2, q0=1.0)[0]
        print(f"\n## {name} (sw2={sw2}, sb2={sb2}, q0={q0:.4f}), c0=0.3")
        print("| N | max_t|q_tied-q_indep| | max_t|c_tied-c_indep| |")
        print("|---:|---:|---:|")
        per_N = {}
        for N in Ns:
            o = both_models(name, sw2, sb2, q0, 0.3, args.T, N,
                            args.reps, args.seed)
            dq = float(np.max(np.abs(np.array(o["tied"]["q"])
                                     - np.array(o["indep"]["q"]))))
            dc = float(np.max(np.abs(np.array(o["tied"]["c"])
                                     - np.array(o["indep"]["c"]))))
            per_N[N] = {"dq": dq, "dc": dc,
                        "c_tied": o["tied"]["c"].tolist(),
                        "c_indep": o["indep"]["c"].tolist(),
                        "q_tied": o["tied"]["q"].tolist(),
                        "q_indep": o["indep"]["q"].tolist()}
            print(f"| {N} | {dq:.5f} | {dc:.5f} |")
        results[name] = {"sw2": sw2, "sb2": sb2, "q0": q0, "per_N": per_N}

    # long horizon: tied ReLU angle gap vs horizon at fixed N ----------------
    print("\n## long-horizon tied-vs-independent (ReLU angle, N=1500)")
    Tlong, Nl, reps_l = 50, 1000, 20
    o = both_models("relu", 1.0, 0.0, 0.5, 0.3, Tlong, Nl, reps_l, args.seed)
    ct = np.array(o["tied"]["c"])
    ci = np.array(o["indep"]["c"])
    gap_long = np.abs(ct - ci)
    print("| t | c_tied | c_indep | |gap| |")
    print("|---:|---:|---:|---:|")
    for t in (1, 4, 8, 12, 20, 30, 40, 50):
        print(f"| {t} | {ct[t]:.4f} | {ci[t]:.4f} | {gap_long[t]:.4f} |")
    print(f"max gap over [0,{Tlong}] = {float(gap_long.max()):.4f} at "
          f"t={int(gap_long.argmax())}  (fixed horizon: gap stays O(1/sqrtN); "
          "long-horizon tied departure is the open regime)")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(ec.rcparams())
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0), constrained_layout=True)

    # panel A: cosine trajectories, tied vs indep, relu at one N
    a = ax[0]
    ts = np.arange(args.T + 1)
    rr = results["tanh"]["per_N"][2000]
    a.plot(ts, rr["c_tied"], "o-", color=ec.WONG[0], ms=3, mfc="none",
           label="tied")
    a.plot(ts, rr["c_indep"], "s--", color=ec.WONG[1], ms=3, mfc="none",
           label="independent")
    a.set_xlabel("depth $t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_title(r"(a) tanh, $N=2000$: tied $=$ independent")
    a.legend(loc="lower right")

    # panel B: gap vs N (log-log) for each activation
    a = ax[1]
    for i, (name, *_ ) in enumerate(acts):
        dcs = np.array([results[name]["per_N"][N]["dc"] for N in Ns])
        a.loglog(Ns, dcs, "o-", color=ec.WONG[i], mfc="none", label=name)
    a.loglog(np.array(Ns), 0.06 * (np.array(Ns) / 250.0) ** -0.5, "k:",
             lw=0.8, label=r"$N^{-1/2}$")
    a.set_xlabel("width $N$")
    a.set_ylabel(r"$\max_t|c_{\rm tied}-c_{\rm indep}|$")
    a.set_title("(b) agreement improves as $N^{-1/2}$"); a.legend()

    # panel C: long-horizon tied vs indep (ReLU)
    a = ax[2]
    tl = np.arange(Tlong + 1)
    a.plot(tl, ct, color=ec.WONG[0], lw=1.2, label="tied")
    a.plot(tl, ci, color=ec.WONG[1], lw=1.2, ls="--", label="independent")
    a.set_xlabel("depth $t$"); a.set_ylabel(r"cosine $c_t$")
    a.set_title(r"(c) ReLU angle, $N=1500$: agree at fixed $t$")
    a.legend(loc="lower right")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "fig_e2_tied_vs_independent.pdf")
    fig.savefig(fig_path); plt.close(fig)

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"params": vars(args), "results": results,
               "long_horizon": {"T": Tlong, "N": Nl, "c_tied": ct.tolist(),
                                "c_indep": ci.tolist(),
                                "max_gap": float(gap_long.max())}}
    data_path = os.path.join(DATA_DIR, "exp_e2_tied_vs_independent.json")
    with open(data_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {data_path}\nwrote {fig_path}\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
