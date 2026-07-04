"""Experiment B: one-step gain versus self-parallelism for fixed-W ReLU.

For a fixed matrix W (N x N, i.i.d. N(0,1/N)) and the map f(u)=ReLU(Wu), define
for a nonnegative unit direction u

    g(u)   = ||ReLU(Wu)||                 (one-step gain)
    F(u)   = ReLU(Wu)/g(u)                (normalized image, a unit vector)
    s(u)   = <u, F(u)> = cos(u, F(u))     (self-overlap; here >= 0)

A ReLU ray is a fixed point of F, with g(u) = multiplier.  We test two statements:

  (structural cap) among near-rays, i.e. directions with ||F(u)-u|| <= delta
    (equivalently s(u) >= 1 - delta^2/2), the largest gain stays close to and
    above 2^{-1/2} = 0.70711 by a margin that does not grow with N; and

  (no high-gain rays) a large single-step gain need not be self-parallel: at
    finite N the largest one-step gains come from non-self-parallel directions
    and are upward fluctuations that shrink toward 2^{-1/2} as N grows, so
    sustained gain (a ray) is what is capped, consistent with the no-large-rays
    argument.

Directions are sampled from three families:
  (1) random nonnegative directions (|Gaussian|);
  (2) points along normalized trajectories u <- F(u) (generic to near-ray);
  (3) nonnegative directions from leading eigenvectors of principal submatrices.

Each direction's (gain, self-overlap) is computed once, at generation.

Writes data/experimentB.json and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SQRT_HALF = 1.0 / math.sqrt(2.0)  # 0.70710678...
DELTAS = [0.05, 0.10, 0.20, 0.30]
GAIN_EDGES = [0.0, 0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 2.0]
SELFOV_EDGES = [-1.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]


def relu(z):
    return np.maximum(z, 0.0)


def normalize_rows(U):
    n = np.linalg.norm(U, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return U / n


def step_batch(U, W):
    """One ReLU step for a batch of row-directions U (m x N)."""
    Y = relu(U @ W.T)
    g = np.linalg.norm(Y, axis=1)
    F = Y.copy()
    nz = g > 0.0
    F[nz] = Y[nz] / g[nz, None]
    return g, F


def submatrix_eigen_directions(W, alphas, per_alpha, q_cap, rng):
    """Nonnegative directions from leading eigenvectors of blocks W_{S,S}."""
    N = W.shape[0]
    out = []
    for alpha in alphas:
        q = max(2, min(int(round(alpha * N)), q_cap))
        for _ in range(per_alpha):
            S = rng.choice(N, size=q, replace=False)
            wvals, vecs = np.linalg.eig(W[np.ix_(S, S)])
            v = vecs[:, int(np.argmax(wvals.real))].real
            for cand in (relu(v), relu(-v)):
                nrm = np.linalg.norm(cand)
                if nrm == 0.0:
                    continue
                u = np.zeros(N)
                u[S] = cand / nrm
                out.append(u)
    return np.array(out) if out else np.zeros((0, N))


def collect_stats(W, rng, n_random, n_traj, traj_steps, traj_burn,
                  eig_alphas, eig_per_alpha, eig_qcap):
    """Return (gain, selfov, tag). tag: 0 random, 1 traj-early, 2 traj-late,
    3 submatrix-eigvec."""
    N = W.shape[0]
    gains, selfs, tags = [], [], []

    def record(U, tag_value):
        if U.shape[0] == 0:
            return
        g, F = step_batch(U, W)
        s = np.einsum("ij,ij->i", U, F)
        alive = g > 0.0
        gains.append(g[alive])
        selfs.append(s[alive])
        tags.append(np.full(int(np.count_nonzero(alive)), tag_value, dtype=int))

    record(normalize_rows(np.abs(rng.standard_normal((n_random, N)))), 0)

    U = normalize_rows(np.abs(rng.standard_normal((n_traj, N))))
    for t in range(traj_steps):
        g, F = step_batch(U, W)
        s = np.einsum("ij,ij->i", U, F)
        alive = g > 0.0
        tag_value = 1 if t < traj_burn else 2
        gains.append(g[alive])
        selfs.append(s[alive])
        tags.append(np.full(int(np.count_nonzero(alive)), tag_value, dtype=int))
        U = F

    record(submatrix_eigen_directions(W, eig_alphas, eig_per_alpha,
                                      eig_qcap, rng), 3)
    return (np.concatenate(gains), np.concatenate(selfs), np.concatenate(tags))


def binned(x, y, edges, reducer):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        n = int(np.count_nonzero(mask))
        if n:
            out.append({"lo": lo, "hi": hi, "n": n, **reducer(y[mask])})
    return out


def nearray_table(gain, selfov):
    table = []
    for d in DELTAS:
        thr = 1.0 - d * d / 2.0
        mask = selfov >= thr
        n = int(np.count_nonzero(mask))
        entry = {"delta": d, "thr": thr, "n": n}
        if n:
            entry.update({
                "max": float(np.max(gain[mask])),
                "p99": float(np.quantile(gain[mask], 0.99)),
                "mean": float(np.mean(gain[mask])),
            })
        table.append(entry)
    return table


def scatter_subsample(gain, selfov, tag, rng, cap_random=2500, cap_traj=3500):
    """Stratified subsample so both the random high-gain tail and the near-ray
    arm are represented."""
    idx_rand = np.where(tag == 0)[0]
    idx_traj = np.where((tag == 1) | (tag == 2))[0]
    idx_eig = np.where(tag == 3)[0]
    if idx_rand.size > cap_random:
        idx_rand = rng.choice(idx_rand, cap_random, replace=False)
    if idx_traj.size > cap_traj:
        idx_traj = rng.choice(idx_traj, cap_traj, replace=False)
    keep = np.concatenate([idx_rand, idx_traj, idx_eig])
    return (gain[keep].tolist(), selfov[keep].tolist(), tag[keep].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--scatter-N", type=int, default=1500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    eig_alphas = (0.3, 0.4, 0.5, 0.6)
    plan = ([(1000, 1), (1500, 1)] if args.quick
            else [(1000, 2), (1500, 2), (2000, 2), (3000, 1)])

    print("# Experiment B: one-step gain vs self-parallelism")
    print(f"seed={args.seed}; 2^(-1/2) = {SQRT_HALF:.6f}")

    per_N = {}
    scatter = None
    for N, n_matrices in plan:
        print(f"\n## N = {N}  (matrices={n_matrices})")
        gA, sA, tA = [], [], []
        for _ in range(n_matrices):
            W = rng.standard_normal((N, N)) / math.sqrt(N)
            g, s, t = collect_stats(
                W, rng, n_random=2500, n_traj=200, traj_steps=300, traj_burn=40,
                eig_alphas=eig_alphas, eig_per_alpha=8, eig_qcap=600)
            gA.append(g)
            sA.append(s)
            tA.append(t)
        gain = np.concatenate(gA)
        selfov = np.concatenate(sA)
        tag = np.concatenate(tA)

        corr = float(np.corrcoef(gain, selfov)[0, 1])
        overall_max = float(np.max(gain))
        gb = binned(gain, selfov, GAIN_EDGES,
                    lambda y: {"mean_selfov": float(np.mean(y)),
                               "median_selfov": float(np.median(y))})
        sb = binned(selfov, gain, SELFOV_EDGES,
                    lambda y: {"mean_gain": float(np.mean(y)),
                               "max_gain": float(np.max(y)),
                               "p99_gain": float(np.quantile(y, 0.99))})
        nt = nearray_table(gain, selfov)
        per_N[N] = {"corr": corr, "overall_max": overall_max,
                    "gain_bins": gb, "selfov_bins": sb, "nearray": nt,
                    "samples": int(gain.size)}

        print(f"samples={gain.size}, corr(gain, self-overlap)={corr:+.4f}, "
              f"overall max gain={overall_max:.4f} "
              f"(={overall_max - SQRT_HALF:+.4f} vs 0.7071)")
        print("self-ov bin -> mean/max/p99 gain:")
        for b in sb:
            print(f"  [{b['lo']:.2f},{b['hi']:.2f})  n={b['n']:>6}  "
                  f"mean={b['mean_gain']:.4f}  max={b['max_gain']:.4f}  "
                  f"p99={b['p99_gain']:.4f}")
        print("near-rays (||F(u)-u||<=delta):")
        for e in nt:
            if e["n"]:
                print(f"  delta={e['delta']:.2f} (self-ov>={e['thr']:.4f})  "
                      f"n={e['n']:>6}  max gain={e['max']:.4f}  "
                      f"({e['max']-SQRT_HALF:+.4f})")
            else:
                print(f"  delta={e['delta']:.2f} (self-ov>={e['thr']:.4f})  "
                      f"n=0")

        if N == args.scatter_N:
            scatter = {"N": N}
            gl, sl, tl = scatter_subsample(gain, selfov, tag, rng)
            scatter["gain"], scatter["selfov"], scatter["tag"] = gl, sl, tl

    Ns = [N for N, _ in plan]
    print("\n## max gain over near-rays vs N")
    print("| delta \\ N | " + " | ".join(map(str, Ns)) + " |")
    for d in DELTAS:
        cells = []
        for N in Ns:
            e = next(x for x in per_N[N]["nearray"] if x["delta"] == d)
            cells.append(f"{e['max']:.4f}" if e["n"] else "-")
        print(f"| {d:.2f} | " + " | ".join(cells) + " |")
    print("\n## overall max one-step gain vs N (finite-N fluctuation)")
    print("| N | " + " | ".join(map(str, Ns)) + " |")
    print("| max gain | "
          + " | ".join(f"{per_N[N]['overall_max']:.4f}" for N in Ns) + " |")

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"seed": args.seed, "sqrt_half": SQRT_HALF, "deltas": DELTAS,
               "per_N": {str(N): per_N[N] for N in Ns}, "scatter": scatter}
    with open(os.path.join(DATA_DIR, "experimentB.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved {os.path.join(DATA_DIR, 'experimentB.json')}")


if __name__ == "__main__":
    main()
