"""Experiment C: finite-time diagnostics for fixed-W iterated ReLU at N ~ 1500.

For one reused Gaussian matrix W (N x N, i.i.d. N(0,1/N)), a deterministic
all-equal start x_0 = (1/sqrt N) 1 (coordinate mean m_nu = 1), and a nonnegative
random second start y_0, we reproduce the finite-time state-evolution
diagnostics:

  * norm halving        2^t ||x_t||^2 -> 1
  * consecutive cosine  cos(x_{t-1}, x_t) vs the arccosine-kernel orbit
                        r_{t+1} = Khat(r_t) started at cos(x_0,x_1) = 1/sqrt(pi)
  * support density     |S_t|/N -> 1/2
  * two-start cosine    cos(x_t, y_t) vs the kernel orbit from cos(x_0, y_0)

and verify the kernel-orbit angle law theta_t = arccos(r_t) ~ 3 pi / t
deterministically out to large t.

Writes data/experimentC.json and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def khat(rho: float) -> float:
    rho = min(1.0, max(-1.0, rho))
    return (math.sqrt(max(0.0, 1.0 - rho * rho))
            + rho * (math.pi - math.acos(rho))) / math.pi


def cosine(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else float("nan")


def run(n, steps, reps, seed):
    rng = np.random.default_rng(seed)
    stats = np.zeros((steps, 6))
    for _ in range(reps):
        w = rng.standard_normal((n, n)) / math.sqrt(n)
        x = np.full(n, 1.0 / math.sqrt(n))     # deterministic all-equal start
        y = np.abs(rng.standard_normal(n))     # nonnegative random second start
        y /= np.linalg.norm(y)
        rho_cons = 1.0 / math.sqrt(math.pi)    # kernel value of cos(x_0, x_1)
        rho_pair = cosine(x, y)
        for t in range(steps):
            zx = w @ x
            x_new = np.maximum(zx, 0.0)
            y_new = np.maximum(w @ y, 0.0)
            supp = zx > 0.0
            rho_pair = khat(rho_pair)
            row = stats[t]
            row[0] += (2.0 ** (t + 1)) * float(x_new @ x_new)
            row[1] += cosine(x, x_new)
            row[2] += rho_cons
            row[3] += float(np.mean(supp))
            row[4] += cosine(x_new, y_new)
            row[5] += rho_pair
            rho_cons = khat(rho_cons)
            x, y = x_new, y_new
    stats /= reps

    diag = []
    for t in range(steps):
        r = stats[t]
        diag.append({
            "t": t + 1, "norm2": r[0], "cos": r[1], "kernel": r[2],
            "support": r[3], "twostart_cos": r[4], "twostart_kernel": r[5],
            "angle_obs": math.acos(min(1.0, max(-1.0, r[1]))),
            "angle_kernel": math.acos(min(1.0, max(-1.0, r[2]))),
            "three_pi_over_t": 3.0 * math.pi / (t + 1),
        })

    print(f"\n## N={n}, reps={reps}, seed={seed}")
    print("| t | 2^t||x_t||^2 | cos(x_{t-1},x_t) | kernel | support | "
          "two-start cos | two-start kernel | angle_obs | 3pi/t |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in diag:
        print(f"| {d['t']} | {d['norm2']:.4f} | {d['cos']:.4f} | "
              f"{d['kernel']:.4f} | {d['support']:.4f} | "
              f"{d['twostart_cos']:.4f} | {d['twostart_kernel']:.4f} | "
              f"{d['angle_obs']:.4f} | {d['three_pi_over_t']:.4f} |")
    return diag


def kernel_orbit(t_max=4000, marks=(10, 30, 100, 300, 1000, 3000)):
    """Deterministic kernel orbit from cos=1/sqrt(pi); theta_t ~ 3 pi/t."""
    r = 1.0 / math.sqrt(math.pi)
    out = []
    markset = set(marks)
    print("\n## kernel-orbit angle law: theta_t ~ 3 pi / t")
    print("| t | r_t | theta_t | 3 pi/t | t*theta_t/(3 pi) |")
    for t in range(1, t_max + 1):
        if t in markset:
            theta = math.acos(min(1.0, max(-1.0, r)))
            ratio = t * theta / (3.0 * math.pi)
            out.append({"t": t, "r": r, "theta": theta,
                        "three_pi_over_t": 3.0 * math.pi / t, "ratio": ratio})
            print(f"| {t} | {r:.6f} | {theta:.6f} | {3.0*math.pi/t:.6f} | "
                  f"{ratio:.4f} |")
        r = khat(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--reps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("# Experiment C: finite-time diagnostics")
    print("Paper reference (N=1200): t=1 (0.976, cos 0.559, ker 0.564, "
          "supp 0.492, 2s 0.825); t=4 (0.966, 0.762, 0.762, 0.502, 0.880); "
          "t=7 (0.923, 0.831, 0.847, 0.486, 0.913); "
          "t=10 (0.863, 0.875, 0.893, 0.496, 0.933).")
    diag = run(args.N, args.steps, args.reps, args.seed)
    orbit = kernel_orbit()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "experimentC.json"), "w") as fh:
        json.dump({"N": args.N, "reps": args.reps, "seed": args.seed,
                   "diag": diag, "orbit": orbit}, fh, indent=2)
    print(f"\nsaved {os.path.join(DATA_DIR, 'experimentC.json')}")


if __name__ == "__main__":
    main()
