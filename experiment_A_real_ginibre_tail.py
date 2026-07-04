"""Experiment A: real-Ginibre exterior eigenvalue tail (real-count and modulus).

For a q x q real Ginibre block G with i.i.d. N(0, 1/N) entries and q = alpha*N,
the spectral radius is sqrt(alpha).  In the exterior range sqrt(alpha) < r <=
2 sqrt(alpha) we estimate, by Monte Carlo over many independent blocks, two
expected counts:

    E[ #{real eigenvalues with value  > r} ]        (real-eigenvalue count)
    E[ #{eigenvalues with modulus |lambda| > r} ]   (modulus count; conjugate
                                                     pairs counted twice)

and compare the empirical tail rates  -log(E[count]) / N  against the claimed
large-deviation rate

    I_alpha(gamma) = 0.5 * (gamma - alpha + alpha*log(alpha/gamma)),   gamma = r^2.

Because {real count} <= {modulus count} pointwise, the modulus rate is the
smaller (more conservative) of the two; if it too matches I_alpha(r^2), then
controlling the modulus count discharges the real-Ginibre exterior tail
assumption used in the no-large-rays argument.  As N grows both empirical rates
approach I_alpha from above (an O(log N)/N prefactor correction).

Real eigenvalues of a real matrix are returned by LAPACK dgeev with imaginary
part exactly zero; we use a tiny tolerance for safety.

Writes data/experimentA.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def I_alpha(alpha: float, gamma: float) -> float:
    if gamma <= alpha:
        return 0.0
    return 0.5 * (gamma - alpha + alpha * math.log(alpha / gamma))


def rate(Ecount: float, N: int) -> float:
    return -math.log(Ecount) / N if Ecount > 0 else float("nan")


def run_case(N, alpha, r_values, M, tol, rng):
    q = int(round(alpha * N))
    scale = 1.0 / math.sqrt(N)
    nr = len(r_values)
    real_counts = np.zeros((M, nr))
    mod_counts = np.zeros((M, nr))
    t0 = time.time()
    for m in range(M):
        G = rng.standard_normal((q, q)) * scale
        w = np.linalg.eigvals(G)
        re = w.real[np.abs(w.imag) <= tol]
        mod = np.abs(w)
        for j, r in enumerate(r_values):
            real_counts[m, j] = np.count_nonzero(re > r)
            mod_counts[m, j] = np.count_nonzero(mod > r)
    elapsed = time.time() - t0

    rows = []
    for j, r in enumerate(r_values):
        cr, cm = real_counts[:, j], mod_counts[:, j]
        Er, Em = float(np.mean(cr)), float(np.mean(cm))
        rows.append({
            "alpha": alpha, "N": N, "q": q, "M": M, "r": r, "gamma": r * r,
            "I_alpha": I_alpha(alpha, r * r),
            "E_real": Er, "tot_real": int(np.sum(cr)), "rate_real": rate(Er, N),
            "E_mod": Em, "tot_mod": int(np.sum(cm)), "rate_mod": rate(Em, N),
        })
    print(f"\n### N={N}, alpha={alpha}, q={q}, M={M}, elapsed={elapsed:.1f}s")
    print("| r | gamma | I_alpha | E[real] | rate_real | real-I | "
          "E[mod] | rate_mod | mod-I | tot_real | tot_mod |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def f(x):
        return f"{x:.5f}" if x == x else "  -  "

    for row in rows:
        dr = row["rate_real"] - row["I_alpha"]
        dm = row["rate_mod"] - row["I_alpha"]
        fr = "" if row["tot_real"] >= 20 else "*"
        fm = "" if row["tot_mod"] >= 20 else "*"
        print(f"| {row['r']:.3f} | {row['gamma']:.4f} | {row['I_alpha']:.5f} | "
              f"{row['E_real']:.3e} | {f(row['rate_real'])} | {f(dr)} | "
              f"{row['E_mod']:.3e} | {f(row['rate_mod'])} | {f(dm)} | "
              f"{row['tot_real']}{fr} | {row['tot_mod']}{fm} |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plan = [
        (0.3, [0.58, 0.60, 0.62, 0.64, 0.66, 0.70]),
        (0.5, [0.74, 0.76, 0.78, 0.80, 0.82, 0.85]),
    ]
    if args.quick:
        M_by_N, Ns = {200: 4000, 400: 2000, 800: 800}, [200, 400]
    else:
        M_by_N, Ns = {200: 30000, 400: 8000, 800: 1500}, [200, 400, 800]

    print("# Experiment A: real-Ginibre exterior eigenvalue tail")
    print(f"seed={args.seed}, tol={args.tol}")
    print("I_alpha(gamma) = 0.5*(gamma - alpha + alpha*log(alpha/gamma)), "
          "gamma=r^2; '*' = fewer than 20 total events (rate unreliable).")

    all_rows = []
    for alpha, r_values in plan:
        print(f"\n## alpha = {alpha}  (sqrt(alpha)={math.sqrt(alpha):.4f}, "
              f"2 sqrt(alpha)={2*math.sqrt(alpha):.4f})")
        for N in Ns:
            all_rows.extend(run_case(N, alpha, r_values, M_by_N[N],
                                     args.tol, rng))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "experimentA.json"), "w") as fh:
        json.dump({"seed": args.seed, "tol": args.tol, "rows": all_rows}, fh,
                  indent=2)
    print(f"\nsaved {os.path.join(DATA_DIR, 'experimentA.json')}")


if __name__ == "__main__":
    main()
