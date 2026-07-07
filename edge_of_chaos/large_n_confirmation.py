"""Large-N confirmation for the edge-of-chaos paper, run on the Azure box.

Self-contained, numpy only (no scipy). For tanh with a bias, at N up to 8192:
 (1) tied vs independent weights give the same finite-horizon cosine trajectory,
     gap ~ N^{-1/2} (the innovation-miracle claim, Theorem: tied=independent);
 (2) the finite-N length and cosine track the deterministic mean-field maps,
     gap shrinking with N (the e^{-cN} concentration).

Mean-field maps via numpy Gauss-Hermite (numpy.polynomial.hermite.hermgauss).
Kept within the 4-core / 8G aiwork slice: one N at a time, modest seed count.
"""
import argparse
import json
import math
import time

import numpy as np

SW2, SB2 = 1.5, 0.05
GX, GW = np.polynomial.hermite.hermgauss(120)
GZ = math.sqrt(2.0) * GX
GP = GW / math.sqrt(math.pi)


def v_map(q):
    s = math.sqrt(SW2 * q + SB2)
    return float(GP @ np.tanh(s * GZ) ** 2)


def fixed_point(q=1.0):
    for _ in range(400):
        qn = v_map(q)
        if abs(qn - q) < 1e-13:
            return qn
        q = qn
    return q


def corr_next(q, c):
    qhat = SW2 * q + SB2
    rho = min(1.0, (SW2 * q * c + SB2) / qhat)
    s = math.sqrt(qhat)
    r = math.sqrt(max(0.0, 1.0 - rho * rho))
    u1 = s * GZ[:, None]
    u2 = s * (rho * GZ[:, None] + r * GZ[None, :])
    m = float(GP @ (np.tanh(u1) * np.tanh(u2)) @ GP)
    return v_map(q), m / v_map(q)


def mf_orbit(q0, c0, T):
    q, c = q0, c0
    qs, cs = [q0], [c0]
    for _ in range(T):
        q, c = corr_next(q, c)
        qs.append(q)
        cs.append(c)
    return np.array(qs), np.array(cs)


def make_starts(rng, n, q0, c0):
    xh = rng.standard_normal(n); xh /= np.linalg.norm(xh)
    y = rng.standard_normal(n); y -= (y @ xh) * xh; yh = y / np.linalg.norm(y)
    r = math.sqrt(n * q0)
    return r * xh, r * (c0 * xh + math.sqrt(1 - c0 * c0) * yh)


def run_pair(x0, y0, T, sw, sb, rng, tied):
    n = x0.size
    x, y = x0.copy(), y0.copy()
    c = np.empty(T + 1); q = np.empty(T + 1)
    def rec(t):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        q[t] = nx * nx / n
        c[t] = float(x @ y / (nx * ny))
    rec(0)
    if tied:
        W = sw * rng.standard_normal((n, n)); b = sb * rng.standard_normal(n)
    for t in range(1, T + 1):
        if not tied:
            W = sw * rng.standard_normal((n, n)); b = sb * rng.standard_normal(n)
        x = np.tanh(W @ x + b); y = np.tanh(W @ y + b)
        rec(t)
    return q, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--Ns", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()

    qstar = fixed_point()
    c0 = 0.3
    q_th, c_th = mf_orbit(qstar, c0, args.T)
    print(f"# Azure large-N (tanh sw2={SW2} sb2={SB2}) q*={qstar:.4f} "
          f"reps={args.reps} T={args.T}")
    print("| N | max|c_tied-c_indep| | max|c_sim-c_mf| | max|q_sim-q_mf| |")
    print("|---:|---:|---:|---:|")
    out = {"qstar": qstar, "T": args.T, "reps": args.reps, "rows": []}
    for N in args.Ns:
        ct = np.zeros(args.T + 1); ci = np.zeros(args.T + 1)
        qt = np.zeros(args.T + 1)
        sw = math.sqrt(SW2 / N); sb = math.sqrt(SB2)
        for r in range(args.reps):
            rng0 = np.random.default_rng([args.seed, r, 0])
            x0, y0 = make_starts(rng0, N, qstar, c0)
            rng = np.random.default_rng([args.seed, r, 1])
            q, c = run_pair(x0, y0, args.T, sw, sb, rng, True); ct += c; qt += q
            rng = np.random.default_rng([args.seed, r, 1])
            _, c = run_pair(x0, y0, args.T, sw, sb, rng, False); ci += c
        ct /= args.reps; ci /= args.reps; qt /= args.reps
        dti = float(np.max(np.abs(ct - ci)))
        dcm = float(np.max(np.abs(ct - c_th)))
        dqm = float(np.max(np.abs(qt - q_th)))
        out["rows"].append({"N": N, "tied_indep": dti, "cos_mf": dcm,
                            "q_mf": dqm})
        print(f"| {N} | {dti:.5f} | {dcm:.5f} | {dqm:.5f} |")
        print(f"  ... N={N} done, elapsed {time.time()-t0:.0f}s", flush=True)
    with open("azure_largeN_result.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nsaved azure_largeN_result.json  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
