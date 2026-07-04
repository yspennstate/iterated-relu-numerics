"""Estimate a_m = sup_{||x||=1} ||f^m(x)|| for f(x)=ReLU(Wx), W iid N(0,1/N) fixed.

If a_m < 1 for some finite m (whp over W), Fekete gives uniform decay for every input --
the worst-case (all-input) infinite-time statement. We search for the worst-case direction
by a sign-feasible power iteration: fix the activation masks of the current x's m-step path,
so f^m is the linear block M = D_m W ... D_1 W (D_t = diag of masks); take the top right
singular vector of M by power iteration (applying M and M^T as sequences of masked matvecs,
never forming M); iterate until the masks stabilize; keep the best over random restarts.
This is a LOWER bound on a_m (local maxima), so best gain^(1/m) < 1 is strong evidence.
"""
import numpy as np

def forward_masks(W, x, m):
    """Run the ReLU map m steps from x; return the list of activation masks (float 0/1)."""
    masks = []
    for _ in range(m):
        z = W @ x
        mk = (z > 0).astype(np.float64)
        masks.append(mk)
        x = mk * z
        n = np.linalg.norm(x)
        if n == 0:
            return masks, False
        x = x / n
    return masks, True

def apply_M(W, masks, v):
    for mk in masks:
        v = mk * (W @ v)
    return v

def apply_MT(W, masks, u):
    for mk in reversed(masks):
        u = W.T @ (mk * u)
    return u

def worst_gain(W, m, restarts, iters, rng):
    N = W.shape[0]
    best, best_so = 0.0, float('nan')
    for _ in range(restarts):
        x = rng.standard_normal(N); x /= np.linalg.norm(x)
        for _ in range(iters):
            masks, ok = forward_masks(W, x, m)
            if not ok:
                break
            v = x.copy()
            for _ in range(40):                      # power iteration on M^T M
                v = apply_MT(W, masks, apply_M(W, masks, v))
                nv = np.linalg.norm(v)
                if nv == 0:
                    break
                v /= nv
            gain = np.linalg.norm(apply_M(W, masks, v))
            x = v
            if gain > best:
                best = gain
                fx = np.maximum(W @ v, 0.0); nf = np.linalg.norm(fx)
                best_so = float(v @ (fx / nf)) if nf > 0 else float('nan')
    return best, best_so

def main():
    rng = np.random.default_rng(7)
    print(f"{'N':>5} {'m':>3} {'best_gain':>10} {'gain^(1/m)':>11} {'self-ovlp':>10}", flush=True)
    for N in (400, 800, 1500):
        W = rng.standard_normal((N, N)) / np.sqrt(N)
        for m in (1, 2, 4, 6, 8, 10, 14, 20):
            g, so = worst_gain(W, m, restarts=30, iters=6, rng=rng)
            print(f"{N:5d} {m:3d} {g:10.4f} {g**(1.0/m):11.4f} {so:10.4f}", flush=True)

if __name__ == "__main__":
    main()
