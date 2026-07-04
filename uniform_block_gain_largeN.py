"""Larger-N confirmation that a_m = sup ||f^m(x)|| < 1 at finite m, and that the crossover
is stable (does not drift up) as N grows. Focused on the crossover band m in {5,6,8,10,12}.
Same sign-feasible power iteration as uniform_block_gain.py."""
import numpy as np
from uniform_block_gain import worst_gain

def main():
    rng = np.random.default_rng(11)
    print(f"{'N':>5} {'m':>3} {'a_m':>8} {'a_m^(1/m)':>10}", flush=True)
    for N in (1000, 1500, 2200):
        W = rng.standard_normal((N, N)) / np.sqrt(N)
        for m in (5, 6, 8, 10, 12):
            g, _ = worst_gain(W, m, restarts=36, iters=6, rng=rng)
            print(f"{N:5d} {m:3d} {g:8.4f} {g**(1.0/m):10.4f}", flush=True)

if __name__ == "__main__":
    main()
