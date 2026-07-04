# Numerical experiments for iterated ReLU maps with a fixed Gaussian matrix

Code and figures accompanying the paper *Finite-horizon decay of iterated ReLU maps with a
fixed Gaussian matrix: state evolution, parallel paths, and the remaining infinite-time
obstruction*.

The map is `x_{t+1} = ReLU(W x_t)` with a single `N x N` matrix `W` of independent
`N(0, 1/N)` entries reused at every step. The scripts reproduce the paper's diagnostics and
the two experiments that support its analytic claims.

## Experiments

- `experiment_A_real_ginibre_tail.py` — estimates the expected number of eigenvalues of the
  `alpha N` principal Gaussian block with modulus (and, separately, real part) exceeding `r`, and
  compares the empirical exponential rate `-(1/N) log E[count]` with
  `I_alpha(r^2) = (r^2 - alpha + alpha log(alpha / r^2)) / 2`. This is the rate that makes the
  paper's no-large-ray theorem unconditional.
- `experiment_B_gain_vs_selfparallel.py` — samples nonnegative directions and measures the
  one-step gain `g(u) = ||ReLU(W u)||` against the self-overlap `cos(u, F(u))`, and records the
  largest gain over near-fixed directions as `N` grows. It illustrates that sustained gain is
  capped near `2^{-1/2}` while an isolated large one-step gain need not be self-parallel.
- `experiment_C_finite_time_diagnostics.py` — checks the finite-horizon state evolution: the
  norm halving `2^t ||x_t||^2 -> 1`, the activation density `-> 1/2`, the consecutive-step cosine
  against the arccosine-kernel orbit, and the angle law `theta_t ~ 3 pi / t`.
- `uniform_block_gain.py` — searches for the worst-case direction of the `m`-step map by a
  sign-feasible power iteration and reports the block gain `a_m = sup_{||x||=1} ||f^m(x)||`. It
  finds `a_m < 1` from about `m = 6`, evidence for uniform (all-input) finite-block contraction;
  `make_uniform_figure.py` plots it from `uniform_block_gain_snap.txt`.
- `make_figures.py` — regenerates the figures in `figures/` from the JSON in `data/`.

Each experiment writes a JSON summary to `data/` and prints a table; `RESULTS.md` records the
tables from a reference run.

## Usage

```
pip install -r requirements.txt
python experiment_A_real_ginibre_tail.py
python experiment_B_gain_vs_selfparallel.py
python experiment_C_finite_time_diagnostics.py
python make_figures.py
```

All scripts use fixed seeds and depend only on NumPy and Matplotlib.

## Figures

`figures/` holds the vector PDFs used in the paper: the finite-time diagnostics, the tail-rate
comparison against `I_alpha`, and the near-ray gain cap.
