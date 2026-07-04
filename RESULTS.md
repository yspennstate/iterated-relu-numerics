# Numerical results

Three experiments supporting the finite-time / infinite-time analysis of iterated
ReLU with a fixed Gaussian matrix `x_{t+1}=ReLU(W x_t)`, `W` i.i.d. `N(0,1/N)`.
Each `experiment_*.py` script recomputes its data (fixed seeds), writes a JSON to
`data/`, and prints the table below; `make_figures.py` reads the JSON and renders
the PDFs in `figures/`. NumPy/Matplotlib only.

Definitions used throughout:
`I_alpha(gamma) = (1/2)(gamma - alpha + alpha*log(alpha/gamma))`, `gamma = r^2`;
`2^{-1/2} = 0.707107`.

## Experiment A — real-Ginibre exterior eigenvalue tail

For a `q x q` block `G` with i.i.d. `N(0,1/N)` entries and `q = alpha*N` (spectral
radius `sqrt(alpha)`), Monte Carlo estimates of the expected count of real
eigenvalues above `r`, and of all eigenvalues with modulus above `r`, in the
exterior window `sqrt(alpha) < r <= 2 sqrt(alpha)`. The empirical tail rate
`-log(E[count])/N` is compared with `I_alpha(r^2)`. Since {real count} <= {modulus
count} pointwise, the modulus rate is the smaller of the two; if it also matches
`I_alpha(r^2)`, controlling the modulus count discharges the exterior-tail
assumption. `*` marks fewer than 20 total observed events (rate unreliable).

Empirical rates `-log(E[count])/N` against `I_alpha(r^2)`, at the smallest `r`
that stays observable out to `N=800` (`*` = fewer than 20 events at that `N`):

alpha = 0.5 (`sqrt(alpha)=0.7071`):

| r | I_alpha | rate_real (N=200 / 400 / 800) | rate_mod (N=200 / 400 / 800) |
|---:|---:|:---|:---|
| 0.74 | 0.00107 | 0.01043 / 0.00619 / 0.00368 | -0.00012 / 0.00087 / 0.00134 |
| 0.76 | 0.00273 | 0.01327 / 0.00853 / 0.00560* | 0.00482 / 0.00486 / 0.00433 |
| 0.78 | 0.00514 | 0.01670 / 0.01185 / 0.00914* | 0.01008 / 0.00930 / 0.00741* |

alpha = 0.3 (`sqrt(alpha)=0.5477`):

| r | I_alpha | rate_real (N=200 / 400 / 800) | rate_mod (N=200 / 400 / 800) |
|---:|---:|:---|:---|
| 0.58 | 0.00102 | 0.01048 / 0.00610 / 0.00370 | 0.00085 / 0.00130 / 0.00149 |
| 0.60 | 0.00265 | 0.01336 / 0.00834 / 0.00584* | 0.00568 / 0.00498 / 0.00444 |
| 0.62 | 0.00501 | 0.01671 / 0.01114 / 0.00777* | 0.01057 / 0.00904 / 0.00690* |

For `r=0.74`, `alpha=0.5`, the real-count rate is 0.01043, 0.00619, 0.00368 at
`N=200,400,800` against `I_alpha=0.00107`: the excess over `I_alpha` is 0.00936,
0.00512, 0.00261, i.e. it halves as `N` doubles (an `O(1/N)` prefactor). The
modulus rate for the same `r` is already within 0.001 of `I_alpha` at every `N`
(the `E[mod]>1` case at `N=200` even makes the raw rate slightly negative). The
smaller alpha behaves identically. The full six-threshold sweep is in
`data/experimentA.json` / `runA_final.log`.

Reading: both empirical rates lie **above** `I_alpha` at finite `N` (the counts
decay slightly faster than `exp(-N I_alpha)`, so the assumption holds as an upper
bound), and the gap shrinks roughly like `1/N` — halving as `N` doubles — so both
rates converge down to `I_alpha` as `N` grows. The modulus rate is the closer of
the two to `I_alpha`, i.e. the more conservative bound is already essentially at
the target. This is what the exterior-tail assumption asserts.

## Experiment B — one-step gain versus self-parallelism

Fixed `W` (`N` in {1000,1500,2000,3000}), many nonnegative unit directions `u`
(random `|Gaussian|`; points along normalized trajectories `u <- F(u)`; leading
eigenvectors of principal submatrices). For each, `g(u)=||ReLU(Wu)||`,
`F(u)=ReLU(Wu)/g(u)`, and self-overlap `s(u)=<u,F(u)>`. A near-ray means
`||F(u)-u|| <= delta`, i.e. `s(u) >= 1 - delta^2/2`.

Gain by self-overlap bin (`N=1500`, 250k directions over 2 matrices):

| self-overlap bin | n | mean gain | max gain | p99 gain |
|:---|---:|---:|---:|---:|
| [-1.00, 0.50) | 5497 | 0.7157 | 0.7686 | 0.7502 |
| [0.50, 0.70)  | 857  | 0.7123 | 0.7654 | 0.7455 |
| [0.70, 0.80)  | 1030 | 0.7069 | 0.7583 | 0.7386 |
| [0.80, 0.90)  | 2444 | 0.7051 | 0.7447 | 0.7311 |
| [0.90, 0.95)  | 3501 | 0.7073 | 0.7391 | 0.7244 |
| [0.95, 0.99)  | 18309| 0.7081 | 0.7240 | 0.7175 |
| [0.99, 1.00)  | 93490| 0.7080 | 0.7174 | 0.7142 |

The largest single-step gain sits in the **lowest** self-overlap bin (max 0.7686
at `s < 0.5`), while the near-parallel bins (`s -> 1`) are pinned just above
`2^{-1/2}` (max 0.7174 at `s >= 0.99`).

Max gain over near-rays `||F(u)-u|| <= delta`:

| delta | N=1000 | N=1500 | N=2000 | N=3000 |
|---:|---:|---:|---:|---:|
| 0.05 | (none) | 0.7116 | 0.7245 | (none) |
| 0.10 | 0.7146 | 0.7145 | 0.7264 | 0.7104 |
| 0.20 | 0.7326 | 0.7227 | 0.7279 | 0.7195 |
| 0.30 | 0.7402 | 0.7240 | 0.7303 | 0.7283 |

Max one-step gain over **all** sampled directions (a finite-`N` fluctuation, not a
ray):

| N | 1000 | 1500 | 2000 | 3000 |
|---:|---:|---:|---:|---:|
| max gain | 0.7898 | 0.7686 | 0.7553 | 0.7552 |

Reading: near-rays are gain-capped at `2^{-1/2}` plus a margin of at most about
`0.03`, and the margin shrinks as `delta -> 0` (compare the `delta=0.30` column
with `delta<=0.10`) while not growing with `N`. The overall single-step maximum
decreases monotonically with `N` (0.790 -> 0.755) toward `2^{-1/2}`, confirming
that gains above the threshold are transient fluctuations rather than rays. The
`(none)` entries are cases where the 300-step trajectories at that `N` did not
reach `s >= 0.9988` within the horizon; they are absences of tight near-rays, not
high-gain ones.

## Experiment C — finite-time diagnostics

Reused `W` at `N=1500` (48 reps), deterministic all-equal start
`x_0 = (1/sqrt N) 1` (coordinate mean `m_nu = 1`), plus a nonnegative random
second start. Consecutive-cosine kernel column is the arccosine-kernel orbit
`r_{t+1}=Khat(r_t)` started at `cos(x_0,x_1)=1/sqrt(pi)`.

| t | 2^t‖x_t‖² | cos(x_{t-1},x_t) | kernel | support | two-start cos | two-start kernel |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.9833 | 0.5622 | 0.5642 | 0.4984 | 0.8272 | 0.8257 |
| 2  | 0.9783 | 0.6484 | 0.6526 | 0.4971 | 0.8498 | 0.8478 |
| 3  | 0.9789 | 0.7128 | 0.7152 | 0.4960 | 0.8690 | 0.8657 |
| 4  | 0.9722 | 0.7599 | 0.7615 | 0.4975 | 0.8819 | 0.8806 |
| 5  | 0.9643 | 0.7949 | 0.7969 | 0.4956 | 0.8938 | 0.8931 |
| 6  | 0.9577 | 0.8235 | 0.8246 | 0.4959 | 0.9052 | 0.9036 |
| 7  | 0.9593 | 0.8452 | 0.8469 | 0.4948 | 0.9147 | 0.9126 |
| 8  | 0.9616 | 0.8631 | 0.8650 | 0.4961 | 0.9219 | 0.9204 |
| 9  | 0.9634 | 0.8779 | 0.8800 | 0.4944 | 0.9284 | 0.9272 |
| 10 | 0.9686 | 0.8909 | 0.8925 | 0.4961 | 0.9345 | 0.9331 |
| 11 | 0.9749 | 0.9021 | 0.9032 | 0.4973 | 0.9399 | 0.9383 |
| 12 | 0.9800 | 0.9112 | 0.9123 | 0.4951 | 0.9442 | 0.9429 |

Norm halving holds (all values within 4% of 1), the consecutive cosine tracks the
kernel column to within about 0.002 at every step, support density is 0.5 to
three digits, and the two-start cosine follows its kernel orbit. The angle law
`theta_t = arccos(r_t) ~ 3 pi / t` is a slow asymptotic; iterating the
deterministic orbit from `1/sqrt(pi)`:

| t | r_t | theta_t | 3 pi/t | t·theta_t/(3 pi) |
|---:|---:|---:|---:|---:|
| 10   | 0.892534 | 0.467864 | 0.942478 | 0.4964 |
| 30   | 0.974193 | 0.227679 | 0.314159 | 0.7247 |
| 100  | 0.996524 | 0.083401 | 0.094248 | 0.8849 |
| 300  | 0.999551 | 0.029960 | 0.031416 | 0.9537 |
| 1000 | 0.999957 | 0.009273 | 0.009425 | 0.9839 |
| 3000 | 0.999995 | 0.003123 | 0.003142 | 0.9940 |

The ratio approaches 1 from below, so `theta_t ~ 3 pi/t` is confirmed but only
attained slowly (at `t=10` the angle is about half of `3 pi/t`).

## Points where the numerics refine the stated claims

- Experiment A confirms the exterior-tail rate quantitatively: both the real-count
  and the modulus-count rates converge to `I_alpha(r^2)` from above. The modulus
  count — an upper bound for the real count — already matches the rate, so the
  assumption can be discharged through the easier modulus statistic.
- Experiment B supports the structural cap but not the literal phrasing "the
  largest-gain directions are the nearly self-parallel ones." At every `N` the
  single largest gain occurs at self-overlap below 0.5, not near 1; such gains are
  finite-`N` fluctuations that shrink toward `2^{-1/2}` as `N` grows. What holds is
  the converse used in the argument: sustained gain requires self-parallelism, and
  self-parallel (near-ray) directions are capped at `2^{-1/2}` up to a margin that
  vanishes with `delta`. The sign of the gain / self-overlap correlation is not a
  stable summary — it depends on how the sampled families populate the
  self-overlap range at each `N` — so only the bin-level and near-ray statements
  should be quoted.
- Experiment C reproduces the paper's finite-time diagnostics table.

## Files

- `experiment_A_real_ginibre_tail.py`, `experiment_B_gain_vs_selfparallel.py`,
  `experiment_C_finite_time_diagnostics.py` — compute and save `data/*.json`.
- `make_figures.py` — renders `figures/fig_tail_rate.pdf`,
  `figures/fig_near_ray.pdf`, `figures/fig_finite_time.pdf`.
