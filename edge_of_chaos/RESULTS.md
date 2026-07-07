# Reference results

Key numbers from a reference run of each experiment (fixed seeds). Reproduce with the
commands in `README.md`; each script also prints its full tables and writes JSON to `data/`.

## E1 — state evolution and edge-of-chaos rates

Fixed points and multipliers match the closed forms: ReLU at `sw^2=2` gives `q*=1`, `chi1=1`;
tanh moves from ordered (`chi1=0.94` at `sw^2=1.5`) to chaotic (`chi1=1.21` at `sw^2=3`).

- finite-width tracking (tanh, depth 14): max cosine gap `0.040, 0.010, 0.007, 0.004` at
  `N = 250, 500, 1000, 2000` — the `N^{-1/2}` fluctuation.
- ordered collapse geometric: measured rate `0.5733` vs `chi1=0.5733` (erf, `sw^2=1`).
- critical collapse (ReLU): `1-c_t ~ t^{-1.97}` (theory `-2`), angle ratio
  `t*theta_t/(3pi) -> 0.994` at `t=3000` (the `theta_t ~ 3pi/t` law).
- one-step surrogate: for ReLU (`chi1_angle=1`) it predicts inputs never align while the true
  orbit reaches cosine 0.9 by depth 13; for near-critical tanh it mis-estimates the
  depth-to-align by 40% (25 layers vs 35).

## E2 — tied weights equal fresh weights

Reusing one matrix and drawing a fresh matrix per layer give the same depth-12 trajectory:
max cosine gap for tanh `0.037 -> 0.012` over `N = 250 -> 2000`, for gelu near `0.004`
throughout. A long ReLU run (`N=1500`) agrees to within `0.02` in cosine through 50 layers,
the tied trajectory aligning marginally more slowly at the largest depths.

## E3 — bias-tuned edge of chaos (tanh)

The `chi1=1` curve is solved in the `(sw^2, sb^2)` plane; at `sb^2=0.05` the critical
`sw^2=1.761`. Three regimes:

- ordered (`sw^2=1.32`, `chi1=0.887`): geometric rate `0.884` vs `chi1=0.887`, `xi_c=8.3`.
- critical (`sw^2=1.761`, `chi1=1.000`): `1-c_t ~ t^{-0.94}` — the smooth-activation power law
  `t^{-1}`, one power slower than the ReLU kink's `t^{-2}`.
- chaotic (`sw^2=2.82`, `chi1=1.18`): `c_t -> c* = 0.32 < 1` (inputs decorrelate).

## E4 — convolutional signal propagation

A random circular CNN (`C` channels, `L=32`, `k=5`). The two-input correlation tracks the
mean-field recursion (`0.320` vs predicted `0.318` after one layer), and the within-image
patch correlation coincides with it, so all spatial patches become parallel in the ordered
phase. An ordered configuration (`chi1=1/2`, with bias) drives `1-cbar` to `1e-9` in 30
layers geometrically; the angle-critical configuration reaches only `0.04`, polynomially.
Finite-channel concentration: std of the two-input correlation falls `0.069 -> 0.025` over
`C = 32 -> 256` (fitted slope `-0.44`), mean-to-MF gap `4e-5` at `C=256`. The tanh critical
scale solved from `chi1=1` is `sw^2=1.761` at `sb^2=0.05`.

## E5 — self-attention rank collapse

Pure attention drives all tokens to a common vector. In the value-free form (the Dobrushin
object) the mean token cosine reaches one and the stable rank falls to one within 2--3
layers, and the deviation `R(X^t)` follows the doubly-exponential rate (fitted step exponent
near 3). The Dobrushin oscillation bound `osc(X^t) <= prod delta(A^s) osc(X^0)` holds with no
violations. Skip connections, then a feed-forward block, then pre-layer-normalization
progressively arrest the collapse: with pre-normalization the tokens never reach `1e-3` of
parallel over 40 layers, while pure attention gets there in two.

## E6 — residual depth scaling

Residual blocks `x + beta phi(W x)` with a centered activation (tanh). The identity branch
pins `chi1^res = 1 + O(beta^2)`, so the cosine evolves on the depth scale `1/beta^2`
(Yang-Schoenholz, resnets on the edge of chaos). The `c_t` curves for
`beta in {0.15, 0.2, 0.3, 0.5}` fall onto one profile under `t -> beta^2 t`, mean absolute
spread `9e-4`; the linearized multiplier has `|1-m|/beta^2 ~ 0.013` at the smallest beta; a
finite-width run at `N=400` matches the mean-field orbit to `0.03` in cosine. The length
grows (`1 -> 8.3` over 120 layers) while a plain ReLU network's halves. The direction of the
drift depends on the configuration (random tanh residual nets decorrelate slowly, keeping
inputs distinguishable), but the depth scale is always `1/beta^2`, against the `O(1)` scale
of a plain ordered network (geometric rate `0.885`).

## E7 — activation smoothness sets the critical exponent

Five activations placed exactly at criticality (`chi1=1`), deterministic cosine maps. Kinked
activations approach parallel as `1-c_t ~ t^{-2}` (ReLU `-1.97`, leaky `-1.96`); smooth ones
as `t^{-1}` (erf `-1.00`, tanh `-1.01`, sin `-0.99`). The angle follows `t^{-1}` and
`t^{-1/2}`. All five share the same multiplier `chi1=1`, so the exponent is set by the local
geometry of the map at `c=1` (analytic for smooth phi, non-analytic arccosine term for a
kink), not by the multiplier.
