# Edge-of-chaos experiments

Numerics for the multi-step signal-propagation results: general activations, biases, tied
versus independent weights, convolutions, self-attention, and residual connections. These
extend the iterated-ReLU experiments in the parent directory to the edge-of-chaos setting,
where the object of interest is the whole depth trajectory of the length and correlation
maps rather than a single map's fixed point.

`eoc_common.py` holds the shared pieces: an activation library (ReLU, leaky ReLU, linear,
tanh, erf, GELU, Swish, sin) with derivatives; Gauss--Hermite quadrature for the length map
`V`, the correlation map `C`, and the multiplier `chi1`; and closed forms for the
activations that have them. Run `python eoc_common.py` for the quadrature self-test. The
self-test also records that Gauss--Hermite is only about `1e-3` accurate on the ReLU kink and
wrong on the step derivative, which is why the ReLU family is routed through closed forms.

## Experiments

- `exp_e1_state_evolution.py` — fixed points and multipliers for each activation; the
  finite-width length and cosine trajectories tracking the deterministic maps, with the gap
  shrinking as `N^{-1/2}`; the collapse rate (geometric off criticality, `t^{-2}` for the
  ReLU kink at criticality with `theta_t ~ 3 pi / t`); and the one-step surrogate failing in
  the transient.
- `exp_e2_tied_vs_independent.py` — reusing one matrix versus a fresh matrix per layer gives
  the same finite-horizon state evolution; the gap is `O(N^{-1/2})`, and a long run shows the
  small tied departure at large depth.
- `exp_e3_bias_criticality.py` — the `chi1 = 1` curve for tanh in the `(sw^2, sb^2)` plane,
  and simulations in the ordered, critical (`1-c_t ~ t^{-1}` for a smooth activation) and
  chaotic phases.
- `exp_e4_cnn_signal_propagation.py` — a random circular CNN: per-position length and
  two-point correlations following the mean-field recursion, all spatial patches becoming
  parallel in the ordered phase, and the reduced order parameters concentrating as the
  channel count grows.
- `exp_e5_attention_rank_collapse.py` — a stack of self-attention layers: token uniformity
  (rank collapse) under pure attention, doubly exponential in depth, the Dobrushin
  oscillation bound, and the effect of skip connections, feed-forward blocks and layer
  normalization.
- `exp_e6_resnet_depth_scaling.py` — residual blocks `x + beta phi(W x)`: the cosine curves
  collapsing onto one profile under `t -> beta^2 t`, the depth scale `1/beta^2`, and the
  contrast between residual (`1/beta^2` scale) and plain (`O(1)` scale) evolution.
- `exp_e7_activation_smoothness.py` — at criticality (`chi1=1` for all), kinked activations
  (ReLU, leaky) approach parallel as `t^{-2}` while smooth ones (erf, tanh, sin) approach as
  `t^{-1}`: the critical exponent is set by the activation's smoothness, not by the
  multiplier. Deterministic (no finite-N simulation).

## Usage

```
python eoc_common.py                       # self-test
python exp_e1_state_evolution.py
python exp_e2_tied_vs_independent.py
python exp_e3_bias_criticality.py
python exp_e4_cnn_signal_propagation.py
python exp_e5_attention_rank_collapse.py
python exp_e6_resnet_depth_scaling.py
python exp_e7_activation_smoothness.py
```

Each script writes a JSON summary to `data/`, prints markdown tables, and draws a vector PDF
into `figures/`. All runs use fixed seeds.
