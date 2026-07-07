"""Shared numerics for the edge-of-chaos experiments E1-E3.

Implements the objects of THEORY_SPEC.md exactly, in the length convention
q = ||x||^2 / N:

  * activation library: phi and phi' for relu, leaky_relu (a = 0.2), linear,
    tanh, erf (rescaled), gelu, swish/silu, sin;
  * Gauss--Hermite quadrature (201 nodes) for E[f(Z)], Z ~ N(0,1), and for
    correlated Gaussian pairs (U1, U2);
  * length map V(q) = E[phi(sqrt(sw^2 q + sb^2) Z)^2] and its fixed point q*;
  * the correlation map: one step (q, c) -> (q', c') for two equal-length
    trajectories, with pre-activation correlation
    c' = (sw^2 q c + sb^2) / (sw^2 q + sb^2); at q = q* this is the spec's C;
  * chi1 = sw^2 E[phi'(sqrt(qhat*) Z)^2] with qhat* = sw^2 q* + sb^2;
  * finite-N pair simulation x -> phi(W x + b), tied or independent weights;
  * closed forms (ReLU arccosine kernel Khat, erf kernel, sin, linear, leaky
    relu) used as ground truth by the self-test.

Run `python eoc_common.py` for the quadrature-vs-closed-form self-test.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

LEAKY_A = 0.2

# ---------------------------------------------------------------------------
# activation library: name -> (phi, dphi), both vectorized
# ---------------------------------------------------------------------------


def _relu(z):
    return np.maximum(z, 0.0)


def _drelu(z):
    return (z > 0.0).astype(float)


def _leaky(z):
    return np.where(z > 0.0, z, LEAKY_A * z)


def _dleaky(z):
    return np.where(z > 0.0, 1.0, LEAKY_A)


def _linear(z):
    return np.asarray(z, dtype=float)


def _dlinear(z):
    return np.ones_like(np.asarray(z, dtype=float))


def _tanh(z):
    return np.tanh(z)


def _dtanh(z):
    t = np.tanh(z)
    return 1.0 - t * t


def _erf(z):
    # rescaled erf: phi(z) = erf(z / sqrt(2)) so that phi'(0) = sqrt(2/pi)
    return special.erf(z / math.sqrt(2.0))


def _derf(z):
    return math.sqrt(2.0 / math.pi) * np.exp(-0.5 * z * z)


def _gelu(z):
    return z * special.ndtr(z)


def _dgelu(z):
    return special.ndtr(z) + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _swish(z):
    s = special.expit(z)
    return z * s


def _dswish(z):
    s = special.expit(z)
    return s * (1.0 + z * (1.0 - s))


def _sin(z):
    return np.sin(z)


def _dsin(z):
    return np.cos(z)


ACTIVATIONS = {
    "relu": (_relu, _drelu),
    "leaky_relu": (_leaky, _dleaky),
    "linear": (_linear, _dlinear),
    "tanh": (_tanh, _dtanh),
    "erf": (_erf, _derf),
    "gelu": (_gelu, _dgelu),
    "swish": (_swish, _dswish),
    "sin": (_sin, _dsin),
}

HOMOGENEOUS = {"relu", "leaky_relu", "linear"}   # scale-free cosine map

# colorblind-safe palette (Wong 2011) and shared Matplotlib style
WONG = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
        "#56B4E9", "#CC79A7", "#000000", "#666666"]


def rcparams():
    return {
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

# ---------------------------------------------------------------------------
# Gauss--Hermite quadrature (>= 200 nodes; 201 keeps the node at 0)
# ---------------------------------------------------------------------------

GH_NODES = 201
_gx, _gw = special.roots_hermite(GH_NODES)
GH_Z = math.sqrt(2.0) * _gx          # standard-normal nodes
GH_P = _gw / math.sqrt(math.pi)      # weights, sum to 1


def gauss_e(f):
    """E[f(Z)], Z ~ N(0,1)."""
    return float(GH_P @ f(GH_Z))


def gauss_e2(f, g, s1, s2, rho):
    """E[f(U1) g(U2)], (U1,U2) centered Gaussian, sd s1 and s2, corr rho."""
    rho = min(1.0, max(-1.0, rho))
    u1 = s1 * GH_Z
    u2 = s2 * (rho * GH_Z[:, None]
               + math.sqrt(max(0.0, 1.0 - rho * rho)) * GH_Z[None, :])
    return float(GH_P @ (g(u2) @ GH_P * f(u1)))


# ---------------------------------------------------------------------------
# the deterministic maps of the spec
# ---------------------------------------------------------------------------


def v_map(q, phi, sw2, sb2):
    """Length map V(q) = E[phi(sqrt(sw^2 q + sb^2) Z)^2]."""
    s = math.sqrt(max(0.0, sw2 * q + sb2))
    return gauss_e(lambda z: phi(s * z) ** 2)


def fixed_point_q(phi, sw2, sb2, q0=1.0, tol=1e-13, max_iter=4000):
    """Solve q* = V(q*) by iteration from q0.

    Returns (qstar, marginal).  marginal=True means V(q) = q identically
    (homogeneous phi at its critical sw^2: a line of fixed points); then the
    convention q* = q0 is returned.
    """
    if (abs(v_map(0.7 * q0, phi, sw2, sb2) - 0.7 * q0) < 1e-10 * max(1.0, q0)
            and abs(v_map(1.3 * q0, phi, sw2, sb2) - 1.3 * q0)
            < 1e-10 * max(1.0, q0)):
        return q0, True
    q = q0
    for _ in range(max_iter):
        qn = v_map(q, phi, sw2, sb2)
        if abs(qn - q) < tol * max(1.0, abs(q)):
            q = qn
            break
        q = qn
    return q, False


def corr_next(phi, sw2, sb2, q, c):
    """One step of the joint (length, cosine) map for two equal-length
    trajectories: pre-activation variance qhat = sw^2 q + sb^2, correlation
    c' = (sw^2 q c + sb^2)/qhat; returns (q_next, c_next).

    At q = q* this is exactly the spec's cosine map C."""
    qhat = sw2 * q + sb2
    if qhat <= 0.0:
        return 0.0, 1.0
    rho = (sw2 * q * c + sb2) / qhat
    s = math.sqrt(qhat)
    qn = gauss_e(lambda z: phi(s * z) ** 2)
    m = gauss_e2(phi, phi, s, s, rho)
    return qn, m / qn


def c_map(phi, sw2, sb2, qstar, c):
    """The spec's C(c) at the length fixed point qstar."""
    return corr_next(phi, sw2, sb2, qstar, c)[1]


def theory_orbit(phi, sw2, sb2, q0, c0, T):
    """Iterate the deterministic (q, c) maps; returns arrays of length T+1."""
    qs, cs = [q0], [c0]
    q, c = q0, c0
    for _ in range(T):
        q, c = corr_next(phi, sw2, sb2, q, c)
        qs.append(q)
        cs.append(c)
    return np.array(qs), np.array(cs)


def chi1(dphi, sw2, sb2, qstar):
    """chi1 = sw^2 E[phi'(sqrt(qhat*) Z)^2], qhat* = sw^2 qstar + sb^2."""
    s = math.sqrt(max(0.0, sw2 * qstar + sb2))
    return sw2 * gauss_e(lambda z: dphi(s * z) ** 2)


def xi_c(chi):
    """Correlation depth scale xi_c = -1/log chi1 (ordered phase)."""
    return -1.0 / math.log(chi)


# ---------------------------------------------------------------------------
# finite-N simulation: x -> phi(W x + b), two trajectories
# ---------------------------------------------------------------------------


def make_starts(rng, n, q0, c0):
    """Two starts with ||x0||^2 = ||y0||^2 = n q0 and cos(x0, y0) = c0."""
    xhat = rng.standard_normal(n)
    xhat /= np.linalg.norm(xhat)
    y = rng.standard_normal(n)
    y -= (y @ xhat) * xhat
    yhat = y / np.linalg.norm(y)
    r = math.sqrt(n * q0)
    return r * xhat, r * (c0 * xhat + math.sqrt(1.0 - c0 * c0) * yhat)


def iterate_pair(x0, y0, T, phi, sw2, sb2, rng, tied=True):
    """Iterate x -> phi(Wx+b) for both starts; same W, b for both trajectories,
    reused every step if tied, freshly drawn each step if not.

    Returns (qx, qy, c), arrays of length T+1 including t=0."""
    n = x0.size
    sw = math.sqrt(sw2 / n)
    sb = math.sqrt(sb2)
    x, y = x0.copy(), y0.copy()
    qx = np.empty(T + 1)
    qy = np.empty(T + 1)
    c = np.empty(T + 1)

    def record(t):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        qx[t] = nx * nx / n
        qy[t] = ny * ny / n
        c[t] = float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else 1.0

    record(0)
    if tied:
        w = sw * rng.standard_normal((n, n))
        b = sb * rng.standard_normal(n)
    for t in range(1, T + 1):
        if not tied:
            w = sw * rng.standard_normal((n, n))
            b = sb * rng.standard_normal(n)
        x = phi(w @ x + b)
        y = phi(w @ y + b)
        record(t)
    return qx, qy, c


# ---------------------------------------------------------------------------
# closed forms (ground truth for the self-test and for long ReLU orbits)
# ---------------------------------------------------------------------------


def khat(rho):
    """Normalized ReLU arccosine kernel: the exact ReLU cosine map."""
    rho = min(1.0, max(-1.0, rho))
    return (math.sqrt(max(0.0, 1.0 - rho * rho))
            + rho * (math.pi - math.acos(rho))) / math.pi


def relu_kernel(v, rho):
    """E[relu(U1) relu(U2)], var v each, corr rho."""
    rho = min(1.0, max(-1.0, rho))
    return v * (math.sqrt(max(0.0, 1.0 - rho * rho))
                + rho * (math.pi - math.acos(rho))) / (2.0 * math.pi)


def erf_kernel(s2, rho):
    """E[erf(U1/sqrt2) erf(U2/sqrt2)], var s2 each, corr rho (Williams 1997)."""
    return (2.0 / math.pi) * math.asin(rho * s2 / (1.0 + s2))


def sin_kernel(v1, v2, cov):
    """E[sin(U1) sin(U2)] for centered Gaussians."""
    return math.exp(-0.5 * (v1 + v2)) * math.sinh(cov)


CHI1_CLOSED = {
    # chi1 closed forms as functions of (sw2, qhat*)
    "relu": lambda sw2, qhat: sw2 / 2.0,
    "leaky_relu": lambda sw2, qhat: sw2 * (1.0 + LEAKY_A ** 2) / 2.0,
    "linear": lambda sw2, qhat: sw2,
    "erf": lambda sw2, qhat: sw2 * (2.0 / math.pi) / math.sqrt(1.0 + 2.0 * qhat),
    "sin": lambda sw2, qhat: sw2 * (1.0 + math.exp(-2.0 * qhat)) / 2.0,
}


# ---------------------------------------------------------------------------
# accurate dispatched maps
#
# Gauss--Hermite is spectrally accurate for SMOOTH phi but converges slowly on
# the ReLU/leaky kink and is wrong for the step-function derivative (it puts a
# stray weight on the node at 0), e.g. it returns E[1{Z>0}] ~ 0.456 not 0.5.
# So we route the non-smooth / closed-form activations through their exact
# kernels and use quadrature only where phi is smooth.  This is a real accuracy
# statement, not a workaround: for ReLU the arccosine kernel Khat IS the map.
# ---------------------------------------------------------------------------

CLOSED_FORM = {"relu", "leaky_relu", "linear", "erf", "sin"}


def _leaky_kernel(v, rho):
    """E[leaky(U1) leaky(U2)], var v each, corr rho; leaky(u)=u_+ - a u_-."""
    a = LEAKY_A
    kp = relu_kernel(v, rho)          # E[u1_+ u2_+] = E[u1_- u2_-]
    kx = relu_kernel(v, -rho)         # E[u1_+ u2_-] = E[u1_- u2_+]
    return (1.0 + a * a) * kp - 2.0 * a * kx


def v_of(q, name, sw2, sb2):
    """Length map V(q), accurate: closed form where available."""
    phi = ACTIVATIONS[name][0]
    qhat = sw2 * q + sb2
    if name == "relu":
        return qhat / 2.0
    if name == "leaky_relu":
        return qhat * (1.0 + LEAKY_A ** 2) / 2.0
    if name == "linear":
        return qhat
    if name == "erf":
        return erf_kernel(qhat, 1.0)
    if name == "sin":
        return 0.5 * (1.0 - math.exp(-2.0 * qhat))
    return gauss_e(lambda z: phi(math.sqrt(max(0.0, qhat)) * z) ** 2)


def fixed_point_of(name, sw2, sb2, q0=1.0, **kw):
    phi = ACTIVATIONS[name][0]
    return fixed_point_q(phi, sw2, sb2, q0=q0, **kw)


def chi1_of(name, sw2, sb2, qstar):
    """chi1 at the length fixed point, accurate."""
    qhat = sw2 * qstar + sb2
    if name in CHI1_CLOSED:
        return CHI1_CLOSED[name](sw2, qhat)
    dphi = ACTIVATIONS[name][1]
    return sw2 * gauss_e(lambda z: dphi(math.sqrt(max(0.0, qhat)) * z) ** 2)


def corr_next_of(name, sw2, sb2, q, c):
    """One accurate step of the (length, cosine) map for two equal-length
    trajectories.  Returns (q_next, c_next)."""
    phi = ACTIVATIONS[name][0]
    qhat = sw2 * q + sb2
    if qhat <= 0.0:
        return 0.0, 1.0
    rho = (sw2 * q * c + sb2) / qhat
    qn = v_of(q, name, sw2, sb2)
    if name == "relu":
        m = relu_kernel(qhat, rho)
    elif name == "leaky_relu":
        m = _leaky_kernel(qhat, rho)
    elif name == "linear":
        m = qhat * rho
    elif name == "erf":
        m = erf_kernel(qhat, rho)
    elif name == "sin":
        m = math.exp(-qhat) * math.sinh(qhat * rho)
    else:
        s = math.sqrt(qhat)
        m = gauss_e2(phi, phi, s, s, rho)
    return qn, m / qn


def c_of(name, sw2, sb2, qstar, c):
    """The spec's cosine map C(c) at the length fixed point, accurate."""
    return corr_next_of(name, sw2, sb2, qstar, c)[1]


def theory_orbit_of(name, sw2, sb2, q0, c0, T):
    """Iterate the accurate deterministic (q, c) maps; arrays of length T+1."""
    qs, cs = [q0], [c0]
    q, c = q0, c0
    for _ in range(T):
        q, c = corr_next_of(name, sw2, sb2, q, c)
        qs.append(q)
        cs.append(c)
    return np.array(qs), np.array(cs)


# ---------------------------------------------------------------------------
# self-test: quadrature vs closed forms
# ---------------------------------------------------------------------------


def _selftest():
    rows = []

    def add(name, got, want):
        rows.append((name, got, want, abs(got - want)))

    add("E[Z^2]", gauss_e(lambda z: z ** 2), 1.0)
    add("E[Z^4]", gauss_e(lambda z: z ** 4), 3.0)
    add("E[Z^6]", gauss_e(lambda z: z ** 6), 15.0)

    # length maps with closed forms
    for q, sw2, sb2 in [(1.0, 2.0, 0.0), (0.4, 1.5, 0.05)]:
        add(f"V_relu(q={q},sw2={sw2},sb2={sb2})",
            v_map(q, _relu, sw2, sb2), (sw2 * q + sb2) / 2.0)
        a2 = sw2 * q + sb2
        add(f"V_sin(q={q},sw2={sw2},sb2={sb2})",
            v_map(q, _sin, sw2, sb2), 0.5 * (1.0 - math.exp(-2.0 * a2)))
        add(f"V_linear(q={q},sw2={sw2},sb2={sb2})",
            v_map(q, _linear, sw2, sb2), a2)

    # 2D kernels for SMOOTH phi vs closed forms (GH is spectrally accurate here;
    # the ReLU kink is handled by the dispatched closed form, checked below).
    for rho in (-0.5, 0.0, 0.3, 0.9, 0.999, 1.0):
        add(f"erf kernel rho={rho}",
            gauss_e2(_erf, _erf, math.sqrt(0.7), math.sqrt(0.7), rho),
            erf_kernel(0.7, rho))
        add(f"sin kernel rho={rho}",
            gauss_e2(_sin, _sin, math.sqrt(0.6), math.sqrt(0.6), rho),
            sin_kernel(0.6, 0.6, 0.6 * rho))

    # accurate chi1 at a genuine length fixed point (dispatched)
    q_lin, _ = fixed_point_q(_linear, 0.8, 0.2)
    add("linear q* (=1 exactly)", q_lin, 1.0)
    add("chi1 linear (disp)", chi1_of("linear", 0.8, 0.2, q_lin), 0.8)
    add("chi1 relu He (disp)", chi1_of("relu", 2.0, 0.0, 1.0), 1.0)
    add("chi1 leaky critical (disp)",
        chi1_of("leaky_relu", 2.0 / (1.0 + LEAKY_A ** 2), 0.0, 1.0), 1.0)
    q_erf, _ = fixed_point_q(_erf, 1.5, 0.05)
    add("chi1 erf (disp) vs closed",
        chi1_of("erf", 1.5, 0.05, q_erf),
        CHI1_CLOSED["erf"](1.5, 1.5 * q_erf + 0.05))

    # C'(1) finite difference vs chi1 for smooth phi (quadrature path)
    for name in ("tanh", "gelu", "swish"):
        qt, _ = fixed_point_q(ACTIVATIONS[name][0], 1.5, 0.05)
        eps = 1e-6
        slope = (1.0 - c_of(name, 1.5, 0.05, qt, 1.0 - eps)) / eps
        add(f"{name} C'(1) fd vs chi1 (disp)", slope,
            chi1_of(name, 1.5, 0.05, qt))
        add(f"{name} C(1)=1 (disp)", c_of(name, 1.5, 0.05, qt, 1.0), 1.0)

    # ReLU dispatched cosine map = Khat, at any length (homogeneity / scale-free)
    worst = max(abs(c_of("relu", 2.0, 0.0, 1.0, c) - khat(c))
                for c in np.linspace(-1, 1, 41))
    add("max|C_relu - Khat| (He)", worst, 0.0)
    worst2 = max(abs(c_of("relu", 1.0, 0.0, 0.37, c) - khat(c))
                 for c in np.linspace(-1, 1, 41))
    add("max|C_relu - Khat| (sw2=1, q=0.37: scale-free)", worst2, 0.0)

    print("| check | value | reference | abs diff |")
    print("|---|---:|---:|---:|")
    bad = 0
    for name, got, want, d in rows:
        flag = "" if d < 5e-6 else "  <-- LARGE"
        bad += d >= 5e-6
        print(f"| {name} | {got:.12f} | {want:.12f} | {d:.2e} |{flag}")
    print(f"\nself-test {'PASSED' if bad == 0 else 'FAILED'} "
          f"({len(rows)} checks, tolerance 5e-6, GH nodes {GH_NODES})")

    # informative diagnostic: raw GH on the ReLU kink is only ~1e-3 accurate,
    # which is exactly why the dispatched maps use the closed form for ReLU.
    raw = gauss_e2(_relu, _relu, math.sqrt(2.0), math.sqrt(2.0), 0.0)
    raw_chi = chi1(_drelu, 2.0, 0.0, 1.0)
    print(f"\n[diagnostic] raw GH ReLU kernel(rho=0) = {raw:.6f} vs exact "
          f"{relu_kernel(2.0, 0.0):.6f} (gap {abs(raw-relu_kernel(2.0,0.0)):.1e}); "
          f"raw GH chi1_relu(He) = {raw_chi:.6f} vs exact 1.0 "
          f"-- GH is inaccurate on the kink/step, dispatched maps use closed forms.")


if __name__ == "__main__":
    _selftest()
