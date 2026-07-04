"""Figure: worst-case m-block gain a_m = sup_{||x||=1} ||f^m(x)|| drops below 1 at finite m,
evidence for uniform (all-input) finite-block contraction. Reads uniform_block_gain_snap.txt."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 10, "axes.linewidth": 0.7, "lines.linewidth": 1.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
WONG = ["#0072B2", "#D55E00", "#009E73"]

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
with open(os.path.join(HERE, "uniform_block_gain_snap.txt")) as fh:
    for ln in fh:
        p = ln.split()
        if len(p) >= 4 and p[0].isdigit():
            N, m = int(p[0]), int(p[1]); g, gr = float(p[2]), float(p[3])
            rows.setdefault(N, []).append((m, g, gr))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
for i, (N, data) in enumerate(sorted(rows.items())):
    data.sort()
    ms = [d[0] for d in data]; gs = [d[1] for d in data]; grs = [d[2] for d in data]
    ax1.plot(ms, gs, marker="o", ms=3.5, color=WONG[i], label=f"$N={N}$")
    ax2.plot(ms, grs, marker="s", ms=3.5, color=WONG[i], label=f"$N={N}$")
ax1.axhline(1.0, ls=":", color="0.4", lw=1.0)
ax1.set_xlabel("block length $m$"); ax1.set_ylabel(r"$a_m=\sup_{\|x\|=1}\|f^m(x)\|$")
ax1.legend(frameon=False, fontsize=8)
ax2.axhline(2**-0.5, ls="--", color="0.4", lw=1.0)
ax2.text(13.5, 2**-0.5+0.012, r"$2^{-1/2}$", fontsize=8, color="0.35")
ax2.set_xlabel("block length $m$"); ax2.set_ylabel(r"$a_m^{1/m}$")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "fig_uniform_block.pdf")
fig.savefig(out)
print("wrote", os.path.abspath(out), os.path.getsize(out), "bytes")
