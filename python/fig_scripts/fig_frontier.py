"""Frontier panels at N = 4M from v05_frontier.mat: feasible (mu, sigma^2) cells per arm.
Rows: learner (Hedge, family F; EXP3-IX, family B). Columns: first-moment rectangle (valid Markov), direct tail.
Single-hue fill for feasible cells (palette sequential step 300), truth as a hollow marker, areas in the panel labels.
"""
import sys, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

path, out = sys.argv[1], sys.argv[2]
f = h5py.File(path, "r")
IDTOL = float(f["IDTOL"][()].squeeze()); N = int(f["N"][()].squeeze())
def s(ref):
    return "".join(chr(c) for c in f[ref][()].squeeze())
names = [s(r) for r in f["name"][0]]; area = f["area"][()].squeeze()
VV = [f[r][()].squeeze() for r in f["VV"][0]]; G = [f[r][()] for r in f["grids"][0]]
nV = nM = 61
def grid(k):
    v = VV[k].reshape(nM, nV).T; mu = G[k][0].reshape(nM, nV).T; sg = G[k][1].reshape(nM, nV).T
    return v[1:, 1:] <= IDTOL, mu[0, 1:], sg[1:, 0]
rows = [("hedge", "Hedge, full feedback (family F)"), ("exp3ix", "EXP3-IX, bandit feedback (family B)")]
cols = [("momentF", "momentB", "First-moment rectangle (valid Markov)"), ("tailF", "tailB", "Direct tail")]
SURF, FILL, INK, MUTED, GRID, AXIS = "#fcfcfb", "#6da7ec", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
cmap = ListedColormap([SURF, FILL])
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
fig.patch.set_facecolor(SURF)
for r, (lrn, rlabel) in enumerate(rows):
    for c, (cf, cb, clabel) in enumerate(cols):
        ax = axes[r, c]; nm = f"{lrn}_{cf if lrn == 'hedge' else cb}"; k = names.index(nm)
        feas, mu, sg = grid(k)
        ax.set_facecolor(SURF)
        ax.pcolormesh(mu, sg, feas.astype(float), cmap=cmap, vmin=0, vmax=1, shading="nearest", edgecolors="none")
        ax.plot([3.0], [1.0], marker="o", ms=8, mfc="none", mec=INK, mew=1.4, ls="none")
        ax.text(0.02, 0.96, f"area {area[k]:.4f}, share {100*feas.mean():.1f}%", transform=ax.transAxes,
                fontsize=8.5, color=INK, va="top", ha="left")
        if r == 0: ax.set_title(clabel, fontsize=9.5, color=INK)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[r, 0].set_ylabel(rlabel + "\n" + r"cost variance $\sigma^2$", color=INK, fontsize=9)
for ax in axes[-1]: ax.set_xlabel(r"mean cost $\mu$", color=INK, fontsize=9)
fig.suptitle(f"Two-seller lab at N = {N:,}: feasible candidates (shaded) on the 61 x 61 grid, L1 consistency; truth = circle",
             fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print("areas:", dict(zip(names, np.round(area, 4))))
