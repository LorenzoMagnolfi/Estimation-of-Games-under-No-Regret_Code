"""Sweeps from v05_sweeps.csv: identified-set area against the action grid |A| (s = 5) and against the cost support s (|A| = 5).
Two panels; lines: Hedge tailF at 1M and 4M (local grid), EXP3-IX tailB at 4M (global grid). Open circle = region hits the grid boundary;
down-triangle on the floor = no off-truth cell. Palette: categorical slots 1-3 (validated), 2px lines, 8px markers, direct labels, legend.
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
series = [("hedge", 1000000, "#2a78d6", "Hedge tail, 1M"), ("hedge", 4000000, "#eb6834", "Hedge tail, 4M"), ("exp3ix", 4000000, "#1baf7a", "EXP3-IX tail, 4M")]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
fig.patch.set_facecolor(SURF)
for ax, (sweep, xcol, xlabel, fixed) in zip(axes, [("A", "A", "number of prices |A| (s = 5)", ("s", 5)), ("s", "s", "cost support s (|A| = 5)", ("A", 5))]):
    ax.set_facecolor(SURF)
    sub = d[d[fixed[0]] == fixed[1]]
    for lrn, Nv, col, lab in series:
        s = sub[(sub.learner == lrn) & (sub.N == Nv)].sort_values(xcol)
        if s.empty: continue
        floor = s.area.where(s.n_cells > 0).min() if (s.n_cells > 0).any() else np.nan
        y = np.where(s.n_cells.values > 0, s.area.values, np.nan)
        ax.plot(s[xcol].values, y, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=1.2, label=lab)
        hb = s.hit_boundary.values.astype(bool)
        if hb.any(): ax.plot(s[xcol].values[hb], y[hb], ls="none", marker="o", ms=8, mfc="none", mec=INK, mew=1.0)
        empty = s.n_cells.values == 0
        if empty.any():
            ax.plot(s[xcol].values[empty], np.full(empty.sum(), np.nanmin(y) if np.isfinite(np.nanmin(y)) else 1e-4), ls="none", marker="v", ms=9, mfc=SURF, mec=col, mew=1.6)
        ok = np.isfinite(y)
        if ok.any():
            dy = 7 if '1M' in lab else (-7 if '4M' in lab and 'Hedge' in lab else 0)
            ax.annotate(lab, (s[xcol].values[ok][-1], y[ok][-1]), xytext=(7, dy), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    xs = sorted(sub[xcol].unique()); ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs]); ax.set_xticks([], minor=True)
    ax.set_xlim(min(xs) / 1.3, max(xs) * 2.2)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
axes[0].set_ylabel(r"identified-set area in $(\mu, \sigma^2)$ units", color=INK, fontsize=9)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.925))
fig.suptitle("Two-seller lab under the certified tails: region area against the action grid and the cost support (open circle = boundary; triangle = no off-truth cell)",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print(d[["sweep", "A", "s", "N", "learner", "grid", "area", "share", "n_cells", "width_mu", "width_sigma2", "hit_boundary", "truth_feasible"]].round(4).to_string(index=False))
