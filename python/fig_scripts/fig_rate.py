"""Rate landscape from v05_rate.csv: identified-set area against N, log-log, one panel per learner.
Lines: first-moment rectangle (valid Markov), direct tail, and for Hedge the legacy July radius (no M_R factor).
Palette: categorical slots 1-3 in fixed order (validated), 2px lines, 8px markers, direct labels, legend.
A region with no off-truth feasible cell is drawn as an open marker on the one-cell floor ("below one cell").
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"moment": "#2a78d6", "tail": "#eb6834", "legacy": "#1baf7a"}
LAB = {"moment": "first-moment rectangle (valid Markov)", "tail": "direct tail", "legacy": "legacy July radius (no M_R)"}
panels = [("hedge", "Hedge, full feedback (family F)", [("hedge_momentF", "moment"), ("hedge_tailF", "tail"), ("hedge_legacyMarkov10", "legacy")]),
          ("exp3ix", "EXP3-IX, bandit feedback (family B)", [("exp3ix_momentB", "moment"), ("exp3ix_tailB", "tail")])]
floor = 0.0038 * (60 / 40) ** 2   # one cell of the 40x40 grid in (mu, sigma^2) units
Ns = sorted(d.N.unique())
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
fig.patch.set_facecolor(SURF)
for ax, (lrn, title, arms) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for arm, kind in arms:
        s = d[d.arm == arm].sort_values("N")
        if s.empty: continue
        a = s.area.values; below = a <= 0
        y = np.where(below, floor, a)
        ax.plot(s.N.values, y, color=COL[kind], lw=2, marker="o", ms=8, mec=SURF, mew=1.2, label=LAB[kind])
        hb = s.hit_boundary.values.astype(bool)
        if hb.any():
            ax.plot(s.N.values[hb], y[hb], ls="none", marker="o", ms=8, mfc="none", mec=INK, mew=1.0)
        if below.any():
            ax.plot(s.N.values[below], y[below], ls="none", marker="v", ms=9, mfc=SURF, mec=COL[kind], mew=1.6)
            ax.annotate("no off-truth cell", (s.N.values[below][-1], floor), xytext=(0, -14), textcoords="offset points",
                        fontsize=7.5, color=MUTED, ha="center", va="top")
        ax.annotate(LAB[kind].split(" (")[0], (s.N.values[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    ax.axhline(floor, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(Ns[0], floor * 1.18, "one grid cell", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([f"{n/1e6:g}M" for n in Ns]); ax.set_xticks([], minor=True)
    ax.set_xlim(Ns[0] / 1.25, Ns[-1] * 2.6)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel("periods N", color=INK, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
axes[0].set_ylabel(r"identified-set area in $(\mu, \sigma^2)$ units", color=INK, fontsize=9)
axes[0].set_ylim(floor / 3, 40)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.925))
fig.suptitle("Two-seller lab: confidence-region area against the horizon (40 x 40 grid, L1 consistency); open circle = region hits the grid boundary",
             fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
