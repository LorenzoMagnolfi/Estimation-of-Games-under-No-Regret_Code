"""Nonparametric cost-distribution clouds from v05_nonparam.csv (10,201 candidate probability vectors on the type support).
Three panels against N (s = 5): identified share of the cloud (log), width of the mean projection, width of the variance projection.
Lines: Hedge tailF, Hedge momentF, EXP3-IX tailB; Hedge tailF at s = 10 (two points) dashed in slot 1.
Palette: categorical slots 1-3 (validated); 2px lines; 8px markers; direct labels; legend; no dual axes.
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
d["mean_w"] = d.mean_hi - d.mean_lo; d["var_w"] = d.var_hi - d.var_lo
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
series = [("hedge", 14, 5, "#2a78d6", "-", "Hedge, direct tail (s = 5)", "Hedge, tail"),
          ("hedge", 16, 5, "#1baf7a", "-", "Hedge, valid first moment (s = 5)", "Hedge, first moment"),
          ("exp3ix", 15, 5, "#eb6834", "-", "EXP3-IX, direct tail (s = 5)", "EXP3-IX, tail"),
          ("hedge", 14, 10, "#2a78d6", (0, (4, 3)), "Hedge, direct tail (s = 10)", "Hedge, tail (s = 10)")]
panels = [("share", "identified share of the candidate cloud", True), ("mean_w", "width of the mean projection", False), ("var_w", "width of the variance projection", False)]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9)); fig.patch.set_facecolor(SURF)
for ax, (col, ylabel, ylog) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for lrn, sw, s, c, ls, lab, short in series:
        sub = d[(d.learner == lrn) & (d.switch_eps == sw) & (d.s == s)].sort_values("N")
        if sub.empty: continue
        y = sub[col].values.astype(float)
        if ylog: y = np.where(y > 0, y, np.nan)
        ax.plot(sub.N.values, y, color=c, lw=2, ls=ls, marker="o", ms=8, mec=SURF, mew=1.2, label=lab)
        nf = sub.truth_identified.values.astype(bool)
        if (~nf).any(): ax.plot(sub.N.values[~nf], y[~nf], ls="none", marker="x", ms=9, color=INK, mew=1.5)
        ok = np.isfinite(y)
        if ok.any() and col == "share":
            ax.annotate(short, (sub.N.values[ok][-1], y[ok][-1]), xytext=(7, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    Ns = sorted(d.N.unique()); ax.set_xticks(Ns); ax.set_xticklabels([f"{n/1e6:g}M" for n in Ns]); ax.set_xticks([], minor=True)
    ax.set_xlim(min(Ns) / 1.4, max(Ns) * (3.4 if col == "share" else 1.4))
    ax.set_xlabel("periods N", color=INK, fontsize=9); ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Nonparametric cost distributions in the two-seller lab: the certified cloud against N (an x marks a run where the true vector is infeasible)", fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print(d[["s", "N", "learner", "switch_eps", "n_identified", "share", "truth_identified", "mean_lo", "mean_hi", "var_lo", "var_hi", "max_realized_regret"]].round(4).to_string(index=False))
