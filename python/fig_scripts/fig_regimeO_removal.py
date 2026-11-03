"""Exercise O (contexts latent) in Design A: share of the bids-only region removed by the obedience restriction, against N,
for the certified tails (Hedge tailF from tailF_v1, EXP3-IX tailB from exp3ix_v1) and the historical Markov rule (Hedge),
one line per role (thin) with the across-role median (thick), three panels. Palette: slots 1-3; 2px lines; direct labels; legend.
Usage: python fig_regimeO_removal.py <designA_dir> <out_prefix>
"""
import sys, os, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

da, out = sys.argv[1], sys.argv[2]
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
tf = pd.read_csv(os.path.join(da, "tailF_v1", "summary_O.csv"))
ex = pd.read_csv(os.path.join(da, "exp3ix_v1", "summary_O.csv"))
panels = [(tf, "removed_by_tailF", "#2a78d6", "Hedge, direct tail (family F)"),
          (ex, "removed_by_tailB", "#eb6834", "EXP3-IX, direct tail (family B)"),
          (tf, "removed_by_markov", "#1baf7a", "Hedge, Markov (historical)")]
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True); fig.patch.set_facecolor(SURF)
for ax, (d, col, c, lab) in zip(axes, panels):
    ax.set_facecolor(SURF)
    d = d.copy(); d[col] = d[col].clip(lower=0)
    for role, g in d.groupby("role"):
        g = g.sort_values("N"); ax.plot(g.N, g[col], color=c, lw=1.0, alpha=0.45)
    med = d.groupby("N")[col].median().sort_index()
    ax.plot(med.index, med.values, color=c, lw=2.4, marker="o", ms=7, mec=SURF, mew=1.2, label=lab)
    ax.annotate(lab.split(" (")[0], (med.index[-1], med.values[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xscale("log"); Ns = sorted(d.N.unique()); ax.set_xticks(Ns); ax.set_xticklabels([f"{n/1e6:g}M" if n >= 1e6 else f"{n/1e3:g}k" for n in Ns]); ax.set_xticks([], minor=True)
    ax.set_xlim(min(Ns) / 1.4, max(Ns) * 5); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("auctions N", color=INK, fontsize=9); ax.set_title(lab, fontsize=9, color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
axes[0].set_ylabel("share of the bids-only region removed", color=INK, fontsize=9)
fig.suptitle("Design A, contexts latent: what the no-regret restriction removes from the bids-only region (six roles, thin; median, thick)", fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print(tf.groupby("N")[["removed_by_tailF", "removed_by_markov"]].median().round(3).to_string()); print(ex.groupby("N")[["removed_by_tailB"]].median().round(3).to_string())
