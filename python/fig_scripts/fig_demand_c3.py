"""Demand-cost ridge and the C3 projection from v05_demand_c3.csv.
Two panels against the candidate demand slope eta: the feasible range of the mean cost (left) and of the cost variance
(right) at each slope, as vertical segments. Set A (known payoff at each slope, alpha_R = 0.045; the union over slopes is the
demand-free projection) in slot 1; set B (slopes inside the first-stage 99 percent interval, alpha_R = 0.035; the union is
the C3 set) in slot 2 over the shaded interval; the truth as a marker. Open circle on a segment: the wide window's edge was
hit at that slope. Palette: categorical slots 1-2 (validated); 2px segments; 8px markers; direct labels; legend.
Usage: python fig_demand_c3.py <csv> <out_prefix> [ci_lo ci_hi]
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
ci = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2 = "#2a78d6", "#eb6834"
truth = (-1 / 3, 3.0, 1.0)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0)); fig.patch.set_facecolor(SURF)
for ax, (lo, hi, ylabel, ytruth) in zip(axes, [("mu_lo", "mu_hi", "feasible mean cost at the slope", truth[1]), ("sigma2_lo", "sigma2_hi", "feasible cost variance at the slope", truth[2])]):
    ax.set_facecolor(SURF)
    if ci is not None:
        ax.axvspan(ci[0], ci[1], color=C2, alpha=0.10, lw=0)
    for sname, col, lab, dx in [("A", C1, "known slope, each slope separately (demand-free union)", 0.0), ("B", C2, "slopes in the first-stage interval (C3 union)", 0.0)]:
        s = d[(d.set == sname) & (d.n_feasible_local > 0)].sort_values("eta")
        if s.empty: continue
        for _, r in s.iterrows():
            ax.plot([r.eta + dx, r.eta + dx], [r[lo], r[hi]], color=col, lw=2.2, solid_capstyle="butt")
            if bool(r.wide_hits_edge):
                ax.plot([r.eta + dx], [r[hi]], ls="none", marker="o", ms=8, mfc="none", mec=INK, mew=1.0)
        ax.plot([], [], color=col, lw=2.2, label=lab)
        r0 = s.iloc[-1]; ax.annotate("demand-free" if sname == "A" else "C3", (r0.eta + dx, r0[hi]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.plot([truth[0]], [ytruth], ls="none", marker="D", ms=7, color=INK, label="truth")
    ax.set_xlabel("candidate demand slope eta", color=INK, fontsize=9); ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Joint demand and cost in the two-seller lab (N = 4M, certified tail): the cost set at each slope, and the C3 projection", fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
pd.set_option("display.width", 220)
print(d[["set", "eta", "alpha_R", "n_feasible_wide", "wide_hits_edge", "n_feasible_local", "mu_lo", "mu_hi", "sigma2_lo", "sigma2_hi", "truth_cell_feasible", "seconds"]].round(4).to_string(index=False))
