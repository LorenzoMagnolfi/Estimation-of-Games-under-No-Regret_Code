"""Sparse-outcome exhibit from v05_sparse.csv: identified-set area at N = 4M against the sale-probability scale q, three learners.
Left: area (log-log) for expected-profit Hedge (tailF), Hedge on realized potential profits (tail + bridge), EXP3-IX on the realized own
profit (tail + bridge). Right: the certified tolerance divided by the cap at the truth (median over types), the mechanism.
Palette: categorical slots 1-3 (validated); 2px lines; 8px markers; direct labels; legend.
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
series = [("hedge", "#2a78d6", "expected profits (Hedge, tail)"), ("hedge_noisy", "#eb6834", "realized potential profits (Hedge, tail + bridge)"),
          ("exp3ix_noisy", "#1baf7a", "realized own profit (EXP3-IX, tail + bridge)")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
fig.patch.set_facecolor(SURF)
for ax, (ycol, ylabel, ylog) in zip(axes, [("area", r"identified-set area in $(\mu, \sigma^2)$ units", True), ("tol_over_cap_median", "certified tolerance over cap at the truth (median)", True)]):
    ax.set_facecolor(SURF)
    for lrn, col, lab in series:
        s = d[d.learner == lrn].sort_values("q")
        if s.empty: continue
        y = s[ycol].values.astype(float)
        if ycol == "area": y = np.where(y > 0, y, np.nan)
        ax.plot(s.q.values, y, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=1.2, label=lab)
        if ycol == "area":
            hb = s.hit_boundary.values.astype(bool)
            if hb.any(): ax.plot(s.q.values[hb], y[hb], ls="none", marker="o", ms=8, mfc="none", mec=INK, mew=1.0)
        ok = np.isfinite(y)
        if ok.any() and lrn != "exp3ix_noisy":
            txt = "expected profits" if lrn == "hedge" else "realized feedback (both learners)"
            ax.annotate(txt, (s.q.values[ok][0], y[ok][0]), xytext=(-7, 9 if lrn != "hedge" else 0), textcoords="offset points", fontsize=8, color=INK, va="center", ha="right")
    ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    qs = sorted(d.q.unique()); ax.set_xticks(qs); ax.set_xticklabels([f"{q:g}" for q in qs]); ax.set_xticks([], minor=True)
    ax.set_xlim(min(qs) / 1.6, max(qs) * 1.4)
    ax.invert_xaxis()
    ax.set_xlabel("sale-probability scale q (sparser outcomes to the right)", color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
axes[1].axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 3))); axes[1].text(max(d.q) * 1.35, 0.72, "tolerance = cap (vacuous above)", fontsize=7.5, color=MUTED, ha="left", va="top")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Sparse outcomes in the two-seller lab (N = 4M, 61 x 61 grid): the identified set does not depend on q; the certified allowance does",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print(d[["q", "learner", "switch_eps", "area", "share", "hit_boundary", "truth_feasible", "max_realized_regret", "tol_over_cap_max", "tol_over_cap_median"]].round(4).to_string(index=False))
