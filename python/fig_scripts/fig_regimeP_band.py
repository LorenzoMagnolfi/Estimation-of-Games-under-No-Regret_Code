"""Exercise P (contexts observed) in Design A, role 28: the interval for the value per conversion against N under the
historical Markov rule and the certified direct tail (Hedge, family F), as bands around the truth. Palette: slot 1 (Markov,
light fill) and slot 2 (direct tail, fill); 2px edges; direct labels; legend. Usage: python fig_regimeP_band.py <csv> <out>
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
truth = 115.44
fig, ax = plt.subplots(figsize=(6.8, 4.0)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
for rule, col, fill, lab in [("markov", "#2a78d6", "#6da7ec", "Markov conversion (historical)"), ("tailF", "#eb6834", "#f4a27e", "direct tail, family F (certified)")]:
    s = d[d.rule == rule].sort_values("N")
    ax.fill_between(s.N, s.lam_lo, s.lam_hi, color=fill, alpha=0.35, lw=0)
    ax.plot(s.N, s.lam_lo, color=col, lw=2); ax.plot(s.N, s.lam_hi, color=col, lw=2, label=lab)
ax.axhline(truth, color=INK, lw=1, ls=(0, (3, 3))); ax.annotate("truth 115.4", (1e8, truth), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
ax.set_xscale("log"); Ns = sorted(d.N.unique()); ax.set_xticks(Ns); ax.set_xticklabels([f"{n/1e6:g}M" if n >= 1e6 else f"{n/1e3:g}k" for n in Ns]); ax.set_xticks([], minor=True)
ax.set_xlim(min(Ns) / 1.4, max(Ns) * 4); ax.set_ylim(-700, 700)
ax.set_xlabel("auctions N", color=INK, fontsize=9); ax.set_ylabel("value per conversion", color=INK, fontsize=9)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=8, which="both")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("Design A, role 28, contexts observed: the certified interval against the horizon", fontsize=9.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
print(d[d.rule.isin(["markov", "tailF"])].pivot_table(index="N", columns="rule", values=["lam_lo", "lam_hi", "width"]).round(1).to_string())
