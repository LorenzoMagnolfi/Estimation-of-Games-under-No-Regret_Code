"""Price of bandit feedback in the auction: Regime P interval width against horizon for role 28 under the certified tails
(Hedge tailF from tailF_v1, EXP3-IX tailB from exp3ix_v1) and the historical Markov rules (Hedge markov, plain EXP3 markov).
Palette: categorical slots 1-3 and slot 2 dashed for the historical companions; 2px lines; 8px markers; direct labels; legend.
"""
import sys, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base, out, role = sys.argv[1], sys.argv[2], sys.argv[3]
SURF, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
h = pd.read_csv(f"{base}/tailF_v1/role{role}/hedge_1e8_regimeP.csv")
x = pd.read_csv(f"{base}/exp3ix_v1/role{role}/exp3ix_1e7_regimeP.csv")
series = [(h[h.rule == "tailF"], "#2a78d6", "-", "Hedge, direct tail (family F)"),
          (x[x.rule == "tailB"], "#eb6834", "-", "EXP3-IX, direct tail (family B)"),
          (h[h.rule == "markov"], "#2a78d6", (0, (4, 3)), "Hedge, Markov (historical)")]
fig, ax = plt.subplots(figsize=(6.8, 4.0)); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
for s, col, ls, lab in series:
    s = s[s.N <= 1e8].sort_values("N")
    ax.plot(s.N.values, s.width.values, color=col, lw=2, ls=ls, marker="o", ms=8, mec=SURF, mew=1.2, label=lab)
    ax.annotate(lab.split(",")[0] + ("" if "Markov" not in lab else " (Markov)"), (s.N.values[-1], s.width.values[-1]), xytext=(7, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
Ns = sorted(set(h.N[h.N <= 1e8]) | set(x.N)); ax.set_xticks(Ns); ax.set_xticklabels([f"{n/1e6:g}M" if n >= 1e6 else f"{n/1e3:g}k" for n in Ns], rotation=0); ax.set_xticks([], minor=True)
ax.set_xlim(min(Ns) / 1.4, max(Ns) * 4)
ax.set_xlabel("auctions N", color=INK, fontsize=9); ax.set_ylabel("width of the interval for the value per conversion", color=INK, fontsize=9)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=8, which="both")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title(f"Design A, role {role}, contexts observed: interval width against horizon (truth inside throughout)", fontsize=9.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out + ".png", dpi=160, facecolor=SURF); fig.savefig(out + ".pdf", facecolor=SURF)
m = pd.merge(h[h.rule == "tailF"][["N", "width"]], x[x.rule == "tailB"][["N", "width"]], on="N", suffixes=("_hedge_tailF", "_exp3ix_tailB"))
m["ratio"] = m.width_exp3ix_tailB / m.width_hedge_tailF
print(m.round(2).to_string(index=False))
