"""Figure for the single-learner counterfactual (output of cf_single.py, one role).

Left: the focal advertiser's long-run spend per auction as a function of its value per conversion, under the recorded rule
and three alternatives; shaded bands = the value intervals identified at 1e7 and 1e8 auctions (Regime P, Markov); markers =
the spend of the learner at the true value (truth) and the spend if it kept its recorded bidding (naive).
Right: the predicted change in spend relative to the recorded rule (ratio), as a function of the value.
Usage: python plot_cf_single.py <cf_json> <out_png_prefix>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

d = json.load(open(sys.argv[1])); out = sys.argv[2]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
COL = {"gsp": "#2a78d6", "fp": "#eb6834", "fp_top": "#1baf7a", "gsp_r10": "#4a3aa7"}
LAB = {"gsp": "recorded rule (pay the next bid)", "fp": "first price on every slot", "fp_top": "first price on slot 1 only",
       "gsp_r10": "recorded rule, reserve price 0.10"}
L = np.array(d["lam_grid"]); lam0 = d["lam0"]; iv = d["intervals"]
rows = {r["rule"]: r for r in d["rows"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.28))
for ax in (a1, a2):
    ax.axvspan(*iv["10000000"], color="#e6e5e0", zorder=0)
    ax.axvspan(*iv["100000000"], color="#cdcbc4", zorder=0)
    ax.axvline(lam0, color="#0b0b0b", lw=0.8, ls="dashed")
for rule in ["gsp", "fp", "fp_top", "gsp_r10"]:
    a1.plot(L, d["curves"][rule], color=COL[rule], lw=2, label=LAB[rule])
    a1.plot(lam0, rows[rule]["truth"], "o", color=COL[rule], ms=6, zorder=4)
    if rule != "gsp":
        a1.plot(lam0, rows[rule]["naive"], "x", color=COL[rule], ms=7, mew=2, zorder=4)
        base = np.array(d["curves"]["gsp"]); new = np.array(d["curves"][rule]); ok = base > 0
        a2.plot(L[ok], new[ok] / base[ok], color=COL[rule], lw=2, label=LAB[rule])
        a2.plot(lam0, rows[rule]["truth_ratio"], "o", color=COL[rule], ms=6, zorder=4)
a1.set_xlabel("value per conversion"); a1.set_ylabel("long-run spend per auction")
a1.set_title("A. Long-run spend under each rule", loc="left", fontsize=10)
a1.legend(loc="upper left", frameon=False, fontsize=8)
a2.axhline(1.0, color="#52514e", lw=0.6)
a2.set_xlabel("value per conversion"); a2.set_ylabel("spend relative to the recorded rule")
a2.set_title("B. Spend relative to the recorded rule", loc="left", fontsize=10)
a2.set_ylim(0.5, 1.4); a2.set_xlim(50, 250)
a1.set_xlim(0, 250)
for ax in (a1, a2):
    ax.grid(color="#e6e5e0", lw=0.5)
for ext in ["png", "pdf"]:
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print("written", out)
