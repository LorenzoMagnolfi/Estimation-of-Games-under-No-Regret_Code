"""Figures that explain Regime O, from the outputs of explain_regimeO.py (one role, three grid variants).

Fig 1  what the analyst sees and what is hidden: the learner's joint distribution over (multiplier, context) at the
       largest checkpoint, and the bid distribution it produces, with the bid window of each context.
Fig 2  why the no-regret restriction bites late: per context, the worst regret available on the multiplier grid
       against the Markov tolerance at each checkpoint (trimmed and wide grids).
Fig 3  the (mu, sigma) regions at 1e7 and 1e8: kept by the bids alone, kept with no regret at the current tolerance,
       kept with a tolerance 22 times smaller; dashed outline = current tolerance computed from the candidate's own
       context shares; star = truth.
Fig 4  share of the bids-only region that survives, as a function of the multiple kappa of the current tolerance.
Usage: python plot_explain_regimeO.py <designA_dir> <role> <out_dir> [explain_subdir=explain]
With the outputs of explain_regimeO.py in 'candidate' profile (subdir with <tag>_markovrealized_<N>.npy files), the regions use
the corrected tolerance and the dashed outline is the earlier tolerance with the realized context shares.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

da, role, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
ex = os.path.join(da, sys.argv[4] if len(sys.argv) > 4 else "explain")
CORR = os.path.exists(os.path.join(ex, f"role{role}_base_markovrealized_100000000.npy"))
ALT_FILE, ALT_LAB = ("markovrealized", "earlier tolerance (realized context shares)") if CORR else ("candphi", "tolerance from the candidate's own context shares")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 10,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb"})
BLUE = {100: "#cde2fb", 150: "#b7d3f6", 250: "#86b6ef", 400: "#3987e5", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281", 700: "#0d366b"}
ORANGE, AQUA, INK, INK2, MUTED = "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f3f0", BLUE[100], BLUE[250], BLUE[400], BLUE[550], BLUE[700]])
VARIANTS = [("base", "Main grid: 12 multipliers, 0.5 to 1.5 x value"), ("wide", "Wide grid: 13 multipliers, 0.25 to 2.5 x value"),
            ("nv5", "Main grid, 5 contexts")]
RUN = {"base": os.path.join(da, f"run_role{role}_hedge_1e8.npz"), "wide": os.path.join(da, "robust", f"role{role}", "wide", "run_hedge_1e8.npz"),
       "nv5": os.path.join(da, "robust", f"role{role}", "nv5", "run_hedge_1e8.npz")}


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 1 ----------
tab = np.load(os.path.join(da, "tables", f"role{role}_base.npz")); run = np.load(RUN["base"])
N = int(run["checks"][-1]); Z = run[f"Z_{N}"] / N
alphas, vgrid, lam0, bid_index, NB = tab["alphas"], tab["vgrid"], float(tab["lam0"]), tab["bid_index"], int(tab["NB"])
NM, NV = Z.shape
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.42))
im = a1.imshow(Z, origin="lower", cmap=SEQ, aspect="auto", vmin=0, vmax=Z.max())
a1.set_xticks(range(NV)); a1.set_xticklabels([f"{j + 1}\n{vgrid[j] * 1e4:.1f}" for j in range(NV)])
a1.set_yticks(range(NM)); a1.set_yticklabels([f"{a / lam0:.2f}" for a in alphas])
a1.set_xlabel("context (top: index; bottom: conversion probability x 1e4)"); a1.set_ylabel("multiplier / value per conversion")
a1.set_title("A. Hidden from the analyst: context and multiplier", loc="left")
for k in range(NM):
    for j in range(NV):
        if Z[k, j] >= 0.02:
            a1.text(j, k, f"{Z[k, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if Z[k, j] > 0.08 else INK)
cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.02); cb.set_label("share of auctions"); cb.outline.set_visible(False)
q = np.zeros(NB)
for k in range(NM):
    for j in range(NV):
        q[bid_index[k, j]] += Z[k, j]
a2.bar(np.arange(NB), q, width=0.8, color=BLUE[450])
for j in range(NV):
    lo, hi = bid_index[:, j].min(), bid_index[:, j].max()
    y = -0.012 - 0.0075 * (j % 3)
    a2.plot([lo - 0.4, hi + 0.4], [y, y], color=INK2, lw=2, solid_capstyle="butt")
    a2.text(lo - 0.8, y, f"{j + 1}", ha="right", va="center", fontsize=7, color=INK2)
a2.axhline(0, color=INK2, lw=0.6)
top = np.argsort(q)[-4:]
for b in top:
    a2.annotate(f"{q[b]:.3f}", (b, q[b]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK)
a2.set_ylim(-0.035, q.max() * 1.12); a2.set_xlim(-2.5, NB)
a2.set_xlabel("bid level (steps of 10.5 percent); grey bars = bids each context 1 to 7 can produce")
a2.set_ylabel("share of auctions")
a2.set_title("B. Observed by the analyst: the bid distribution", loc="left")
save(fig, f"fig1_hidden_vs_seen_role{role}")

# ---------- Fig 2 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, (v, title) in zip(axs, VARIANTS[:2]):
    t = pd.read_csv(os.path.join(ex, f"role{role}_{v}_tolerance.csv"))
    t = t[t.K > 0]
    ctx = sorted(t.context.unique())
    w = t[t.N == t.N.max()].set_index("context")
    ax.scatter(ctx, w.loc[ctx, "worst_regret"], s=60, color=ORANGE, zorder=3, label="worst regret available on the grid")
    for N_, step, lab in [(1_000_000, 250, "tolerance at 1 million auctions"), (10_000_000, 450, "tolerance at 10 million"),
                          (100_000_000, 650, "tolerance at 100 million")]:
        s = t[t.N == N_].set_index("context")
        ax.scatter(ctx, s.loc[ctx, "tol_markov_cond"], s=36, color=BLUE[step], zorder=3, label=lab)
    s8 = t[t.N == 100_000_000].set_index("context")
    ax.scatter(ctx, s8.loc[ctx, "tol_markov_cond"] / 22.2, s=36, facecolors="none", edgecolors=BLUE[650], zorder=3,
               label="100 million, tolerance / 22 (illustrative)")
    ax.set_yscale("log"); ax.set_xticks(ctx); ax.set_xlabel("context")
    ax.set_title(title, loc="left"); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
axs[0].set_ylabel("payoff per auction in the context")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
save(fig, f"fig2_tolerance_vs_regret_role{role}")

# ---------- Fig 3 ----------
LEVELS = ListedColormap(["#fcfcfb", BLUE[150], BLUE[400], BLUE[700]])
fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
for col, (v, title) in enumerate(VARIANTS):
    g = np.load(os.path.join(ex, f"role{role}_{v}_grid.npz"))
    mus, sigs, mu0, sig0 = g["mus"], g["sigs"], float(g["mu0"]), float(g["sig0"])
    for row, N_ in enumerate([10_000_000, 100_000_000]):
        ax = axs[row, col]
        kap = np.load(os.path.join(ex, f"role{role}_{v}_kappa_{N_}.npy")); cand = np.load(os.path.join(ex, f"role{role}_{v}_{ALT_FILE}_{N_}.npy"))
        lev = np.isfinite(kap).astype(int) + (kap <= 1).astype(int) + (kap <= 1 / 22.2).astype(int)
        dmu, dsg = mus[1] - mus[0], sigs[1] - sigs[0]
        ax.imshow(lev, origin="lower", cmap=LEVELS, vmin=-0.5, vmax=3.5, aspect="auto",
                  extent=[sigs[0] - dsg / 2, sigs[-1] + dsg / 2, mus[0] - dmu / 2, mus[-1] + dmu / 2], interpolation="nearest")
        ax.contour(sigs, mus, cand.astype(float), levels=[0.5], colors=INK, linestyles="dashed", linewidths=1.0)
        ax.plot(sig0, mu0, marker="*", ms=11, color=ORANGE, mec="white", mew=0.6, zorder=4)
        none, mk, tight = np.isfinite(kap).mean(), (kap <= 1).mean(), (kap <= 1 / 22.2).mean()
        ax.text(0.02, 0.97, f"bids only {none:.3f}\nno regret {mk:.3f}\n({'earlier tolerance' if CORR else 'candidate shares'} {cand.mean():.3f})\ntolerance / 22: {tight:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK)
        if row == 0:
            ax.set_title(title, loc="left", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{N_ / 1e6:.0f} million auctions\n" + r"$\mu$ (mean of log conversion probability)")
        if row == 1:
            ax.set_xlabel(r"$\sigma$ (spread of log conversion probability)")
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE[150]), plt.Rectangle((0, 0), 1, 1, color=BLUE[400]), plt.Rectangle((0, 0), 1, 1, color=BLUE[700]),
           plt.Line2D([], [], color=INK, ls="dashed"), plt.Line2D([], [], marker="*", color=ORANGE, ls="none", ms=10)]
fig.legend(handles, ["kept by the bids alone (no-regret off)", "also kept by no regret, current tolerance",
                     "also kept with tolerance / 22 (illustrative)", f"no regret, {ALT_LAB}", "truth"],
           loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
save(fig, f"fig3_regions_role{role}")

# ---------- Fig 4 ----------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
grid_k = np.logspace(-3.5, 1.5, 300)
for (v, title), colr in zip(VARIANTS, [BLUE[450], ORANGE, AQUA]):
    kap = np.load(os.path.join(ex, f"role{role}_{v}_kappa_100000000.npy"))
    fin = kap[np.isfinite(kap)]
    share = np.array([(fin <= x).mean() for x in grid_k])
    ax.plot(grid_k, share, color=colr, lw=2, label=title)
ax.axvline(1.0, color=INK2, lw=0.8); ax.text(1.05, 0.03, "current tolerance", fontsize=7.5, color=INK2, rotation=90, va="bottom")
ax.axvline(1 / 22.2, color=INK2, lw=0.8, ls="dotted"); ax.text(1 / 22.2 * 1.05, 0.03, "tolerance / 22", fontsize=7.5, color=INK2, rotation=90, va="bottom")
s = pd.read_csv(os.path.join(ex, f"role{role}_base_summary.csv"))
kt = float(s[s.N == 100_000_000].truth_kappa.iloc[0])
ax.axvline(kt, color=ORANGE, lw=0.8, ls="dashed"); ax.text(kt * 1.05, 0.55, "truth drops out\nbelow this", fontsize=7.5, color=ORANGE, va="bottom")
ax.set_xscale("log"); ax.set_ylim(0, 1.02)
ax.set_xlabel("tolerance as a multiple of the current (Markov) tolerance, 100 million auctions")
ax.set_ylabel("share of the bids-only region that survives\n(grid cells, step 0.05)")
ax.legend(loc="upper left", frameon=False, fontsize=8); ax.grid(color="#e6e5e0", lw=0.6)
save(fig, f"fig4_share_vs_tolerance_role{role}")
print("figures written to", out)
