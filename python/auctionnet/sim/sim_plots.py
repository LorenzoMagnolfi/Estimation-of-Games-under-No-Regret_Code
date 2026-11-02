"""Figures for Design A: Regime O regions in (mu, sigma) by horizon and slack rule; Regime P intervals by horizon.
Usage: python sim_plots.py <env_npz> <inf_prefix> <fig_prefix>
Writes <fig_prefix>_regimeO.pdf/.png and <fig_prefix>_regimeP.pdf/.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env = np.load(sys.argv[1]); pref = sys.argv[2]; fig = sys.argv[3]
mu0, sig0, lam0 = float(env["mu0"]), float(env["sig0"]), float(env["lam0"])
grid = np.load(f"{pref}_grid.npz"); mus, sigs = grid["mus"], grid["sigs"]
O = pd.read_csv(f"{pref}_regimeO.csv"); P = pd.read_csv(f"{pref}_regimeP.csv")
Ns = sorted(O.N.unique())
rules = [r for r in ["tailF", "markov", "empirical"] if r in set(O.rule)]   # tailF present only in the 2026-11-02 reruns
fig1, axes = plt.subplots(len(rules), len(Ns), figsize=(2.6 * len(Ns), 2.6 * len(rules)), squeeze=False)
for r_, rule in enumerate(rules):
    for c_, N in enumerate(Ns):
        ax = axes[r_, c_]
        f = f"{pref}_O_{rule}_{N}.npy"
        if os.path.exists(f):
            feas = np.load(f)
            ax.pcolormesh(mus, sigs, feas.T, cmap="Blues", vmin=0, vmax=1.6, shading="auto")
        ax.plot([mu0], [sig0], "k.", ms=6)
        ax.set_title(f"{rule}, N={N:,.0f}".replace(",000,000", "M").replace(",000", "k"), fontsize=8)
        ax.tick_params(labelsize=7)
        if r_ == len(rules) - 1:
            ax.set_xlabel(r"$\mu$", fontsize=8)
        if c_ == 0:
            ax.set_ylabel(r"$\sigma$", fontsize=8)
fig1.suptitle("Regime O: feasible (mu, sigma) on the grid; dot = truth", fontsize=9)
fig1.tight_layout()
fig1.savefig(f"{fig}_regimeO.pdf"); fig1.savefig(f"{fig}_regimeO.png", dpi=150)

fig2, ax = plt.subplots(figsize=(5, 3))
for rule, col in [("markov", "tab:blue"), ("tailF", "tab:green"), ("empirical", "tab:red")]:
    sub = P[P.rule == rule].sort_values("N")
    if sub.empty:
        continue
    lo = np.maximum(sub.lam_lo.values, -0.5 * lam0); hi = np.minimum(sub.lam_hi.values, 3.0 * lam0)
    ax.fill_between(sub.N.values, lo, hi, alpha=0.25, color=col, label=rule)
    ax.plot(sub.N.values, lo, color=col, lw=1); ax.plot(sub.N.values, hi, color=col, lw=1)
ax.axhline(lam0, color="k", lw=1, ls="--", label="truth")
ax.set_xscale("log"); ax.set_xlabel("auctions N"); ax.set_ylabel("value per conversion")
ax.set_ylim(-0.5 * lam0, 3.0 * lam0); ax.legend(fontsize=7); ax.set_title("Regime P: interval for the value per conversion", fontsize=9)
fig2.tight_layout(); fig2.savefig(f"{fig}_regimeP.pdf"); fig2.savefig(f"{fig}_regimeP.png", dpi=150)
print("wrote", f"{fig}_regimeO.*", f"{fig}_regimeP.*")
