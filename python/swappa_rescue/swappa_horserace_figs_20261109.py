"""Figures for the horse race against the equilibrium benchmark (9 November 2026).

Figure 1 (frontier): for GadgetPickup, the smallest share of daily expected profit the seller must be forgoing,
against the eight level shifts, for each candidate Lerner index to be its true one; with the equilibrium interval, the
steady-state set and the test-of-record set below on the same axis. Two panels: Decluttr reference (declared) and
internal reference.
Figure 2 (laboratory): coverage of the true cost ratio by procedure across data-generating processes (GadgetPickup),
and the equilibrium point's error against the seller's realized forgone profit.

Usage: python swappa_horserace_figs_20261109.py [--dir output/swappa_horserace_20261109] [--out docs/evidence]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
C_EQ, C_ST, C_RC = '#eb6834', '#1baf7a', '#2a78d6'      # equilibrium, steady state, test of record
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
                     'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})


def frontier_panel(ax_top, ax_bot, r, title):
    L = np.array(r['frontier_curve']['lerner'])
    s = 100 * np.array(r['frontier_curve']['regret_share'])
    eq = r['equilibrium']
    keep = np.isfinite(s) & (s <= 12) & (L > -5) & (L < 32)
    ax_top.plot(L[keep], s[keep], color=INK, lw=1.6)
    ax_top.axhline(100 * r['allowance_share_at_thN'], color=C_RC, lw=1, ls='--')
    ax_top.text(-1.5, 100 * r['allowance_share_at_thN'] + .2, 'test-of-record\nallowance', ha='left', va='bottom',
                color=MUTED, fontsize=8, linespacing=1.1)
    mr = 100 * r['monthly_interval_regret_share_at_thN']
    ax_top.axhline(mr, color=MUTED, lw=.8, ls=':')
    ax_top.text(-1.5, mr + .2, 'monthly forgone\nprofit at equilibrium', ha='left', va='bottom', color=MUTED, fontsize=8,
                linespacing=1.1)
    ax_top.set_ylim(0, 10)
    ax_top.set_xlim(-2, 32)
    ax_top.set_ylabel(r'forgone expected profit, \% of daily profit')
    ax_top.set_title(title, fontsize=10, color=INK)
    rows = [('equilibrium', eq['lerner_iv'], C_EQ), ('steady state', r['steady']['lerner'], C_ST),
            ('no regret (test of record)', r['record']['lerner'], C_RC)]
    for k, (lab, iv, c) in enumerate(rows):
        y = 2 - k
        if iv is not None:
            a, b = max(iv[0], -2), min(iv[1], 32)
            ax_bot.plot([a, b], [y, y], color=c, lw=5, solid_capstyle='round')
            ax_bot.text(a, y + .28, lab + f' [{iv[0]:.1f}, {iv[1]:.1f}]', va='bottom', ha='left', fontsize=8, color=INK)
    ax_bot.plot([eq['lerner']], [2], 'o', color='white', ms=4, mec=C_EQ, mew=1.2)
    ax_bot.set_ylim(-.5, 3.1)
    ax_bot.set_yticks([])
    ax_bot.spines['left'].set_visible(False)
    ax_bot.set_xlabel(r'Lerner index of sold phones, \%')


def fig_frontier(ev, out):
    gp = ev['GadgetPickup']
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[3.0, 1.4], hspace=.08, wspace=.15))
    frontier_panel(axs[0, 0], axs[1, 0], gp['dec'], r'GadgetPickup, cost $\theta\times$ Decluttr quote')
    frontier_panel(axs[0, 1], axs[1, 1], gp['int'], r'GadgetPickup, cost $\theta\times$ internal market index')
    axs[0, 1].set_ylabel('')
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'swappa_horserace_frontier.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)


ORDER = ['S', 'S16', 'S784', 'S0', 'OFFm', 'OFFp', 'C', 'SHm5', 'SHm3', 'SHm2', 'SHm1', 'SHp1', 'SHp2', 'SHp3', 'SHp5', 'SHm3h16', 'SHp3h16']
LABEL = {'S': 'maintained', 'S16': r'$H/16$', 'S784': r'$H{=}784$', 'S0': r'$H{=}0$', 'C': 'calendar', 'OFFm': 'off-grid $-$', 'OFFp': 'off-grid $+$',
         'SHm5': r'$-5\%$', 'SHm3': r'$-3\%$', 'SHm2': r'$-2\%$', 'SHm1': r'$-1\%$', 'SHp1': r'$+1\%$', 'SHp2': r'$+2\%$',
         'SHp3': r'$+3\%$', 'SHp5': r'$+5\%$', 'SHm3h16': r'$-3\%$, $H/16$', 'SHp3h16': r'$+3\%$, $H/16$'}


def fig_lab(lab, out, ev=None):
    d = lab['dgps']
    keys = [k for k in ORDER if k in d]
    x = np.arange(len(keys))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=.25))
    for off, (lab_, key, c, mk) in zip([-.18, 0, .18], [('equilibrium', 'cover_eq', C_EQ, 'o'), ('steady state', 'cover_steady', C_ST, 's'),
                                                        ('no regret (test of record)', 'cover_record', C_RC, 'D')]):
        a1.plot(x + off, [d[k][key] for k in keys], mk, color=c, ms=5, label=lab_, mec='white', mew=.6)
    a1.axhline(.95, color=MUTED, lw=.8, ls=':')
    s_grp = [i for i, k in enumerate(keys) if k in ('S', 'S16', 'S784', 'S0', 'OFFm', 'OFFp')]
    c_grp = [i for i, k in enumerate(keys) if k.startswith('SH')]
    if s_grp:
        a1.axvspan(min(s_grp) - .5, max(s_grp) + .5, color='#f2f1ee', zorder=0)
        a1.text(np.mean(s_grp), .5, r'Assumption S holds', ha='center', fontsize=8, color=MUTED)
    if c_grp:
        a1.text(np.mean(c_grp), .5, r'experience predates a change in cost', ha='center', fontsize=8, color=MUTED)
    a1.set_xticks(x)
    a1.set_xticklabels([LABEL[k] for k in keys], rotation=55, ha='right', fontsize=8)
    a1.set_ylim(-.02, 1.04)
    a1.set_ylabel(r'coverage of the true $\theta$')
    a1.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, .02))
    reg = np.array([100 * d[k]['regret_share']['mean'] for k in keys])
    err = np.array([d[k]['eq_err']['mean'] for k in keys])
    lo = np.array([d[k]['eq_err']['p05'] for k in keys])
    hi = np.array([d[k]['eq_err']['p95'] for k in keys])
    a2.errorbar(reg, err, yerr=[err - lo, hi - err], fmt='o', color=C_EQ, ms=4, lw=1, mec='white', mew=.6)
    a2.axhline(0, color=MUTED, lw=.8)
    rec = (ev or {}).get(lab['seller'], {}).get('dec')
    if rec is not None:                     # envelope implied by the data's regret frontier
        Lc = np.array(rec['frontier_curve']['lerner'])
        sc = 100 * np.array(rec['frontier_curve']['regret_share'])
        Le = rec['equilibrium']['lerner']
        grid = np.linspace(0, max(3.2, reg.max() * 1.05), 200)
        up = [np.max(Lc[sc <= g] - Le) if np.any(sc <= g) else np.nan for g in grid]
        dn = [np.min(Lc[sc <= g] - Le) if np.any(sc <= g) else np.nan for g in grid]
        a2.plot(grid, up, color=INK, lw=.9, ls='--')
        a2.plot(grid, dn, color=INK, lw=.9, ls='--')
        a2.text(grid[-1], up[-1] + .15, 'frontier from the data', ha='right', va='bottom', fontsize=8, color=MUTED)
    w = np.array([d[k]['width_eq']['mean'] for k in keys]).mean() / 2
    a2.axhspan(-w, w, color=C_EQ, alpha=.12, lw=0)
    a2.text(.98, w - .05, r'equilibrium interval, mean half-width', transform=a2.get_yaxis_transform(), fontsize=8, color=MUTED, va='top', ha='right')
    a2.set_xlabel(r"seller's realized forgone profit in the window, \%")
    a2.set_ylabel(r'equilibrium error, Lerner points')
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'swappa_horserace_lab_{lab["seller"].replace(" ", "_").replace(",", "")}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    d = Path(args[args.index('--dir') + 1]) if '--dir' in args else ROOT / 'output' / 'swappa_horserace_20261109'
    out = Path(args[args.index('--out') + 1]) if '--out' in args else ROOT / 'docs' / 'evidence'
    out.mkdir(parents=True, exist_ok=True)
    ev = json.loads((d / 'horserace_eval_raw_d0.json').read_text())
    fig_frontier(ev, out)
    for p in sorted(d.glob('horserace_lab_*.json')):
        fig_lab(json.loads(p.read_text()), out, ev)
    print('FIGS_DONE', out)


if __name__ == '__main__':
    main()
