"""Compact evidence record and markdown tables for the horse race of 9 November 2026 (reads the run outputs of
Unity jobs 11392145 and 11392198; seconds). Writes docs/evidence/swappa_horserace_20261109.json and prints tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'output' / 'swappa_horserace_20261109'
OUT = ROOT / 'docs' / 'evidence' / 'swappa_horserace_20261109.json'
FOCAL = ['GadgetPickup', 'AMS Traders Inc.', 'UPGRADE SOLUTION INC', 'We Buy Mobile Corp']


def main():
    ev = json.loads((D / 'horserace_eval_raw_d0.json').read_text())
    labs = {json.loads(p.read_text())['seller']: json.loads(p.read_text()) for p in sorted(D.glob('horserace_lab_*.json'))}
    comp = dict(date='2026-11-09', jobs=dict(tables=11392145, run=11392198), sellers={}, lab={})
    for name, refs in ev.items():
        comp['sellers'][name] = {}
        for ref, r in refs.items():
            comp['sellers'][name][ref] = dict(
                H_cell=r['H_cell'], sales=r['sales'], equilibrium=r['equilibrium'], steady=r['steady'], record=r['record'],
                allowance_share=r['allowance_share_at_thN'], monthly_regret_share=r['monthly_interval_regret_share_at_thN'],
                frontier={k: v for k, v in r['frontier'].items()}, halves=r['halves'], halves_disjoint=r['halves_disjoint'],
                months={k: v['eq_lerner_iv'] for k, v in r['months'].items()},
                months_eq_pairs_disjoint=r['months_eq_pairs_disjoint'], months_eq_common_point=r['months_eq_common_point'])
    for name, lab in labs.items():
        comp['lab'][name] = dict(H_measured=lab['H_measured'], theta_data=lab['theta_data'], truths=lab['truths'], reps=lab['reps'],
                                 dgps={k: {kk: (vv if not isinstance(vv, dict) or kk == 'level_hist' else {a: round(b, 4) for a, b in vv.items()})
                                           for kk, vv in v.items()} for k, v in lab['dgps'].items()})
    files = sorted((ROOT / 'python' / 'swappa_rescue').glob('swappa_horserace_*_20261109.py'))
    comp['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    comp['sha256'].update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.npz'))})
    OUT.write_text(json.dumps(comp, indent=1, default=float))
    # tables
    print('| Seller | H (cells) | Sales | Equilibrium | Steady state | Test of record | Equilibrium, halves | Disjoint halves (eq / st / rec) | Monthly forgone profit at equilibrium |')
    print('|---|---:|---:|---|---|---|---|---|---:|')
    for name in sorted(ev, key=lambda n: -ev[n]['dec']['H_cell']):
        for ref in ['dec', 'int']:
            r = ev[name][ref]
            e = r['equilibrium']
            h = r['halves']
            dj = r['halves_disjoint']
            f = lambda iv: 'empty' if iv is None else f'[{iv[0]:.1f}, {iv[1]:.1f}]'  # noqa: E731
            print(f"| {name if ref == 'dec' else ''} {'(Decluttr)' if ref == 'dec' else '(internal)'} | {r['H_cell']:,.0f} | {r['sales']} | "
                  f"{e['lerner']:.1f} {f(e['lerner_iv'])} | {f(r['steady']['lerner'])} | {f(r['record']['lerner'])} | "
                  f"{f(h['first']['eq_lerner_iv'])}, {f(h['second']['eq_lerner_iv'])} | "
                  f"{'yes' if dj['equilibrium'] else 'no'} / {'yes' if dj['steady'] else 'no'} / {'yes' if dj['record'] else 'no'} | "
                  f"{100 * r['monthly_interval_regret_share_at_thN']:.2f}% |")
    for ref in ['dec', 'int']:
        n = len(ev)
        cnt = {k: sum(ev[s][ref]['halves_disjoint'][k] for s in ev) for k in ['equilibrium', 'steady', 'record']}
        mon = sum(ev[s][ref]['months_eq_pairs_disjoint'] > 0 for s in ev)
        print(f'{ref}: halves disjoint {cnt} of {n}; sellers with >=1 disjoint monthly equilibrium pair {mon} of {n}')
    print()
    for name, lab in labs.items():
        print(f'LAB {name} (H={lab["H_measured"]:.0f}, theta*={lab["theta_data"]})')
        print('| DGP | H | Forgone profit, % (mean) | Equilibrium error, Lerner points (mean, sd) | Coverage: equilibrium (no demand term) | steady state | test of record | Width: eq / st / rec |')
        print('|---|---:|---:|---|---|---|---|---|')
        for k, v in lab['dgps'].items():
            print(f"| {k} | {v['H']:,.0f} | {100 * v['regret_share']['mean']:.2f} | {v['eq_err']['mean']:+.2f} ({v['eq_err']['sd']:.2f}) | "
                  f"{v['cover_eq']:.2f} ({v['cover_eq_nodemand']:.2f}) | {v['cover_steady']:.2f} | {v['cover_record']:.2f} | "
                  f"{v['width_eq']['mean']:.1f} / {v['width_steady']['mean']:.1f} / {v['width_record']['mean']:.1f} |")
        print()
    print('frontier, GadgetPickup dec:', ev['GadgetPickup']['dec']['frontier'])
    print('frontier, GadgetPickup int:', ev['GadgetPickup']['int']['frontier'])
    print('EVIDENCE', OUT)


if __name__ == '__main__':
    main()
