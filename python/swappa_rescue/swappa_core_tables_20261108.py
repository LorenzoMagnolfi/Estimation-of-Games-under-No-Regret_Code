"""Assemble the 8 November Swappa results into one table (the "one old/new table under common
demand specifications" that the 6 November memo asked for).

Reads output/swappa_core_20261108/*_core.json (native-reference panel: raw_d0, raw_d1, raw_d2,
raw_d6, raw_d8_fresh; existing-reference panel: d0_revival_like, d1_rich_competition,
d6_flexible_gap) and docs/evidence/swappa_bootstrap_analysis_20261108.json, and writes
docs/evidence/swappa_core_tables_20261108.csv plus a Markdown rendering printed to stdout.

Columns: Lerner index of sold phones (percent) under the level-shift comparator with the
Bonferroni test as run, the normal least-favorable max-t, its GMS refinement, the percentile-t
bootstrap (where available), the fixed-level comparator relative to the internal market index
(the paper's main regret notion on a level action), and integer-rounded deviations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'swappa_core_20261108'
EVID = ROOT / 'docs' / 'evidence'
SPEC_LABEL = {'raw_d0': 'baseline', 'raw_d1': 'richer rivals', 'raw_d2': 'richer rivals + SKU-week', 'raw_d6': 'flexible gap',
              'raw_d8_fresh': 'richer rivals + freshness', 'd0_revival_like': 'baseline (existing ref.)',
              'd1_rich_competition': 'richer rivals (existing ref.)', 'd6_flexible_gap': 'flexible gap (existing ref.)'}


def fmt(iv):
    if iv is None:
        return 'empty'
    return f'[{iv[0]:.1f}, {iv[1]:.1f}]'


def main():
    boot = json.loads((EVID / 'swappa_bootstrap_analysis_20261108.json').read_text()) if (EVID / 'swappa_bootstrap_analysis_20261108.json').exists() else {'results': {}}
    rows = []
    for f in sorted(OUT.glob('*_core.json')):
        d = json.loads(f.read_text())
        spec = d['spec']
        for name, rec in d['sellers'].items():
            c = rec['comparators']
            b = boot['results'].get(spec, {}).get('sellers', {}).get(name, {})
            rows.append(dict(spec=spec, label=SPEC_LABEL.get(spec, spec), panel=d.get('panel', 'raw_reference_panel.parquet'), seller=name, sales=rec['sales'],
                             fit=fmt(c['phi_base']['lerner']['fit']), bonferroni=fmt(c['phi_base']['lerner']['bonf']),
                             maxt=fmt(c['phi_base']['lerner']['maxt']), gms=fmt(c['phi_base']['lerner']['maxt_gms']),
                             with_cross=fmt(c['phi_base']['lerner']['maxt_cross']),
                             boot_pt=fmt(b['lerner']['percentile_t_boot']) if b else '',
                             g1_maxt=fmt(c['phi_g1']['lerner']['maxt']),
                             ext_int_fit=fmt(c['ext_int']['lerner']['fit']), ext_int_maxt=fmt(c['ext_int']['lerner']['maxt']),
                             ext_rmin_maxt=fmt(c['ext_rmin']['lerner']['maxt']), ext_ref_maxt=fmt(c['ext_ref']['lerner']['maxt']),
                             rounded_bonf=fmt(c['phi_base_rounded']['lerner']['bonf']),
                             size_bonf=b['null_point_size_check']['reject_bonferroni'] if b and b.get('null_point_size_check') else '',
                             size_maxt=b['null_point_size_check']['reject_normal_maxt'] if b and b.get('null_point_size_check') else '',
                             size_pt=b['null_point_size_check']['reject_percentile_t'] if b and b.get('null_point_size_check') else ''))
    df = pd.DataFrame(rows)
    order = ['raw_d0', 'raw_d1', 'raw_d2', 'raw_d6', 'raw_d8_fresh', 'd0_revival_like', 'd1_rich_competition', 'd6_flexible_gap']
    df['o'] = df.spec.map({s: i for i, s in enumerate(order)})
    df = df.sort_values(['o', 'sales'], ascending=[True, False]).drop(columns='o')
    df.to_csv(EVID / 'swappa_core_tables_20261108.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200)
    for spec in order:
        sub = df[df.spec == spec]
        if len(sub) == 0:
            continue
        print(f'\n### {SPEC_LABEL.get(spec, spec)} ({spec}; panel {sub.panel.iloc[0]})')
        print(sub[['seller', 'sales', 'fit', 'bonferroni', 'maxt', 'gms', 'boot_pt', 'g1_maxt', 'ext_int_fit', 'ext_int_maxt', 'ext_rmin_maxt', 'ext_ref_maxt', 'rounded_bonf', 'size_bonf', 'size_maxt', 'size_pt']].to_string(index=False))
    print('\nwritten', EVID / 'swappa_core_tables_20261108.csv')


if __name__ == '__main__':
    main()
