"""Descriptive pricing audit; no causal, adoption, or cost-identification claim.

Run with SWAPPA_DATA_VERSION=fullsource_refscreen_known. Only pairs of
consecutive calendar dates in the retained demand sample enter change rates.
A retained zero change is not evidence that the seller did not review price.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from audit_panel import OUT, DATA_VERSION
from demand_timing_sensitivity import terminal_status_data

SELLERS = ['UPGRADE SOLUTION INC', 'AMS Traders Inc.', 'GadgetPickup',
           'Certified Cell, Inc', 'SaveGadget']


def main():
    assert DATA_VERSION == 'fullsource_refscreen_known'
    d = terminal_status_data().sort_values(['Code', 'date']).copy()
    assert not d.duplicated(['Code', 'date']).any()
    grp = d.groupby('Code', sort=False)
    d['days_since_previous'] = grp.date.diff().dt.days
    d['price_change'] = grp.Price.diff()
    d['previous_rival_min'] = grp.rival_min.shift()
    d['previous_price'] = grp.Price.shift()
    records = []
    for seller in SELLERS:
        s = d[d.Seller.eq(seller)].copy()
        pairs = s[s.days_since_previous.eq(1)].copy()
        changed = pairs[pairs.price_change.abs().gt(1e-8)]
        all_changes = s[s.price_change.abs().gt(1e-8)]
        counts = changed.groupby('date').size()
        groups = s.groupby(['date', 'device_id', 'Storage', 'Condition']).Price.agg(['size', 'nunique'])
        multi = groups[groups['size'].gt(1)]
        # Price responses are compared with lagged other-seller minimum to
        # avoid claiming a within-day ordering absent from daily snapshots.
        lag_gap = changed.Price - changed.previous_rival_min
        same_day_gap = changed.Price - changed.rival_min
        record = {
            'seller': seller, 'rows': len(s), 'listings': s.Code.nunique(),
            'dates': s.date.nunique(), 'consecutive_pairs': len(pairs),
            'consecutive_changes': len(changed),
            'consecutive_increases': int(changed.price_change.gt(0).sum()),
            'consecutive_decreases': int(changed.price_change.lt(0).sum()),
            'change_rate_consecutive': len(changed)/len(pairs),
            'dates_with_consecutive_change': len(counts),
            'max_consecutive_changes_on_date': int(counts.max()) if len(counts) else 0,
            'share_changes_on_dates_with_at_least_5_changes': float(counts[counts.ge(5)].sum()/len(changed)) if len(changed) else None,
            'all_observed_pair_changes_including_gaps': len(all_changes),
            'dates_with_all_pair_changes': all_changes.date.nunique(),
            'multi_listing_category_days': len(multi),
            'share_multi_listing_category_days_one_price': float(multi['nunique'].eq(1).mean()),
            'changed_price_within_1_of_previous_rival_min': int(lag_gap.abs().le(1).sum()),
            'changed_price_within_1_of_current_rival_min': int(same_day_gap.abs().le(1).sum()),
            'median_change_dollars': float(changed.price_change.median()),
        }
        records.append(record)
    payload = {
        'created': '2026-10-20', 'data_version': DATA_VERSION,
        'input': str(OUT/'panel.parquet'),
        'input_sha256': hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'sample': 'terminal_status_data: reference available, rivals present, outcome observed, final date excluded',
        'interpretation': 'Descriptive only. Daily snapshots cannot identify review frequency, simultaneous updating, vendor adoption, causal response, or marginal cost. The five sellers were selected for this documentation audit, not as the top five by size.',
        'records': records,
    }
    dest = OUT/'pricing_practices_audit_20261020.json'
    dest.write_text(json.dumps(payload, indent=2, default=int)+'\n', encoding='utf-8')
    # Tracked compact artifact makes the research note reproducible without
    # requiring readers to find the ignored output directory.
    evidence = Path('docs/evidence')
    evidence.mkdir(exist_ok=True)
    (evidence/dest.name).write_text(dest.read_text(encoding='utf-8'), encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))
    print(dest)


if __name__ == '__main__':
    main()
