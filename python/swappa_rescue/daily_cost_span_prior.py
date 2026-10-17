"""Exact prior mean-cost upper bound under a primitive within-SKU cost span.

If max(c)-min(c)<=span and c_i<=receipt_cap_i, then every c_i is at most
min(receipt_cap)+span. This bound is jointly attainable by clipping each
receipt cap at that level. It holds for every realization of sale weights.
"""
import json
import numpy as np
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts


def main():
    s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].copy();s['receipt']=recover_receipts(s.Price.to_numpy());s['event']=s.sale_terminal_status*s.since_seen.gt(0);total=s.event.sum();price=float(s.event@s.receipt/total);rows=[]
    for span in [0,5,10,20]:
        upper=0
        for key,sub in s.groupby(['device_id','Storage','Condition']):
            cap=sub.groupby('Code').receipt.min();weight=sub.groupby('Code').event.sum();upper+=float(weight@np.minimum(cap,cap.min()+span))/total
        rows.append({'within_sku_cost_span':span,'prior_cost_interval':[0,upper],'prior_margin_interval':[price-upper,price],'mean_receipt':price})
    (OUT/'daily_cost_span_prior.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
