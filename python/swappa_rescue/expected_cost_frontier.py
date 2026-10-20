"""Separate expected-sales cost target, with all unrestricted listing costs."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from policy_cost_frontier import endpoint,minimax
from demand_timing_sensitivity import terminal_status_data
from empirical_pricing_rules import DESIGN,RUN,ROOT
from ask_price_grid import recover_receipts

SPEC=ROOT/'docs/evidence/gadgetpickup_expected_cost_design_20261020.json'

def main():
    d=json.loads(DESIGN.read_text());spec=json.loads(SPEC.read_text());data=terminal_status_data();out=[]
    for seller,stem in [('GadgetPickup','gp'),('UPGRADE SOLUTION INC','up')]:
        s=data[data.Seller.eq(seller)&data.date.ge(d['test_start'])].copy()
        f=pd.read_parquet(RUN/(seller.replace(' ','_').replace(',','').replace('.','')+'_forecasts.parquet'))
        s=s.merge(f[f.date.ge(d['test_start'])][['Code','date','eligible']],on=['Code','date'],validate='one_to_one').reset_index(drop=True)
        codes,idx=np.unique(s.Code,return_inverse=True);J=len(codes);N=int(s.eligible.sum());x=recover_receipts(s.Price.to_numpy())
        for fit in ['seller','common']:
            z=np.load(RUN/(stem+'_'+fit+'_coef.npz'),allow_pickle=True)
            assert np.array_equal(codes,z['codes'])
            qa=z['q_observed'];mass=np.bincount(idx,weights=qa,minlength=J);Q=float(mass.sum());w=mass/Q
            receipt=float(x@qa/Q);aw=-float(x@qa/N);bw=-mass/N
            for family in spec['comparisons']:
                a=z['A'] if family=='price_only' else (np.array([aw]) if family=='withdrawal_only' else np.r_[z['A'],aw])
                b=z['B'] if family=='price_only' else (bw[None,:] if family=='withdrawal_only' else np.vstack([z['B'],bw]))
                for prior in spec['priors']:
                    bounds=[(0,None)]*J if prior=='nonnegative_only' else [(0,float(v)) for v in z['caps']]
                    r=dict(seller=seller,fit=fit,comparison_family=family,prior=prior,eligible_listing_days=N,expected_sales=Q,expected_mean_receipt=receipt,prior_upper=None if prior=='nonnegative_only' else float(w@z['caps']),minimum_feasible_budget=minimax(a,b,bounds),budget_at_zero_mean_cost=minimax(a,b,bounds,w,0),profiles=[])
                    for eps in spec['budgets']:
                        lo=endpoint(w,a,b,eps,bounds,1);hi=endpoint(w,a,b,eps,bounds,-1)
                        assert lo['status'] in [0,2,3] and hi['status'] in [0,2,3]
                        r['profiles'].append(dict(budget=eps,lower=lo,upper=hi,margin_lower=receipt-hi['value'] if hi['value'] is not None else None,margin_upper=receipt-lo['value'] if lo['value'] is not None else None))
                        if family=='withdrawal_only' and prior=='nonnegative_only':
                            assert abs(hi['value']-(receipt+N*eps/Q))<1e-6
                    print(seller,fit,family,prior,r['profiles'][0],flush=True);out.append(r)
    payload=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),design_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='Fitted-demand sensitivity only. Different estimand from observed-sale-weighted costs; no cost homogeneity or cap imposed in nonnegative_only cases.',results=out)
    for p in [RUN/'expected_frontier.json',ROOT/'docs/evidence/gadgetpickup_expected_frontier_20261020.json']:p.write_text(json.dumps(payload,indent=2)+'\n')
    print('EXPECTED_FRONTIER_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
