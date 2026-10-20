"""Supplementary, predeclared withholding comparison; no confidence claim."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from policy_cost_frontier import endpoint, minimax
from demand_timing_sensitivity import terminal_status_data
from empirical_pricing_rules import DESIGN, RUN, ROOT
from ask_price_grid import recover_receipts

SUPPLEMENT=ROOT/'docs/evidence/gadgetpickup_withdrawal_design_20261020.json'

def main():
    original=json.loads(DESIGN.read_text());spec=json.loads(SUPPLEMENT.read_text())
    data=terminal_status_data();out=[];audits=[]
    for seller,stem in [('GadgetPickup','gp'),('UPGRADE SOLUTION INC','up')]:
        s=data[data.Seller.eq(seller)&data.date.ge(original['test_start'])].copy()
        f=pd.read_parquet(RUN/(seller.replace(' ','_').replace(',','').replace('.','')+'_forecasts.parquet'))
        f=f[f.date.ge(original['test_start'])][['Code','date','eligible']]
        s=s.merge(f,on=['Code','date'],validate='one_to_one').reset_index(drop=True)
        codes,idx=np.unique(s.Code,return_inverse=True);J=len(codes)
        N=int(s.eligible.sum());x=recover_receipts(s.Price.to_numpy())
        events=s.sale_terminal_status.to_numpy()*s.eligible.to_numpy()
        mean_receipt=float(events@x/events.sum())
        retained=s.groupby('market_id').Code.transform('size').to_numpy()
        audits.append(dict(seller=seller,retained_rows=len(s),retained_listings=J,
            fraction_rows_raw_own_count_exceeds_retained=float(np.mean(s.own_n.to_numpy()>retained)),
            mean_raw_own_count=float(s.own_n.mean()),mean_retained_own_count=float(retained.mean()),
            maximum_omitted_own_count=float(np.max(s.own_n.to_numpy()-retained))))
        for fit in spec['demand_fits']:
            artifact=RUN/(stem+'_'+fit+'_coef.npz');z=np.load(artifact,allow_pickle=True)  # Trusted local artifacts written by policy_cost_frontier.
            assert np.array_equal(codes,z['codes'])
            A=z['A'];B=z['B'];w=z['weights'];cap=z['caps'];qa=z['q_observed']
            assert len(qa)==len(s) and np.min(qa)>0
            aw=-float(x@qa)/N;bw=-np.bincount(idx,weights=qa,minlength=J)/N
            # Withdrawal gain is negative actual profit, at arbitrary unknown costs.
            c=.37*cap
            assert abs(aw-bw@c+float((x-c[idx])@qa)/N)<1e-10
            for family in spec['comparison_sets']:
                keep=np.zeros(len(A),dtype=bool) if family=='withdrawal_only' else (np.array([not str(v).startswith('ask_shift_') for v in z['policies']]) if family=='fitted_rules_plus_withdrawal' else np.ones(len(A),dtype=bool))
                a=np.r_[A[keep],aw];b=np.vstack([B[keep],bw])
                for prior in spec['priors']:
                    bounds=[(0,None)]*J if prior=='nonnegative_only' else [(0,float(v)) for v in cap]
                    r=dict(seller=seller,fit=fit,comparison_family=family,prior=prior,eligible_listing_days=N,calendar_rounds=int(s.date.nunique()),target_sales=int(events.sum()),mean_receipt=mean_receipt,policy_count=len(a),minimum_feasible_budget=minimax(a,b,bounds),budget_at_zero_mean_cost=minimax(a,b,bounds,w,0),profiles=[])
                    prevlo=float('inf');prevhi=-float('inf')
                    for eps in spec['budgets_dollars_per_eligible_listing_day']:
                        lo=endpoint(w,a,b,eps,bounds,1);hi=endpoint(w,a,b,eps,bounds,-1)
                        assert lo['status']==0 and hi['status']==0
                        assert lo['value']<=prevlo+1e-5 and hi['value']>=prevhi-1e-5
                        prevlo=lo['value'];prevhi=hi['value']
                        r['profiles'].append(dict(budget=eps,lower=lo['value'],upper=hi['value'],margin_lower=mean_receipt-hi['value'],margin_upper=mean_receipt-lo['value']))
                        if family=='withdrawal_only' and prior=='nonnegative_only':
                            # Analytic knapsack endpoint with one budget constraint.
                            expected_exposure=-bw
                            analytic=(eps-aw)*np.max(w/expected_exposure)
                            assert abs(analytic-hi['value'])<1e-5
                    out.append(r)
                    print(seller,fit,family,prior,r['profiles'][0],flush=True)
    payload=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),design_sha256=hashlib.sha256(SUPPLEMENT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_coefficients_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*_coef.npz')},interpretation='Supplementary fitted-demand sensitivity sets. Historical feasibility, zero baseline payoff, modeled inventory coverage, and a common regret allowance are maintained assumptions; no confidence coverage is claimed.',inventory_audit=audits,results=out)
    for p in [RUN/'withdrawal_frontier.json',ROOT/'docs/evidence/gadgetpickup_withdrawal_frontier_20261020.json']:
        p.write_text(json.dumps(payload,indent=2)+'\n')
    print('WITHDRAWAL_FRONTIER_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
