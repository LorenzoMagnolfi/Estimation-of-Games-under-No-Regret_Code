"""Exact linear cost inversion for frozen history-based policy comparisons.

These are fitted-demand sensitivity sets, NOT confidence intervals. Costs
remain unknown, persistent, and otherwise unrestricted across listings.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from audit_panel import OUT, DATA_VERSION
from ask_price_grid import recover_receipts
from demand_timing_sensitivity import terminal_status_data
from empirical_pricing_rules import DESIGN, RUN, ROOT


def changed_data(s,price):
    d=s.copy();delta=price-d.Price.to_numpy()
    d['Price']=price;d['residual']=d.Price-d.Ref_Price
    d['gap_min']=d.Price-d.rival_min;d['gap_mean']=d.Price-d.rival_mean
    total=pd.Series(delta,index=d.index).groupby(d.market_id).transform('sum')
    # Preserve the raw-inventory peer count; prices of omitted owned listings
    # are held fixed. Their omitted profit spillovers are a stated limitation.
    d['own_total']=d.own_total+total
    return d


def endpoint(w,A,B,budget,bounds,sign):
    r=linprog(sign*w,A_ub=csr_matrix(-B),b_ub=budget-A,bounds=bounds,method='highs')
    return dict(status=int(r.status),value=float(sign*r.fun) if r.success else None)


def minimax(A,B,bounds,weights=None,value=None):
    # Minimize nonnegative average foregone profit over unknown costs.
    J=B.shape[1];obj=np.r_[np.zeros(J),1.]
    eq=None;rhs=None
    if weights is not None: eq=csr_matrix(np.r_[weights,0.][None,:]);rhs=[value]
    r=linprog(obj,A_ub=csr_matrix(np.c_[-B,-np.ones(len(A))]),b_ub=-A,A_eq=eq,b_eq=rhs,bounds=bounds+[(0,None)],method='highs')
    return float(r.fun) if r.success else None


def main():
    assert DATA_VERSION=='fullsource_refscreen_known'
    design=json.loads(DESIGN.read_text());data=terminal_status_data();results=[]
    for seller in design['sellers'][:2]:
        stem=seller.replace(' ','_').replace(',','').replace('.','')
        path=RUN/(stem+'_forecasts.parquet');f=pd.read_parquet(path)
        s=data[data.Seller.eq(seller)&data.date.ge(design['test_start'])].copy()
        f=f[f.date.ge(design['test_start'])].drop(columns=['Seller','Price'])
        s=s.merge(f,on=['Code','date'],validate='one_to_one').reset_index(drop=True)
        eligible=s.eligible.to_numpy();N=int(eligible.sum());T=s.date.nunique()
        codes,idx=np.unique(s.Code,return_inverse=True);J=len(codes)
        event=s.sale_terminal_status.to_numpy()*eligible
        w=np.bincount(idx,weights=event,minlength=J)/event.sum()
        x=recover_receipts(s.Price.to_numpy());cap=pd.Series(x,index=s.Code).groupby(level=0).min().reindex(codes).to_numpy()
        observed_receipt=float(event@x/event.sum())
        columns=[c for c in f if '__' in c or c.startswith('ask_shift_')]
        proposed=[]
        for name in columns:
            px=np.where(eligible,s[name].to_numpy(),s.Price.to_numpy())
            assert np.isfinite(px).all()
            proposed.append(px)
        pnew=np.column_stack(proposed);xnew=recover_receipts(pnew)
        for fitname in ['continuation_seller_full_1.joblib','multiproduct_initial_terminal.joblib']:
            fit=joblib.load(OUT/fitname)
            qa=fit.predict(s)
            assert np.max(abs(fit.predict(changed_data(s,s.Price.to_numpy()))-qa))<1e-12
            qnew=np.column_stack([fit.predict(changed_data(s,pnew[:,k])) for k in range(len(columns))])
            A=(xnew*qnew-x[:,None]*qa[:,None]).sum(axis=0)/N
            B=np.zeros((len(columns),J))
            for k in range(len(columns)):
                B[k]=np.bincount(idx,weights=qnew[:,k]-qa,minlength=J)/N
            # Direct versus coefficient identity at a heterogeneous test cost.
            c=.37*cap
            direct=((xnew-c[idx,None])*qnew-(x-c[idx])[:,None]*qa[:,None]).sum(axis=0)/N
            assert np.max(abs(direct-(A-B@c)))<1e-10
            artifact=RUN/(stem+'_'+fitname.replace('.joblib','')+'_policy_coefficients.npz')
            np.savez_compressed(artifact,A=A,B=B,weights=w,caps=cap,codes=codes,policies=np.array(columns),q_observed=qa,q_proposed=qnew,proposed_prices=pnew)
            for family in ['all_frozen','fitted_rules_only']:
                keep=np.ones(len(columns),dtype=bool) if family=='all_frozen' else np.array([not c.startswith('ask_shift_') for c in columns])
                a,b=A[keep],B[keep]
                for prior in ['nonnegative_only','no_below_observed_receipt']:
                    bounds=[(0,None)]*J if prior=='nonnegative_only' else [(0,float(v)) for v in cap]
                    record=dict(seller=seller,fit=fitname,comparison_family=family,prior=prior,eligible_listing_days=N,calendar_rounds=int(T),cost_variables=J,target_sales=int(event.sum()),mean_receipt=observed_receipt,policy_count=int(keep.sum()),prior_mean_cost_upper=None if prior=='nonnegative_only' else float(w@cap),minimum_feasible_budget=minimax(a,b,bounds),budget_at_zero_mean_cost=minimax(a,b,bounds,w,0.),profiles=[])
                    if prior!='nonnegative_only':record['budget_at_prior_upper_mean_cost']=minimax(a,b,bounds,w,float(w@cap))
                    for budget in design['regret_budget_dollars_per_eligible_listing_day']:
                        lo=endpoint(w,a,b,budget,bounds,1);hi=endpoint(w,a,b,budget,bounds,-1)
                        if lo['status'] not in [0,2,3] or hi['status'] not in [0,2,3]:raise RuntimeError((lo,hi))
                        record['profiles'].append(dict(budget=budget,lower=lo,upper=hi,margin_lower=observed_receipt-hi['value'] if hi['value'] is not None else None,margin_upper=observed_receipt-lo['value'] if lo['value'] is not None else None))
                    # Optimistic prefix-only diagnostic; not seller adoption.
                    record['idealized_exp3ix_normalized_allowance_at_95pct']=float(2*np.sqrt(2*len(a)*T*np.log(len(a)))+(np.sqrt(2*len(a)*T/np.log(len(a)))+1)*np.log(40))
                    record['coefficient_artifact']=str(artifact)
                    results.append(record)
                    print(json.dumps({k:v for k,v in record.items() if k!='profiles'}),flush=True)
    payload=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),forecasts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*_forecasts.parquet')},demand_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['continuation_seller_full_1.joblib','multiproduct_initial_terminal.joblib']},interpretation='Exact LP sensitivity sets at fixed fitted demand. No statistical coverage, algorithm adoption, or causal demand validation is claimed. Persistent unknown costs; no cost-dispersion restriction. Price rules were frozen before loading demand.',results=results)
    for dest in [RUN/'policy_cost_frontier.json',ROOT/'docs/evidence/gadgetpickup_policy_cost_frontier_20261020.json']:
        dest.write_text(json.dumps(payload,indent=2)+'\n')
    print('POLICY_COST_FRONTIER_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
