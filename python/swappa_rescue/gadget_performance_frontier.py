"""Invert limited relative-profit performance at corrected fitted demand.

No algorithm adoption, cost caps, common costs, or statistical coverage is
claimed. The design records the comparison family and sensitivity axes.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit, logit
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from ask_price_grid import recover_receipts
from audit_panel import ROOT, DATA_VERSION
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_demand_crossfit import WeekDemand
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_performance_design_20261020.json'
RESULT=ROOT/'docs/evidence/gadgetpickup_performance_results_20261020.json'


def solve(w, a, b, sign):
    """Optimize the mean cost subject to a - b c <= 0 and c >= 0."""
    r=linprog(sign*w, A_ub=csr_matrix(-b), b_ub=-a,
              bounds=(0,None), method='highs')
    out=dict(status=int(r.status), value=float(w@r.x) if r.success else None)
    if r.success:
        violation=float(np.max(a-b@r.x))
        assert violation<1e-5 and np.min(r.x)>-1e-6
        out.update(max_violation=violation, largest_cost=float(r.x.max()),
                   positive_costs=int(np.sum(r.x>1e-7)),
                   binding_rows=np.flatnonzero(abs(a-b@r.x)<1e-5).tolist())
    assert r.status in (0,2,3),r.message
    return out


def main():
    assert DATA_VERSION=='fullsource_refscreen_known'
    design=json.loads(DESIGN.read_text()); shifts=design['shifts_integer_ask']
    cached=pd.read_parquet(RUN/'observation_corrected_predictions.parquet')
    data=terminal_status_data()
    s=data[data.Seller.eq('GadgetPickup')].merge(cached[['Code','date','q','fold']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    price=s.Price.to_numpy(); x=recover_receipts(price); ask=np.rint(x/1.03)
    eligible=s.since_seen.gt(0).to_numpy()
    xnew=np.column_stack([1.03*np.maximum(1,ask+delta*eligible) for delta in shifts])
    pnew=np.ceil(xnew-1e-9)
    assert np.allclose(recover_receipts(pnew),xnew)
    changed=[changed_data(s,pnew[:,k]) for k in range(len(shifts))]
    q=np.zeros(len(s)); qnew=np.zeros_like(xnew); folds=[]
    for fold in range(5):
        tr=s.fold.ne(fold);te=~tr
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning)
            fit=WeekDemand('additive').fit(s[tr],s.loc[tr,'sale_terminal_status'])
        q[te]=fit.predict(s[te])
        for k in range(len(shifts)):qnew[te,k]=fit.predict(changed[k].loc[te])
        folds.append(dict(fold=fold,iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs)))
        print('PERFORMANCE DEMAND FOLD',fold,flush=True)
    assert np.max(abs(q-s.q.to_numpy()))<1e-10
    prediction=s[['Code','date','Price','sale_terminal_status','fold','category']].copy();prediction['q']=q
    for k,delta in enumerate(shifts):
        prediction['q_shift_'+str(delta)]=qnew[:,k];prediction['receipt_shift_'+str(delta)]=xnew[:,k]
    pp=RUN/'performance_predictions.parquet';prediction.to_parquet(pp,index=False)
    rows=[];summaries=[];certificates=[]
    for window in design['windows']:
        use=np.ones(len(s),bool) if window=='full' else s.date.ge('2023-11-01').to_numpy()
        z=s.loc[use]; codes,ix=np.unique(z.Code,return_inverse=True);J=len(codes)
        q0=q[use];x0=x[use];y=z.sale_terminal_status.to_numpy();Q0=np.bincount(ix,weights=q0,minlength=J); R0=float(x0@q0)
        w=Q0/Q0.sum();wy=np.bincount(ix,weights=y,minlength=J)/y.sum()
        base=dict(window=window,rows=len(z),dates=int(z.date.nunique()),listings=J,sales=int(y.sum()),expected_sales=float(Q0.sum()),expected_revenue=R0,expected_mean_receipt=R0/Q0.sum(),realized_mean_receipt=float(x0@y/y.sum()),continuation_rows=int(eligible[use].sum()))
        for scale in design['demand_response_scales']:
            qc=expit(logit(q0[:,None])+scale*(logit(qnew[use])-logit(q0[:,None])))
            rc=(xnew[use]*qc).sum(axis=0)
            Qc=np.column_stack([np.bincount(ix,weights=qc[:,k],minlength=J) for k in range(len(shifts))]).T
            diagnostics=[]
            for k,delta in enumerate(shifts):
                diagnostics.append(dict(shift=delta,expected_sales=float(Qc[k].sum()),revenue=float(rc[k]),revenue_gain=float(rc[k]-R0),extra_sales=float(Qc[k].sum()-Q0.sum()),zero_cost_shortfall=max(0.,float(1-R0/rc[k]))))
            summaries.append(dict(**base,response_scale=scale,comparisons=diagnostics))
            for family,ds in design['families'].items():
                ks=[shifts.index(delta) for delta in ds]
                critical=max(0.,float(1-R0/max(rc[ks])))
                for rho in design['rho_grid']:
                    a=np.r_[-R0,(1-rho)*rc[ks]-R0]
                    b=np.vstack([-Q0,(1-rho)*Qc[ks]-Q0])
                    for target,weights in [('expected',w),('realized',wy)]:
                        lo=solve(weights,a,b,1);hi=solve(weights,a,b,-1)
                        rows.append(dict(window=window,response_scale=scale,family=family,rho=rho,target=target,zero_cost_critical_rho=critical,lower=lo,upper=hi))
                    if window=='novdec' and scale==1 and family=='four_shifts':
                        ratios=np.max(b/w,axis=1)
                        lowercert=np.where((a>0)&(ratios>0),a/np.maximum(ratios,1e-300),0)
                        certificates.append(dict(rho=rho,row_labels=['participation']+[str(k) for k in ds],a=a.tolist(),max_b_over_w=ratios.tolist(),individual_lower_certificates=lowercert.tolist()))
            # The participation upper bound is exact when used alone.
            check=solve(w,np.array([-R0]),-Q0[None,:],-1)
            assert abs(check['value']-R0/Q0.sum())<1e-6
        print('PERFORMANCE WINDOW COMPLETE',window,flush=True)
    # Economically meaningful monotonicity: loosening performance expands sets.
    for window in design['windows']:
        for scale in design['demand_response_scales']:
            for family in design['families']:
                for target in ['expected','realized']:
                    rr=[r for r in rows if (r['window'],r['response_scale'],r['family'],r['target'])==(window,scale,family,target)]
                    prev=None
                    for r in rr:
                        if r['lower']['status']==0 and r['upper']['status']==0:
                            pair=(r['lower']['value'],r['upper']['value'])
                            if prev is not None:assert pair[0]<=prev[0]+1e-6 and pair[1]>=prev[1]-1e-6
                            prev=pair
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),input_prediction_sha256=hashlib.sha256((RUN/'observation_corrected_predictions.parquet').read_bytes()).hexdigest(),output_prediction_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),folds=folds,summaries=summaries,profiles=rows,lower_certificates=certificates,interpretation=design['interpretation'],checks=['Exact ask-grid compatibility','Factual predictions reproduce corrected audit within 1e-10','LP feasibility residuals below 1e-5','Analytic participation endpoint','Nested sets as rho increases'])
    for p in [RESULT,RUN/'performance_frontier.json']:p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PERFORMANCE_FRONTIER_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
