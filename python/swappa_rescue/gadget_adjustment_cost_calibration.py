"""Test whether changes and inaction jointly bound a stable adjustment cost.

This tests a daily-payoff approximation. It is not an estimated regret
allowance when the inequalities fail, and does not price multi-day benefits.
"""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from ask_price_grid import recover_receipts
from audit_panel import ROOT
from gadget_demand_audit import RUN

DESIGN=ROOT/'docs/evidence/gadgetpickup_empirical_allowance_demand_design_20261020.json'


def optimize(obj,A,b):
    # Scale dollar cost coordinates and constraint rows without changing the set.
    scale=np.r_[np.full(len(obj)-1,100.),1.]
    row_scale=np.maximum(1.,np.maximum(abs(b),np.max(abs(A*scale),axis=1)))
    AA=A*scale/row_scale[:,None];bb=b/row_scale
    for method,presolve in [('highs',True),('highs-ds',False),('highs-ipm',False)]:
        r=linprog(obj*scale,A_ub=csr_matrix(AA),b_ub=bb,bounds=(0,None),method=method,options={'presolve':presolve})
        if r.status in [0,2,3]:break
    assert r.status in [0,2,3],r.message
    if r.success:assert np.max(A@(r.x*scale)-b)<1e-4
    return dict(status=int(r.status),value=float(obj@(r.x*scale)) if r.success else None)


def farkas(A,b):
    # y>=0, A'y>=0 and b'y<0 certify infeasibility for nonnegative unknowns.
    rs=np.maximum(1.,np.maximum(abs(b),np.max(abs(A),axis=1)));AA=A/rs[:,None];bb=b/rs
    r=linprog(bb,A_ub=csr_matrix(-AA.T),b_ub=np.zeros(A.shape[1]),A_eq=np.ones((1,len(b))),b_eq=[1.],bounds=(0,None),method='highs')
    assert r.success and r.fun < -1e-8
    assert np.min(AA.T@r.x)>-1e-7
    return dict(normalized_certificate_margin=float(-r.fun),minimum_dual_component=float(np.min(AA.T@r.x)))


def main():
    p=pd.read_parquet(RUN/'performance_predictions.parquet');r=pd.read_parquet(RUN/'repricing_performance_predictions.parquet')
    s=p.merge(r[['Code','date','device_id','previous_date','previous_price','undo_all_q','undo_all_receipt']],on=['Code','date'],validate='one_to_one')
    s['valid']=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    s['change']=s.valid&s.Price.ne(s.previous_price);s['x']=recover_receipts(s.Price.to_numpy())
    batch=s[s.change].groupby('date').agg(changes=('Code','size'),models=('device_id','nunique'))
    batchdates=set(batch.index[(batch.changes>=10)&(batch.models>=3)])
    records=[];summary=[]
    for window in ['full','novdec']:
        z=s if window=='full' else s[s.date.ge('2023-11-01')]
        for schedule in ['all_retained_dates','batch_dates']:
            for complete in [False,True]:
                partial=[]
                for model,g in z.groupby('device_id'):
                    g=g.reset_index(drop=True);ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1
                    Q=np.bincount(ix,weights=g.q,minlength=J);R=float(g.x@g.q);w=Q/Q.sum()
                    rows=[np.r_[Q,0.]];rhs=[R];labels=['participation'];changed=0;inactive=0;days=[]
                    for date,gg in g.groupby('date'):
                        if not gg.valid.all() or (schedule=='batch_dates' and date not in batchdates):continue
                        ids=gg.index.to_numpy();idx=ix[ids];q=gg.q.to_numpy();x=gg.x.to_numpy();qold=gg.undo_all_q.to_numpy();xold=gg.undo_all_receipt.to_numpy()
                        if gg.change.any():
                            changed+=1
                            rows.append(np.r_[np.bincount(idx,weights=q-qold,minlength=J),1.]);rhs.append(float(x@q-xold@qold));labels.append('change:'+str(date.date()))
                            if complete:
                                for shift in [-10,-5,5,10]:
                                    qk=gg['q_shift_'+str(shift)].to_numpy();xk=gg['receipt_shift_'+str(shift)].to_numpy()
                                    rows.append(np.r_[np.bincount(idx,weights=q-qk,minlength=J),0.]);rhs.append(float(x@q-xk@qk));labels.append('changed_optimal:'+str(date.date())+':'+str(shift))
                        else:
                            inactive+=1
                            for shift in [-10,-5,5,10]:
                                qk=gg['q_shift_'+str(shift)].to_numpy();xk=gg['receipt_shift_'+str(shift)].to_numpy()
                                rows.append(np.r_[np.bincount(idx,weights=q-qk,minlength=J),-1.]);rhs.append(float(x@q-xk@qk));labels.append('inaction:'+str(date.date())+':'+str(shift))
                        days.append(str(date.date()))
                    A=np.array(rows);b=np.array(rhs);check=optimize(np.zeros(J+1),A,b)
                    rec=dict(window=window,schedule=schedule,complete_menu=complete,model=str(model),rows=len(g),expected_sales=float(Q.sum()),eligible_change_days=changed,eligible_inaction_days=inactive,eligible_dates=days,feasible=check['status']==0)
                    if rec['feasible']:
                        rec['cost_lower']=optimize(np.r_[w,0.],A,b);hi=optimize(np.r_[-w,0.],A,b);hi['value']=-hi['value'] if hi['value'] is not None else None;rec['cost_upper']=hi
                        rec['adjustment_lower']=optimize(np.r_[np.zeros(J),1.],A,b);hi=optimize(np.r_[np.zeros(J),-1.],A,b);hi['value']=-hi['value'] if hi['value'] is not None else None;rec['adjustment_upper']=hi
                    else:
                        # Report an irreducible subset of behavioral rows; no slack chosen to rescue it.
                        active=list(range(len(b)))
                        for j in list(active):
                            if j==0:continue
                            reduced=[i for i in active if i!=j]
                            if optimize(np.zeros(J+1),A[reduced],b[reduced])['status']==2:active=reduced
                        rec['infeasible_subset']=[labels[j] for j in active]
                        rec['infeasibility_certificate']=farkas(A[active],b[active])
                    records.append(rec);partial.append(rec)
                total=sum(a['expected_sales'] for a in partial);feas=all(a['feasible'] for a in partial)
                summary.append(dict(window=window,schedule=schedule,complete_menu=complete,models=len(partial),infeasible_models=[a['model'] for a in partial if not a['feasible']],models_with_changes_and_inaction=sum(a['eligible_change_days']>0 and a['eligible_inaction_days']>0 for a in partial),identified_adjustment_share=sum(a['expected_sales'] for a in partial if a['eligible_change_days']>0 and a['eligible_inaction_days']>0)/total,joint_feasible=feas,cost_lower=sum(a['expected_sales']*a['cost_lower']['value'] for a in partial)/total if feas else None,cost_upper=sum(a['expected_sales']*a['cost_upper']['value'] for a in partial)/total if feas else None))
                print('ADJUSTMENT COMPLETE',window,schedule,complete,summary[-1],flush=True)
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),batch_dates=[str(x.date()) for x in sorted(batchdates)],summaries=summary,models=records,interpretation='Daily-payoff stable adjustment-cost model, jointly unknown listing costs and K. Complete-menu variant additionally requires no profitable listed alternative on adjustment dates. No profit percentage is calibrated if the model is infeasible; no inference on multi-day adjustment benefits.')
    for p in [RUN/'adjustment_cost_calibration.json',ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json']:p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('ADJUSTMENT_CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':main()
