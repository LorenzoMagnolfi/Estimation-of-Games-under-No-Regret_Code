"""Aggregate review comparisons allow individual pricing mistakes."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from audit_panel import ROOT
from gadget_demand_audit import RUN
from gadget_persistent_review import probability
from gadget_adjustment_cost_calibration import optimize,farkas

DESIGN=ROOT/'docs/evidence/gadgetpickup_aggregate_review_design_20261020.json'


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').read_text());s=pd.read_parquet(RUN/'persistent_review_predictions.parquet')
    dates=pd.to_datetime(json.loads((ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json').read_text())['batch_dates'])
    schedules={'preperiod_gap_distribution':source['gap_distribution'],'H1':[1],'H3':[3],'H7':[7],'H14':[14]};records=[];summaries=[]
    for spec,z in s.groupby('spec'):
        z=z[z.date.ge('2023-11-01')].copy();z=z[z.groupby(['device_id','date']).valid.transform('all')&z.date.isin(dates)].reset_index(drop=True)
        for horizon,hs in schedules.items():
            z['P']=probability(z.q.to_numpy(),hs);z['Pold']=probability(z.qold.to_numpy(),hs)
            for shift in [-10,-5,5,10]:z['P'+str(shift)]=probability(z['q_shift_'+str(shift)].to_numpy(),hs)
            for complete in [False,True]:
                partial=[]
                for model,g in z.groupby('device_id'):
                    g=g.reset_index(drop=True);ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1
                    Q=np.bincount(ix,weights=g.P,minlength=J);R=float(g.x@g.P);w=Q/Q.sum()
                    change=g.groupby('date').changed.transform('any').to_numpy();C=int(g.loc[change,'date'].nunique());I=int(g.loc[~change,'date'].nunique())
                    assert C>0 and I>0
                    rows=[np.r_[Q,0.]];rhs=[R]
                    m=change;rows.append(np.r_[np.bincount(ix[m],weights=(g.P-g.Pold).to_numpy()[m],minlength=J),float(C)]);rhs.append(float((g.x*g.P-g.undo_all_receipt*g.Pold)[m].sum()))
                    for shifting in [False,True] if complete else [False]:
                        m=change if shifting else ~change
                        for shift in [-10,-5,5,10]:
                            qa=g['P'+str(shift)].to_numpy();xa=g['receipt_shift_'+str(shift)].to_numpy()
                            rows.append(np.r_[np.bincount(ix[m],weights=g.P.to_numpy()[m]-qa[m],minlength=J),0. if shifting else -float(I)])
                            rhs.append(float((g.x.to_numpy()[m]*g.P.to_numpy()[m]-xa[m]*qa[m]).sum()))
                    A=np.array(rows);b=np.array(rhs);check=optimize(np.zeros(J+1),A,b)
                    rec=dict(spec=spec,horizon=horizon,complete_menu=complete,model=str(model),changes=C,inactions=I,expected_review_sales=float(Q.sum()),expected_receipt=R/Q.sum(),feasible=check['status']==0)
                    if rec['feasible']:
                        for name,obj in [('cost',np.r_[w,0.]),('adjustment',np.r_[np.zeros(J),1.])]:
                            lo=optimize(obj,A,b);hi=optimize(-obj,A,b);rec[name+'_lower']=lo['value'];rec[name+'_upper']=-hi['value'] if hi['value'] is not None else None
                    else:rec['certificate']=farkas(A,b)
                    partial.append(rec);records.append(rec)
                feasible=all(a['feasible'] for a in partial);total=sum(a['expected_review_sales'] for a in partial)
                rec=dict(spec=spec,horizon=horizon,complete_menu=complete,models=len(partial),infeasible_models=[a['model'] for a in partial if not a['feasible']],joint_feasible=feasible,expected_review_sales=total,expected_receipt=sum(a['expected_review_sales']*a['expected_receipt'] for a in partial)/total,cost_lower=sum(a['expected_review_sales']*a['cost_lower'] for a in partial)/total if feasible else None,cost_upper=sum(a['expected_review_sales']*a['cost_upper'] for a in partial)/total if feasible else None,regret_allowance_lower=sum(a['inactions']*a['adjustment_lower'] for a in partial) if feasible else None,regret_allowance_upper=sum(a['inactions']*a['adjustment_upper'] for a in partial) if feasible and all(a['adjustment_upper'] is not None for a in partial) else None)
                summaries.append(rec);print('AGGREGATE REVIEW',spec,horizon,complete,rec['cost_lower'],rec['cost_upper'],rec['regret_allowance_upper'],len(rec['infeasible_models']),flush=True)
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),summaries=summaries,models=records,interpretation='Joint projection of costs and adjustment costs under aggregate maintained performance assumptions. No arbitrary percentage, individual-review optimality, algorithm adoption, or statistical coverage is asserted. Core regret allowance only covers shifts on inaction reviews.')
    (ROOT/'docs/evidence/gadgetpickup_aggregate_review_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
