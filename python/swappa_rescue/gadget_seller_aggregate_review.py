"""Seller-wide aggregate comparisons retain all models and heterogeneous K."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from audit_panel import ROOT
from gadget_demand_audit import RUN
from gadget_persistent_review import probability

DESIGN=ROOT/'docs/evidence/gadgetpickup_seller_aggregate_review_design_20261020.json'


def endpoint(obj,A,b):
    scale=np.maximum(1.,np.maximum(abs(b),np.max(abs(A),axis=1)))
    r=linprog(obj,A_ub=csr_matrix(A/scale[:,None]),b_ub=b/scale,bounds=(0,None),method='highs')
    assert r.status in [0,2,3],r.message
    if r.success:assert np.max(A@r.x-b)<1e-5
    return dict(status=int(r.status),value=float(r.fun) if r.success else None)


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').read_text());s=pd.read_parquet(RUN/'persistent_review_predictions.parquet')
    dates=pd.to_datetime(json.loads((ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json').read_text())['batch_dates']);records=[];maxerror=0.
    schedules={'preperiod_gap_distribution':source['gap_distribution'],'H1':[1],'H3':[3],'H7':[7],'H14':[14]}
    for spec,g in s.groupby('spec'):
        g=g[g.date.ge('2023-11-01')].copy();g=g[g.groupby(['device_id','date']).valid.transform('all')&g.date.isin(dates)].reset_index(drop=True)
        ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1;mi,models=pd.factorize(g.device_id,sort=True);M=len(models)
        changed=g.groupby(['device_id','date']).changed.transform('any').to_numpy()
        reviews=g[['device_id','date']].copy();reviews['changed']=changed;reviews=reviews.drop_duplicates();C=reviews[reviews.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy();I=reviews[~reviews.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy();assert np.all(C>0)
        x=g.x.to_numpy();xo=g.undo_all_receipt.to_numpy();test=np.r_[np.linspace(37.,413.,J),np.linspace(.1,3.,M)]
        for horizon,hs in schedules.items():
            q=probability(g.q.to_numpy(),hs);qo=probability(g.qold.to_numpy(),hs);Q=np.bincount(ix,weights=q,minlength=J);R=float(x@q);w=Q/Q.sum()
            for complete in [False,True]:
                rows=[np.r_[Q,np.zeros(M)]];rhs=[R];m=changed
                row=np.r_[np.bincount(ix[m],weights=q[m]-qo[m],minlength=J),C];b=float((x[m]*q[m]-xo[m]*qo[m]).sum());rows.append(row);rhs.append(b)
                direct=float(((xo[m]-test[ix[m]])*qo[m]-(x[m]-test[ix[m]])*q[m]).sum()+C@test[J:]);maxerror=max(maxerror,abs(row@test-b-direct))
                for on_change in [False,True] if complete else [False]:
                    m=changed if on_change else ~changed
                    for shift in [-10,-5,5,10]:
                        qa=probability(g['q_shift_'+str(shift)].to_numpy(),hs);xa=g['receipt_shift_'+str(shift)].to_numpy()
                        row=np.r_[np.bincount(ix[m],weights=q[m]-qa[m],minlength=J),np.zeros(M) if on_change else -I];b=float((x[m]*q[m]-xa[m]*qa[m]).sum());rows.append(row);rhs.append(b)
                        direct=float(((xa[m]-test[ix[m]])*qa[m]-(x[m]-test[ix[m]])*q[m]).sum()-(0. if on_change else I@test[J:]));maxerror=max(maxerror,abs(row@test-b-direct))
                A=np.array(rows);b=np.array(rhs);obj=np.r_[w,np.zeros(M)];lo=endpoint(obj,A,b);hi=endpoint(-obj,A,b);allow=np.r_[np.zeros(J),I];al=endpoint(allow,A,b);ah=endpoint(-allow,A,b)
                rec=dict(spec=spec,horizon=horizon,complete_menu=complete,models=M,change_reviews=int(C.sum()),inaction_reviews=int(I.sum()),expected_review_sales=float(Q.sum()),expected_receipt=R/Q.sum(),cost_lower=lo['value'],cost_upper=-hi['value'] if hi['value'] is not None else None,regret_allowance_lower=al['value'],regret_allowance_upper=-ah['value'] if ah['value'] is not None else None,lower_status=lo['status'],upper_status=hi['status'])
                records.append(rec);print('SELLER AGGREGATE',spec,horizon,complete,rec['cost_lower'],rec['cost_upper'],rec['regret_allowance_upper'],flush=True)
    assert maxerror<1e-8
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),profit_identity_max_error=maxerror,results=records,interpretation='Maintained aggregate seller-wide comparison, heterogeneous adjustment costs, all models retained. Core bounds inaction-only shift regret. Dollar allowances and cost intervals are joint-projection sensitivity sets, not confidence intervals or externally measured losses.')
    (ROOT/'docs/evidence/gadgetpickup_seller_aggregate_review_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
