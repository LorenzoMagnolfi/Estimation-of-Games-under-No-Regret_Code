"""Weaker aggregate rollback comparison for persistent price decisions."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from audit_panel import ROOT
from gadget_demand_audit import RUN
from gadget_persistent_review import probability

DESIGN=ROOT/'docs/evidence/gadgetpickup_persistent_purposeful_design_20261020.json'


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads((ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').read_text())
    s=pd.read_parquet(RUN/'persistent_review_predictions.parquet')
    batch=json.loads((ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json').read_text())['batch_dates']
    schedules={'preperiod_gap_distribution':source['gap_distribution'],'H1':[1],'H3':[3],'H7':[7],'H14':[14]}
    records=[];maxerror=0.
    for spec,g in s.groupby('spec'):
        g=g[g.date.ge('2023-11-01')].copy();keep=g.groupby(['device_id','date']).valid.transform('all')&g.date.isin(pd.to_datetime(batch));g=g[keep].reset_index(drop=True)
        ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1;x=g.x.to_numpy();xo=g.undo_all_receipt.to_numpy()
        for horizon,hs in schedules.items():
            q=probability(g.q.to_numpy(),hs);qo=probability(g.qold.to_numpy(),hs);Q=np.bincount(ix,weights=q,minlength=J);w=Q/Q.sum();R=float(x@q);test=np.linspace(37.,413.,J)
            for level in ['pooled','phone_model']:
                masks=[np.ones(len(g),bool)] if level=='pooled' else [g.device_id.eq(k).to_numpy() for k in sorted(g.device_id.unique())]
                aa=[];bb=[]
                for m in masks:
                    aa.append(np.bincount(ix[m],weights=q[m],minlength=J));bb.append(float(x[m]@q[m]))
                    row=np.bincount(ix[m],weights=q[m]-qo[m],minlength=J);b=float(x[m]@q[m]-xo[m]@qo[m]);aa.append(row);bb.append(b)
                    direct=float(((xo[m]-test[ix[m]])*qo[m]-(x[m]-test[ix[m]])*q[m]).sum());maxerror=max(maxerror,abs(row@test-b-direct))
                A=csr_matrix(np.array(aa));b=np.array(bb);lo=linprog(w,A_ub=A,b_ub=b,bounds=(0,None),method='highs');hi=linprog(-w,A_ub=A,b_ub=b,bounds=(0,None),method='highs')
                assert lo.status in [0,2,3] and hi.status in [0,2,3]
                rec=dict(spec=spec,horizon=horizon,level=level,review_listing_rows=len(g),listings=int(J),expected_review_sales=float(Q.sum()),expected_receipt=R/Q.sum(),lower_status=int(lo.status),upper_status=int(hi.status),cost_lower=float(lo.fun) if lo.success else None,cost_upper=float(-hi.fun) if hi.success else None,actual_minus_rollback_revenue=float(x@q-xo@qo),actual_minus_rollback_expected_sales=float(q.sum()-qo.sum()))
                records.append(rec);print('PERSISTENT PURPOSEFUL',spec,horizon,level,rec['cost_lower'],rec['cost_upper'],flush=True)
    assert maxerror<1e-8
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),profit_identity_max_error=maxerror,results=records,interpretation='Conditional on fixed demand and review law, aggregate no-harm rollback assumption. This is not an empirically calibrated regret allowance against unimplemented prices or a confidence region.')
    (ROOT/'docs/evidence/gadgetpickup_persistent_purposeful_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
