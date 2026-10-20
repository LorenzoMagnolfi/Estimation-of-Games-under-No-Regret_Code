"""Transfer review restrictions to original expected and realized cost targets."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from audit_panel import ROOT
from gadget_demand_audit import RUN
from gadget_persistent_review import probability
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def endpoint(obj,A,b):
    scale=np.full(len(obj),100.)
    row=np.maximum(1.,np.maximum(abs(b),np.max(abs(A*scale),axis=1)))
    for method,presolve in [("highs",True),("highs-ds",False),("highs-ipm",False)]:
        r=linprog(obj*scale,A_ub=csr_matrix(A*scale/row[:,None]),b_ub=b/row,bounds=(0,None),method=method,options={"presolve":presolve,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        if r.status in [2,3]:return dict(status=int(r.status),value=None)
        if r.success and np.max(A@(r.x*scale)-b)<1e-5:return dict(status=0,value=float(obj@(r.x*scale)))
    raise AssertionError((r.message,float(np.max(A@(r.x*scale)-b)) if r.x is not None else None))

DESIGN=ROOT/'docs/evidence/gadgetpickup_review_target_transfer_design_20261020.json'


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s=pd.read_parquet(RUN/'persistent_review_predictions.parquet');prior=json.loads((ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').read_text());dates=pd.to_datetime(json.loads((ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json').read_text())['batch_dates']);records=[]
    schedules={'preperiod_gap_distribution':prior['gap_distribution'],'H1':[1],'H3':[3],'H7':[7],'H14':[14]}
    for spec,z in s.groupby('spec'):
        z=z[z.date.ge('2023-11-01')].reset_index(drop=True);iz=pd.factorize(z.Code,sort=True)[0];J=iz.max()+1;keep=z.groupby(['device_id','date']).valid.transform('all')&z.date.isin(dates);g=z[keep].reset_index(drop=True);ix=iz[keep];models=sorted(g.device_id.unique());M=len(models)
        Qdaily=np.bincount(iz,weights=z.q,minlength=J);Qreal=np.bincount(iz,weights=z.sale_terminal_status,minlength=J);Rdaily=float(z.x@z.q)
        changed=g.groupby(['device_id','date']).changed.transform('any').to_numpy();reviews=g[['device_id','date']].copy();reviews['changed']=changed;reviews=reviews.drop_duplicates();C=reviews[reviews.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy();I=reviews[~reviews.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy();x=g.x.to_numpy();xo=g.undo_all_receipt.to_numpy()
        for horizon,hs in schedules.items():
            q=probability(g.q,hs);qo=probability(g.qold,hs);Qreview=np.bincount(ix,weights=q,minlength=J);R=float(x@q)
            for complete in [False,True]:
                rows=[np.r_[Qreview,np.zeros(M)]];rhs=[R];m=changed;rows.append(np.r_[np.bincount(ix[m],weights=q[m]-qo[m],minlength=J),C]);rhs.append(float((x[m]*q[m]-xo[m]*qo[m]).sum()))
                for on_change in [False,True] if complete else [False]:
                    m=changed if on_change else ~changed
                    for shift in [-10,-5,5,10]:
                        qa=probability(g['q_shift_'+str(shift)],hs);xa=g['receipt_shift_'+str(shift)].to_numpy();rows.append(np.r_[np.bincount(ix[m],weights=q[m]-qa[m],minlength=J),np.zeros(M) if on_change else -I]);rhs.append(float((x[m]*q[m]-xa[m]*qa[m]).sum()))
                for daily_participation in [False,True]:
                    A=np.array(rows+([np.r_[Qdaily,np.zeros(M)]] if daily_participation else []));b=np.array(rhs+([Rdaily] if daily_participation else []))
                    for target,Q in [('review_exposure_expected_sales',Qreview),('original_daily_expected_sales',Qdaily),('original_realized_sales',Qreal)]:
                        obj=np.r_[Q/Q.sum(),np.zeros(M)];lo=endpoint(obj,A,b);hi=endpoint(-obj,A,b)
                        records.append(dict(spec=spec,horizon=horizon,complete_menu=complete,daily_participation=daily_participation,target=target,lower_status=lo['status'],upper_status=hi['status'],cost_lower=lo['value'],cost_upper=-hi['value'] if hi['value'] is not None else None,listings=int(J),reviewed_listings=int(np.count_nonzero(Qreview)),unreviewed_daily_mass_share=float(Qdaily[Qreview==0].sum()/Qdaily.sum()),unreviewed_realized_mass_share=float(Qreal[Qreview==0].sum()/Qreal.sum())))
            print('TARGET TRANSFER',spec,horizon,flush=True)
    out=dict(code_commit=commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),results=records,interpretation='Same review restrictions, all original-window listing costs included. Daily participation is an additional retained prior assumption; review and original cost targets are not interchangeable. Status 3 means unbounded.')
    (ROOT/'docs/evidence/gadgetpickup_review_target_transfer_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
