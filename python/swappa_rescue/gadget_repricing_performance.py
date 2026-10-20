"""Purposeful repricing: compare actual choices with the previous fresh quote."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from ask_price_grid import recover_receipts
from audit_panel import ROOT,OUT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_demand_crossfit import WeekDemand
from gadget_performance_frontier import solve
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_repricing_performance_design_20261020.json'


def main():
    design=json.loads(DESIGN.read_text())
    cached=pd.read_parquet(RUN/'observation_corrected_predictions.parquet')
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].merge(cached[['Code','date','fold','q']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    raw=pd.read_parquet(OUT/'panel.parquet');raw=raw[raw.Seller.eq('GadgetPickup')].sort_values(['Code','date'])
    raw['previous_date']=raw.groupby('Code').date.shift();raw['previous_price']=raw.groupby('Code').Price.shift()
    s=s.merge(raw[['Code','date','previous_date','previous_price']],on=['Code','date'],validate='one_to_one')
    valid=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    sign=np.sign(s.Price-s.previous_price).fillna(0).to_numpy();active=valid.to_numpy()&(sign!=0)
    masks=[active,active&(sign<0),active&(sign>0)];names=['undo_all','undo_cuts','undo_increases']
    p0=s.Price.to_numpy();pold=np.column_stack([np.where(m,s.previous_price.to_numpy(),p0) for m in masks]);xold=recover_receipts(pold);x0=recover_receipts(p0)
    changed=[changed_data(s,pold[:,k]) for k in range(3)]
    q0=s.q.to_numpy();qp=np.zeros_like(pold);folds=[]
    for fold in range(5):
        tr=s.fold.ne(fold);te=~tr
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit=WeekDemand('additive').fit(s[tr],s.loc[tr,'sale_terminal_status'])
        assert np.max(abs(fit.predict(s[te])-q0[te]))<1e-10
        for k in range(3):qp[te,k]=fit.predict(changed[k].loc[te])
        folds.append(dict(fold=fold,iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs)))
        print('REPRICING FOLD',fold,flush=True)
    pred=s[['Code','date','Price','previous_date','previous_price','device_id','sale_terminal_status']].copy();pred['q']=q0
    for k,n in enumerate(names):pred[n+'_q']=qp[:,k];pred[n+'_receipt']=xold[:,k]
    prediction_path=RUN/'repricing_performance_predictions.parquet';pred.to_parquet(prediction_path,index=False)
    results=[];summaries=[]
    for window in design['windows']:
        use=np.ones(len(s),bool) if window=='full' else s.date.ge('2023-11-01').to_numpy();z=s.loc[use];ix=pd.factorize(z.Code,sort=True)[0];J=ix.max()+1
        q=q0[use];x=x0[use];y=z.sale_terminal_status.to_numpy();mass=np.bincount(ix,weights=q,minlength=J);w=mass/mass.sum();wy=np.bincount(ix,weights=y,minlength=J)/y.sum()
        for scale in design['response_scales']:
            qc=expit(logit(q[:,None])+scale*(logit(qp[use])-logit(q[:,None])))
            xnew=xold[use];records=[]
            for k,n in enumerate(names):
                changedrows=masks[k][use]
                records.append(dict(comparison=n,changed_listing_days=int(changedrows.sum()),change_dates=int(z.loc[changedrows,'date'].nunique()),changed_listings=int(z.loc[changedrows,'Code'].nunique()),receipt_change_quantiles=np.quantile((x-xnew[:,k])[changedrows],[0,.25,.5,.75,1]).tolist() if changedrows.any() else [],rollback_revenue_gain=float(xnew[:,k]@qc[:,k]-x@q),rollback_sales_change=float(qc[:,k].sum()-q.sum())))
            summaries.append(dict(window=window,response_scale=scale,comparisons=records))
            for level in ['pooled','device']:
                gm=[np.ones(len(z),bool)] if level=='pooled' else [z.device_id.eq(g).to_numpy() for g in sorted(z.device_id.unique())]
                for family,ks in [('all',[0]),('split',[1,2]),('all_and_split',[0,1,2])]:
                    aa=[];bb=[]
                    for m in gm:
                        aa.append(-float(x[m]@q[m]));bb.append(-np.bincount(ix[m],weights=q[m],minlength=J))
                        for k in ks:
                            aa.append(float(xnew[m,k]@qc[m,k]-x[m]@q[m]));bb.append(np.bincount(ix[m],weights=qc[m,k]-q[m],minlength=J))
                    a=np.array(aa);b=np.array(bb)
                    for target,weights in [('expected',w),('realized',wy)]:
                        results.append(dict(window=window,response_scale=scale,level=level,family=family,target=target,lower=solve(weights,a,b,1),upper=solve(weights,a,b,-1)))
            print('REPRICING COMPLETE',window,scale,flush=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),folds=folds,summaries=summaries,profiles=results,interpretation=design['qualification'])
    for p in [RUN/'repricing_performance.json',ROOT/'docs/evidence/gadgetpickup_repricing_performance_results_20261020.json']:p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('REPRICING_PERFORMANCE_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
