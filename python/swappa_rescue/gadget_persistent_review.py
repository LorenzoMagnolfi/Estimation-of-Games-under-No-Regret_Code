"""Invert persistent-price review payoffs with unknown costs and adjustment costs.

The empirical pre-November review-gap distribution is predetermined for the
November-December inversion. This is a maintained review model, not a theorem
about arbitrary pricing algorithms or a simulation of factual inventory paths.
"""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from audit_panel import ROOT
from ask_price_grid import recover_receipts
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_demand_credibility import NoInitialDemand
from gadget_demand_time_controls import CellWeekDemand
from policy_cost_frontier import changed_data
from gadget_adjustment_cost_calibration import optimize,farkas

DESIGN=ROOT/'docs/evidence/gadgetpickup_persistent_review_design_20261020.json'


def probability(q,horizons):
    q=np.asarray(q)
    return np.mean([-np.expm1(h*np.log1p(-q)) for h in horizons],axis=0)


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old=pd.read_parquet(RUN/'repricing_performance_predictions.parquet')
    base=pd.read_parquet(RUN/'performance_predictions.parquet')
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].merge(base[['Code','date','fold']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s=s.merge(old[['Code','date','previous_date','previous_price','undo_all_q','undo_all_receipt']],on=['Code','date'],validate='one_to_one')
    s['valid']=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    s['changed']=s.valid&s.Price.ne(s.previous_price)
    batch=s[s.changed].groupby('date').agg(n=('Code','size'),models=('device_id','nunique'))
    dates=set(batch.index[(batch.n>=10)&(batch.models>=3)])
    prior=sorted(x for x in dates if x<pd.Timestamp('2023-11-01'))
    gaps=np.diff(np.array(prior,dtype='datetime64[D]')).astype(int).tolist()
    assert len(gaps)>0 and min(gaps)>=1
    schedules={'preperiod_gap_distribution':gaps,'H1':[1],'H3':[3],'H7':[7],'H14':[14]}
    print('PREPERIOD_GAPS',gaps,flush=True)
    cached=pd.read_parquet(RUN/'demand_credibility_predictions.parquet');frames=[];fits=[]
    rollback=changed_data(s,np.where(s.changed,s.previous_price,s.Price))
    for spec in ['cached_corrected_additive','additive_no_initial','category_week_C1']:
        z=s.merge(cached[cached.spec.eq(spec)].drop(columns=['Price','sale_terminal_status','category','spec']),on=['Code','date'],validate='one_to_one')
        if spec=='cached_corrected_additive':qold=z.undo_all_q.to_numpy()
        else:
            qold=np.zeros(len(s))
            for fold in range(5):
                tr=s.fold.ne(fold);te=~tr
                fit=NoInitialDemand() if spec=='additive_no_initial' else CellWeekDemand(False)
                with warnings.catch_warnings(record=True) as logs:
                    warnings.simplefilter('always',ConvergenceWarning);fit.fit(s[tr],s.loc[tr,'sale_terminal_status'])
                discrepancy=float(np.max(abs(fit.predict(s[te])-z.loc[te,'q'])))
                assert discrepancy<1e-10
                qold[te]=fit.predict(rollback.loc[te])
                fits.append(dict(spec=spec,fold=fold,max_factual_difference=discrepancy,convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs)))
                print('PERSISTENCE FIT',spec,fold,flush=True)
        z['qold']=qold;z['spec']=spec;z['x']=recover_receipts(z.Price.to_numpy());frames.append(z)
    pd.concat(frames,ignore_index=True).to_parquet(RUN/'persistent_review_predictions.parquet',index=False)
    records=[];summaries=[];identity_error=0.
    for s in frames:
        spec=s.spec.iloc[0];z=s[s.date.ge('2023-11-01')].copy()
        eligible=z.groupby(['device_id','date']).valid.transform('all')&z.date.isin(dates)
        z=z[eligible].reset_index(drop=True)
        for horizon,hs in schedules.items():
            z['P']=probability(z.q.to_numpy(),hs);z['Pold']=probability(z.qold.to_numpy(),hs)
            for shift in [-10,-5,5,10]:z['P'+str(shift)]=probability(z['q_shift_'+str(shift)].to_numpy(),hs)
            if horizon=='H1':assert np.max(abs(z.P-z.q))<1e-14
            assert np.all(z.P>=z.q-1e-14) and np.all(z.P<=1+1e-14)
            for complete in [False,True]:
                subset=[]
                for model,g in z.groupby('device_id'):
                    g=g.reset_index(drop=True);ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1
                    Q=np.bincount(ix,weights=g.P,minlength=J);R=float(g.x@g.P);w=Q/Q.sum()
                    rows=[np.r_[Q,0.]];rhs=[R];labels=['participation'];nchange=0;nidle=0
                    test_c=np.linspace(37.,413.,J);test_K=2.31
                    for date,gg in g.groupby('date'):
                        ids=gg.index.to_numpy();idx=ix[ids];q=gg.P.to_numpy();x=gg.x.to_numpy()
                        actual=float((q*(x-test_c[idx])).sum())
                        if gg.changed.any():
                            nchange+=1;qa=gg.Pold.to_numpy();xa=gg.undo_all_receipt.to_numpy()
                            row=np.r_[np.bincount(idx,weights=q-qa,minlength=J),1.];b=float(x@q-xa@qa)
                            direct=test_K-actual+float((qa*(xa-test_c[idx])).sum())
                            identity_error=max(identity_error,abs(row@np.r_[test_c,test_K]-b-direct))
                            rows.append(row);rhs.append(b);labels.append('change:'+str(date.date()))
                        else:nidle+=1
                        if not gg.changed.any() or complete:
                            for shift in [-10,-5,5,10]:
                                qa=gg['P'+str(shift)].to_numpy();xa=gg['receipt_shift_'+str(shift)].to_numpy();kc=0. if gg.changed.any() else -1.
                                row=np.r_[np.bincount(idx,weights=q-qa,minlength=J),kc];b=float(x@q-xa@qa)
                                direct=float((qa*(xa-test_c[idx])).sum())-actual+kc*test_K
                                identity_error=max(identity_error,abs(row@np.r_[test_c,test_K]-b-direct))
                                rows.append(row);rhs.append(b);labels.append(('adjusted_menu:' if gg.changed.any() else 'inaction:')+str(date.date())+':'+str(shift))
                    A=np.array(rows);b=np.array(rhs);status=optimize(np.zeros(J+1),A,b)
                    rec=dict(spec=spec,horizon=horizon,complete_menu=complete,model=str(model),reviews=int(g.date.nunique()),change_reviews=nchange,inaction_reviews=nidle,listings=int(J),expected_review_sales=float(Q.sum()),expected_receipt=R/Q.sum(),feasible=status['status']==0)
                    if rec['feasible']:
                        for name,obj in [('cost',np.r_[w,0.]),('adjustment',np.r_[np.zeros(J),1.])]:
                            lo=optimize(obj,A,b);hi=optimize(-obj,A,b)
                            rec[name+'_lower']=lo['value'];rec[name+'_upper']=-hi['value'] if hi['value'] is not None else None
                    else:
                        rec['certificate']=farkas(A,b)
                        rec['infeasible_without_participation']=optimize(np.zeros(J+1),A[1:],b[1:])['status']==2
                    records.append(rec);subset.append(rec)
                feasible=all(a['feasible'] for a in subset);total=sum(a['expected_review_sales'] for a in subset)
                summary=dict(spec=spec,horizon=horizon,complete_menu=complete,models=len(subset),eligible_model_reviews=sum(a['reviews'] for a in subset),review_listing_rows=len(z),expected_review_sales=total,expected_receipt=sum(a['expected_review_sales']*a['expected_receipt'] for a in subset)/total,infeasible_models=[a['model'] for a in subset if not a['feasible']],joint_feasible=feasible,cost_lower=sum(a['expected_review_sales']*a['cost_lower'] for a in subset)/total if feasible else None,cost_upper=sum(a['expected_review_sales']*a['cost_upper'] for a in subset)/total if feasible else None,restricted_regret_upper=sum(a['inaction_reviews']*a['adjustment_upper'] for a in subset) if feasible and complete and all(a['adjustment_upper'] is not None for a in subset) else None)
                summaries.append(summary);print('PERSISTENCE RESULT',spec,horizon,complete,len(summary['infeasible_models']),flush=True)
    assert identity_error<1e-8
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),preperiod_batch_dates=[str(x.date()) for x in prior],gap_distribution=gaps,gap_mean=float(np.mean(gaps)),gap_median=float(np.median(gaps)),fits=fits,profit_identity_max_error=identity_error,summaries=summaries,models=records,interpretation='Persistent stationary-hazard conditional review model. Unknown listing opportunity costs and adjustment costs; no calibrated empirical regret bound or confidence region unless all economic/demand restrictions are established. Review-exposure target differs from prior daily target.')
    (ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PERSISTENT_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
