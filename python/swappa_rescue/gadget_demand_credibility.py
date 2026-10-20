"""Corrected demand response, support, exact fixed effects and calibration audit."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.iv import AbsorbingLS
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from ask_price_grid import recover_receipts
from audit_panel import ROOT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN,metrics
from gadget_demand_crossfit import WeekDemand
from gadget_demand_time_controls import CellWeekDemand
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_empirical_allowance_demand_design_20261020.json'

class NoInitialDemand(WeekDemand):
    def __init__(self):super().__init__('additive')
    def continuous(self,d,action=None):
        return super().continuous(d,action).drop(columns=['initial_residual','initial_reference'])


def main():
    previous=pd.read_parquet(RUN/'performance_predictions.parquet')
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].merge(previous[['Code','date','fold']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s['category_week']=s.category+'::'+s.week.astype(str)
    s=s.merge(previous.drop(columns=['Price','sale_terminal_status','category','fold']),on=['Code','date'],validate='one_to_one')
    x=recover_receipts(s.Price.to_numpy());ask=np.rint(x/1.03);shifts=[-10,-5,5,10];eligible=s.since_seen.gt(0).to_numpy()
    xn=np.column_stack([1.03*np.maximum(1,ask+k*eligible) for k in shifts]);pn=np.ceil(xn-1e-9)
    alt=[changed_data(s,pn[:,k]) for k in range(4)]
    records=[];folds=[];predictions=[]
    for name in ['cached_corrected_additive','category_week_C1','category_week_C10','additive_no_initial']:
        if name=='cached_corrected_additive':q=s.q.to_numpy();qn=s[['q_shift_'+str(k) for k in shifts]].to_numpy()
        else:
            q=np.zeros(len(s));qn=np.zeros_like(xn)
            for fold in range(5):
                tr=s.fold.ne(fold);te=~tr
                fit=NoInitialDemand() if name=='additive_no_initial' else CellWeekDemand(False)
                if name=='category_week_C10':fit.model.set_params(C=10.,max_iter=6000)
                with warnings.catch_warnings(record=True) as logs:
                    warnings.simplefilter('always',ConvergenceWarning);fit.fit(s[tr],s.loc[tr,'sale_terminal_status'])
                q[te]=fit.predict(s[te])
                for k in range(4):qn[te,k]=fit.predict(alt[k].loc[te])
                folds.append(dict(spec=name,fold=fold,iterations=int(fit.model.n_iter_[0]),warning=any(isinstance(w.message,ConvergenceWarning) for w in logs),effects=fit.effects()))
                print('CREDIBILITY FOLD',name,fold,flush=True)
        out=s[['Code','date','category','Price','sale_terminal_status']].copy();out['q']=q;out['spec']=name
        for k,delta in enumerate(shifts):out['q_shift_'+str(delta)]=qn[:,k];out['receipt_shift_'+str(delta)]=xn[:,k]
        predictions.append(out)
        for window,use in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
            rev=float(x[use]@q[use]);rnew=(xn[use]*qn[use]).sum(axis=0)
            records.append(dict(spec=name,window=window,metrics=metrics(s[use],q[use]),expected_revenue=rev,changes=[dict(shift=delta,extra_sales=float((qn[use,k]-q[use]).sum()),revenue_gain=float(rnew[k]-rev),zero_cost_shortfall=max(0.,float(1-rev/rnew[k]))) for k,delta in enumerate(shifts)]))
    pd.concat(predictions,ignore_index=True).to_parquet(RUN/'demand_credibility_predictions.parquet',index=False)
    # Support in observed category-week prices: an envelope check, not a propensity score.
    support=[];lo=s.groupby('category_week').Price.transform('min').to_numpy();hi=s.groupby('category_week').Price.transform('max').to_numpy()
    nprices=s.groupby('category_week').Price.transform('nunique').to_numpy()
    for window,use in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
        for k,delta in enumerate(shifts):
            changed=use&eligible;outside=(pn[:,k]<lo)|(pn[:,k]>hi);dq=s['q_shift_'+str(delta)].to_numpy()-s.q.to_numpy();positive=np.maximum(dq,0.)
            support.append(dict(window=window,shift=delta,continuation_rows=int(changed.sum()),outside_price_envelope_share=float(outside[changed].mean()),single_price_cell_share=float((nprices[changed]==1).mean()),positive_extra_sales_outside_share=float(positive[use&outside].sum()/positive[use].sum()) if positive[use].sum()>0 else None))
    # Two-way cluster descriptive within variation. It is not an instrument or causal estimator.
    model=WeekDemand('additive');X=model.continuous(s).astype(float);DX=model.continuous(alt[0]).to_numpy()-X.to_numpy();y=s.sale_terminal_status.astype(float)
    fe_results=[]
    for name,keys in [('category_date',['category','date']),('category_week_date',['category_week','date']),('listing_date',['Code','date'])]:
        absorb=pd.DataFrame({k:pd.Categorical(s[k]) for k in keys})
        clusters=pd.DataFrame({'category':pd.factorize(s.category)[0],'date':pd.factorize(s.date)[0]})
        with warnings.catch_warnings(record=True) as logs:
            fit=AbsorbingLS(y,X,absorb=absorb,drop_absorbed=True).fit(cov_type='clustered',clusters=clusters,debiased=True)
        columns=list(fit.params.index);positions=[list(X.columns).index(c) for c in columns]
        for window,use in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
            contrast=DX[use][:,positions].sum(axis=0);value=float(contrast@fit.params.to_numpy());variance=float(contrast@fit.cov.to_numpy()@contrast);se=float(np.sqrt(max(variance,0)))
            fe_results.append(dict(spec=name,window=window,nobs=int(fit.nobs),minus10_extra_sales_linear=value,cluster_se=se,normal_interval=[value-1.96*se,value+1.96*se],covariance_negative_for_contrast=variance<0,retained_price_columns=[c for c in columns if c in ['own','gap_positive','gap_negative','price_ref_interaction','own_peer_price']],warnings=[str(w.message)[:300] for w in logs]))
        print('CREDIBILITY ABSORBED',name,flush=True)
    # Calendar-week bootstrap of fixed validation residuals; no refitting interpretation.
    calibration=[];rng=np.random.default_rng(20261020)
    for pred in predictions:
        name=pred.spec.iloc[0];use=pred.date.ge('2023-11-01');z=pred[use].copy();z['x']=recover_receipts(z.Price.to_numpy());z['block']=((z.date-pd.Timestamp('2023-07-11')).dt.days//7)
        z['ry']=z.x*z.sale_terminal_status;z['rq']=z.x*z.q
        a=z.groupby('block').agg(y=('sale_terminal_status','sum'),q=('q','sum'),ry=('ry','sum'),rq=('rq','sum')).to_numpy();B=len(a);draw=rng.integers(B,size=(2000,B));tot=a[draw].sum(axis=1)
        diff=tot[:,2]/tot[:,0]-tot[:,3]/tot[:,1];counts=tot[:,0]-tot[:,1]
        calibration.append(dict(spec=name,blocks=B,receipt_mean_observed_minus_predicted=float(z.ry.sum()/z.sale_terminal_status.sum()-z.rq.sum()/z.q.sum()),receipt_gap_bootstrap_interval=np.quantile(diff,[.025,.975]).tolist(),sales_observed_minus_predicted=float(z.sale_terminal_status.sum()-z.q.sum()),sales_gap_bootstrap_interval=np.quantile(counts,[.025,.975]).tolist(),interpretation='Descriptive calendar-block bootstrap conditional on fitted predictions; training and structural uncertainty not included.'))
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256((RUN/'demand_credibility_predictions.parquet').read_bytes()).hexdigest(),records=records,folds=folds,support=support,fixed_effects=fe_results,calibration=calibration)
    for p in [RUN/'demand_credibility.json',ROOT/'docs/evidence/gadgetpickup_demand_credibility_results_20261020.json']:p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('DEMAND_CREDIBILITY_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
