"""GadgetPickup demand calibration, information timing, and predetermined refits."""
from __future__ import annotations
import hashlib,json,subprocess,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from continuation_demand import make
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from audit_panel import OUT,ROOT,DATA_VERSION
from ask_price_grid import recover_receipts

RUN=OUT/'gadget_demand_20261020'
SPEC=ROOT/'docs/evidence/gadgetpickup_demand_design_20261020.json'

class GadgetDemand(MultiproductDemand):
    def __init__(self,kind):
        super().__init__();self.variant='initial';self.kind=kind
        self.encoder=OneHotEncoder(handle_unknown='ignore',min_frequency=1,sparse_output=True)
        self.model.set_params(C=1.,max_iter=3000,tol=1e-8)
        self.categories=[]
    def cat(self,d):
        if self.kind=='additive':return super().cat(d)
        return pd.DataFrame({'category':d.category.to_numpy(),'weekday':d.date.dt.dayofweek.astype(str).to_numpy()})
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        if self.kind!='additive':z['calendar_time']=(d.date-pd.Timestamp('2023-07-11')).dt.days.to_numpy()/100
        if self.kind=='category_age':
            for cat in self.categories:z['age::'+cat]=np.log1p(d.since_seen.to_numpy())*d.category.eq(cat).to_numpy()
        if self.kind=='category_spline':
            # These are predictive price controls, not a claim of causal monotonicity.
            for knot in [200,400,600,800]:z['price_hinge_'+str(knot)]=np.maximum(d.Price.to_numpy()-knot,0)/100
        return z
    def fit(self,d,y):
        self.categories=sorted(d.category.unique());return super().fit(d,y)

def metrics(s,q):
    y=s.sale_terminal_status.to_numpy();x=recover_receipts(s.Price.to_numpy())
    return dict(rows=len(s),sales=int(y.sum()),predicted_sales=float(q.sum()),mean_receipt_actual=float(x@y/y.sum()) if y.sum() else None,mean_receipt_predicted=float(x@q/q.sum()),receipt_residual=float(x@(y-q)),logloss=float(log_loss(y,q,labels=[0,1])),brier=float(brier_score_loss(y,q)),auc=float(roc_auc_score(y,q)) if y.min()!=y.max() else None)

def groups(s,q,key):
    z=s.copy();z['q']=q;z['x']=recover_receipts(s.Price.to_numpy());z['actual_revenue']=z.x*z.sale_terminal_status;z['predicted_revenue']=z.x*z.q
    a=z.groupby(key,observed=True).agg(rows=('Code','size'),listings=('Code','nunique'),sales=('sale_terminal_status','sum'),expected_sales=('q','sum'),actual_revenue=('actual_revenue','sum'),predicted_revenue=('predicted_revenue','sum'),mean_price=('Price','mean')).reset_index()
    a['sale_residual']=a.sales-a.expected_sales;a['revenue_residual']=a.actual_revenue-a.predicted_revenue
    return a

def main():
    assert DATA_VERSION=='fullsource_refscreen_known';RUN.mkdir(exist_ok=True)
    spec=json.loads(SPEC.read_text());d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s['price_bin']=pd.cut(s.Price,spec['price_bins']).astype(str);s['age_bin']=pd.cut(s.since_seen,spec['age_bins']).astype(str);s['month']=s.date.dt.strftime('%Y-%m');s['entry']=s.since_seen.eq(0).astype(str)
    f=joblib.load(OUT/'continuation_seller_full_1.joblib');q=f.predict(s);tables=[]
    for window,mask in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01'))]:
        for key in ['category','price_bin','age_bin','month','entry']:
            a=groups(s.loc[mask],q[mask],key);a['grouping']=key;a['window']=window;tables.append(a.rename(columns={key:'group'}))
    calibration=pd.concat(tables,ignore_index=True);calibration.to_csv(RUN/'existing_calibration.csv',index=False)
    print('WORST_REVENUE_CELLS',calibration[(calibration.window=='novdec')&calibration.grouping.eq('category')].sort_values('revenue_residual').head(8).to_json(orient='records'),flush=True)
    print('PRICE_BINS',calibration[(calibration.window=='novdec')&calibration.grouping.eq('price_bin')].to_json(orient='records'),flush=True)
    # Mechanical outcome-timing diagnostics, not a causal event study.
    s['gap']=s.groupby('Code').date.diff().dt.days;s['change']=s.groupby('Code').Price.diff();s['consecutive']=s.gap.eq(1)
    s['change_group']=np.select([~s.consecutive,s.change.lt(-.01),s.change.gt(.01)],['entry_or_gap','cut','increase'],default='unchanged')
    timing=s.groupby('change_group').agg(rows=('Code','size'),sales=('sale_terminal_status','sum'),mean_price=('Price','mean')).reset_index();timing['sale_rate']=timing.sales/timing.rows
    s['last_date']=s.groupby('Code').date.transform('max');s['days_to_last']=(s.last_date-s.date).dt.days
    changes=s[s.consecutive&s.change.lt(0)].copy();changes['terminal_or_previous_day']=changes.days_to_last.le(1)&changes.Real_Sold.eq(1)
    batch=changes.groupby('date').agg(changes=('Code','size'),categories=('category','nunique'),terminal_or_previous_day=('terminal_or_previous_day','sum')).reset_index();batch.to_csv(RUN/'cut_dates.csv',index=False)
    original=metrics(s[s.date.ge('2023-11-01')],q[s.date.ge('2023-11-01')]);records=[];predictions=[]
    for end,start,stop in spec['chronological_splits']:
        tr=s.date.le(end);te=s.date.between(start,stop)
        for kind in ['additive','category','category_age','category_spline']:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);fit=GadgetDemand(kind).fit(s[tr],s.loc[tr,'sale_terminal_status'])
            pred=fit.predict(s[te]);rec=dict(train_end=end,test_start=start,test_end=stop,kind=kind,train_rows=int(tr.sum()),train_sales=int(s.loc[tr,'sale_terminal_status'].sum()),unseen_category_rows=int((~s.loc[te,'category'].isin(s.loc[tr,'category'])).sum()),iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in caught),metrics=metrics(s[te],pred),effects=fit.effects())
            # Direction of a joint $1 price change for continuation inventories only.
            from policy_cost_frontier import changed_data
            sub=s[te].copy();delta=sub.since_seen.gt(0).to_numpy().astype(float)
            qup=fit.predict(changed_data(sub,sub.Price.to_numpy()+delta))
            rec['positive_joint_demand_response_share']=float(np.mean((qup-pred)[delta>0]>1e-9))
            records.append(rec);p=s.loc[te,['Code','date','category','Price','sale_terminal_status']].copy();p['prediction']=pred;p['kind']=kind;predictions.append(p)
            print(json.dumps({k:v for k,v in rec.items() if k!='effects'}),flush=True)
            if start=='2023-11-01':joblib.dump(fit,RUN/('gadget_'+kind+'_oct31.joblib'))
    pd.concat(predictions,ignore_index=True).to_parquet(RUN/'chronological_predictions.parquet',index=False)
    raw=pd.read_parquet(OUT/'panel.parquet');raw=raw[raw.Seller.eq('GadgetPickup')]
    payload=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),panel_sha256=hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(),existing_fit_sha256=hashlib.sha256((OUT/'continuation_seller_full_1.joblib').read_bytes()).hexdigest(),existing_fit_novdec=original,calibration=calibration.to_dict('records'),timing=timing.to_dict('records'),raw_rows=len(raw),retained_rows=len(s),raw_unknown_terminal_rows=int((~raw.outcome_observed).sum()),retained_columns=list(s.columns),cut_rows=len(changes),cuts_on_terminal_or_prior_day=int(changes.terminal_or_previous_day.sum()),records=records,interpretation='Retrospective chronological predictive validation; no causal interpretation or new cost estimates. Existing fullsample fit is diagnostic only.')
    for p in [RUN/'demand_audit.json',ROOT/'docs/evidence/gadgetpickup_demand_audit_20261020.json']:p.write_text(json.dumps(payload,indent=2,default=str)+'\n')
    print('GADGET_DEMAND_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
