"""Demand sensitivity to predetermined listing quality and observation timing.

These checks assess predictive stability, not causal identification by themselves.
"""
from __future__ import annotations
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from demand_pilot import Demand
from audit_panel import OUT

class QualityDemand(Demand):
    def __init__(self,variant):
        super().__init__('rich')
        self.variant=variant
        self.model.set_params(max_iter=2500)
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        if self.variant!='original':
            age=np.log1p(d.since_seen.to_numpy())
            z['log_age']=age
            for h in [2,4,8,16,32]:
                z[f'age_hinge_{h}']=np.maximum(age-np.log(h),0)
        if self.variant in ['quality','initial']:
            battery=pd.to_numeric(d.Battery_Life.str.extract(r'(\d+)',expand=False),errors='coerce')
            z['battery']=battery.fillna(90).to_numpy()/100
            z['battery_missing']=battery.isna().astype(float).to_numpy()
            z['battery_full']=battery.eq(100).astype(float).to_numpy()
        if self.variant=='initial':
            z['initial_residual']=d.initial_residual.to_numpy()/100
            z['initial_reference']=d.initial_reference.to_numpy()/500
        return z

def data():
    d=pd.read_parquet(OUT/'panel.parquet').sort_values(['Code','date'])
    d['initial_residual']=d.groupby('Code').residual.transform('first')
    d['initial_reference']=d.groupby('Code').Ref_Price.transform('first')
    d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy()
    d=d[~(d.Real_Sold.eq(1)&d.date.gt(d.Updated))].copy().reset_index(drop=True)
    top=d.groupby('Seller').Code.nunique().nlargest(20).index
    d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    return d

def main():
    from quality_demand import QualityDemand
    d=data(); y=d.sale_last_seen
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
    early=d.date.lt(d.date.min()+pd.Timedelta(days=110)).to_numpy()
    late=d.date.ge(d.date.min()+pd.Timedelta(days=120)).to_numpy()
    splits={'listing':(train,test),'later_dates':(np.flatnonzero(early),np.flatnonzero(late)),'listing_and_later_dates':(train[early[train]],test[late[test]])}
    records=[]
    for variant in ['original','observed_age','quality','initial']:
      for split,(tr,te) in splits.items():
        fit=QualityDemand(variant)
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning)
            fit.fit(d.iloc[tr],y.iloc[tr])
        pred=fit.predict(d.iloc[te]); actual=y.iloc[te]
        # Price derivatives retain initial-price controls at their observed value.
        eff=fit.effects(); ref=d.iloc[te].Ref_Price.to_numpy()
        slopes=(eff['own']+np.where(d.iloc[te].gap_min.to_numpy()>0,eff['gap_positive'],eff['gap_negative'])+eff['price_ref_interaction']*np.log(ref/500))/100
        rec={'variant':variant,'split':split,'train_rows':len(tr),'test_rows':len(te),'train_listings':int(d.iloc[tr].Code.nunique()),'test_listings':int(d.iloc[te].Code.nunique()),'logloss':log_loss(actual,pred),'auc':roc_auc_score(actual,pred),'brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_[0]),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in logs),'own_logodds_derivative_q10':float(np.quantile(slopes,.1)),'own_logodds_derivative_q50':float(np.median(slopes)),'own_logodds_derivative_q90':float(np.quantile(slopes,.9)),'positive_derivative_share':float((slopes>=0).mean()),'effects':eff}
        records.append(rec); print(json.dumps(rec,default=float),flush=True)
        (OUT/'quality_validation.json').write_text(json.dumps(records,indent=2,default=float))
      fit=QualityDemand(variant).fit(d,y)
      joblib.dump(fit,OUT/f'quality_demand_{variant}.joblib')
    print('QUALITY_VALIDATION_COMPLETE',flush=True)

if __name__=='__main__':main()
