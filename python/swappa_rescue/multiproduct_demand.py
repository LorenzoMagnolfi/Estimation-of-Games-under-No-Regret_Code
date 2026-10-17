"""Own-inventory price spillover audit for the Swappa demand model.

This exposes a limitation of separable listing payoffs: other listings owned by
one seller can be substitutes. Predictive coefficients alone are not causal.
"""
import json
import warnings
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from enhanced_demand import EnhancedQualityDemand
from quality_demand import data
from audit_panel import OUT

class MultiproductDemand(EnhancedQualityDemand):
    def __init__(self):super().__init__('calendar')
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        n=d.own_n.to_numpy();present=n>1
        peers=np.divide(d.own_total.to_numpy()-d.Price.to_numpy(),n-1,out=np.zeros(len(d)),where=present)
        z['own_peer_price']=np.where(present,(peers-d.Ref_Price.to_numpy())/100,0.)
        z['own_peer_present']=present.astype(float)
        if 'fullsource_prepared' in d:
            z['reference_observed']=d.reference_observed.to_numpy(dtype=float)
            if 'reference_quote_rejected' in d:z['reference_quote_rejected']=d.reference_quote_rejected.to_numpy(dtype=float)
            z['log_reference_quote_age']=np.log1p(d.reference_age_days.to_numpy())
            z['initial_reference_missing']=d.initial_reference_missing.to_numpy(dtype=float)
        return z

def main():
    from multiproduct_demand import MultiproductDemand
    d=data();y=d.sale_last_seen;tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code));held=((d.date.dt.dayofweek.to_numpy()-2*d.week.to_numpy())%7)<2
    records=[]
    for name,(train,test) in {'listing':(tr,te),'rotating_dates_and_listings':(tr[~held[tr]],te[held[te]])}.items():
        fit=MultiproductDemand()
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit.fit(d.iloc[train],y.iloc[train])
        pred=fit.predict(d.iloc[test]);actual=y.iloc[test];ef=fit.effects();s=d.iloc[test]
        partial=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500)+np.where(s.gap_min.to_numpy()>0,ef['gap_positive'],ef['gap_negative']))/100
        joint=partial+ef['own_peer_price']*s.own_n.gt(1).to_numpy()/100
        rec={'split':name,'logloss':log_loss(actual,pred),'auc':roc_auc_score(actual,pred),'brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_[0]),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in logs),'median_individual_logodds_slope':float(np.median(partial)),'median_common_price_logodds_slope':float(np.median(joint)),'positive_common_slope_share':float((joint>=0).mean()),'effects':ef}
        records.append(rec);print(json.dumps(rec),flush=True);(OUT/'multiproduct_validation.json').write_text(json.dumps(records,indent=2))
    fit=MultiproductDemand().fit(d,y);joblib.dump(fit,OUT/'multiproduct_demand.joblib')
    print('MULTIPRODUCT_DEMAND_COMPLETE',flush=True)
if __name__=='__main__':main()
