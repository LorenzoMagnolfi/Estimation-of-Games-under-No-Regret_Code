"""Held-out sensitivity to seller-specific continuation demand slopes.
Predictive diagnostics only: this does not establish causal price exogeneity.
Never overwrites the demand fit or uncertainty inputs used by the OSC run.
"""
import json,warnings
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from audit_panel import OUT

SELLERS=['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']
class SellerContinuationDemand(MultiproductDemand):
    def __init__(self,heterogeneous=False):
        super().__init__();self.variant='initial';self.heterogeneous=heterogeneous
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        if self.heterogeneous:
            for seller in SELLERS:
                z['own_seller::'+seller]=z['own'].to_numpy()*d.Seller.eq(seller).to_numpy()
        return z

def main():
    from continuation_seller_sensitivity import SellerContinuationDemand
    d=terminal_status_data();y=d.sale_terminal_status
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
    records=[]
    for heterogeneous in [False,True]:
        with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            fit=SellerContinuationDemand(heterogeneous).fit(d.iloc[tr],y.iloc[tr])
        ef=fit.effects();s=d.iloc[te];q=fit.predict(s)
        for seller in ['all']+SELLERS:
            mask=s.since_seen.gt(0).to_numpy().copy()
            if seller!='all':mask &= s.Seller.eq(seller).to_numpy()
            sub=s.iloc[np.flatnonzero(mask)];actual=y.iloc[te[mask]].to_numpy();pred=q[mask]
            slope=(ef['own']+ef['price_ref_interaction']*np.log(sub.Ref_Price.to_numpy()/500)+np.where(sub.gap_min.to_numpy()>0,ef['gap_positive'],ef['gap_negative']))/100
            for name in SELLERS:slope+=ef.get('own_seller::'+name,0)*sub.Seller.eq(name).to_numpy()/100
            rec={'heterogeneous':heterogeneous,'seller':seller,'rows':len(sub),'listings':int(sub.Code.nunique()),'sales':int(actual.sum()),'logloss':log_loss(actual,pred),'auc':roc_auc_score(actual,pred),'brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'median_own_logit_slope':float(np.median(slope)),'positive_slope_share':float((slope>0).mean()),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in caught)}
            records.append(rec);print(json.dumps(rec),flush=True)
        (OUT/'continuation_seller_validation.json').write_text(json.dumps({'records':records,'last_fit_effects':ef},indent=2))
        with threadpool_limits(limits=2):full=SellerContinuationDemand(heterogeneous).fit(d,y)
        joblib.dump(full,OUT/f'continuation_seller_full_{int(heterogeneous)}.joblib')
        (OUT/f'continuation_seller_full_{int(heterogeneous)}_effects.json').write_text(json.dumps(full.effects(),indent=2))
    audits=[]
    for seller in SELLERS:
        sub=d[d.Seller.eq(seller)].sort_values(['Code','date']).copy();change=sub.groupby('Code').Price.diff();con=sub.since_seen.gt(0)
        daily=sub.assign(changed=change.abs().gt(0)).groupby('date').agg(listings=('Code','size'),changes=('changed','sum'))
        p=sub.Price.to_numpy();a=np.arange(0,int(p.max())+1);grids={'ceil':np.ceil(1.03*a-1e-9),'round':np.round(1.03*a),'floor':np.floor(1.03*a+1e-9)}
        audits.append({'seller':seller,'days':int(sub.date.nunique()),'continuation_rows':int(con.sum()),'price_change_rows':int(change.abs().gt(0).sum()),'dates_with_changes':int(daily.changes.gt(0).sum()),'median_concurrent_listings':float(daily.listings.median()),'max_changes_per_day':int(daily.changes.max()),'price_changes_quantiles':change.dropna().quantile([0,.01,.5,.99,1]).to_dict(),'fraction_prices_outside_3pct_rounded_ask_grid':{k:float((~np.isin(p,v)).mean()) for k,v in grids.items()}})
    (OUT/'continuation_clock_grid_audit.json').write_text(json.dumps(audits,indent=2));print('CONTINUATION_SELLER_SENSITIVITY_COMPLETE',flush=True)
if __name__=='__main__':main()
