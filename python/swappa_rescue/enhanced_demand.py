"""Calendar and seller price-sensitivity checks with rotating date holdouts."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from quality_demand import data,QualityDemand
from audit_panel import OUT

SELLERS=['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']
class EnhancedQualityDemand(QualityDemand):
    def __init__(self,extension):
        super().__init__('quality');self.extension=extension
    def cat(self,d):
        z=super().cat(d)
        if self.extension in ['calendar','seller']:
            z=z.copy();z['weekday']=d.date.dt.dayofweek.astype(str).to_numpy()
        return z
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        if self.extension=='seller':
            for seller in SELLERS:z['own_seller::'+seller]=z['own'].to_numpy()*d.Seller.eq(seller).to_numpy()
        return z

def main():
    from enhanced_demand import EnhancedQualityDemand
    d=data();y=d.sale_last_seen;tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
    # Two adjacent days rotate across weeks; every weekday and week occurs
    # in training. Listing groups remain disjoint in the joint holdout.
    held=((d.date.dt.dayofweek.to_numpy()-2*d.week.to_numpy())%7)<2
    splits={'listing':(tr,te),'rotating_dates_and_listings':(tr[~held[tr]],te[held[te]])};records=[]
    for extension in ['base','calendar','seller']:
      for split,(train,test) in splits.items():
        fit=EnhancedQualityDemand(extension)
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit.fit(d.iloc[train],y.iloc[train])
        pred=fit.predict(d.iloc[test]);actual=y.iloc[test];s=d.iloc[test];ef=fit.effects()
        slope=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500)+np.where(s.gap_min.to_numpy()>0,ef['gap_positive'],ef['gap_negative']))/100
        for seller in SELLERS:slope+=ef.get('own_seller::'+seller,0)*s.Seller.eq(seller).to_numpy()/100
        rec={'extension':extension,'split':split,'train_rows':len(train),'test_rows':len(test),'unseen_weeks':len(set(s.week)-set(d.iloc[train].week)),'unseen_weekdays':len(set(s.date.dt.dayofweek)-set(d.iloc[train].date.dt.dayofweek)),'logloss':log_loss(actual,pred),'auc':roc_auc_score(actual,pred),'brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_[0]),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in logs),'positive_price_slope_fraction':float((slope>=0).mean()),'effects':ef}
        records.append(rec);print(json.dumps(rec,default=float),flush=True);(OUT/'enhanced_demand_validation.json').write_text(json.dumps(records,indent=2,default=float))
      full=EnhancedQualityDemand(extension).fit(d,y);joblib.dump(full,OUT/f'enhanced_demand_{extension}.joblib')
    print('ENHANCED_DEMAND_COMPLETE',flush=True)
if __name__=='__main__':main()
