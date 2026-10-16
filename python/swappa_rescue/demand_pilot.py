"""Cross-validated demand and fixed-cost regret diagnostics; not confidence sets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from audit_panel import OUT

class Demand:
    def __init__(self,spec):
        self.spec=spec
        self.encoder=OneHotEncoder(handle_unknown='ignore',min_frequency=100,sparse_output=True)
        self.scaler=StandardScaler()
        self.model=LogisticRegression(C=10,solver='lbfgs',max_iter=500,tol=1e-7)
    def continuous(self,d,action=None):
        p=d.residual.to_numpy()/100 if action is None else np.broadcast_to(action/100,len(d))
        z={'own':p,'reference':d.Ref_Price.to_numpy()/500,'log_age':np.log1p(d.age.fillna(d.since_seen).clip(lower=0).to_numpy()),'log_rivals':np.log1p(d.rival_n.to_numpy()),'log_own':np.log1p(d.own_n.to_numpy())}
        if self.spec in ['mean','both','rich']:
            z['rival_mean']=d.rival_mean_residual.to_numpy()/100
        if self.spec in ['min','both','rich']:
            z['rival_min']=d.rival_min_residual.to_numpy()/100
        if self.spec=='rich':
            gap=p-d.rival_min_residual.to_numpy()/100
            z['gap_positive']=np.maximum(gap,0)
            z['gap_negative']=np.minimum(gap,0)
            z['price_ref_interaction']=p*np.log(d.Ref_Price.to_numpy()/500)
        return pd.DataFrame(z)
    def cat(self,d):
        return d[['device_id','Storage','Condition','week','seller_group']].astype(str)
    def fit(self,d,y):
        x=self.continuous(d)
        self.names=list(x.columns)
        self.model.fit(hstack([csr_matrix(self.scaler.fit_transform(x)),self.encoder.fit_transform(self.cat(d))]).tocsr(),y)
        return self
    def logit(self,d,action=None):
        x=hstack([csr_matrix(self.scaler.transform(self.continuous(d,action))),self.encoder.transform(self.cat(d))]).tocsr()
        return self.model.decision_function(x)
    def predict(self,d,action=None):
        return expit(self.logit(d,action))
    def effects(self):
        return dict(zip(self.names,self.model.coef_[0,:len(self.names)]/self.scaler.scale_))

def main():
    d=pd.read_parquet(OUT/'panel.parquet')
    d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy()
    # Exclude rows unambiguously after the recorded sale date; keep alternate timing rules explicit.
    d=d[~(d.Real_Sold.eq(1)&d.date.gt(d.Updated))].copy()
    top=d.groupby('Seller').Code.nunique().nlargest(20).index
    d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
    records=[]; frontiers=[]
    actions=np.arange(-200,201,5,dtype=float)+.01
    costs=np.arange(-250,151,1,dtype=float)
    focal=['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']
    for outcome in ['sale_legacy','sale_forward','sale_last_seen']:
      for spec in ['mean','min','both','rich']:
        fit=Demand(spec).fit(d.iloc[train],d.iloc[train][outcome])
        pred=fit.predict(d.iloc[test]); y=d.iloc[test][outcome]
        record={'outcome':outcome,'spec':spec,'n_train':len(train),'n_test':len(test),'logloss':log_loss(y,pred),'auc':roc_auc_score(y,pred),'brier':brier_score_loss(y,pred),'actual_rate':float(y.mean()),'predicted_rate':float(pred.mean()),'coefficients':fit.effects()}
        records.append(record)
        print(json.dumps(record),flush=True)
        # Whole-sample fit is exploratory, separated from predictive validation.
        fit=Demand(spec).fit(d,d[outcome])
        for seller in focal:
          sub=d[d.Seller.eq(seller)].copy()
          for unit in ['listing_day','listing_entry']:
            s=sub if unit=='listing_day' else sub.sort_values('date').drop_duplicates('Code')
            qa=fit.predict(s); q=np.column_stack([fit.predict(s,a) for a in actions])
            ra=s.residual.to_numpy()
            # Predeclared feasible deviations: exclude prices <=0 or far beyond input support.
            allowed=(s.Ref_Price.to_numpy()[:,None]+actions[None,:]>0)
            # Restrict a fixed comparator to actions feasible for EVERY included observation.
            ok=allowed.all(axis=0)
            q=q[:,ok]; act=actions[ok]
            intercept=(q*act).mean(axis=0)-(qa*ra).mean()
            slope=qa.mean()-q.mean(axis=0)
            r=np.maximum(0,(intercept[:,None]+slope[:,None]*costs).max(axis=0))
            ix=r.argmin(); best=costs[ix]
            for budget in [0,.1,.25,.5,1,2]:
              feas=costs[r<=budget]
              frontiers.append({'outcome':outcome,'spec':spec,'seller':seller,'unit':unit,'N':len(s),'n_listings':s.Code.nunique(),'R_min':r[ix],'c_minregret':best,'mean_price':s.Price.mean(),'mean_reference':s.Ref_Price.mean(),'cost_at_minregret':s.Ref_Price.mean()+best,'gross_markup_at_minregret':s.Price.mean()-s.Ref_Price.mean()-best,'epsilon_dollars_per_unit':budget,'cost_residual_low':feas.min() if len(feas) else np.nan,'cost_residual_high':feas.max() if len(feas) else np.nan,'grid_boundary':bool(len(feas) and (feas.min()==costs.min() or feas.max()==costs.max()))})
        pd.DataFrame(frontiers).to_csv(OUT/'pilot_frontiers.csv',index=False)
        (OUT/'demand_validation.json').write_text(json.dumps(records,indent=2))
    print('Completed pilot; all structural outputs are exploratory conditional regret sets.',flush=True)
if __name__=='__main__':
    main()
