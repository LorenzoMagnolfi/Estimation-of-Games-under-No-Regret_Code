"""Demand sensitivity to alternative observation models for sale timing.

The terminal-status version retains every observed active listing-day, uses
later sold status to label the final exposure, and does not interpret Updated
as a transaction timestamp. This is an alternative maintained observation model,
not an established correction. Final calendar-day exits are censored.
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from audit_panel import OUT


def terminal_status_data():
    d=pd.read_parquet(OUT/'panel.parquet').sort_values(['Code','date']).copy()
    if 'fullsource_prepared' not in d:
        lastdate=d.groupby('Code').date.transform('max')
        d['sale_terminal_status']=(d.Real_Sold.eq(1)&d.date.eq(lastdate)).astype(int)
        d['initial_residual']=d.groupby('Code').residual.transform('first')
        d['initial_reference']=d.groupby('Code').Ref_Price.transform('first')
    else:
        # Endpoints and entry controls were set on the FULL raw histories.
        assert d.fullsource_prepared.all()
        d=d[d.Ref_Price.notna()].copy()
    d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy().reset_index(drop=True)
    top=d.groupby('Seller').Code.nunique().nlargest(20).index
    d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    return d


def main():
    records=[]
    for outcome in ['sale_forward','sale_legacy','sale_terminal_status']:
        d=terminal_status_data() if outcome=='sale_terminal_status' else data();y=d[outcome]
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
        fit=MultiproductDemand()
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit.fit(d.iloc[tr],y.iloc[tr])
        pred=fit.predict(d.iloc[te]);actual=y.iloc[te]
        rec={'outcome':outcome,'rows':len(d),'positives':int(y.sum()),'listing_holdout_logloss':log_loss(actual,pred),'listing_holdout_auc':roc_auc_score(actual,pred),'listing_holdout_brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_[0]),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in logs),'effects':fit.effects()}
        records.append(rec);print(json.dumps(rec),flush=True)
        (OUT/'multiproduct_timing_validation.json').write_text(json.dumps(records,indent=2))
        full=MultiproductDemand().fit(d,y);joblib.dump(full,OUT/f'multiproduct_{outcome}.joblib')
        if outcome=='sale_terminal_status':d.to_parquet(OUT/'panel_terminal_status.parquet',index=False)
    print('TIMING_DEMAND_COMPLETE',flush=True)
if __name__=='__main__':main()
