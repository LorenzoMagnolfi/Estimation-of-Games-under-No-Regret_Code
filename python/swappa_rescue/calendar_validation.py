"""Contemporaneous blocked-date validation with calendar-week overlap."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from quality_demand import data,QualityDemand
from audit_panel import OUT

def main():
    d=data(); y=d.sale_last_seen; train,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code))
    held=d.date.dt.dayofweek.isin([1,4]).to_numpy()
    splits={'calendar_days':(np.flatnonzero(~held),np.flatnonzero(held)),'listings_and_calendar_days':(train[~held[train]],test[held[test]])}
    records=[]
    for spec in ['mean','min','rich']:
      for split,(tr,te) in splits.items():
        fit=QualityDemand('quality'); fit.spec=spec
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter('always',ConvergenceWarning); fit.fit(d.iloc[tr],y.iloc[tr])
        pred=fit.predict(d.iloc[te]); actual=y.iloc[te]
        rec={'spec':spec,'split':split,'train_rows':len(tr),'test_rows':len(te),'unseen_test_weeks':len(set(d.iloc[te].week)-set(d.iloc[tr].week)),'logloss':log_loss(actual,pred),'auc':roc_auc_score(actual,pred),'brier':brier_score_loss(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_[0]),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in warn)}
        records.append(rec); print(json.dumps(rec),flush=True)
        (OUT/'calendar_validation.json').write_text(json.dumps(records,indent=2))
    print('CALENDAR_VALIDATION_COMPLETE',flush=True)
if __name__=='__main__':main()
