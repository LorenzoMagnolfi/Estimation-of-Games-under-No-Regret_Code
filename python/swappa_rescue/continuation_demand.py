"""Predetermined initial-price controls for continuation pricing only.

The fitted model is predictive on all observed rows. Structural counterfactual
price changes must be restricted to observations after first exposure: an
entry-price deviation cannot hold its own initial-price control fixed.
"""
import json,warnings
import joblib,numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from audit_panel import OUT


def make():
 fit=MultiproductDemand();fit.variant='initial';return fit


def main():
 d=terminal_status_data();y=d.sale_terminal_status;tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code));fit=make()
 with warnings.catch_warnings(record=True) as warnings_seen:
  warnings.simplefilter('always',ConvergenceWarning);fit.fit(d.iloc[tr],y.iloc[tr])
 records=[]
 for group,mask in [('all',np.ones(len(te),bool)),('continuation',d.iloc[te].since_seen.gt(0).to_numpy()),('first_exposure',d.iloc[te].since_seen.eq(0).to_numpy())]:
  ids=te[mask];q=fit.predict(d.iloc[ids]);actual=y.iloc[ids];rec={'group':group,'rows':len(ids),'logloss':log_loss(actual,q),'auc':roc_auc_score(actual,q),'brier':brier_score_loss(actual,q),'actual_rate':float(actual.mean()),'predicted_rate':float(q.mean()),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in warnings_seen),'effects':fit.effects()};records.append(rec);print(json.dumps(rec),flush=True)
 (OUT/'initial_control_validation.json').write_text(json.dumps(records,indent=2))
 full=make().fit(d,y);joblib.dump(full,OUT/'multiproduct_initial_terminal.joblib');print(json.dumps({'full_effects':full.effects()}),flush=True)
 print('INITIAL_CONTROL_DEMAND_COMPLETE',flush=True)
if __name__=='__main__':main()
