"""Log-link demand diagnostic for cost-plus policy learning.

An exact log-linear mean can give a common-markup policy the same ranking
across states. That is useful only if the probability model and prediction
restrictions survive the data. Predictions above one are reported, not clipped.
"""
import json
import warnings
import joblib
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss,mean_poisson_deviance
from sklearn.exceptions import ConvergenceWarning
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from audit_panel import OUT

class LogLinkDemand(MultiproductDemand):
    def __init__(self,kink=False):
        super().__init__();self.kink=kink;self.model=PoissonRegressor(alpha=1e-6,max_iter=2000,tol=1e-8)
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        if not self.kink:z=z.drop(columns=['gap_positive','gap_negative'])
        return z
    def fit(self,d,y):
        self.model.set_params(alpha=1/(10*len(d)));return super().fit(d,y)
    def linear_predictor(self,d,action=None):
        X=hstack([csr_matrix(self.scaler.transform(self.continuous(d,action))),self.encoder.transform(self.cat(d))],format='csr')
        return np.asarray(X@self.model.coef_).ravel()+self.model.intercept_
    def predict(self,d,action=None):return np.exp(self.linear_predictor(d,action))
    def effects(self):return dict(zip(self.names,self.model.coef_[:len(self.names)]/self.scaler.scale_))


def main():
    from loglink_demand import LogLinkDemand
    d=data();y=d.sale_last_seen;tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=16016).split(d,groups=d.Code));records=[]
    for kink in [False,True]:
        fit=LogLinkDemand(kink)
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit.fit(d.iloc[tr],y.iloc[tr])
        pred=fit.predict(d.iloc[te]);actual=y.iloc[te];valid=bool(np.all(pred<=1))
        focal=d.iloc[te].Seller.eq('UPGRADE SOLUTION INC').to_numpy()
        rec={'kink':kink,'test_rows':len(te),'predictions_over_one':int((pred>1).sum()),'maximum_prediction':float(pred.max()),'upgrade_predictions_over_one':int((pred[focal]>1).sum()),'upgrade_maximum_prediction':float(pred[focal].max()),'poisson_deviance':mean_poisson_deviance(actual,pred),'logloss':log_loss(actual,pred) if valid else None,'brier_raw_mean_prediction':float(np.mean((pred-actual.to_numpy())**2)),'auc':roc_auc_score(actual,pred),'actual_rate':float(actual.mean()),'predicted_rate':float(pred.mean()),'iterations':int(fit.model.n_iter_),'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in logs),'effects':fit.effects()}
        records.append(rec);print(json.dumps(rec,default=float),flush=True);(OUT/'loglink_validation.json').write_text(json.dumps(records,indent=2,default=float))
        full=LogLinkDemand(kink).fit(d,y);joblib.dump(full,OUT/f'loglink_demand_kink{int(kink)}.joblib')
        predall=full.predict(d);summary={'kink':kink,'all_predictions_over_one':int((predall>1).sum()),'upgrade_predictions_over_one':int(((predall>1)&d.Seller.eq('UPGRADE SOLUTION INC').to_numpy()).sum()),'full_effects':full.effects()};print(json.dumps(summary,default=float),flush=True)
    print('LOGLINK_DEMAND_COMPLETE',flush=True)
if __name__=='__main__':main()
