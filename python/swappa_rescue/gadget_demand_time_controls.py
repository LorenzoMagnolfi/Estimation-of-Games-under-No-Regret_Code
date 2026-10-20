"""Predetermined sensitivity to category-specific weekly demand shocks."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from gadget_demand_crossfit import WeekDemand
from gadget_demand_audit import metrics,groups,RUN,SPEC
from demand_timing_sensitivity import terminal_status_data
from policy_cost_frontier import changed_data
from audit_panel import ROOT

class CellWeekDemand(WeekDemand):
    def __init__(self,no_initial=False):super().__init__('category');self.no_initial=no_initial
    def cat(self,d):
        z=super().cat(d);z['category_week']=d.category.to_numpy()+'::'+d.week.astype(str).to_numpy();return z
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        return z.drop(columns=['initial_residual','initial_reference']) if self.no_initial else z

def main():
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s['fold']=s.Code.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5)
    s['price_bin']=pd.cut(s.Price,json.loads(SPEC.read_text())['price_bins']).astype(str)
    records=[];predictions=[];folds=[];calibration=[]
    for name,no_initial in [('category_week',False),('category_week_no_initial_price',True)]:
        q=np.zeros(len(s));down=q.copy();up=q.copy()
        for fold in range(5):
            tr=s.fold.ne(fold);te=~tr
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter('always',ConvergenceWarning);fit=CellWeekDemand(no_initial).fit(s[tr],s.loc[tr,'sale_terminal_status'])
            q[te]=fit.predict(s[te]);delta=10*s.since_seen.gt(0).to_numpy()
            down[te]=fit.predict(changed_data(s,np.maximum(1,s.Price.to_numpy()-delta)).loc[te]);up[te]=fit.predict(changed_data(s,s.Price.to_numpy()+delta).loc[te])
            f=dict(kind=name,fold=fold,iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs),effects=fit.effects());folds.append(f)
            print(json.dumps({k:v for k,v in f.items() if k!='effects'}),flush=True)
        for window,mask in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
            cont=mask&s.since_seen.gt(0).to_numpy()
            r=dict(kind=name,window=window,metrics=metrics(s[mask],q[mask]),minus10_sales_change=float((down-q)[cont].sum()),plus10_sales_change=float((up-q)[cont].sum()),minus10_mean_logodds_change=float(np.mean(np.log(down[cont]/(1-down[cont]))-np.log(q[cont]/(1-q[cont])))),positive_price_response_fraction=float(np.mean(up[cont]>q[cont]+1e-9)))
            records.append(r);print(json.dumps(r),flush=True)
            a=groups(s[mask],q[mask],'price_bin');a['kind']=name;a['window']=window;calibration.append(a)
        p=s[['Code','date','category','Price','sale_terminal_status','fold']].copy();p['q']=q;p['q_minus10']=down;p['q_plus10']=up;p['kind']=name;predictions.append(p)
    pd.concat(predictions,ignore_index=True).to_parquet(RUN/'time_controls_predictions.parquet',index=False)
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256((ROOT/'docs/evidence/gadgetpickup_time_controls_design_20261020.json').read_bytes()).hexdigest(),records=records,folds=folds,calibration=pd.concat(calibration,ignore_index=True).to_dict('records'),interpretation='Retrospective predictive sensitivity. Neither controls nor initial price assumptions establish causal demand.')
    for p in [RUN/'time_controls.json',ROOT/'docs/evidence/gadgetpickup_time_controls_results_20261020.json']:p.write_text(json.dumps(out,indent=2)+'\n')
    print('GADGET_TIME_CONTROLS_COMPLETE',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
