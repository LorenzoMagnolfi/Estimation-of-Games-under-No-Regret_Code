"""Retrospective listing-disjoint demand audit; no cost or regret calibration."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from gadget_demand_audit import GadgetDemand,metrics,groups,RUN,SPEC
from demand_timing_sensitivity import terminal_status_data
from policy_cost_frontier import changed_data
from audit_panel import ROOT,OUT

class WeekDemand(GadgetDemand):
    def __init__(self,kind):super().__init__(kind)
    def cat(self,d):
        if self.kind=='additive':return super().cat(d)
        return pd.DataFrame({'category':d.category.to_numpy(),'week':d.week.astype(str).to_numpy(),'weekday':d.date.dt.dayofweek.astype(str).to_numpy()})
    def continuous(self,d,action=None):
        z=super().continuous(d,action)
        return z.drop(columns='calendar_time',errors='ignore')

def main():
    spec=json.loads(SPEC.read_text());d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s['fold']=s.Code.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5)
    s['price_bin']=pd.cut(s.Price,spec['price_bins']).astype(str)
    records=[];predictions=[];calibration=[]
    for kind in ['additive','category','category_spline']:
        q=np.zeros(len(s));down=q.copy();up=q.copy();folds=[]
        for fold in range(5):
            tr=s.fold.ne(fold);te=~tr
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter('always',ConvergenceWarning)
                fit=WeekDemand(kind).fit(s[tr],s.loc[tr,'sale_terminal_status'])
            z=s[te].copy();q[te]=fit.predict(z)
            # Reconstruct owned inventory totals using all rows, then select the
            # held-out rows. Otherwise same-day peers in other folds are missed.
            delta=10*s.since_seen.gt(0).to_numpy()
            down[te]=fit.predict(changed_data(s,np.maximum(1,s.Price.to_numpy()-delta)).loc[te])
            up[te]=fit.predict(changed_data(s,s.Price.to_numpy()+delta).loc[te])
            folds.append(dict(fold=fold,train_rows=int(tr.sum()),test_rows=int(te.sum()),iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs),effects=fit.effects()))
            print(json.dumps(dict(kind=kind,fold=fold,metrics=metrics(z,q[te]))),flush=True)
        for window,mask in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
            cont=mask&s.since_seen.gt(0).to_numpy()
            r=dict(kind=kind,window=window,metrics=metrics(s[mask],q[mask]),continuation_rows=int(cont.sum()),minus10_sales_change=float((down-q)[cont].sum()),plus10_sales_change=float((up-q)[cont].sum()),minus10_mean_logodds_change=float(np.mean(np.log(down[cont]/(1-down[cont]))-np.log(q[cont]/(1-q[cont])))),positive_price_response_fraction=float(np.mean(up[cont]>q[cont]+1e-9)))
            records.append(r)
            a=groups(s[mask],q[mask],'price_bin');a['kind']=kind;a['window']=window;calibration.append(a)
            print(json.dumps(r),flush=True)
        p=s[['Code','date','category','Price','sale_terminal_status','fold']].copy();p['q']=q;p['q_minus10']=down;p['q_plus10']=up;p['kind']=kind;predictions.append(p)
        (RUN/('crossfit_'+kind+'_folds.json')).write_text(json.dumps(folds,indent=2))
    variation=[]
    for window,mask in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
        z=s[mask]
        for keys in [['category'],['category','week'],['category','date']]:
            residual=z.Price-z.groupby(keys).Price.transform('mean')
            variation.append(dict(window=window,controls=keys,rows=len(z),price_std=float(z.Price.std()),residual_rms=float(np.sqrt(np.mean(residual**2))),r_squared=float(1-np.sum(residual**2)/np.sum((z.Price-z.Price.mean())**2)),zero_residual_share=float(np.mean(abs(residual)<1e-8))))
    raw=pd.read_parquet(OUT/'panel.parquet');dated=raw[raw.Code.eq('LXKC36753')]
    historical=dict(code='LXKC36753',in_panel_rows=len(dated),records=dated[[x for x in ['Code','date','Price','Real_Sold','Updated','Seller'] if x in dated]].to_dict('records'))
    cal=pd.concat(calibration,ignore_index=True);cal.to_csv(RUN/'crossfit_calibration.csv',index=False);pd.concat(predictions,ignore_index=True).to_parquet(RUN/'crossfit_predictions.parquet',index=False)
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256((ROOT/'docs/evidence/gadgetpickup_crossfit_design_20261020.json').read_bytes()).hexdigest(),records=records,calibration=cal.to_dict('records'),price_variation=variation,historical_listing=historical,interpretation='Retrospective listing-disjoint validation, not causal identification, independent time holdout, or seller information. Fold peers can share demand shocks.')
    for p in [RUN/'crossfit.json',ROOT/'docs/evidence/gadgetpickup_crossfit_results_20261020.json']:p.write_text(json.dumps(out,indent=2,default=str)+'\n')
    print('GADGET_CROSSFIT_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
