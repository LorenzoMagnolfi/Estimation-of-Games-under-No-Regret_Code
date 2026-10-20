"""Correct one daily observation defect; assess remaining pagination risk."""
import hashlib
import json
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from audit_panel import ROOT, OUT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN, metrics
from gadget_demand_crossfit import WeekDemand
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_observation_correction_design_20261020.json'

def main():
    cal=pd.read_csv(RUN/'snapshot_calendar.csv',parse_dates=['date'])
    valid=cal.loc[cal.one_day_observation_interval,'date']
    fresh=cal.loc[~cal.copied_previous_snapshot,'date']
    a=pd.read_parquet(RUN/'historical_display_order.parquet')
    a['page']=(a.table_position-1)//50
    keys=['device_id','date','source_price_numeric']
    a['boundary']=a.groupby(keys).page.transform('max').gt(a.groupby(keys).page.transform('min'))
    flags=a.groupby(['Code','date']).boundary.max().reset_index()
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')&d.date.isin(valid)].sort_values(['Code','date']).reset_index(drop=True)
    s=s.merge(flags,on=['Code','date'],how='left',validate='one_to_one');assert s.boundary.notna().all()
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    s['fold']=s.Code.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5)
    prior=pd.read_parquet(RUN/'crossfit_predictions.parquet');prior=prior[prior.kind.eq('additive')]
    s=s.merge(prior[['Code','date','q','q_minus10','q_plus10']].rename(columns={'q':'old_q','q_minus10':'old_down','q_plus10':'old_up'}),on=['Code','date'],validate='one_to_one')
    assert s.old_q.notna().all()
    q=np.zeros(len(s));down=q.copy();up=q.copy();folds=[]
    delta=10*s.since_seen.gt(0).to_numpy()
    lower=changed_data(s,np.maximum(1,s.Price.to_numpy()-delta));higher=changed_data(s,s.Price.to_numpy()+delta)
    for fold in range(5):
        tr=s.fold.ne(fold);te=~tr
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always',ConvergenceWarning);fit=WeekDemand('additive').fit(s[tr],s.loc[tr,'sale_terminal_status'])
        q[te]=fit.predict(s[te]);down[te]=fit.predict(lower[te]);up[te]=fit.predict(higher[te])
        folds.append(dict(fold=fold,train_rows=int(tr.sum()),test_rows=int(te.sum()),iterations=int(fit.model.n_iter_[0]),convergence_warning=any(isinstance(w.message,ConvergenceWarning) for w in logs),effects=fit.effects()))
        print(f'CORRECTED FOLD {fold} COMPLETE',flush=True)
    records=[]
    for name,pred,low,high in [('old_fit_same_rows',s.old_q.to_numpy(),s.old_down.to_numpy(),s.old_up.to_numpy()),('corrected_fit',q,down,up)]:
        for window,mask in [('full',np.ones(len(s),bool)),('novdec',s.date.ge('2023-11-01').to_numpy())]:
            for group,use in [('all',mask),('boundary',mask&s.boundary.to_numpy()),('other',mask&~s.boundary.to_numpy())]:
                cont=use&s.since_seen.gt(0).to_numpy()
                records.append(dict(fit=name,window=window,group=group,metrics=metrics(s[use],pred[use]),continuation_rows=int(cont.sum()),minus10_sales_change=float((low-pred)[cont].sum()),plus10_sales_change=float((high-pred)[cont].sum())))
    s['q']=q;s['q_minus10']=down;s['q_plus10']=up
    s[['Code','date','Price','sale_terminal_status','fold','boundary','old_q','q','q_minus10','q_plus10']].to_parquet(RUN/'observation_corrected_predictions.parquet',index=False)
    raw=pd.read_parquet(OUT/'panel.parquet');g=raw[raw.Seller.eq('GadgetPickup')&raw.date.isin(fresh)].merge(flags,on=['Code','date'],validate='one_to_one').sort_values(['Code','date'])
    index=dict(zip(fresh,range(len(fresh))));g['calendar_index']=g.date.map(index)
    g['next_index']=g.groupby('Code').calendar_index.shift(-1);g['next_price']=g.groupby('Code').Price.shift(-1)
    pairs=g[g.next_index.notna()].copy();pairs['missing_fresh_dates']=pairs.next_index-pairs.calendar_index-1;pairs['gap']=pairs.missing_fresh_dates.gt(0);pairs['same_endpoint_price']=pairs.Price.eq(pairs.next_price)
    gapstats=pairs.groupby(['boundary','gap']).agg(transitions=('Code','size'),same_endpoint_price=('same_endpoint_price','sum'),missing_fresh_dates=('missing_fresh_dates','sum')).reset_index()
    pairs[['Code','date','Price','boundary','missing_fresh_dates','same_endpoint_price']].to_csv(RUN/'pagination_gap_diagnostic.csv',index=False)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),sample=dict(rows=len(s),sales=int(s.sale_terminal_status.sum()),dates=int(s.date.nunique()),listings=int(s.Code.nunique()),boundary_rows=int(s.boundary.sum())),folds=folds,records=records,gaps=gapstats.to_dict('records'),interpretation='Corrected one-day observation diagnostic, not a causal demand estimate or a validated finite-regret model. Retained outcomes are the existing terminal proxy. Gap statistics condition on future reappearance; boundary flags do not establish missingness or buyer exposure.')
    for p in [RUN/'observation_correction.json',ROOT/'docs/evidence/gadgetpickup_observation_correction_results_20261020.json']:p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True);print('OBSERVATION_CORRECTION_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
