"""Predetermined-price diagnostics on a paired daily risk set."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from linearmodels.iv import AbsorbingLS
from threadpoolctl import threadpool_limits
from ask_price_grid import recover_receipts
from audit_panel import ROOT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_demand_crossfit import WeekDemand
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_demand_timing_design_20261020.json'

def main():
    prior=pd.read_parquet(RUN/'repricing_performance_predictions.parquet');d=terminal_status_data()
    s=d[d.Seller.eq('GadgetPickup')].merge(prior[['Code','date','previous_date','previous_price']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    valid=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1);s['category_week']=s.category+'::'+s.week.astype(str)
    lag=changed_data(s,np.where(valid,s.previous_price,s.Price));ask=np.rint(recover_receipts(s.Price.to_numpy())/1.03);plow=np.ceil(1.03*np.maximum(1,ask-10*s.since_seen.gt(0).to_numpy())-1e-9)
    lagask=np.rint(recover_receipts(lag.Price.to_numpy())/1.03);laglow=np.ceil(1.03*np.maximum(1,lagask-10*valid.to_numpy())-1e-9)
    model=WeekDemand('additive');xcurrent=model.continuous(s);xlag=model.continuous(lag)
    dxcurrent=model.continuous(changed_data(s,plow))-xcurrent;dxlag=model.continuous(changed_data(lag,laglow))-xlag
    z=s[valid].reset_index(drop=True);y=z.sale_terminal_status.astype(float);clusters=pd.DataFrame({'category':pd.factorize(z.category)[0],'date':pd.factorize(z.date)[0]})
    results=[]
    for spec in ['current_price','lagged_own_price_vector','current_plus_lagged_own_residual']:
        X=(xlag if spec=='lagged_own_price_vector' else xcurrent).loc[valid].reset_index(drop=True).copy();DX=(dxlag if spec=='lagged_own_price_vector' else dxcurrent).loc[valid].reset_index(drop=True).copy()
        if spec=='current_plus_lagged_own_residual':X['lagged_own_residual']=xlag.loc[valid,'own'].to_numpy();DX['lagged_own_residual']=0.
        for fe,keys in [('category_week_date',['category_week','date']),('listing_date',['Code','date'])]:
            absorb=pd.DataFrame({k:pd.Categorical(z[k]) for k in keys})
            with warnings.catch_warnings(record=True) as logs:fit=AbsorbingLS(y,X,absorb=absorb,drop_absorbed=True).fit(cov_type='clustered',clusters=clusters,debiased=True)
            for window,use in [('full',np.ones(len(z),bool)),('novdec',z.date.ge('2023-11-01').to_numpy())]:
                contrast=DX.loc[use,list(fit.params.index)].sum().to_numpy();value=float(contrast@fit.params.to_numpy());var=float(contrast@fit.cov.to_numpy()@contrast);se=float(np.sqrt(max(0,var)))
                results.append(dict(spec=spec,fixed_effects=fe,window=window,rows=int(use.sum()),sales=int(y[use].sum()),price_contrast_extra_sales=value,se=se,normal_interval=[value-1.96*se,value+1.96*se],warnings=[str(w.message)[:200] for w in logs]))
        print('PRICE TIMING',spec,flush=True)
    z['cut']=z.Price.lt(z.previous_price);z['increase']=z.Price.gt(z.previous_price)
    counts=z.groupby(['category','date']).cut.transform('sum');z['price_change_group']=np.select([z.increase,z.cut&counts.ge(3),z.cut],['increase','cut_with_at_least_3_category_cuts','other_cut'],default='unchanged')
    raw=[]
    for window,use in [('full',np.ones(len(z),bool)),('novdec',z.date.ge('2023-11-01').to_numpy())]:
        for group,g in z[use].groupby('price_change_group'):raw.append(dict(window=window,group=group,rows=len(g),sales=int(g.sale_terminal_status.sum()),sale_rate=float(g.sale_terminal_status.mean())))
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),regressions=results,raw=raw,interpretation='Paired-sample descriptive price-timing audit. Lagged prices are not asserted exogenous; binary terminal outcomes and risk-set selection remain. Normal intervals describe clustered regression uncertainty only, not causal identification.')
    for p in [RUN/'demand_price_timing.json',ROOT/'docs/evidence/gadgetpickup_demand_timing_results_20261020.json']:p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('PRICE_TIMING_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
