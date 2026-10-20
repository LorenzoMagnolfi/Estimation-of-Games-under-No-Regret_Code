"""Coordinated markdown, stable-price, and frozen-hazard demand diagnostics."""
import hashlib,json,subprocess,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2
from linearmodels.iv import AbsorbingLS
from threadpoolctl import threadpool_limits
from audit_panel import ROOT
from ask_price_grid import recover_receipts
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_demand_crossfit import WeekDemand
from policy_cost_frontier import changed_data

DESIGN=ROOT/'docs/evidence/gadgetpickup_persistent_review_design_20261020.json'


def contrast(fit,weights):
    w=np.array([weights.get(k,0.) for k in fit.params.index]);v=float(w@fit.params.to_numpy());var=float(w@fit.cov.to_numpy()@w)
    return dict(estimate=v,se=float(np.sqrt(max(0.,var))),negative_variance=var<0)


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=terminal_status_data();prior=pd.read_parquet(RUN/'repricing_performance_predictions.parquet');pred=pd.read_parquet(RUN/'performance_predictions.parquet')
    s=raw[raw.Seller.eq('GadgetPickup')].merge(prior[['Code','date','previous_date','previous_price']],on=['Code','date'],validate='one_to_one').merge(pred[['Code','date','q']],on=['Code','date'],validate='one_to_one').sort_values(['Code','date']).reset_index(drop=True)
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1);s['category_week']=s.category+'::'+s.week.astype(str);s['category_date']=s.category+'::'+s.date.astype(str)
    s['valid']=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    s['cut']=s.valid&s.Price.lt(s.previous_price);s['increase']=s.valid&s.Price.gt(s.previous_price);s['changed']=s.cut|s.increase
    s['ask']=recover_receipts(s.Price.to_numpy())/1.03;s['logage']=np.log1p(s.since_seen)
    batch=s[s.changed].groupby('date').agg(n=('Code','size'),models=('device_id','nunique'));dates=sorted(batch.index[(batch.n>=10)&(batch.models>=3)])
    cd=s.groupby(['category','date']).agg(n=('Code','size'),sales=('sale_terminal_status','sum'),ask=('ask','mean'),logage=('logage','mean'),rival_min=('rival_min','mean'),valid=('valid','sum'),cuts=('cut','sum'),increases=('increase','sum')).reset_index();cd['rate']=cd.sales/cd.n
    stacks=[];details=[]
    for date in dates:
        if date<s.date.min()+pd.Timedelta(days=7) or date>s.date.max()-pd.Timedelta(days=7):continue
        a=cd[cd.date.eq(date)].copy();a['treated']=(a.cuts>=3)&a.increases.eq(0);a['control']=(a.valid>=3)&a.cuts.eq(0)&a.increases.eq(0)
        a=a[a.treated|a.control];base=cd[cd.date.eq(date-pd.Timedelta(days=1))][['category','n']].rename(columns={'n':'weight'})
        a=a.merge(base,on='category',validate='one_to_one');a['treated']=a.treated.astype(int)
        if a.treated.sum()==0 or a.treated.sum()==len(a):continue
        grid=a[['category','treated','weight']].merge(pd.DataFrame({'rel':range(-7,8)}),how='cross');grid['date']=date+pd.to_timedelta(grid.rel,unit='D');grid=grid.merge(cd,on=['category','date'],how='left',validate='one_to_one');grid['anchor']=date
        grid['stack_category']=str(date.date())+'::'+grid.category;grid['stack_date']=str(date.date())+'::'+grid.date.astype(str)
        stacks.append(grid)
        details.append(dict(anchor=str(date.date()),treated_categories=int(a.treated.sum()),control_categories=int((1-a.treated).sum()),missing_category_days=int(grid.rate.isna().sum()),potential_category_days=len(grid),treated_categories_with_subsequent_changes=int(grid[(grid.rel>0)&grid.treated.eq(1)&((grid.cuts+grid.increases)>0)].category.nunique()),control_categories_with_subsequent_changes=int(grid[(grid.rel>0)&grid.treated.eq(0)&((grid.cuts+grid.increases)>0)].category.nunique())))
    stacked=pd.concat(stacks,ignore_index=True);stacked.to_parquet(RUN/'batch_event_stack.parquet',index=False)
    events=[]
    for window in ['full','novdec']:
        z=stacked if window=='full' else stacked[stacked.anchor.ge('2023-11-01')]
        z=z[z.rate.notna()].reset_index(drop=True)
        X=pd.DataFrame({str(k):(z.treated*z.rel.eq(k)).astype(float) for k in range(-7,8) if k!=-1})
        absorb=pd.DataFrame({k:pd.Categorical(z[k]) for k in ['stack_category','stack_date']});clusters=pd.DataFrame({'category':pd.factorize(z.category)[0],'date':pd.factorize(z.date)[0]})
        for outcome in ['rate','ask','logage','rival_min']:
            with warnings.catch_warnings(record=True) as logs:fit=AbsorbingLS(z[outcome],X,absorb=absorb,weights=z.weight,drop_absorbed=True).fit(cov_type='clustered',clusters=clusters,debiased=True)
            leads=[str(k) for k in range(-7,-1) if str(k) in fit.params.index];vc=fit.cov.loc[leads,leads].to_numpy();bb=fit.params[leads].to_numpy();rank=int(np.linalg.matrix_rank(vc));stat=float(bb@np.linalg.pinv(vc)@bb)
            rec=dict(window=window,outcome=outcome,stacked_rows=len(z),anchors=int(z.anchor.nunique()),categories=int(z.category.nunique()),coefficients={str(k):contrast(fit,{str(k):1.}) for k in range(-7,8) if k!=-1},contrasts={name:contrast(fit,{str(k):1/len(ks) for k in ks}) for name,ks in [('leads',list(range(-7,-1))),('post_0_3',list(range(4))),('post_1_3',[1,2,3]),('post_4_7',[4,5,6,7])]},lead_wald_stat=stat,lead_wald_df=rank,lead_wald_p=float(chi2.sf(stat,rank)) if rank>0 and stat>=0 else None,lead_covariance_min_eigenvalue=float(np.linalg.eigvalsh(vc).min()),warnings=[str(w.message)[:200] for w in logs])
            events.append(rec);print('BATCH EVENT',window,outcome,rec['contrasts'],flush=True)
    # Stable-quote and tight fixed-effect diagnostics, with the same shift vector.
    model=WeekDemand('additive');X0=model.continuous(s).astype(float);newprice=np.ceil(1.03*np.maximum(1,np.rint(s.ask)-10*s.since_seen.gt(0).to_numpy())-1e-9);DX=model.continuous(changed_data(s,newprice))-X0
    stable=[]
    for sample,mask in [('fresh_previous_quote',s.valid),('unchanged_quote',s.valid&~s.changed)]:
        z=s[mask].reset_index(drop=True);X=X0.loc[mask].reset_index(drop=True);delta=DX.loc[mask].reset_index(drop=True)
        clusters=pd.DataFrame({'category':pd.factorize(z.category)[0],'date':pd.factorize(z.date)[0]})
        for fe,keys in [('category_week_date',['category_week','date']),('listing_date',['Code','date']),('listing_category_date',['Code','category_date'])]:
            with warnings.catch_warnings(record=True) as logs:fit=AbsorbingLS(z.sale_terminal_status.astype(float),X,absorb=pd.DataFrame({k:pd.Categorical(z[k]) for k in keys}),drop_absorbed=True).fit(cov_type='clustered',clusters=clusters,debiased=True)
            for window,use in [('full',np.ones(len(z),bool)),('novdec',z.date.ge('2023-11-01').to_numpy())]:
                weights=delta.loc[use,list(fit.params.index)].sum().to_dict();c=contrast(fit,weights)
                stable.append(dict(sample=sample,fixed_effects=fe,window=window,rows=int(use.sum()),sales=int(z.loc[use,'sale_terminal_status'].sum()),extra_sales_contrast=c,retained_columns=list(fit.params.index),warnings=[str(w.message)[:300] for w in logs]))
            print('STABLE DEMAND',sample,fe,flush=True)
    # Censor before changed quotes or gaps; never use a sale-ending spell as an ex ante horizon.
    indexed={(row.Code,row.date):row for row in s.itertuples()};history=[];endings=[]
    for date in dates:
        for anchor in s[s.date.eq(date)].itertuples():
            end='horizon_14'
            for h in range(14):
                row=indexed.get((anchor.Code,date+pd.Timedelta(days=h)))
                if row is None:end='missing_or_exit';break
                if row.Price!=anchor.Price:end='new_quote';break
                history.append(dict(anchor=date,Code=anchor.Code,date=row.date,depth=h,actual=int(row.sale_terminal_status),frozen=float(anchor.q),updated=float(row.q)))
                if row.sale_terminal_status:end='sale';break
            endings.append(dict(anchor=date,reason=end,depth=h))
    hh=pd.DataFrame(history);ee=pd.DataFrame(endings);hh['band']=pd.cut(hh.depth,[-1,0,2,6,13],labels=['0','1-2','3-6','7-13']);hazard=[]
    for window in ['full','novdec']:
        z=hh if window=='full' else hh[hh.anchor.ge('2023-11-01')];e=ee if window=='full' else ee[ee.anchor.ge('2023-11-01')]
        for band,g in z.groupby('band',observed=True):
            hazard.append(dict(window=window,band=str(band),duplicated_anchor_exposures=len(g),distinct_listing_days=int(g[['Code','date']].drop_duplicates().shape[0]),actual_sales=int(g.actual.sum()),frozen_expected_sales=float(g.frozen.sum()),updated_expected_sales=float(g.updated.sum()),frozen_brier=float(np.mean((g.actual-g.frozen)**2)),updated_brier=float(np.mean((g.actual-g.updated)**2))))
        print('HAZARD ENDINGS',window,e.reason.value_counts().to_dict(),flush=True)
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),batch_dates=[str(x.date()) for x in dates],stack_details=details,events=events,stable_price=stable,hazard=hazard,hazard_endings={w:(ee if w=='full' else ee[ee.anchor.ge('2023-11-01')]).reason.value_counts().to_dict() for w in ['full','novdec']},interpretation='Descriptive event studies and hazard diagnostics. Assignment and review dates are endogenous; clustered standard errors do not establish causal price response. Constant hazards, changing risk sets, and future control repricing are audited explicitly.')
    (ROOT/'docs/evidence/gadgetpickup_batch_demand_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('BATCH_DEMAND_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
