"""Inventory diagnostics and full-history information bounds, not cost estimates."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import betaln, xlogy
from audit_panel import ROOT, OUT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts
from gadget_demand_audit import RUN

DESIGN = ROOT / 'docs/evidence/gadgetpickup_inventory_feedback_extension_design_20261020.json'

def band(n, s, alpha):
    """Invert the bounded-outcome beta mixture at a fixed error budget."""
    if n == 0:
        return (0., 1.)
    k = betaln(s+.5, n-s+.5)-betaln(.5,.5)+np.log(alpha)
    def f(q):
        return k-xlogy(s,q)-xlogy(n-s,1-q)
    return (brentq(f,0,s/n,xtol=1e-14) if s>0 else 0.,
            brentq(f,s/n,1,xtol=1e-14) if s<n else 1.)

def shape_bands(prices, lower, upper):
    """Project marginal bands under nonincreasing probability in price."""
    order=np.argsort(prices)
    lo=np.array(lower)[order]; hi=np.array(upper)[order]
    lo=np.maximum.accumulate(lo[::-1])[::-1]
    hi=np.minimum.accumulate(hi)
    undo=np.argsort(order)
    return lo[undo],hi[undo]

def inventory():
    source=ROOT/'output/swappa_rescue_20261016/fullsource/reference_screened/panel.parquet'
    d=pd.read_parquet(source); begin=d.date.min(); end=d.date.max()
    s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).copy()
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    first=s.groupby('Code').date.transform('min'); last=s.groupby('Code').date.transform('max')
    s['entry']=s.date.eq(first); s['exit']=s.date.eq(last)
    s['gap_return']=s.groupby('Code').date.diff().dt.days.gt(1)
    s['gap_start']=s.groupby('Code').date.shift(-1).sub(s.date).dt.days.gt(1)
    s['nonsale_exit']=s.exit & s.Real_Sold.eq(0) & s.date.lt(end)
    s['sale']=s.sale_terminal_status.eq(1)&s.date.lt(end)
    s['inventory_age']=(s.date-first).dt.days
    g=s.groupby(['category','date']).agg(stock=('Code','size'),entry=('entry','sum'),exits=('exit','sum'),sale=('sale','sum'),nonsale=('nonsale_exit','sum'),gap_return=('gap_return','sum'),gap_start=('gap_start','sum'))
    cats=sorted(s.category.unique()); dates=pd.date_range(begin,end)
    g=g.reindex(pd.MultiIndex.from_product([cats,dates],names=['category','date']),fill_value=0).reset_index()
    records=[]; stockouts=[]
    for cat,z in g.groupby('category',sort=False):
        z=z.sort_values('date').copy()
        active=z.stock.gt(0).to_numpy(); ix=np.flatnonzero(active)
        if len(ix)>1:
            gaps=np.diff(ix)-1
            stockouts.extend([dict(category=cat,days=int(k)) for k in gaps if k>0])
        for h in [1,3,7,14]:
            future=sum(z.entry.shift(-k,fill_value=0) for k in range(1,h+1))
            past=sum(z.entry.shift(k,fill_value=0) for k in range(1,h+1))
            inside=z.date.between(begin+pd.Timedelta(days=14),end-pd.Timedelta(days=14))
            for typ in ['sale','nonsale','active_day']:
                w=z[typ].to_numpy() if typ!='active_day' else active.astype(int)
                use=inside.to_numpy()&(w>0)
                if not use.any():continue
                records.append(dict(category=cat,horizon=h,event=typ,events=int(w[use].sum()),future_entry_sum=float(np.dot(w[use],future[use])),past_entry_sum=float(np.dot(w[use],past[use])),future_any_sum=float(np.dot(w[use],future[use].gt(0))),past_any_sum=float(np.dot(w[use],past[use].gt(0)))))
    r=pd.DataFrame(records); r.to_csv(RUN/'inventory_entry_windows.csv',index=False)
    windows=r.groupby(['horizon','event'])[['events','future_entry_sum','past_entry_sum','future_any_sum','past_any_sum']].sum().reset_index()
    for v in ['future_entry','past_entry','future_any','past_any']:
        windows[v+'_mean']=windows[v+'_sum']/windows.events
    continuing=s[~s.entry].groupby(['category','date']).agg(continuing_prices=('Price','nunique'),continuing_price=('Price','first'),continuing_age=('inventory_age','mean'),continuing_n=('Code','size'))
    entrants=s[s.entry&s.date.gt(begin)].merge(continuing,on=['category','date'],how='left',validate='many_to_one')
    matched=entrants[entrants.continuing_prices.eq(1)].copy()
    matched['same_price']=matched.Price.eq(matched.continuing_price)
    matched.to_csv(RUN/'inventory_entrant_prices.csv',index=False)
    daily=s.groupby('date').agg(stock=('Code','size'),entry=('entry','sum'),exits=('exit','sum'),gap_return=('gap_return','sum'),gap_start=('gap_start','sum')).reindex(dates,fill_value=0)
    error=daily.stock.diff()-daily.entry-daily.gap_return+daily.exits.shift(1)+daily.gap_start.shift(1)
    assert error.dropna().abs().max()==0
    listings=s.groupby('Code').agg(first=('date','min'),last=('date','max'),rows=('Code','size'),sold=('Real_Sold','first'))
    interior=g[g.date.between(begin+pd.Timedelta(days=14),end-pd.Timedelta(days=14))]
    sold=listings[listings.sold.eq(1)&listings['last'].lt(end)]
    return dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=len(s),listings=len(listings),categories=len(cats),dates=len(dates),known_sales=int(s.sale.sum()),nonsale_exits=int(s.nonsale_exit.sum()),right_edge_listings=int(listings['last'].eq(end).sum()),gap_returns=int(s.gap_return.sum()),stock_identity_max_error=float(error.abs().max()),interior_stock_mean=float(interior.groupby('date').stock.sum().mean()),interior_entries=int(interior.entry.sum()),interior_sales=int(interior.sale.sum()),sold_observed_days_median=float(sold.rows.median()),sold_observed_days_p90=float(sold.rows.quantile(.9)),entrants_after_first_date=len(entrants),entrants_with_continuing_stock=int(entrants.continuing_n.notna().sum()),entrants_with_unique_continuing_price=len(matched),entrant_same_price_fraction=float(matched.same_price.mean()),matched_continuing_age_mean=float(matched.continuing_age.mean()),category_stockout_spells=len(stockouts),category_stockout_median=float(np.median([a['days'] for a in stockouts])),entry_windows=windows.to_dict('records'))

def feedback():
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].copy()
    s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    chosen=s[s.date.le('2023-09-30')].groupby('category').size().sort_values(ascending=False).head(5).index
    g=s.groupby(['date','category']).agg(rows=('Code','size'),sales=('sale_terminal_status','sum'),price=('Price','first'),distinct=('Price','nunique')).reset_index()
    g=g[g.distinct.eq(1)].sort_values(['date','category'])
    records=[]
    for unit in ['day','listing']:
        states={}; indices={}
        for r in g.itertuples():
            if r.category not in states:
                states[r.category]={};indices[r.category]=len(indices)+1
            state=states[r.category]; i=indices[r.category];p=int(r.price)
            if p not in state:state[p]=[0.,0.,len(state)+1]
            if r.date>=pd.Timestamp('2023-10-01'):
                prices=np.array(list(state)); intervals=np.array([band(n,z,.05/(i*(i+1)*j*(j+1))) for n,z,j in state.values()])
                x=recover_receipts(prices);at=int(np.flatnonzero(prices==p)[0])
                for restriction in ['exact_price','monotone_price']:
                    lo,hi=intervals.T if restriction=='exact_price' else shape_bands(prices,*intervals.T)
                    empty=bool(np.any(lo>hi+1e-10))
                    best=max(0.,float(np.max(x*lo)))
                    records.append(dict(unit=unit,restriction=restriction,category=r.category,category_index=i,date=str(r.date.date()),selected=r.category in chosen,rows=r.rows,sales=r.sales,price=p,prior_count=state[p][0],lower=float(lo[at]),upper=float(hi[at]),width=float(hi[at]-lo[at]),empty=empty,zero_cost_dominated=(float(x[at]*hi[at])+1e-10<best) if not empty else None,prices=len(state)))
            state[p][0]+=1 if unit=='day' else r.rows
            state[p][1]+=r.sales/r.rows if unit=='day' else r.sales
    a=pd.DataFrame(records);a.to_csv(RUN/'feedback_full_history.csv',index=False)
    out=[]
    for (unit,restriction,cat),z in a.groupby(['unit','restriction','category']):
        valid=z[~z.empty]
        out.append(dict(unit=unit,restriction=restriction,category=cat,selected=bool(z.selected.iloc[0]),category_index=int(z.category_index.iloc[0]),days=len(z),empty_days=int(z.empty.sum()),mean_width=float(valid.width.mean()) if len(valid) else None,zero_cost_dominated_days=int(valid.zero_cost_dominated.eq(True).sum()),max_prior_count=float(z.prior_count.max())))
    return out

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),inventory=inventory(),feedback=feedback(),interpretation='Inventory events are listing events, not identified acquisitions. Feedback confidence requires constant conditional means and valid timely outcomes; monotonicity adds a restriction. No actual regret or cost interval is estimated.')
    for p in [RUN/'inventory_feedback_extension.json',ROOT/'docs/evidence/gadgetpickup_inventory_feedback_extension_results_20261020.json']:
        p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['inventory'],indent=2),flush=True)
    print(json.dumps([r for r in result['feedback'] if r['selected']],indent=2),flush=True)
    print('INVENTORY_FEEDBACK_EXTENSION_COMPLETE',flush=True)

if __name__=='__main__':main()
