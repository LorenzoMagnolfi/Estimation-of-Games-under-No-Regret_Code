"""Sequential sales-information audit, not a seller-algorithm or cost estimate."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import betaln,xlogy
from scipy.optimize import brentq
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts
from gadget_demand_audit import RUN
from audit_panel import ROOT

def interval(n,s,j,method):
    if n==0:return (0.,1.)
    alpha=.05/(5*j*(j+1))
    if method=='hoeffding':
        r=np.sqrt(np.log(2*n*(n+1)/alpha)/(2*n));return max(0,s/n-r),min(1,s/n+r)
    const=betaln(s+.5,n-s+.5)-betaln(.5,.5)-np.log(1/alpha)
    def f(q):return const-xlogy(s,q)-xlogy(n-s,1-q)
    center=s/n
    lo=brentq(f,0,center,xtol=1e-14) if s>0 else 0.
    hi=brentq(f,center,1,xtol=1e-14) if s<n else 1.
    assert lo<=center<=hi
    return lo,hi

def main():
    d=terminal_status_data();s=d[d.Seller.eq('GadgetPickup')].copy();s['category']=s[['device_id','Storage','Condition']].astype(str).agg('|'.join,axis=1)
    chosen=s[s.date.le('2023-09-30')].groupby('category').size().sort_values(ascending=False).head(5).index
    g=s[s.category.isin(chosen)&s.date.between('2023-10-01','2023-12-22')].groupby(['category','date']).agg(rows=('Code','size'),sales=('sale_terminal_status','sum'),price=('Price','first'),distinct=('Price','nunique')).reset_index()
    g=g[g.distinct.eq(1)].copy();records=[]
    for method in ['hoeffding','mixture']:
      for unit in ['day','listing']:
       for cat in chosen:
        state={}
        for r in g[g.category.eq(cat)].sort_values('date').itertuples():
            p=int(r.price)
            if p not in state:state[p]=[0.,0.,len(state)+1]
            intervals={a:interval(*v,method) for a,v in state.items()}
            lo,hi=intervals[p];x=float(recover_receipts(np.array([p]))[0])
            best_lower=max([0.]+[float(recover_receipts(np.array([a]))[0])*v[0] for a,v in intervals.items()])
            records.append(dict(method=method,unit=unit,category=cat,date=str(r.date.date()),rows=r.rows,sales=r.sales,price=p,prior_count=state[p][0],lower=lo,upper=hi,width=hi-lo,zero_cost_not_dominated=x*hi+1e-10>=best_lower,zero_cost_chosen_profit_width=x*(hi-lo),introduced_prices=len(state)))
            state[p][0]+=1 if unit=='day' else r.rows;state[p][1]+=r.sales/r.rows if unit=='day' else r.sales
    a=pd.DataFrame(records);a.to_csv(RUN/'feedback_intervals.csv',index=False)
    summary=[]
    for (method,unit,cat),z in a.groupby(['method','unit','category']):
        summary.append(dict(method=method,unit=unit,category=cat,category_days=len(z),exposures=int(z.rows.sum()),sales=int(z.sales.sum()),positive_prior_lower_days=int(z.lower.gt(0).sum()),mean_width=float(z.width.mean()),median_width=float(z.width.median()),zero_cost_dominated_days=int((~z.zero_cost_not_dominated).sum()),mean_zero_cost_chosen_profit_width=float(np.average(z.zero_cost_chosen_profit_width,weights=z.rows)),max_prior_count=float(z.prior_count.max()),prices=int(z.price.nunique())))
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256((ROOT/'docs/evidence/gadgetpickup_feedback_design_20261020.json').read_bytes()).hexdigest(),summary=summary,interpretation='Simultaneous demand bounds only under the stated constant conditional mean and timely-feedback observation model. No seller adoption, optimality, or cost interval claimed. Zero-cost non-dominance is an information diagnostic against previously encountered prices, not full application.')
    for p in [RUN/'feedback_audit.json',ROOT/'docs/evidence/gadgetpickup_feedback_results_20261020.json']:p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('GADGET_FEEDBACK_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
