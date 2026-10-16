"""Check whether explicit full-information regret envelopes can be informative.

Conditional on a known demand primitive and stated payoff support; not a final
confidence region. Includes arbitrary-state Hoeffding rewards, an anytime horizon
allowance, and explicit initialization sensitivity. The observed-state payoff
range must be justified as a support bound before using coverage language.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, lambertw
from demand_pilot import Demand
from quality_demand import data
from audit_panel import OUT

def profit_range(s,fit,cost):
    """Exact maximum/minimum over each of two monotone-logit price segments."""
    eff=fit.effects(); price=s.Price.to_numpy(); ref=s.Ref_Price.to_numpy(); rival=s.rival_min.to_numpy()
    b0=(eff['own']+eff['price_ref_interaction']*np.log(ref/500))/100
    gp=eff['gap_positive']/100; gn=eff['gap_negative']/100
    offset=fit.logit(s)-b0*price-gp*np.maximum(price-rival,0)-gn*np.minimum(price-rival,0)
    lower=np.maximum(1,np.rint(ref)-250); upper=np.rint(ref)+250
    maxima=np.zeros(len(s)); minima=np.zeros(len(s))
    for lo,hi,g in [(lower,np.minimum(upper,rival),gn),(np.maximum(lower,rival),upper,gp)]:
        slope=b0+g; intercept=offset-g*rival
        if np.any(slope>=0): raise ValueError('Monotonicity fails; analytic maximum invalid')
        valid=lo<=hi
        z=intercept+slope*cost-1
        w=lambertw(np.exp(np.clip(z,-700,700))).real
        candidate=np.clip(cost+(1+w)/(-slope),lo,hi)
        def u(p):return expit(intercept+slope*p)*(p-cost)
        best=np.maximum.reduce([u(lo),u(hi),u(candidate)])
        worst=np.minimum(u(lo),u(hi))
        maxima=np.maximum(maxima,np.where(valid,best,0))
        minima=np.minimum(minima,np.where(valid,worst,0))
    return maxima-minima,maxima

def main():
    d=data(); fit=joblib.load(OUT/'demand_rich_sale_last_seen.joblib')
    records=[]; allpoints=[]
    for seller in ['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']:
        s=d[d.Seller.eq(seller)].sort_values(['Code','date']).copy()
        ref0=s.groupby('Code').Ref_Price.transform('first').to_numpy(); ref=s.Ref_Price.to_numpy(); price=s.Price.to_numpy(); qa=fit.predict(s)
        P=np.maximum(1,np.rint(ref[:,None])+np.arange(-250,251,10)[None,:]); Q=np.empty_like(P)
        for a in range(P.shape[1]):
            x=s.copy(); x['residual']=P[:,a]-ref; Q[:,a]=fit.predict(x)
        P=np.column_stack([P,np.zeros(len(s))]);Q=np.column_stack([Q,np.zeros(len(s))])
        N=len(s); A=502; alpha=.05; Nmax=30000
        for model,grid in [('residual',np.arange(-200,101,1.)),('proportional',np.arange(.3,1.201,.002))]:
          rows=[]
          for parameter in grid:
            cost=ref0+parameter if model=='residual' else ref0*parameter
            if cost.min()<0:continue
            gain=(Q*(P-cost[:,None])).mean(axis=0)-(qa*(price-cost)).mean()
            regret=max(0,gain.max())
            ranges,positive=profit_range(s,fit,cost)
            K=float(ranges.max()); Kpos=float(positive.max()); ir=bool(np.all(cost<=price+1e-8))
            row={'seller':seller,'model':model,'parameter':parameter,'N':N,'n_listings':s.Code.nunique(),'n_days':s.date.nunique(),'regret':regret,'K':K,'K_positive':Kpos,'ir_feasible':ir,'mean_cost':cost.mean(),'mean_markup':(price-cost).mean()}
            for init in [1,2,6]:
                # Chernov's anytime mixed-loss bound + a union over possible
                # sample stopping horizons for the randomized-action term.
                factor=(np.sqrt(init*np.log(A))+np.sqrt(np.log(Nmax/alpha)/2))/np.sqrt(N)
                row[f'epsilon_init{init}']=K*factor
                row[f'epsilon_positive_init{init}']=Kpos*factor
            rows.append(row)
          frame=pd.DataFrame(rows); allpoints.extend(rows)
          for init in [1,2,6]:
            for mode in ['full_range','no_loss']:
                mask=frame.regret<=frame[f'epsilon_init{init}'] if mode=='full_range' else (frame.ir_feasible&(frame.regret<=frame[f'epsilon_positive_init{init}']))
                z=frame[mask]
                rec={'seller':seller,'model':model,'initial_entropy_multiplier':init,'mode':mode,'N':N,'candidate_count':len(frame),'accepted_count':len(z),'cost_low':z.mean_cost.min() if len(z) else None,'cost_high':z.mean_cost.max() if len(z) else None,'markup_low':z.mean_markup.min() if len(z) else None,'markup_high':z.mean_markup.max() if len(z) else None,'hits_grid_edge':bool(len(z) and (z.parameter.min()==grid.min() or z.parameter.max()==grid.max()))}
                records.append(rec); print(json.dumps(rec,default=float),flush=True)
          pd.DataFrame(records).to_csv(OUT/'scalar_envelope_summary.csv',index=False)
          pd.DataFrame(allpoints).to_csv(OUT/'scalar_envelope_points.csv',index=False)
    print('SCALAR_ENVELOPE_COMPLETE',flush=True)
if __name__=='__main__':main()
