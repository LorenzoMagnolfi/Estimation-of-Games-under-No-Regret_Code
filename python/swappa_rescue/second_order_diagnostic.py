"""Comparator-specific second-order regret diagnostic, NOT a confidence region.

The randomized ML-Prod extension is derived in docs/SWAPPA_SECOND_ORDER.md.
Observed-state bounds require an ex-ante support justification; demand is fitted.
The grid and initialization/feedback sensitivities are fixed in this script.
"""
from __future__ import annotations
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, lambertw
from quality_demand import data, QualityDemand
from demand_pilot import Demand
from audit_panel import OUT


def max_profit(s, fit, cost):
    ef=fit.effects(); p=s.Price.to_numpy(); ref=s.Ref_Price.to_numpy(); rival=s.rival_min.to_numpy()
    base=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100
    gp=ef['gap_positive']/100; gn=ef['gap_negative']/100
    intercept=fit.logit(s)-base*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    low=np.maximum(1,np.rint(ref)-250); high=np.rint(ref)+250
    best=np.zeros(len(s))
    for lo,hi,g in [(low,np.minimum(high,rival),gn),(np.maximum(low,rival),high,gp)]:
        b=base+g; a=intercept-g*rival
        if np.any(b>=0):raise ValueError('Nonnegative demand slope')
        optimum=np.clip(cost+(1+lambertw(np.exp(np.clip(a+b*cost-1,-700,700))).real)/(-b),lo,hi)
        value=expit(a+b*optimum)*(optimum-cost)
        best=np.maximum(best,np.where(lo<=hi,value,0))
    return best


def second_order_bound(V, K, N, log_prior, copies=1, alpha=.05, rates=10):
    """D separate learners: sum their valid bounds, same rate per comparator.

    Union across D learners. log_prior includes log(A*J) and warm-start penalty.
    For D=1 this follows directly from a single wealth martingale and Ville.
    """
    eta=2.**(-np.arange(rates))[:,None]/(2*np.maximum(K,1e-10)[None,:])
    L=log_prior+np.log(copies/alpha)
    return (copies*L/eta+eta*V[None,:]).min(axis=0)/N


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--demand',choices=['rich','quality'],default='quality'); ap.add_argument('--support-mode',choices=['observed','universal'],default='observed'); ap.add_argument('--alpha',type=float,default=.05); args=ap.parse_args()
    d=data(); name='demand_rich_sale_last_seen' if args.demand=='rich' else 'quality_demand_quality'
    fit=joblib.load(OUT/f'{name}.joblib'); records=[]; points=[]
    sellers=['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']
    for seller in sellers:
        s=d[d.Seller.eq(seller)].sort_values(['Code','date']).copy(); N=len(s)
        ref=s.Ref_Price.to_numpy(); ref0=s.groupby('Code').Ref_Price.transform('first').to_numpy(); p=s.Price.to_numpy(); qa=fit.predict(s)
        P=np.maximum(1,np.rint(ref[:,None])+np.arange(-250,251,10)[None,:]); Q=np.empty_like(P)
        for a in range(P.shape[1]):
            x=s.copy(); x['residual']=P[:,a]-ref; Q[:,a]=fit.predict(x)
        P=np.column_stack([P,np.zeros(N)]);Q=np.column_stack([Q,np.zeros(N)])
        U=Q*P-qa[:,None]*p[:,None]; V=Q-qa[:,None]; W=V*ref0[:,None]
        moments=[(U*U).sum(0),(W*W).sum(0),(V*V).sum(0),(U*W).sum(0),(U*V).sum(0),(W*V).sum(0)]
        means=[U.mean(0),W.mean(0),V.mean(0)]; sold=s.sale_last_seen.eq(1).to_numpy(); entry=~s.Code.duplicated().to_numpy()
        for model,grid in [('residual',np.arange(-300.,201.)),('proportional',np.arange(.3,1.201,.002))]:
            results=[]
            for parameter in grid:
                beta,delta=(1.,parameter) if model=='residual' else (parameter,0.)
                cost=beta*ref0+delta
                if cost.min()<0 or (args.support_mode=='universal' and cost.max()>1500):continue
                R=means[0]-beta*means[1]-delta*means[2]
                sq=moments[0]+beta**2*moments[1]+delta**2*moments[2]-2*beta*moments[3]-2*delta*moments[4]+2*beta*delta*moments[5]
                sq=np.maximum(sq,0)
                maxu=max_profit(s,fit,cost)
                # One-sided bound: u(a)-u(chosen) >= -K_a, allowing positive
                # excess rewards to be arbitrarily large in log(1+x)>=x-x^2.
                K=np.maximum(1e-8,(maxu[:,None]-Q*(P-cost[:,None])).max(axis=0))
                if args.support_mode=='universal':
                    assert P.max()<=2000 and p.max()<=2000
                    # All profits lie in [-c,2000-c], including zero for 0<=c<=1500.
                    K=np.full_like(K,2000.)
                rec={'demand':args.demand,'support_mode':args.support_mode,'alpha':args.alpha,'seller':seller,'model':model,'parameter':parameter,'N':N,'n_listings':int(entry.sum()),'regret':float(max(0,R.max())),'K_nooffer':K[-1],'mean_cost':cost.mean(),'mean_markup':(p-cost).mean(),'sold_cost':cost[sold].mean(),'sold_markup':(p-cost)[sold].mean(),'listing_cost':cost[entry].mean(),'ir_feasible':bool(np.all(cost<=p))}
                for copies in [1,10,100]:
                  for warm in [0,1,5]:
                    for scale in [1,2]:
                        log_prior=np.log(502*10)+warm*np.log(502)
                        bound=second_order_bound(sq,K*scale,N,log_prior,copies,alpha=args.alpha)
                        key=f'D{copies}_H{warm}_S{scale}'
                        rec[f'violation_{key}']=float((R-bound).max())
                        rec[f'nooffer_bound_{key}']=float(bound[-1])
                results.append(rec)
            z=pd.DataFrame(results); points.extend(results)
            for copies in [1,10,100]:
              for warm in [0,1,5]:
                for scale in [1,2]:
                    key=f'D{copies}_H{warm}_S{scale}'; acc=z[z[f'violation_{key}']<=1e-10]
                    row={'demand':args.demand,'support_mode':args.support_mode,'alpha':args.alpha,'seller':seller,'model':model,'copies':copies,'warm_logA':warm,'support_scale':scale,'accepted':len(acc),'candidates':len(z),'cost_low':acc.mean_cost.min(),'cost_high':acc.mean_cost.max(),'markup_low':acc.mean_markup.min(),'markup_high':acc.mean_markup.max(),'sold_cost_low':acc.sold_cost.min(),'sold_cost_high':acc.sold_cost.max(),'sold_markup_low':acc.sold_markup.min(),'sold_markup_high':acc.sold_markup.max(),'grid_boundary':bool(len(acc) and (acc.parameter.min()==grid.min() or acc.parameter.max()==grid.max()))}
                    records.append(row)
                    if copies==1 and warm==0:print(json.dumps(row,default=float),flush=True)
            pd.DataFrame(records).to_csv(OUT/f'second_order_{args.demand}_{args.support_mode}_a{args.alpha}_summary.csv',index=False)
            pd.DataFrame(points).to_csv(OUT/f'second_order_{args.demand}_{args.support_mode}_a{args.alpha}_points.csv',index=False)
    print('SECOND_ORDER_COMPLETE',flush=True)

if __name__=='__main__':main()
