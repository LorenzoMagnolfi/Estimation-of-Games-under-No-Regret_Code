"""Outer bounds on persistent continuous costs from contextual regret.

x[k,l] is latent probability of interval k for listing l; z[k,l] is its
cost-residual first moment. Interval inequalities are valid for continuous costs.
The identified object is an empirical mean, not a population distribution.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, diags, eye, hstack, vstack
from demand_pilot import Demand
from audit_panel import OUT

def arrays(s,fit,actions,nooffer=False):
    s=s.sort_values(['Code','date']).reset_index(drop=True)
    codes,idx=np.unique(s.Code,return_inverse=True)
    first=s.groupby('Code',sort=True).first(); last=s.groupby('Code',sort=True).last()
    L=len(codes); N=len(s); ref0=first.Ref_Price.to_numpy(); ref=s.Ref_Price.to_numpy()
    qa=fit.predict(s)
    prices=np.maximum(1,ref[:,None]+actions)
    q=np.empty_like(prices)
    for a in range(len(actions)):
        sc=s.copy(); sc['residual']=prices[:,a]-ref
        q[:,a]=fit.predict(sc)
    if nooffer:
        q=np.column_stack([q,np.zeros(len(s))])
        prices=np.column_stack([prices,np.zeros(len(s))])
    B=np.zeros((L,q.shape[1])); C=B.copy()
    np.add.at(B,idx,q-qa[:,None])
    np.add.at(C,idx,q*(prices-ref0[idx,None])-qa[:,None]*(s.Price.to_numpy()[:,None]-ref0[idx,None]))
    sold=s.groupby('Code',sort=True).sale_last_seen.max().to_numpy()
    # All objectives use the SAME listings and demand coefficients.
    weights={'listing_mean':np.ones(L)/L,'sold_mean':sold/sold.sum()}
    return C/N,B/N,ref0,last.Price.to_numpy(),weights,{'N':N,'L':L,'sold_listings':int(sold.sum()),'cost_upper_by_listing':(s.groupby('Code',sort=True).Price.min().to_numpy()-ref0).tolist()}

def program(C,B,ref0,edges):
    L,A=C.shape; K=len(edges)-1; V=L*K
    regx=block_diag([csr_matrix(C.T)]*K,format='csr')
    regz=block_diag([csr_matrix(-B.T)]*K,format='csr')
    epsmap=block_diag([np.ones((A,1))]*K,format='csr')
    R=hstack([regx,regz,-epsmap],format='csr')
    lower=np.maximum(np.repeat(edges[:-1],L),np.tile(-ref0,K))
    upper=np.repeat(edges[1:],L)
    # Empty interval support is handled by x=z=0.
    valid=lower<=upper
    lower=np.where(valid,lower,0); upper=np.where(valid,upper,0)
    limits_low=hstack([diags(lower),-eye(V),csr_matrix((V,K))],format='csr')
    limits_high=hstack([-diags(upper),eye(V),csr_matrix((V,K))],format='csr')
    U=vstack([R,limits_low,limits_high],format='csr')
    E=hstack([hstack([eye(L)]*K),csr_matrix((L,V+K))],format='csr')
    bounds=[(0,1) if v else (0,0) for v in valid]+[(None,None) if v else (0,0) for v in valid]+[(0,None)]*K
    return U,E,bounds,V,K

def endpoint_program(C,B,ref0,edges,cost_upper=None):
    """Equivalent interval moments using nonnegative endpoint weights.

    Endpoint atoms encode only each interval's probability and first moment;
    they impose no discreteness restriction on the underlying costs because
    all payoff differences and targets are affine in cost within an interval.
    """
    L,A=C.shape; K=len(edges)-1
    lows=np.maximum(np.asarray(edges[:-1],float)[:,None],-ref0[None,:])
    ups=np.broadcast_to(np.asarray(edges[1:],float)[:,None],(K,L))
    if cost_upper is not None: ups=np.minimum(ups,np.asarray(cost_upper)[None,:])
    valid=lows<=ups
    lows=np.where(valid,lows,0); ups=np.where(valid,ups,0)
    value=np.stack([lows,ups],axis=1).reshape(K,2*L)
    blocks=[csr_matrix(np.hstack([(C-lows[k,:,None]*B).T,(C-ups[k,:,None]*B).T])) for k in range(K)]
    reg=block_diag(blocks,format='csr')
    epsmap=block_diag([csr_matrix(np.ones((A,1)))]*K,format='csr')
    U=hstack([reg,-epsmap],format='csr')
    E=hstack([hstack([eye(L)]*(2*K)),csr_matrix((L,K))],format='csr')
    bounds=[(0,1) if v else (0,0) for v in np.repeat(valid[:,None,:],2,axis=1).ravel()]+[(0,None)]*K
    return U,E,bounds,value,K

def solve(C,B,ref0,last_price,weights,meta,edges,budgets,individual_ir=False):
    cost_upper=meta.pop('cost_upper_by_listing')
    U,E,bounds,values,K=endpoint_program(C,B,ref0,edges,cost_upper if individual_ir else None); L=len(ref0); V=L*K
    b=np.zeros(U.shape[0]); one=np.ones(L)
    o=np.r_[np.zeros(2*V),np.ones(K)]
    options={'time_limit':180,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8}
    r=linprog(o,A_ub=U,b_ub=b,A_eq=E,b_eq=one,bounds=bounds,method='highs',options=options)
    print('minimum',r.message,r.fun,flush=True)
    if not r.success:return []
    U=vstack([U,csr_matrix(o[None,:])],format='csr')
    records=[]
    for budget in budgets:
      for name,wt in weights.items():
        obj=np.r_[(values*np.tile(wt,2)[None,:]).ravel(),np.zeros(K)]
        row={**meta,'estimand':name,'intervals':K,'support_low':edges[0],'support_high':edges[-1],'epsilon':budget,'min_total_regret':r.fun,'mean_anchor_ref':ref0@wt,'mean_last_price':last_price@wt}
        for sign,label in [(1,'low'),(-1,'high')]:
          start=time.time()
          result=linprog(sign*obj,A_ub=U,b_ub=np.r_[b,budget],A_eq=E,b_eq=one,bounds=bounds,method='highs',options=options)
          row[label+'_success']=result.success; row[label+'_seconds']=time.time()-start
          row[label+'_message']=result.message
          if result.success:
            row['cost_'+label]=float(ref0@wt+sign*result.fun)
            row[label+'_violation']=float(max(0,(U@result.x-np.r_[b,budget]).max(),np.abs(E@result.x-one).max()))
            mass=result.x[:2*V].reshape(K,2,L).sum(axis=1)@wt
            row[label+'_edge_mass']=float(mass[0]+mass[-1])
        if row.get('low_success') and row.get('high_success'):
            row['gross_markup_low']=row['mean_last_price']-row['cost_high']
            row['gross_markup_high']=row['mean_last_price']-row['cost_low']
        records.append(row); print(json.dumps(row,default=float),flush=True)
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--spec',default='rich'); p.add_argument('--outcome',default='sale_last_seen'); p.add_argument('--price-step',type=int,default=10); p.add_argument('--cost-step',type=int,default=25); p.add_argument('--support-high',type=int,default=500); p.add_argument('--budgets',default='0,.1,.25,.5,1'); p.add_argument('--nooffer',action='store_true'); p.add_argument('--individual-ir',action='store_true'); args=p.parse_args()
    d=pd.read_parquet(OUT/'panel.parquet'); d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy(); d=d[~(d.Real_Sold.eq(1)&d.date.gt(d.Updated))].copy()
    top=d.groupby('Seller').Code.nunique().nlargest(20).index; d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    cache=OUT/f'demand_{args.spec}_{args.outcome}.joblib'
    if cache.exists():fit=joblib.load(cache)
    else:
        fit=Demand(args.spec); fit.model.set_params(max_iter=2000); fit.fit(d,d[args.outcome]); joblib.dump(fit,cache)
    print(json.dumps({'effects':fit.effects(),'n_iter':fit.model.n_iter_.tolist()}),flush=True)
    actions=np.arange(-250,251,args.price_step)+.01
    edges=np.r_[-1500,-500,np.arange(-250,151,args.cost_step),args.support_high]
    tag=f'{args.spec}_{args.outcome}_p{args.price_step}_c{args.cost_step}_u{args.support_high}_optout{int(args.nooffer)}_ir{int(args.individual_ir)}'
    records=[]
    for seller in ['AMS Traders Inc.','GadgetPickup']:
        s=d[d.Seller.eq(seller)].copy()
        C,B,ref0,last,wt,meta=arrays(s,fit,actions,args.nooffer)
        meta.update(seller=seller,spec=args.spec,outcome=args.outcome,price_step=args.price_step,nooffer=args.nooffer,individual_ir=args.individual_ir)
        rows=solve(C,B,ref0,last,wt,meta,edges,[float(x) for x in args.budgets.split(',')],args.individual_ir)
        records.extend(rows); pd.DataFrame(records).to_csv(OUT/f'continuous_{tag}.csv',index=False)
if __name__=='__main__':main()



