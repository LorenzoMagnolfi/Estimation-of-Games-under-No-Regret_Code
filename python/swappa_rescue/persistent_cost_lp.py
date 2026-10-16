"""Persistent listing-cost outer bounds under budgeted contextual regret.

Empirical cost distributions are unrestricted. No iid composition restriction or
finite-sample coverage claim is made. The LP allows fractional latent assignments,
so its projections are conservative relaxations of deterministic listing costs.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, hstack, vstack
from demand_pilot import Demand
from audit_panel import OUT

def prepare(s,fit,actions,costs):
    s=s.sort_values(['Code','date']).reset_index(drop=True)
    codes,idx=np.unique(s.Code,return_inverse=True)
    L=len(codes); N=len(s); K=len(costs); A=len(actions)
    first=s.groupby('Code',sort=True).first()
    ref0=first.Ref_Price.to_numpy()
    ref=s.Ref_Price.to_numpy()
    qa=fit.predict(s)
    q=np.column_stack([fit.predict(s,a) for a in actions])
    # A comparison action is a reference-price residual clipped at $1.
    # Re-evaluate demand at clipped residuals when needed.
    prices=np.maximum(1,ref[:,None]+actions)
    for a in range(A):
        clip=ref+actions[a]<=0
        if clip.any():
            sc=s.loc[clip].copy(); sc['residual']=1-sc.Ref_Price
            q[clip,a]=fit.predict(sc)
    diffq=q-qa[:,None]
    diffr=q*(prices-ref0[idx,None])-qa[:,None]*(s.Price.to_numpy()[:,None]-ref0[idx,None])
    B=np.zeros((L,A)); C=np.zeros((L,A))
    np.add.at(B,idx,diffq); np.add.at(C,idx,diffr)
    blocks=[csr_matrix((C-c*B).T/N) for c in costs]
    regret=block_diag(blocks,format='csr')
    # Variables x are cost-major: x[k,l]. e[k] is dollars per observed listing-day.
    epsmap=block_diag([np.ones((A,1)) for _ in costs],format='csr')
    Aub=hstack([regret,-epsmap],format='csr')
    Aeq=hstack([csr_matrix(np.tile(np.eye(L),(1,K))),csr_matrix((L,K))],format='csr')
    bnds=[(0,1) if ref0[l]+costs[k]>=0 else (0,0) for k in range(K) for l in range(L)]+[(0,None)]*K
    summary={'N':N,'L':L,'K':K,'A':A,'mean_entry_reference':float(ref0.mean()),'mean_last_price':float(s.groupby('Code').last().Price.mean())}
    return Aub,Aeq,bnds,summary,ref0

def solve_model(s,fit,actions,costs,seller):
    Aub,Aeq,bnds,summary,ref0=prepare(s,fit,actions,costs)
    L=summary['L']; K=summary['K']
    zero=np.zeros(Aub.shape[0]); ones=np.ones(L)
    obj=np.r_[np.zeros(L*K),np.ones(K)]
    start=time.time()
    minimum=linprog(obj,A_ub=Aub,b_ub=zero,A_eq=Aeq,b_eq=ones,bounds=bnds,method='highs',options={'time_limit':300})
    print(json.dumps({'seller':seller,**summary,'minimum_status':minimum.message,'minimum_total_regret':minimum.fun if minimum.success else None,'seconds':time.time()-start}),flush=True)
    if not minimum.success:return []
    budgetrow=csr_matrix(np.r_[np.zeros(L*K),np.ones(K)][None,:])
    constrained=vstack([Aub,budgetrow],format='csr')
    objective=np.r_[np.repeat(costs,L)/L,np.zeros(K)]
    results=[]
    for budget in [0,.1,.25,.5,1,2]:
      row={'seller':seller,**summary,'epsilon':budget,'minimum_total_regret':minimum.fun}
      for sign,label in [(1,'low'),(-1,'high')]:
        result=linprog(sign*objective,A_ub=constrained,b_ub=np.r_[zero,budget],A_eq=Aeq,b_eq=ones,bounds=bnds,method='highs',options={'time_limit':300})
        row[label+'_status']=result.message
        row['mean_residual_'+label]=float(sign*result.fun) if result.success else None
        row['mean_cost_'+label]=float(ref0.mean()+sign*result.fun) if result.success else None
        if result.success:
          row[label+'_max_constraint_violation']=float(max(0,(constrained@result.x-np.r_[zero,budget]).max()))
          row[label+'_max_matching_error']=float(np.abs(Aeq@result.x-ones).max())
          mass=result.x[:K*L].reshape(K,L).sum(axis=1)/L
          row[label+'_boundary_mass']=float(mass[0]+mass[-1])
      results.append(row)
      print(json.dumps(row),flush=True)
      pd.DataFrame(results).to_csv(OUT/('persistent_cost_'+seller.replace(' ','_').replace('.','')+'.csv'),index=False)
    return results

def main():
    p=argparse.ArgumentParser(); p.add_argument('--outcome',default='sale_last_seen'); p.add_argument('--spec',default='rich'); args=p.parse_args()
    d=pd.read_parquet(OUT/'panel.parquet')
    d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy()
    d=d[~(d.Real_Sold.eq(1)&d.date.gt(d.Updated))].copy()
    top=d.groupby('Seller').Code.nunique().nlargest(20).index
    d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    fit=Demand(args.spec).fit(d,d[args.outcome])
    print(json.dumps({'outcome':args.outcome,'spec':args.spec,'coefficients':fit.effects()}),flush=True)
    # Price residual grid is separate from the cost grid.
    actions=np.arange(-200,201,20,dtype=float)+.01
    costs=np.arange(-200,101,20,dtype=float)
    for seller in ['AMS Traders Inc.','GadgetPickup']:
        solve_model(d[d.Seller.eq(seller)],fit,actions,costs,seller)
if __name__=='__main__':main()
