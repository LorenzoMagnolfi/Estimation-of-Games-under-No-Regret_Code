"""Persistent heterogeneous-cost LP with second-order policy regret.

Finite dollar cost grid and fractional listing assignments: exploratory outer
relaxation of assignments, not continuous-cost confidence bounds. Includes
cost-aware markup policies and reference-price policies in a single learner.
"""
from __future__ import annotations
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack, eye
from scipy.optimize import linprog
from quality_demand import data, QualityDemand
from demand_pilot import Demand
from second_order_diagnostic import max_profit
from audit_panel import OUT


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seller',default='UPGRADE SOLUTION INC'); ap.add_argument('--cost-step',type=float,default=25); args=ap.parse_args()
    d=data(); s=d[d.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True)
    fit=joblib.load(OUT/'quality_demand_quality.joblib'); N=len(s); codes,idx=np.unique(s.Code,return_inverse=True); L=len(codes)
    ref=s.Ref_Price.to_numpy(); price=s.Price.to_numpy(); qa=fit.predict(s)
    assert price.max()<=2000
    lo=np.maximum(1,np.rint(ref)-250); hi=np.minimum(2000,np.rint(ref)+250)
    ef=fit.effects(); b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100; gp=ef['gap_positive']/100; gn=ef['gap_negative']/100; rival=s.rival_min.to_numpy()
    offset=fit.logit(s)-b*price-gp*np.maximum(price-rival,0)-gn*np.minimum(price-rival,0)
    def q(p):return expit(offset[:,None]+b[:,None]*p+gp*np.maximum(p-rival[:,None],0)+gn*np.minimum(p-rival[:,None],0))
    refP=np.clip(np.rint(ref[:,None])+np.arange(-250,251,50)[None,:],lo[:,None],hi[:,None]); refQ=q(refP)
    marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]); Atest=refP.shape[1]+len(marks)+1; Aall=502+len(marks); J=6
    costs=np.arange(0,1500+args.cost_step/2,args.cost_step); G=len(costs)
    R=np.empty((Atest,G*L)); V=np.empty_like(R); K=np.zeros(Atest)
    mzero=max_profit(s,fit,np.zeros(N)); mtop=max_profit(s,fit,np.full(N,1500.))
    # Reference policies are affine in cost. The supremum over costs of the
    # maximum payoff difference is convex, so endpoints suffice for K_a.
    for c,mp in [(0.,mzero),(1500.,mtop)]:
        K[:refP.shape[1]]=np.maximum(K[:refP.shape[1]],(mp[:,None]-refQ*(refP-c)).max(0))
    K[refP.shape[1]:]=mzero.max() # Markup and no-offer policies have nonnegative payoff.
    for k,c in enumerate(costs):
        p=np.clip(c+marks[None,:],lo[:,None],hi[:,None]); qm=q(p); qm=np.where(c<=hi[:,None],qm,0.)
        U=np.column_stack([refQ*(refP-c),qm*(p-c),np.zeros(N)])
        r=U-qa[:,None]*(price[:,None]-c)
        rs=np.zeros((L,Atest)); vs=np.zeros_like(rs); np.add.at(rs,idx,r);np.add.at(vs,idx,r*r)
        R[:,k*L:(k+1)*L]=rs.T/N; V[:,k*L:(k+1)*L]=vs.T/N
    eq=hstack([eye(L,format='csr')]*G,format='csr'); sold=s.groupby('Code').sale_last_seen.max().to_numpy(); w=sold/sold.sum(); minprice=s.groupby('Code').Price.min().to_numpy(); lastprice=s.groupby('Code').Price.last().to_numpy(); meanprice=float(w@lastprice)
    objective=(costs[:,None]*w[None,:]).ravel(); records=[]
    for support in ['observed','universal']:
      KK=K if support=='observed' else np.r_[np.full(refP.shape[1],3500.),np.full(len(marks)+1,2000.)]
      eta=2.**(-np.arange(J))[None,:]/(2*np.maximum(KK,1e-6)[:,None])
      # All J predeclared rate copies impose a restriction.
      rows=csr_matrix((R[:,None,:]-eta[:,:,None]*V[:,None,:]).reshape(Atest*J,G*L))
      for copies,warm in [(1,0),(1,5),(10,0)]:
        logterm=np.log(Aall*J)+warm*np.log(Aall)+np.log(copies/.05)
        rhs=(copies*logterm/(N*eta)).ravel()
        for ir in [False,True]:
          allowed=np.ones((G,L),bool) if not ir else costs[:,None]<=minprice[None,:]
          bounds=np.column_stack([np.zeros(G*L),allowed.ravel().astype(float)])
          for direction,sign in [('low',1.),('high',-1.)]:
            start=time.time(); sol=linprog(sign*objective,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=np.ones(L),bounds=bounds,method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
            cost=float(sign*sol.fun) if sol.success else None
            rec={'seller':args.seller,'cost_step':args.cost_step,'N':N,'L':L,'support_mode':support,'copies':copies,'warm_logA':warm,'no_below_cost':ir,'direction':direction,'success':sol.success,'status':sol.message,'sold_cost_bound':cost,'sold_markup_bound':meanprice-cost if cost is not None else None,'sold_mean_price':meanprice,'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rhs).max()) if sol.success else None,'max_equality_violation':float(np.abs(eq@sol.x-1).max()) if sol.success else None}
            records.append(rec); print(json.dumps(rec,default=float),flush=True)
            safe=''.join(x if x.isalnum() else '_' for x in args.seller)
            pd.DataFrame(records).to_csv(OUT/f'second_order_heterogeneous_{safe}.csv',index=False)
    print('HETEROGENEOUS_COMPLETE',flush=True)
if __name__=='__main__':main()
