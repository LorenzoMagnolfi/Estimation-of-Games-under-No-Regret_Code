"""Entry-cohort cost-distribution consistency with persistent listing costs.

Exploratory finite-ratio-grid outer relaxation. DKW allowances are separate from
regret allowances. Conditional iid entry cost ratios and exogeneity of entry
reference/rival states are maintained assumptions, not established data facts.
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix,hstack,vstack,eye
from scipy.optimize import linprog
from quality_demand import data,QualityDemand
from second_order_diagnostic import max_profit
from audit_panel import OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seller',default='UPGRADE SOLUTION INC');args=ap.parse_args()
    d=data();s=d[d.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'quality_demand_quality.joblib')
    codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);first=s.groupby('Code',sort=True).first();last=s.groupby('Code',sort=True).last()
    anchor=first.Ref_Price.to_numpy();ref=s.Ref_Price.to_numpy();price=s.Price.to_numpy();qa=fit.predict(s)
    ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;rival=s.rival_min.to_numpy()
    offset=fit.logit(s)-b*price-gp*np.maximum(price-rival,0)-gn*np.minimum(price-rival,0)
    ratios=np.arange(0,1.5001,.05);G=len(ratios);costs=ratios[:,None]*anchor[None,:]
    marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);A=len(marks)+1;Aall=502+len(marks);J=6
    lo=np.maximum(1,np.rint(ref)-250);hi=np.minimum(2000,np.rint(ref)+250)
    R=np.empty((A,G*L));V=np.empty_like(R)
    for k in range(G):
        c=costs[k,idx];p=np.clip(c[:,None]+marks[None,:],lo[:,None],hi[:,None]);q=expit(offset[:,None]+b[:,None]*p+gp*np.maximum(p-rival[:,None],0)+gn*np.minimum(p-rival[:,None],0));q=np.where(c[:,None]<=hi[:,None],q,0.)
        r=np.column_stack([q*(p-c[:,None]),np.zeros(N)])-qa[:,None]*(price[:,None]-c[:,None]);rs=np.zeros((L,A));vs=np.zeros_like(rs);np.add.at(rs,idx,r);np.add.at(vs,idx,r*r)
        R[:,k*L:(k+1)*L]=rs.T/N;V[:,k*L:(k+1)*L]=vs.T/N
    # Initial listings are left-truncated. Keep them in regret calculations,
    # but impose entry-composition restrictions only on later first appearances.
    entrants=first.date.gt(d.date.min()+pd.Timedelta(days=7)).to_numpy();Ne=int(entrants.sum());groups=[];descriptions=[]
    for name,values in [('reference',first.Ref_Price),('rival_residual',first.rival_min_residual)]:
        bins=pd.qcut(values[entrants],3,labels=False,duplicates='drop')
        for k in sorted(bins.unique()):
            mask=np.zeros(L,bool);mask[np.flatnonzero(entrants)[bins.to_numpy()==k]]=True;groups.append(mask);descriptions.append(name+'_'+str(int(k)))
    alphaC=.005;alphaR=.005
    eall=np.sqrt(np.log(2*(len(groups)+1)/alphaC)/(2*Ne));cdfrows=[];cdfrhs=[];audit=[]
    for name,g in zip(descriptions,groups):
        Ng=int(g.sum());err=eall+np.sqrt(np.log(2*(len(groups)+1)/alphaC)/(2*Ng));a=g.astype(float)/Ng-entrants.astype(float)/Ne
        for threshold in range(G-1):
            row=np.zeros((G,L));row[:threshold+1]=a;rr=csr_matrix(row.reshape(1,-1));cdfrows.extend([rr,-rr]);cdfrhs.extend([err,err])
        audit.append({'group':name,'N':Ng,'cdf_discrepancy_allowance':err})
    C=vstack(cdfrows,format='csr');crhs=np.array(cdfrhs)
    eq=hstack([eye(L,format='csr')]*G,format='csr');sold=last.sale_last_seen.eq(1).to_numpy();w=sold/sold.sum();objective=(costs*w[None,:]).ravel();meanprice=float(w@last.Price.to_numpy());minprice=s.groupby('Code').Price.min().to_numpy()
    records=[];Kobserved=max_profit(s,fit,np.zeros(N)).max()
    print(json.dumps({'seller':args.seller,'entrants':Ne,'alpha_C':alphaC,'alpha_R':alphaR,'groups':audit,'K_observed':Kobserved,'ratio_grid_step':.05,'interpretation':'Finite ratio support; observed-support mode is not a certified regret bound.'}),flush=True)
    for support,K in [('observed',Kobserved),('universal',2000.)]:
      eta=2.**(-np.arange(J))/(2*K);regrows=csr_matrix((R[:,None,:]-eta[None,:,None]*V[:,None,:]).reshape(A*J,G*L))
      for warm in [0,5]:
        rhs=np.tile((np.log(Aall*J)+warm*np.log(Aall)+np.log(1/alphaR))/(N*eta),A)
        for composition in [False,True]:
          rows=vstack([regrows,C],format='csr') if composition else regrows;rrhs=np.r_[rhs,crhs] if composition else rhs
          for ir in [False,True]:
            allowed=(costs<=1500)&((costs<=minprice[None,:]) if ir else True);bounds=np.column_stack([np.zeros(G*L),allowed.ravel().astype(float)])
            for direction,sign in [('low',1.),('high',-1.)]:
              start=time.time();sol=linprog(sign*objective,A_ub=rows,b_ub=rrhs,A_eq=eq,b_eq=np.ones(L),bounds=bounds,method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8});c=float(sign*sol.fun) if sol.success else None
              rec={'seller':args.seller,'N':N,'L':L,'entrants':Ne,'support':support,'warm_logA':warm,'composition':composition,'no_below_cost':ir,'direction':direction,'success':sol.success,'message':sol.message,'sold_cost_bound':c,'sold_markup_bound':meanprice-c if c is not None else None,'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rrhs).max()) if sol.success else None}
              records.append(rec);print(json.dumps(rec),flush=True);safe=''.join(x if x.isalnum() else '_' for x in args.seller);pd.DataFrame(records).to_csv(OUT/f'composition_pilot_{safe}.csv',index=False)
    print('COMPOSITION_PILOT_COMPLETE',flush=True)
if __name__=='__main__':main()
