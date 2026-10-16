"""Continuous persistent-cost outer bounds with an economic payoff envelope.

Assume no below-cost posting, q(c)<=qbar, and -d logit(q)/dp >= k0 on
all model states. Then every nonnegative profit is at most
W(qbar/[e(1-qbar)])/k0. The bound is a primitive restriction, not an
observed maximum. First-stage uncertainty is not included in this pilot.
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit,lambertw
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from quality_demand import data,QualityDemand
from audit_panel import OUT


def profit_cap(qbar,k0):return float(lambertw(qbar/(np.e*(1-qbar))).real/k0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=30);args=ap.parse_args()
    seller='UPGRADE SOLUTION INC';d=data();s=d[d.Seller.eq(seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'quality_demand_quality.joblib')
    codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;first=s.groupby('Code',sort=True).first();last=s.groupby('Code',sort=True).last();p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s)
    ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100
    offset=fit.logit(s)-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    k0=.03;assert ref.min()>=100 and ref.max()<=400
    assert min(-(ef['own']+ef['price_ref_interaction']*np.log(400/500)+g)/100 for g in [ef['gap_positive'],ef['gap_negative']])>=k0
    kmax=np.maximum(-(b+gp),-(b+gn));marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Aref=11;A=Aref+len(marks)+1;Aall=502+len(marks);J=8;alpha=.005
    def prob(price):return expit(offset[:,None]+b[:,None]*price+gp*np.maximum(price-rival[:,None],0)+gn*np.minimum(price-rival[:,None],0))
    baseprice=np.maximum(1,np.rint(ref[:,None])+np.arange(-250,251,50)[None,:]);baseq=prob(baseprice)
    upper=s.groupby('Code',sort=True).Price.min().to_numpy();sold=last.sale_last_seen.eq(1).to_numpy();w=sold/sold.sum();meanprice=float(w@last.Price.to_numpy());eq=hstack([eye(L,format='csr')]*G,format='csr');records=[]
    for qbar in [.5,.75,.9,.99]:
        lq=logit(qbar);cb=(lq-offset+gn*rival)/(b+gn);ca=(lq-offset+gp*rival)/(b+gp);cmin=np.maximum(0,np.where(cb<=rival,cb,ca));lower=np.zeros(L);np.maximum.at(lower,idx,cmin)
        bad=lower>upper+1e-9;K=profit_cap(qbar,k0)
        if bad.any():
            records.append({'seller':seller,'qbar':qbar,'k0':k0,'K':K,'status':'incompatible_cost_domain','incompatible_listings':int(bad.sum())});print(json.dumps(records[-1]),flush=True);continue
        edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];centers=(edges[:-1]+edges[1:])/2;halves=(edges[1:]-edges[:-1])/2
        Rlo=np.empty((A,G*L));Vhi=np.empty_like(Rlo)
        for g in range(G):
            c=centers[g,idx];h=halves[g,idx];cl=edges[g,idx];ch=edges[g+1,idx]
            # Reference policy clips the price at cost; below-cost proposals
            # yield zero profit. This preserves observed prices in the domain.
            uref=baseq*np.maximum(baseprice-c[:,None],0.)
            urlo=baseq*np.maximum(baseprice-cl[:,None],0.);urhi=baseq*np.maximum(baseprice-ch[:,None],0.)
            # Safe derivative bounds, narrowed when a clipping kink cannot occur.
            dactive=np.abs(qa[:,None]-baseq);dinactive=np.broadcast_to(qa[:,None],baseq.shape)
            dr=np.where(baseprice>=ch[:,None],dactive,np.where(baseprice<=cl[:,None],dinactive,np.maximum(dactive,dinactive)))
            pm=c[:,None]+marks[None,:];qm=prob(pm);um=qm*marks[None,:]
            qlo=prob(cl[:,None]+marks[None,:]);qhi=prob(ch[:,None]+marks[None,:]);vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
            dm=np.maximum(qa[:,None],np.abs(qa[:,None]-marks[None,:]*kmax[:,None]*vf))
            u=np.column_stack([uref,um,np.zeros(N)]);r=u-qa[:,None]*(p[:,None]-c[:,None]);deriv=np.column_stack([dr,dm,qa])
            error=h[:,None]*deriv;rlow=r-error;vhigh=(np.abs(r)+error)**2
            rr=np.zeros((L,A));vv=np.zeros_like(rr);np.add.at(rr,idx,rlow);np.add.at(vv,idx,vhigh);Rlo[:,g*L:(g+1)*L]=rr.T/N;Vhi[:,g*L:(g+1)*L]=vv.T/N
        eta=2.**(-np.arange(J))/(2*K);rows=csr_matrix((Rlo[:,None,:]-eta[None,:,None]*Vhi[:,None,:]).reshape(A*J,G*L));priorlo=float(w@lower);priorhi=float(w@upper)
        for copies in [1,10,100]:
          for warm in [0,5]:
            logterm=np.log(Aall*J)+warm*np.log(Aall)+np.log(copies/alpha);rhs=np.tile(copies*logterm/(N*eta),A)
            for direction,sign,values in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
                obj=(values*w[None,:]).ravel();start=time.time();sol=linprog(sign*obj,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8});cost=float(sign*sol.fun) if sol.success else None
                rec={'seller':seller,'qbar':qbar,'k0':k0,'K':K,'cells':G,'N':N,'L':L,'alpha_R':alpha,'copies':copies,'warm_logA':warm,'direction':direction,'success':sol.success,'status':sol.message,'sold_cost_bound':cost,'sold_markup_bound':meanprice-cost if cost is not None else None,'prior_cost_low':priorlo,'prior_cost_high':priorhi,'sold_mean_price':meanprice,'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rhs).max()) if sol.success else None,'max_equality_violation':float(np.abs(eq@sol.x-1).max()) if sol.success else None}
                records.append(rec);print(json.dumps(rec),flush=True)
                pd.DataFrame(records).to_csv(OUT/f'economic_envelope_upgrade_c{G}.csv',index=False)
    print('ECONOMIC_ENVELOPE_COMPLETE',flush=True)
if __name__=='__main__':main()
