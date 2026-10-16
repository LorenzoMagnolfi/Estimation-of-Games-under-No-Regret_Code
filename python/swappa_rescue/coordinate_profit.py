"""Total own-inventory profit bounds for coordinate price deviations.

Known-demand pilot. Cost probability caps hold even at the highest allowed
peer prices. Positive peer-price effects and persistent peer costs are bounded,
not discarded. Counterfactuals keep inventory fixed; no delisting action.
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from economic_envelope import profit_cap
from audit_panel import OUT


def peer_spillover_bounds(price_lo,price_hi,price,log_odds,n_own,groups,
                          margin_lo,margin_hi,peer_slope,unit_profit_cap):
    """Outer bounds on the change in profit of all other owned listings.

    Missing risk-set peers remain in the observed inventory count. Their
    effects are bounded by the uniform profit cap, not assumed to be zero.
    """
    lower=np.zeros_like(price_lo);upper=np.zeros_like(price_hi)
    for ids in groups:
        n=int(n_own[ids[0]]);m=len(ids)
        if n<=1:continue
        assert np.all(n_own[ids]==n) and n>=m
        sl=peer_slope*(price_lo[ids]-price[ids,None])/(n-1)
        sh=peer_slope*(price_hi[ids]-price[ids,None])/(n-1)
        qa=expit(log_odds[ids])[None,None,:]
        dl=expit(log_odds[ids][None,None,:]+sl[:,:,None])-qa
        dh=expit(log_odds[ids][None,None,:]+sh[:,:,None])-qa
        mask=1-np.eye(m);wl=mask*margin_lo[ids][None,:];wh=mask*margin_hi[ids][None,:]
        lower[ids]=np.where(dl>=0,dl*wl[:,None,:],dl*wh[:,None,:]).sum(axis=2)
        upper[ids]=np.where(dh>=0,dh*wh[:,None,:],dh*wl[:,None,:]).sum(axis=2)
        missing=n-m
        if missing:
            lower[ids]-=missing*unit_profit_cap*(-np.expm1(np.minimum(sl,0)))
            upper[ids]+=missing*unit_profit_cap*(-np.expm1(-np.maximum(sh,0)))
    return lower,upper


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=30);ap.add_argument('--peer-bound',type=float,default=.003);args=ap.parse_args()
    seller='UPGRADE SOLUTION INC';d=data();s=d[d.Seller.eq(seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_demand.joblib')
    codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;last=s.groupby('Code',sort=True).last();p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);actual_logit=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values())
    ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100
    offset=actual_logit-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    k0=.03;Pmax=650.;Cmax=400.;B=args.peer_bound
    assert 0<=bp<=B and ref.min()>=100 and ref.max()<=400 and p.max()<=Pmax
    assert min(-(ef['own']+ef['price_ref_interaction']*np.log(400/500)+g)/100 for g in [ef['gap_positive'],ef['gap_negative']])>=k0
    peer=np.divide(s.own_total.to_numpy()-p,n-1,out=np.zeros(N),where=n>1)
    worst_offset=offset+np.where(n>1,bp*(Pmax-peer),0.)
    kmax=np.maximum(-(b+gp),-(b+gn));marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Aref=11;A=Aref+len(marks);Aall=501+len(marks);J=8;alpha=.005
    def prob(price):return expit(offset[:,None]+b[:,None]*price+gp*np.maximum(price-rival[:,None],0)+gn*np.minimum(price-rival[:,None],0))
    baseprice=np.maximum(1,np.rint(ref[:,None])+np.arange(-250,251,50)[None,:]);baseq=prob(baseprice)
    upper=np.minimum(Cmax,s.groupby('Code',sort=True).Price.min().to_numpy());sold=last.sale_last_seen.eq(1).to_numpy();w=sold/sold.sum();meanprice=float(w@last.Price.to_numpy());eq=hstack([eye(L,format='csr')]*G,format='csr');records=[]
    for qbar in [.75,.9,.99]:
        lq=logit(qbar);cb=(lq-worst_offset+gn*rival)/(b+gn);ca=(lq-worst_offset+gp*rival)/(b+gp);cmin=np.maximum(0,np.where(cb<=rival,cb,ca));lower=np.zeros(L);np.maximum.at(lower,idx,cmin)
        bad=lower>upper+1e-9;Kunit=profit_cap(qbar,k0);Ktotal=Kunit*(1+B*Pmax)
        if bad.any():
            records.append({'seller':seller,'qbar':qbar,'k0':k0,'K_unit':Kunit,'K_total':Ktotal,'status':'incompatible_cost_domain','incompatible_listings':int(bad.sum())});print(json.dumps(records[-1]),flush=True);continue
        edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];centers=(edges[:-1]+edges[1:])/2;halves=(edges[1:]-edges[:-1])/2
        marginlo=p-upper[idx];marginhi=p-lower[idx];Rlo=np.empty((A,G*L));Vhi=np.empty_like(Rlo)
        for g in range(G):
            c=centers[g,idx];h=halves[g,idx];cl=edges[g,idx];ch=edges[g+1,idx]
            uref=baseq*np.maximum(baseprice-c[:,None],0.)
            dactive=np.abs(qa[:,None]-baseq);dinactive=np.broadcast_to(qa[:,None],baseq.shape)
            dr=np.where(baseprice>=ch[:,None],dactive,np.where(baseprice<=cl[:,None],dinactive,np.maximum(dactive,dinactive)))
            pm=c[:,None]+marks[None,:];qm=prob(pm);um=qm*marks[None,:]
            qlo=prob(cl[:,None]+marks[None,:]);qhi=prob(ch[:,None]+marks[None,:]);vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
            dm=np.maximum(qa[:,None],np.abs(qa[:,None]-marks[None,:]*kmax[:,None]*vf))
            u=np.column_stack([uref,um]);r=u-qa[:,None]*(p[:,None]-c[:,None]);error=h[:,None]*np.column_stack([dr,dm])
            plo=np.column_stack([np.maximum(baseprice,cl[:,None]),cl[:,None]+marks[None,:]])
            phi=np.column_stack([np.maximum(baseprice,ch[:,None]),ch[:,None]+marks[None,:]])
            assert phi.max()<=Pmax
            spl,sph=peer_spillover_bounds(plo,phi,p,actual_logit,n,groups,marginlo,marginhi,bp,Kunit)
            total_lo=r-error+spl;total_hi=r+error+sph;vhigh=np.maximum(total_lo**2,total_hi**2)
            rr=np.zeros((L,A));vv=np.zeros_like(rr);np.add.at(rr,idx,total_lo);np.add.at(vv,idx,vhigh);Rlo[:,g*L:(g+1)*L]=rr.T/N;Vhi[:,g*L:(g+1)*L]=vv.T/N
        eta=2.**(-np.arange(J))/(2*Ktotal);rows=csr_matrix((Rlo[:,None,:]-eta[None,:,None]*Vhi[:,None,:]).reshape(A*J,G*L));priorlo=float(w@lower);priorhi=float(w@upper)
        for copies in [1,10,100]:
          for warm in [0,5]:
            logterm=np.log(Aall*J)+warm*np.log(Aall)+np.log(copies/alpha);rhs=np.tile(copies*logterm/(N*eta),A)
            for direction,sign,values in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
                obj=(values*w[None,:]).ravel();start=time.time();sol=linprog(sign*obj,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8});cost=float(sign*sol.fun) if sol.success else None
                rec={'seller':seller,'qbar':qbar,'k0':k0,'peer_coefficient':bp,'peer_coefficient_bound':B,'price_support_high':Pmax,'K_unit':Kunit,'K_total':Ktotal,'cells':G,'N':N,'L':L,'alpha_R':alpha,'copies':copies,'warm_logA':warm,'direction':direction,'success':sol.success,'status':sol.message,'sold_cost_bound':cost,'sold_markup_bound':meanprice-cost if cost is not None else None,'prior_cost_low':priorlo,'prior_cost_high':priorhi,'sold_mean_price':meanprice,'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rhs).max()) if sol.success else None,'max_equality_violation':float(np.abs(eq@sol.x-1).max()) if sol.success else None}
                records.append(rec);print(json.dumps(rec),flush=True);pd.DataFrame(records).to_csv(OUT/f'coordinate_profit_upgrade_c{G}_b{B}.csv',index=False)
    print('COORDINATE_PROFIT_COMPLETE',flush=True)
if __name__=='__main__':main()
