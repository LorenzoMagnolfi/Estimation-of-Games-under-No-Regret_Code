"""Combined valid peer bounds, sale-timing and price-support sensitivity.

Any-window shared-Prod envelope with infinitely many learning rates.

Tests finitely many necessary rate restrictions from a countably infinite
learner; the prior on rate j is 1/[(j+1)(j+2)]. The bound includes a known
upper limit on the lifetime decision clock, avoiding a sample-start reset.

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


from coordinate_profit import peer_spillover_bounds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=30);ap.add_argument('--peer-bound',type=float,default=.003);ap.add_argument('--price-cap',type=float,default=650.);ap.add_argument('--k0',type=float,default=.03);ap.add_argument('--outcome',default='sale_last_seen',choices=['sale_last_seen','sale_forward','sale_legacy','sale_terminal_status']);args=ap.parse_args()
    from demand_timing_sensitivity import terminal_status_data
    seller='UPGRADE SOLUTION INC';d=terminal_status_data() if args.outcome=='sale_terminal_status' else data();s=d[d.Seller.eq(seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/('multiproduct_demand.joblib' if args.outcome=='sale_last_seen' else f'multiproduct_{args.outcome}.joblib'))
    codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;last=s.groupby('Code',sort=True).last();p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);actual_logit=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values())
    ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100
    offset=actual_logit-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    k0=args.k0;Pmax=args.price_cap;Cmax=min(400.,Pmax);B=args.peer_bound
    assert 0<=bp<=B and ref.min()>=100 and ref.max()<=400 and p.max()<=Pmax
    assert min(-(ef['own']+ef['price_ref_interaction']*np.log(400/500)+g)/100 for g in [ef['gap_positive'],ef['gap_negative']])>=k0
    peer=np.divide(s.own_total.to_numpy()-p,n-1,out=np.zeros(N),where=n>1)
    worst_offset=offset+np.where(n>1,bp*(Pmax-peer),0.)
    kmax=np.maximum(-(b+gp),-(b+gn));marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Aref=11;A=Aref+len(marks);Aall=501+len(marks);J=8;alpha=.005
    def prob(price):return expit(offset[:,None]+b[:,None]*price+gp*np.maximum(price-rival[:,None],0)+gn*np.minimum(price-rival[:,None],0))
    baseprice=np.clip(np.rint(ref[:,None])+np.arange(-250,251,50)[None,:],1,Pmax);baseq=prob(baseprice)
    upper=np.minimum(Cmax,s.groupby('Code',sort=True).Price.min().to_numpy());events=s[args.outcome].to_numpy();sold=np.bincount(idx,weights=events,minlength=L);w=sold/sold.sum();meanprice=float(events@p/events.sum());eq=hstack([eye(L,format='csr')]*G,format='csr');records=[]
    qweight=np.bincount(idx,weights=qa,minlength=L)/qa.sum()
    targets=[('sold',w,meanprice),('expected_sales',qweight,float(qa@p/qa.sum()))]
    for qbar in [.75,.9,.99]:
        lq=logit(qbar);cb=(lq-worst_offset+gn*rival)/(b+gn);ca=(lq-worst_offset+gp*rival)/(b+gp);cmin=np.maximum(0,np.where(cb<=rival,cb,ca));lower=np.zeros(L);np.maximum.at(lower,idx,cmin)
        bad=lower>upper+1e-9;Kunit=profit_cap(qbar,k0);Ktotal=Kunit*(1+B*Pmax)
        if bad.any():
            records.append({'seller':seller,'qbar':qbar,'k0':k0,'K_unit':Kunit,'K_total':Ktotal,'status':'incompatible_cost_domain','incompatible_listings':int(bad.sum())});print(json.dumps(records[-1]),flush=True);continue
        edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];centers=(edges[:-1]+edges[1:])/2;halves=(edges[1:]-edges[:-1])/2
        marginlo=p-upper[idx];marginhi=p-lower[idx];Rlo=np.empty((A,G*L));Vhi=np.empty_like(Rlo);Runlinked=np.empty_like(Rlo)
        # A global lower price for each policy gives a lower peer probability
        # for every possible focal cost. Crucially, peer profit then uses the
        # SAME peer cost assignment that appears in the cost objective.
        global_pmin=np.column_stack([np.maximum(baseprice,lower[idx,None]),np.minimum(Pmax,lower[idx,None]+marks[None,:])])
        peer_B=np.zeros((L,A));peer_P=np.zeros_like(peer_B);missing_lower=np.zeros(A)
        for ids in groups:
            ni=int(n[ids[0]]);m=len(ids)
            if ni<=1:continue
            shift=bp*(global_pmin[ids]-p[ids,None])/(ni-1)
            dq=expit(actual_logit[ids][None,None,:]+shift[:,:,None])-qa[ids][None,None,:]
            dq*= (1-np.eye(m))[:,None,:]
            per_peer=dq.sum(axis=0).T
            np.add.at(peer_B,idx[ids],per_peer)
            np.add.at(peer_P,idx[ids],per_peer*p[ids,None])
            missing_lower-=(ni-m)*Kunit*(-np.expm1(np.minimum(shift,0))).sum(axis=0)
        missing_lower/=N
        for g in range(G):
            c=centers[g,idx];h=halves[g,idx];cl=edges[g,idx];ch=edges[g+1,idx]
            uref=baseq*np.maximum(baseprice-c[:,None],0.)
            dactive=np.abs(qa[:,None]-baseq);dinactive=np.broadcast_to(qa[:,None],baseq.shape)
            dr=np.where(baseprice>=ch[:,None],dactive,np.where(baseprice<=cl[:,None],dinactive,np.maximum(dactive,dinactive)))
            pm=np.minimum(Pmax,c[:,None]+marks[None,:]);qm=prob(pm);um=qm*(pm-c[:,None])
            qlo=prob(np.minimum(Pmax,cl[:,None]+marks[None,:]));qhi=prob(np.minimum(Pmax,ch[:,None]+marks[None,:]));vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
            dm=np.maximum(qa[:,None],np.abs(qa[:,None]-marks[None,:]*kmax[:,None]*vf))
            capderiv=np.abs(qa[:,None]-prob(np.full((N,1),Pmax)))
            dm=np.where(cl[:,None]+marks[None,:]>=Pmax,capderiv,np.where(ch[:,None]+marks[None,:]<=Pmax,dm,np.maximum(dm,capderiv)))
            u=np.column_stack([uref,um]);r=u-qa[:,None]*(p[:,None]-c[:,None]);error=h[:,None]*np.column_stack([dr,dm])
            plo=np.column_stack([np.maximum(baseprice,cl[:,None]),np.minimum(Pmax,cl[:,None]+marks[None,:])])
            phi=np.column_stack([np.maximum(baseprice,ch[:,None]),np.minimum(Pmax,ch[:,None]+marks[None,:])])
            assert phi.max()<=Pmax
            spl,sph=peer_spillover_bounds(plo,phi,p,actual_logit,n,groups,marginlo,marginhi,bp,Kunit)
            total_lo=r-error+spl;total_hi=r+error+sph;vhigh=np.maximum(total_lo**2,total_hi**2)
            ru=np.zeros((L,A));np.add.at(ru,idx,total_lo);Runlinked[:,g*L:(g+1)*L]=ru.T/N
            rr=np.zeros((L,A));vv=np.zeros_like(rr);np.add.at(rr,idx,r-error);np.add.at(vv,idx,vhigh)
            peer_cost_endpoint=np.where(peer_B>=0,edges[g+1,:,None],edges[g,:,None])
            rr+=peer_P-peer_B*peer_cost_endpoint
            Rlo[:,g*L:(g+1)*L]=rr.T/N;Vhi[:,g*L:(g+1)*L]=vv.T/N
        eta=2.**(-np.arange(J))/(2*Ktotal);rows=csr_matrix(np.concatenate([(Rlo[:,None,:]-eta[None,:,None]*Vhi[:,None,:]).reshape(A*J,G*L),(Runlinked[:,None,:]-eta[None,:,None]*Vhi[:,None,:]).reshape(A*J,G*L)],axis=0));priorlo=float(w@lower);priorhi=float(w@upper)
        for copies in [1,10]:
          for lifetime in [1000000]:
            rate_prior=1./((np.arange(J)+1)*(np.arange(J)+2))
            logterm=np.log(Aall/rate_prior)+np.log(lifetime)+np.log(copies/alpha)
            rhs0=np.tile(copies*logterm/(N*eta),A)
            rhs=np.r_[rhs0-np.repeat(missing_lower,J),rhs0]
            for estimand,target_weight,target_price in targets:
              for direction,sign,values in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
                obj=(values*target_weight[None,:]).ravel();start=time.time();sol=linprog(sign*obj,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8});cost=float(sign*sol.fun) if sol.success else None
                rec={'seller':seller,'outcome':args.outcome,'peer_bounds':'intersection','estimand':estimand,'qbar':qbar,'k0':k0,'peer_coefficient':bp,'peer_coefficient_bound':B,'price_support_high':Pmax,'K_unit':Kunit,'K_total':Ktotal,'cells':G,'N':N,'L':L,'alpha_R':alpha,'copies':copies,'lifetime_clock_max':lifetime,'direction':direction,'success':sol.success,'status':sol.message,'sold_cost_bound':cost,'sold_markup_bound':target_price-cost if cost is not None else None,'prior_cost_low':float(target_weight@lower),'prior_cost_high':float(target_weight@upper),'sold_mean_price':target_price,'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rhs).max()) if sol.success else None,'max_equality_violation':float(np.abs(eq@sol.x-1).max()) if sol.success else None}
                records.append(rec);print(json.dumps(rec),flush=True);pd.DataFrame(records).to_csv(OUT/f'coordinate_profit_combined_{args.outcome}_p{Pmax:g}_k{k0:g}_c{G}.csv',index=False)
    print('COMBINED_COORDINATE_PROFIT_COMPLETE',flush=True)
if __name__=='__main__':main()
