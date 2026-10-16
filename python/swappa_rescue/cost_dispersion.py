"""Any-window shared-Prod envelope with infinitely many learning rates.

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
from scipy.sparse import csr_matrix,hstack,vstack,eye
from scipy.optimize import linprog
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from economic_envelope import profit_cap
from audit_panel import OUT


from coordinate_profit import peer_spillover_bounds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=30);ap.add_argument('--peer-bound',type=float,default=.003);ap.add_argument('--anchor',choices=['entry','absolute'],default='entry');args=ap.parse_args()
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
    qweight=np.bincount(idx,weights=qa,minlength=L)/qa.sum()
    targets=[('sold',w,meanprice),('expected_sales',qweight,float(qa@p/qa.sum()))]
    first=s.groupby('Code',sort=True).first()
    battery=pd.to_numeric(first.Battery_Life.str.extract(r'(\d+)',expand=False),errors='coerce')
    band=pd.cut(battery,[-np.inf,89,94,99,100,np.inf]).cat.codes.astype(str)
    labels=first.device_id.astype(str)+'|'+first.Storage.astype(str)+'|'+first.Condition.astype(str)+'|'+band
    group_names,group_id=np.unique(labels,return_inverse=True);H=len(group_names)
    selector=csr_matrix((np.ones(L),(np.arange(L),group_id)),shape=(L,H))
    cost_anchor=first.initial_reference.to_numpy() if args.anchor=='entry' else np.zeros(L)
    for qbar in [.9]:
        lq=logit(qbar);cb=(lq-worst_offset+gn*rival)/(b+gn);ca=(lq-worst_offset+gp*rival)/(b+gp);cmin=np.maximum(0,np.where(cb<=rival,cb,ca));lower=np.zeros(L);np.maximum.at(lower,idx,cmin)
        bad=lower>upper+1e-9;Kunit=profit_cap(qbar,k0);Ktotal=Kunit*(1+B*Pmax)
        if bad.any():
            records.append({'seller':seller,'qbar':qbar,'k0':k0,'K_unit':Kunit,'K_total':Ktotal,'status':'incompatible_cost_domain','incompatible_listings':int(bad.sum())});print(json.dumps(records[-1]),flush=True);continue
        edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];centers=(edges[:-1]+edges[1:])/2;halves=(edges[1:]-edges[:-1])/2
        marginlo=p-upper[idx];marginhi=p-lower[idx];Rlo=np.empty((A,G*L));Vhi=np.empty_like(Rlo)
        # A global lower price for each policy gives a lower peer probability
        # for every possible focal cost. Crucially, peer profit then uses the
        # SAME peer cost assignment that appears in the cost objective.
        global_pmin=np.column_stack([np.maximum(baseprice,lower[idx,None]),lower[idx,None]+marks[None,:]])
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
            pm=c[:,None]+marks[None,:];qm=prob(pm);um=qm*marks[None,:]
            qlo=prob(cl[:,None]+marks[None,:]);qhi=prob(ch[:,None]+marks[None,:]);vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
            dm=np.maximum(qa[:,None],np.abs(qa[:,None]-marks[None,:]*kmax[:,None]*vf))
            u=np.column_stack([uref,um]);r=u-qa[:,None]*(p[:,None]-c[:,None]);error=h[:,None]*np.column_stack([dr,dm])
            plo=np.column_stack([np.maximum(baseprice,cl[:,None]),cl[:,None]+marks[None,:]])
            phi=np.column_stack([np.maximum(baseprice,ch[:,None]),ch[:,None]+marks[None,:]])
            assert phi.max()<=Pmax
            spl,sph=peer_spillover_bounds(plo,phi,p,actual_logit,n,groups,marginlo,marginhi,bp,Kunit)
            total_lo=r-error+spl;total_hi=r+error+sph;vhigh=np.maximum(total_lo**2,total_hi**2)
            rr=np.zeros((L,A));vv=np.zeros_like(rr);np.add.at(rr,idx,r-error);np.add.at(vv,idx,vhigh)
            peer_cost_endpoint=np.where(peer_B>=0,edges[g+1,:,None],edges[g,:,None])
            rr+=peer_P-peer_B*peer_cost_endpoint
            Rlo[:,g*L:(g+1)*L]=rr.T/N;Vhi[:,g*L:(g+1)*L]=vv.T/N
        eta=2.**(-np.arange(J))/(2*Ktotal);rows=csr_matrix((Rlo[:,None,:]-eta[None,:,None]*Vhi[:,None,:]).reshape(A*J,G*L));priorlo=float(w@lower);priorhi=float(w@upper)
        CL=csr_matrix((edges[:-1].ravel(),(np.tile(np.arange(L),G),np.arange(G*L))),shape=(L,G*L))
        CU=csr_matrix((edges[1:].ravel(),(np.tile(np.arange(L),G),np.arange(G*L))),shape=(L,G*L))
        dispersion=vstack([hstack([CL,-selector]),hstack([-CU,selector])],format='csr')
        full_eq=hstack([eq,csr_matrix((L,H))],format='csr')
        full_reg=hstack([rows,csr_matrix((rows.shape[0],H))],format='csr')
        all_bounds=[(0,1)]*(G*L)+[(None,None)]*H
        for copies in [1]:
          for lifetime in [1000000]:
            rate_prior=1./((np.arange(J)+1)*(np.arange(J)+2))
            logterm=np.log(Aall/rate_prior)+np.log(lifetime)+np.log(copies/alpha)
            rhs=np.tile(copies*logterm/(N*eta),A)-np.repeat(missing_lower,J)
            for diameter in [0.,10.,25.,50.,1000000.]:
              drhs=np.r_[cost_anchor+diameter/2,-cost_anchor+diameter/2]
              for behavioral in [False,True]:
                U=vstack([full_reg,dispersion],format='csr') if behavioral else dispersion
                brhs=np.r_[rhs,drhs] if behavioral else drhs
                for estimand,target_weight,target_price in targets:
                  for direction,sign,values in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
                    obj=np.r_[(values*target_weight[None,:]).ravel(),np.zeros(H)]
                    start=time.time();sol=linprog(sign*obj,A_ub=U,b_ub=brhs,A_eq=full_eq,b_eq=np.ones(L),bounds=all_bounds,method='highs',options={'time_limit':120,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8});cost=float(sign*sol.fun) if sol.success else None
                    rec={'seller':seller,'anchor':args.anchor,'cost_groups':H,'diameter':diameter,'behavioral':behavioral,'estimand':estimand,'qbar':qbar,'k0':k0,'peer_bound':B,'cells':G,'copies':copies,'lifetime_clock_max':lifetime,'direction':direction,'success':sol.success,'status':sol.message,'cost_bound':cost,'markup_bound':target_price-cost if cost is not None else None,'mean_price':target_price,'seconds':time.time()-start,'max_inequality_violation':float((U@sol.x-brhs).max()) if sol.success else None}
                    records.append(rec);print(json.dumps(rec),flush=True)
                    pd.DataFrame(records).to_csv(OUT/f'cost_dispersion_upgrade_{args.anchor}_c{G}.csv',index=False)
    print('COST_DISPERSION_COMPLETE',flush=True)
if __name__=='__main__':main()
