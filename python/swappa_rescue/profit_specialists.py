"""Persistent-cost outer program for profit-range sleeping specialists.

No sale-probability-at-cost restriction. Each specialist is active only on
predictable states where a certified total-inventory payoff range is <=2**h.
The awake rule depends on the seller's known, persistent focal cost. This is
an explicit stronger contextual-learning class, not a consequence of ANR alone.
Known-demand pilot; continuous-cost cell bounds are conservative.
"""
import argparse,json,time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,lambertw
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from quality_demand import data
from multiproduct_demand import MultiproductDemand
from coordinate_profit import peer_spillover_bounds
from audit_panel import OUT


def coordinate_range(cost,price,log_odds,n_own,groups,peer_slope,price_cap,
                     cost_log_odds,min_slope,peer_margin=None):
    """Total-profit range bound for a focal price in [cost, price_cap].

    Own profit is nonnegative and bounded by the log-odds slope envelope.
    Peers' costs need only be nonnegative and no greater than posted prices.
    All peer probabilities increase with the focal price. Thus their joint
    range is bounded by sum p_j*[q_j(Pmax)-q_j(c)], with missing peers bounded
    analytically. The sum of ranges bounds the range of the sum.
    """
    peer_margin=price if peer_margin is None else peer_margin
    own=np.minimum(price_cap-cost,lambertw(np.exp(cost_log_odds-1)).real/min_slope)
    result=own.copy()
    for ids in groups:
        n=int(n_own[ids[0]]);m=len(ids)
        if n<=1:continue
        assert n>=m and np.all(n_own[ids]==n)
        lo=peer_slope*(cost[ids]-price[ids])/(n-1)
        hi=peer_slope*(price_cap-price[ids])/(n-1)
        dq=expit(log_odds[ids][None,:]+hi[:,None])-expit(log_odds[ids][None,:]+lo[:,None])
        result[ids]+=(dq*(1-np.eye(m))*peer_margin[ids][None,:]).sum(axis=1)
        result[ids]+=(n-m)*price_cap*(-np.expm1(-peer_slope*(price_cap-cost[ids])/(n-1)))
    return result


def sleeping_cell_bounds(lo,hi,range_lo,range_hi,cap):
    """Range decreases with cost; lo/hi bound regret throughout a cost cell."""
    always=range_lo<=cap
    possible=range_hi<=cap
    r=np.where(always[:,None],lo,np.where(possible[:,None],np.minimum(lo,0),0))
    v=np.where(possible[:,None],np.maximum(lo*lo,hi*hi),0)
    return r,v



def confidence_cell_bounds(lo,hi,range_lo,range_hi,cap):
    """Confidence-weighted regret, f=min(1,cap/certified_range).

    f depends predictably on true costs and the current payoff function.
    Interval products retain all possible correlations within the cost box.
    """
    flo=np.minimum(1,cap/np.maximum(range_lo,1e-300))[:,None]
    fhi=np.minimum(1,cap/np.maximum(range_hi,1e-300))[:,None]
    lower=np.where(lo>=0,lo*flo,lo*fhi)
    upper=np.where(hi>=0,hi*fhi,hi*flo)
    return lower,np.minimum(np.maximum(lower**2,upper**2),cap**2)



def markup_regret_interval(cl,ch,marks,price,actual_q,offset,b,gp,gn,rival,price_cap):
    """Exact scalar extrema of own-profit gains in each cost interval.

    A clipped cost-plus policy has smooth logit pieces. Interior stationary
    costs solve q(1-q)=actual_q/(k*markup). Include every such root, the kink,
    price-cap transition, and endpoints; evaluating clamped candidates is safe.
    """
    M=marks[None,:];left=cl[:,None];right=ch[:,None]
    def value(cost):
        proposed=np.minimum(price_cap,cost+M)
        z=offset[:,None]+b[:,None]*proposed+gp*np.maximum(proposed-rival[:,None],0)+gn*np.minimum(proposed-rival[:,None],0)
        return expit(z)*(proposed-cost)-actual_q[:,None]*(price[:,None]-cost)
    candidates=[np.broadcast_to(left,(len(cl),len(marks))),np.broadcast_to(right,(len(cl),len(marks))),rival[:,None]-M,np.full((len(cl),1),price_cap)-M]
    for gap in [gp,gn]:
        slope=(b+gap)[:,None];k=-slope;ratio=actual_q[:,None]/(k*M)
        root=np.sqrt(np.maximum(0,1-4*ratio))
        for q in [(1-root)/2,(1+root)/2]:
            q=np.clip(q,1e-14,1-1e-14)
            z=np.log(q/(1-q));pstar=(z-offset[:,None]+gap*rival[:,None])/slope
            candidates.append(pstar-M)
    values=[value(np.clip(c,left,right)) for c in candidates]
    return np.minimum.reduce(values),np.maximum.reduce(values)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=20);ap.add_argument('--price-cap',type=float,default=350.);ap.add_argument('--outcome',default='sale_last_seen',choices=['sale_last_seen','sale_forward','sale_legacy','sale_terminal_status']);ap.add_argument('--copies',type=int,nargs='+',default=[1,10]);ap.add_argument('--rates',type=int,default=5);ap.add_argument('--activation',choices=['sleeping','confidence'],default='sleeping');args=ap.parse_args()
    from demand_timing_sensitivity import terminal_status_data
    d=terminal_status_data() if args.outcome=='sale_terminal_status' else data()
    s=d[d.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True)
    fit=joblib.load(OUT/('multiproduct_demand.joblib' if args.outcome=='sale_last_seen' else f'multiproduct_{args.outcome}.joblib'))
    codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells
    p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);actual_logit=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values())
    ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100
    offset=actual_logit-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
    kmin=np.minimum(-(b+gp),-(b+gn));kmax=np.maximum(-(b+gp),-(b+gn));Pmax=args.price_cap
    assert bp>=0 and kmin.min()>0 and p.max()<=Pmax
    def logits(price):return offset[:,None]+b[:,None]*price+gp*np.maximum(price-rival[:,None],0)+gn*np.minimum(price-rival[:,None],0)
    def prob(price):return expit(logits(price))
    upper=s.groupby('Code',sort=True).Price.min().to_numpy();lower=np.zeros(L)
    edges=np.linspace(0,1,G+1)[:,None]*upper[None,:];centers=(edges[:-1]+edges[1:])/2;halves=(edges[1:]-edges[:-1])/2
    marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);A=len(marks);Aall=int(Pmax)+1+A
    hs=np.arange(2,11);caps=2.**hs;H=len(hs);J=args.rates;alpha=.005;Tmax=1000000
    # Countable cap and rate priors; testing these finite subsets is necessary.
    cap_prior=1/((hs+1)*(hs+2));js=np.arange(J);rate_prior=1/((js+1)*(js+2))
    eta=2.**(-js[None,:])/(2*caps[:,None])
    Rlo=np.zeros((H,A,G*L));Vhi=np.zeros_like(Rlo)
    print(json.dumps({'stage':'building','N':N,'L':L,'cells':G,'caps':caps.tolist(),'price_cap':Pmax,'outcome':args.outcome}),flush=True)
    for g in range(G):
        c=centers[g,idx];h=halves[g,idx];cl=edges[g,idx];ch=edges[g+1,idx]
        pm=np.minimum(Pmax,c[:,None]+marks);qm=prob(pm);r=qm*(pm-c[:,None])-qa[:,None]*(p[:,None]-c[:,None])
        plo=np.minimum(Pmax,cl[:,None]+marks);phi=np.minimum(Pmax,ch[:,None]+marks)
        qlo=prob(plo);qhi=prob(phi);vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
        dm=np.maximum(qa[:,None],np.abs(qa[:,None]-marks*kmax[:,None]*vf))
        dcap=np.abs(qa[:,None]-prob(np.full((N,1),Pmax)))
        dm=np.where(cl[:,None]+marks>=Pmax,dcap,np.where(ch[:,None]+marks<=Pmax,dm,np.maximum(dm,dcap)))
        error=h[:,None]*dm
        spl,sph=peer_spillover_bounds(plo,phi,p,actual_logit,n,groups,p-upper[idx],p,bp,Pmax)
        lo=r-error+spl;hi=r+error+sph
        range_lo=coordinate_range(cl,p,actual_logit,n,groups,bp,Pmax,logits(cl[:,None]).ravel(),kmin)
        range_hi=coordinate_range(ch,p,actual_logit,n,groups,bp,Pmax,logits(ch[:,None]).ravel(),kmin)
        assert np.all(range_lo>=range_hi-1e-8)
        for z,cap in enumerate(caps):
            rl,vh=(sleeping_cell_bounds if args.activation=='sleeping' else confidence_cell_bounds)(lo,hi,range_lo,range_hi,cap)
            # On an awake state, the true squared difference is also <=cap^2.
            vh=np.minimum(vh,cap*cap)
            rr=np.zeros((L,A));vv=np.zeros_like(rr);np.add.at(rr,idx,rl);np.add.at(vv,idx,vh)
            Rlo[z,:,g*L:(g+1)*L]=rr.T/N;Vhi[z,:,g*L:(g+1)*L]=vv.T/N
    rows=csr_matrix((Rlo[:,:,None,:]-eta[:,None,:,None]*Vhi[:,:,None,:]).reshape(H*A*J,G*L))
    del Rlo,Vhi
    eq=hstack([eye(L,format='csr')]*G,format='csr');events=s[args.outcome].to_numpy();sw=np.bincount(idx,weights=events,minlength=L)/events.sum();qw=np.bincount(idx,weights=qa,minlength=L)/qa.sum()
    targets=[('sold',sw,float(events@p/events.sum())),('expected_sales',qw,float(qa@p/qa.sum()))];records=[]
    for copies in args.copies:
        term=np.log(Aall/cap_prior[:,None]/rate_prior[None,:])+np.log(Tmax)+np.log(copies/alpha)
        rhs=np.broadcast_to((copies*term/(N*eta))[:,None,:],(H,A,J)).ravel()
        for estimand,w,price in targets:
            for direction,sign,val in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
                start=time.time();obj=(w[None,:]*val).ravel();sol=linprog(sign*obj,A_ub=rows,b_ub=rhs,A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs',options={'time_limit':240,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
                cost=float(sign*sol.fun) if sol.success else None
                rec={'outcome':args.outcome,'activation':args.activation,'estimand':estimand,'N':N,'L':L,'price_cap':Pmax,'cells':G,'copies':copies,'alpha_R':alpha,'lifetime_clock_max':Tmax,'direction':direction,'success':sol.success,'status':sol.message,'cost_bound':cost,'markup_bound':price-cost if cost is not None else None,'mean_price':price,'prior_cost_low':0.,'prior_cost_high':float(w@upper),'seconds':time.time()-start,'max_inequality_violation':float((rows@sol.x-rhs).max()) if sol.success else None}
                records.append(rec);print(json.dumps(rec),flush=True)
                pd.DataFrame(records).to_csv(OUT/f'profit_specialists_{args.activation}_{args.outcome}_p{Pmax:g}_c{G}.csv',index=False)
    print('PROFIT_SPECIALISTS_COMPLETE',flush=True)
if __name__=='__main__':main()
