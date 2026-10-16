"""State-aware best-response policy: persistent nonparametric cost bounds.

STRONGER MODEL: current competitors' prices and expected payoff information
are available before the action. The learner's fixed policy menu includes a
function selecting the best action from the base price-policy menu in each
current state. This is NOT implied by ordinary external ANR with simultaneous
unobserved rival actions. The oracle excess profit is nonnegative each round.

A single exact-wealth inequality at cap index h=0 and rate j=0 dominates all
other cap/rate inequalities for that nonnegative oracle excess. Its prior is
1/[4*(A_base+1)]. Continuous cost cells and a nuisance logit band give outer
bounds. No common cost fraction, cost density, or bounded residual is imposed.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from demand_band_bounds import LogitBand


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cells',type=int,default=30);ap.add_argument('--grid-points',type=int,default=17);ap.add_argument('--critical',type=float);args=ap.parse_args();P=350.
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/'demand_uniform_initial_p350_b1000.npz');critical=float(data['critical']) if args.critical is None else args.critical
 band=LogitBand(fit,s,data['covariance'],critical);codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;upper=s.groupby('Code',sort=True).Price.min().to_numpy();edges=np.linspace(0,1,G+1)[:,None]*upper[None,:];p=s.Price.to_numpy();n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events,minlength=L)/events.sum();targets={'sold':(sw,float(events@p/events.sum())),'listing_entry':(np.ones(L)/L,float(s.groupby('Code',sort=True).Price.first().mean()))}
 assert np.all(p==np.round(p)), 'Observed prices must be in the stated integer-price menu.'
 marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Abase=362;coeff=np.zeros((G,L));start=time.time();priorlow=np.zeros(N);priorhigh=upper[idx]
 print(json.dumps({'stage':'start','model':'state_aware_oracle_policy','critical':critical,'cells':G,'grid_points':args.grid_points,'N':N,'L':L,'active_rows':int(active.sum())}),flush=True)
 with threadpool_limits(limits=2):
  for g in range(G):
   cl=edges[g,idx];ch=edges[g+1,idx];ownlo,_=band.own_regret_box(cl,ch,marks,P,args.grid_points);spill,rangelo,rangehi=band.peer_and_range_box(cl,ch,marks,P,n,groups,priorlow,priorhigh)
   oraclelo=np.maximum(0,(ownlo+spill).max(axis=1));normalized=oraclelo/np.maximum(1,rangehi);loglower=np.log1p(.99*normalized)*active
   np.add.at(coeff[g],idx,loglower)
  eq=hstack([eye(L,format='csr')]*G,format='csr');row=csr_matrix(coeff.ravel()[None,:]);records=[]
  for copies in [1,10]:
   threshold=copies*(np.log(4*(Abase+1))+np.log(1000000)+np.log(copies/.005))
   for estimand,(w,price) in targets.items():
    for direction,sign,val in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
     obj=(w[None,:]*val).ravel();sol=linprog(sign*obj,A_ub=row,b_ub=[threshold],A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs');cost=float(sign*sol.fun) if sol.success else None
     rec={'model':'state_aware_oracle_policy','estimand':estimand,'critical':critical,'cells':G,'copies':copies,'direction':direction,'success':sol.success,'cost_bound':cost,'gross_margin_bound':price-cost if cost is not None else None,'mean_observed_price':price,'prior_cost_low':0.,'prior_cost_high':float(w@upper),'alpha_R':.005,'lifetime_clock_max':1000000,'seconds':time.time()-start,'max_violation':float((row@sol.x)[0]-threshold) if sol.success else None};records.append(rec);print(json.dumps(rec),flush=True)
 stem=f'oracle_policy_nonparam_crit{critical:.4f}_c{G}_g{args.grid_points}';pd.DataFrame(records).to_csv(OUT/f'{stem}.csv',index=False);np.savez_compressed(OUT/f'{stem}_coefficients.npz',coefficients=coeff,edges=edges,codes=codes)
 print('ORACLE_POLICY_NONPARAM_COMPLETE',flush=True)
if __name__=='__main__':main()
