"""State-aware best-response policy: persistent nonparametric cost bounds.

STRONGER MODEL: current competitors' prices and expected payoff information
are available before the action. The learner's fixed policy menu includes a
function selecting the best action from the base price-policy menu in each
current state. This is NOT implied by ordinary external ANR with simultaneous
unobserved rival actions. The oracle excess profit is nonnegative each round.

The fully normalized oracle has coefficient .99/C_t, with prior 1/[4*A]
in an enlarged learner that retains legacy cap/rate copies to preserve ANR.
Continuous cost cells and a joint relative-logit band give outer bounds.
No common cost fraction or cost distribution is imposed. Reference loading
optionally scales each persistent cost component by a public price index.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from normalized_demand_bounds import NormalizedBand


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--rho-low',type=float);ap.add_argument('--rho-high',type=float);ap.add_argument('--delta',type=float,default=35.);ap.add_argument('--copies',type=int,nargs='+',default=[1,2,5,10]);ap.add_argument('--reference-loading',type=float,default=0.);ap.add_argument('--single-inventory',action='store_true');ap.add_argument('--seller',default='UPGRADE SOLUTION INC');ap.add_argument('--price-cap',type=float,default=350.);ap.add_argument('--cells',type=int,default=30);ap.add_argument('--grid-points',type=int,default=17);ap.add_argument('--critical',type=float);args=ap.parse_args();P=args.price_cap;safe='' if args.seller=='UPGRADE SOLUTION INC' else '_'+''.join(x if x.isalnum() else '_' for x in args.seller)
 if args.bundle:
  assert not args.single_inventory and args.critical is None
  prefix=str(args.bundle);meta=json.loads(open(prefix+'.json').read());assert meta['seller']==args.seller and meta['price_cap']==P
  s=pd.read_csv(prefix+'.csv');arrays=np.load(prefix+'.npz');band=object.__new__(NormalizedBand)
  for key in arrays.files:setattr(band,key,arrays[key].item() if arrays[key].ndim==0 else arrays[key])
  critical=float(band.critical)
 else:
  from multiproduct_demand import MultiproductDemand
  from demand_timing_sensitivity import terminal_status_data
  s=terminal_status_data();s=s[s.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/f'demand_uniform_pairs_initial_p{P:g}_b1000{safe}.npz');critical=float(data['critical']) if args.critical is None else args.critical
 s['cost_factor']=(1-args.reference_loading)+args.reference_loading*s.Ref_Price/s.initial_reference
 assert 0<=args.reference_loading<=1
 assert s.cost_factor.min()>0
 all_minimum=(s.Price/s.cost_factor).groupby(s.Code).min()
 if args.single_inventory:s=s[s.own_n.eq(1)].reset_index(drop=True)
 factor=s.cost_factor.to_numpy()
 if not args.bundle:band=NormalizedBand(fit,s,data['covariance'],critical)
 codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;upper=all_minimum.reindex(codes).to_numpy();lower=np.zeros(L)
 if args.rho_low is not None:
  assert args.rho_high is not None and args.rho_high>=args.rho_low
  anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();lower=np.maximum(0,args.rho_low*anchor-args.delta);upper=np.minimum(upper,args.rho_high*anchor+args.delta)
  if np.any(lower>upper):
   print(json.dumps({'infeasible_prior':True,'rho_low':args.rho_low,'rho_high':args.rho_high,'delta':args.delta}),flush=True);return
 edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];p=s.Price.to_numpy();n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events*factor,minlength=L)/events.sum();targets={'sold':(sw,float(events@p/events.sum())),'listing_entry':(s.groupby('Code',sort=True).cost_factor.first().to_numpy()/L,float(s.groupby('Code',sort=True).Price.first().mean()))}
 band.prepare_peers(groups)
 cevents=events*active
 if cevents.sum()>0:targets['continuation_sold']=(np.bincount(idx,weights=cevents*factor,minlength=L)/cevents.sum(),float(cevents@p/cevents.sum()))
 assert p.max()<=P
 assert np.all(p==np.round(p)), 'Observed prices must be in the stated integer-price menu.'
 marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Abase=int(P)+12;coeff=np.zeros((G,L));start=time.time();priorlow=lower[idx]*factor;priorhigh=upper[idx]*factor
 print(json.dumps({'stage':'start','model':'fully_normalized_oracle','seller':args.seller,'single_inventory':args.single_inventory,'reference_loading':args.reference_loading,'critical':critical,'cells':G,'grid_points':args.grid_points,'N':N,'L':L,'active_rows':int(active.sum())}),flush=True)
 with threadpool_limits(limits=2):
  for g in range(G):
   cl=edges[g,idx]*factor;ch=edges[g+1,idx]*factor;total,rangehi=band.bounds(cl,ch,marks,P,n,args.grid_points,priorlow,priorhigh)
   oraclelo=np.maximum(0,total.max(axis=1));normalized=oraclelo/np.maximum(1e-300,rangehi);loglower=np.log1p(.99*normalized)*active
   np.add.at(coeff[g],idx,loglower)
  eq=hstack([eye(L,format='csr')]*G,format='csr');row=csr_matrix(coeff.ravel()[None,:]);records=[]
  for copies in args.copies:
   threshold=copies*(np.log(4*(Abase+1))+np.log(1000000)+np.log(copies/.005))
   for estimand,(w,price) in targets.items():
    for direction,sign,val in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
     obj=(w[None,:]*val).ravel();sol=linprog(sign*obj,A_ub=row,b_ub=[threshold],A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs');cost=float(sign*sol.fun) if sol.success else None
     rec={'model':'fully_normalized_oracle','seller':args.seller,'single_inventory':args.single_inventory,'reference_loading':args.reference_loading,'estimand':estimand,'critical':critical,'cells':G,'copies':copies,'continuation_sales':int(cevents.sum()),'all_sales':int(events.sum()),'direction':direction,'success':sol.success,'solver_status':sol.status,'solver_message':sol.message,'cost_bound':cost,'gross_margin_bound':price-cost if cost is not None else None,'mean_observed_price':price,'rho_low':args.rho_low,'rho_high':args.rho_high,'delta':args.delta,'prior_cost_low':float(w@lower),'prior_cost_high':float(w@upper),'alpha_R':.005,'lifetime_clock_max':1000000,'seconds':time.time()-start,'max_violation':float((row@sol.x)[0]-threshold) if sol.success else None};records.append(rec);print(json.dumps(rec),flush=True)
 stem=f'normalized_nonparam_p{P:g}_crit{critical:.4f}_c{G}_g{args.grid_points}_single{int(args.single_inventory)}_ref{args.reference_loading:g}_rho{args.rho_low}_{args.rho_high}_delta{args.delta:g}{safe}';pd.DataFrame(records).to_csv(OUT/f'{stem}.csv',index=False);np.savez_compressed(OUT/f'{stem}_coefficients.npz',coefficients=coeff,edges=edges,codes=codes)
 print('NORMALIZED_NONPARAM_COMPLETE',flush=True)
if __name__=='__main__':main()
