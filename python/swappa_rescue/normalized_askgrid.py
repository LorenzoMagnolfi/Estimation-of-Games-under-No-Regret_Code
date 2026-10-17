"""Persistent-cost outer bounds under an integer asking-price model.
Demand uses the rounded displayed price; payoff uses 1.03*integer ask.
The range certificate retains the conservative continuous-display upper bound.
Fixed feasible grid proposals bound the state-aware oracle within each cost
cell without interpolating over discontinuous price recommendations.
This remains a stronger information/learning model than ordinary ANR.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.sparse import csr_matrix,hstack,eye
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from ask_price_grid import AskGridBand as NormalizedBand,recover_receipts


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shared-slope-band',action='store_true');ap.add_argument('--output-tag');ap.add_argument('--band-file');ap.add_argument('--alpha-regret',type=float,default=.005);ap.add_argument('--bundle');ap.add_argument('--rho-low',type=float);ap.add_argument('--rho-high',type=float);ap.add_argument('--delta',type=float,default=35.);ap.add_argument('--copies',type=int,nargs='+',default=[1,2,5,10]);ap.add_argument('--reference-loading',type=float,default=0.);ap.add_argument('--single-inventory',action='store_true');ap.add_argument('--seller',default='UPGRADE SOLUTION INC');ap.add_argument('--price-cap',type=float,default=350.);ap.add_argument('--cells',type=int,default=30);ap.add_argument('--grid-points',type=int,default=17);ap.add_argument('--critical',type=float);args=ap.parse_args();assert 0<args.alpha_regret<1;P=args.price_cap;safe='' if args.seller=='UPGRADE SOLUTION INC' else '_'+''.join(x if x.isalnum() else '_' for x in args.seller)
 assert not (args.bundle and args.shared_slope_band), 'Shared-slope calibration currently requires direct inputs'
 if args.bundle:
  assert not args.single_inventory and args.critical is None
  prefix=str(args.bundle);meta=json.loads(open(prefix+'.json').read());assert meta['seller']==args.seller and meta['price_cap']==P
  s=pd.read_csv(prefix+'.csv');arrays=np.load(prefix+'.npz');band=object.__new__(NormalizedBand)
  for key in arrays.files:setattr(band,key,arrays[key].item() if arrays[key].ndim==0 else arrays[key])
  critical=float(band.critical)
 else:
  from multiproduct_demand import MultiproductDemand
  from demand_timing_sensitivity import terminal_status_data
  s=terminal_status_data();s=s[s.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/(args.band_file if args.band_file else f'demand_uniform_pairs_initial_p{P:g}_b1000{safe}.npz'));critical=float(data['critical']) if args.critical is None else args.critical
 s['cost_factor']=1. if args.reference_loading==0 else (1-args.reference_loading)+args.reference_loading*s.Ref_Price/s.initial_reference
 assert 0<=args.reference_loading<=1
 assert s.cost_factor.min()>0
 s['receipt']=recover_receipts(s.Price.to_numpy());all_minimum=(s.receipt/s.cost_factor).groupby(s.Code).min()
 if args.single_inventory:s=s[s.own_n.eq(1)].reset_index(drop=True)
 factor=s.cost_factor.to_numpy()
 calibration={}
 if not args.bundle:
  if args.shared_slope_band:
   assert args.output_tag, 'Use a distinct output tag for the shared-slope calibration'
   from shared_slope_calibration import calibration as calibrate_shared_slopes
   assert args.critical is None
   calibration=calibrate_shared_slopes(data,args.alpha_regret);sc=calibration['slope_critical'];pc=calibration['peer_critical']
   band=NormalizedBand(fit,s,data['covariance'],critical,slope_critical=sc,peer_critical=pc)
  else:band=NormalizedBand(fit,s,data['covariance'],critical)
 codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);G=args.cells;upper=all_minimum.reindex(codes).to_numpy();lower=np.zeros(L)
 if args.rho_low is not None:
  assert args.rho_high is not None and args.rho_high>=args.rho_low
  anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();lower=np.where(anchor>0,np.maximum(0,args.rho_low*anchor-args.delta),0);upper=np.where(anchor>0,np.minimum(upper,args.rho_high*anchor+args.delta),upper)
  if np.any(lower>upper):
   print(json.dumps({'infeasible_prior':True,'rho_low':args.rho_low,'rho_high':args.rho_high,'delta':args.delta}),flush=True);return
 edges=lower[None,:]+np.linspace(0,1,G+1)[:,None]*(upper-lower)[None,:];p=s.receipt.to_numpy();n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events*factor,minlength=L)/events.sum();targets={'sold':(sw,float(events@p/events.sum())),'listing_entry':(s.groupby('Code',sort=True).cost_factor.first().to_numpy()/L,float(s.groupby('Code',sort=True).receipt.first().mean()))}
 band.prepare_peers(groups)
 cevents=events*active
 if cevents.sum()>0:targets['continuation_sold']=(np.bincount(idx,weights=cevents*factor,minlength=L)/cevents.sum(),float(cevents@p/cevents.sum()))
 assert p.max()<=P
 assert np.all(s.Price.to_numpy()==np.ceil(p-1e-9))
 marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);Abase=int(np.floor(P/1.03))+11;coeff=np.zeros((G,L));start=time.time();priorlow=lower[idx]*factor;priorhigh=upper[idx]*factor
 print(json.dumps({'stage':'start','model':'integer_ask_joint_range_oracle',**calibration,'seller':args.seller,'single_inventory':args.single_inventory,'reference_loading':args.reference_loading,'critical':critical,'cells':G,'grid_points':args.grid_points,'N':N,'L':L,'active_rows':int(active.sum())}),flush=True)
 with threadpool_limits(limits=2):
  for g in range(G):
   cl=edges[g,idx]*factor;ch=edges[g+1,idx]*factor;total,rangehi=band.bounds(cl,ch,marks,P,n,args.grid_points,priorlow,priorhigh)
   oraclelo=np.maximum(0,total.max(axis=1));normalized=np.maximum(oraclelo/np.maximum(1e-300,rangehi),band.last_joint_fraction_lower);loglower=np.log1p(.99*normalized)*active
   np.add.at(coeff[g],idx,loglower)
   if (g+1)%20==0:print(json.dumps({'stage':'cells','completed':g+1,'seconds':time.time()-start}),flush=True)
  eq=hstack([eye(L,format='csr')]*G,format='csr');row=csr_matrix(coeff.ravel()[None,:]);records=[]
  for copies in args.copies:
   threshold=copies*(np.log(4*(Abase+1))+np.log(1000000)+np.log(copies/args.alpha_regret))
   for estimand,(w,price) in targets.items():
    for direction,sign,val in [('low',1.,edges[:-1]),('high',-1.,edges[1:])]:
     obj=(w[None,:]*val).ravel();sol=linprog(sign*obj,A_ub=row,b_ub=[threshold],A_eq=eq,b_eq=np.ones(L),bounds=(0,1),method='highs');cost=float(sign*sol.fun) if sol.success else None
     rec={'model':'integer_ask_joint_range_oracle',**calibration,'seller':args.seller,'single_inventory':args.single_inventory,'reference_loading':args.reference_loading,'estimand':estimand,'critical':critical,'cells':G,'copies':copies,'continuation_sales':int(cevents.sum()),'all_sales':int(events.sum()),'direction':direction,'success':sol.success,'solver_status':sol.status,'solver_message':sol.message,'cost_bound':cost,'gross_margin_bound':price-cost if cost is not None else None,'mean_observed_price':price,'rho_low':args.rho_low,'rho_high':args.rho_high,'delta':args.delta,'prior_cost_low':float(w@lower),'prior_cost_high':float(w@upper),'alpha_R':args.alpha_regret,'lifetime_clock_max':1000000,'seconds':time.time()-start,'max_violation':float((row@sol.x)[0]-threshold) if sol.success else None};records.append(rec);print(json.dumps(rec),flush=True)
 stem=f'normalized_askgrid_jointU_alpha{args.alpha_regret:g}_p{P:g}_crit{critical:.4f}_c{G}_g{args.grid_points}_single{int(args.single_inventory)}_ref{args.reference_loading:g}_rho{args.rho_low}_{args.rho_high}_delta{args.delta:g}{safe}';stem=args.output_tag or stem;assert '/' not in stem and chr(92) not in stem;pd.DataFrame(records).to_csv(OUT/f'{stem}.csv',index=False);np.savez_compressed(OUT/f'{stem}_coefficients.npz',coefficients=coeff,edges=edges,codes=codes.astype(str))
 print('NORMALIZED_NONPARAM_COMPLETE',flush=True)
if __name__=='__main__':main()
