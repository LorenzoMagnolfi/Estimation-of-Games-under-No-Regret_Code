"""Outer cost screening with a simultaneous first-stage logit band.

Statistical validity remains conditional on the asymptotic Gaussian demand
band; no finite-sample demand coverage is claimed. Cost/payoff/range bounds
are deterministic enclosures on that event. Sold targets are empirical costs
of the observed sale-labelled listings, not population transaction means.
"""
import argparse,json,time
import numpy as np,pandas as pd,joblib
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from demand_band_bounds import LogitBand
from threadpoolctl import threadpool_limits


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--oracle',action='store_true');ap.add_argument('--rho-low',type=float,default=0.);ap.add_argument('--rho-high',type=float,default=1.2);ap.add_argument('--step',type=float,default=.005);ap.add_argument('--delta',type=float,default=5.);ap.add_argument('--grid-points',type=int,default=33);ap.add_argument('--critical',type=float);args=ap.parse_args();P=350.
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');bandfile=OUT/'demand_uniform_initial_p350_b1000.npz';data=np.load(bandfile);critical=float(data['critical']) if args.critical is None else args.critical
 band=LogitBand(fit,s,data['covariance'],critical);codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();cmax=s.groupby('Code',sort=True).Price.min().to_numpy();p=s.Price.to_numpy();n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events,minlength=L)/events.sum();targets={'sold':(sw,float(events@p/events.sum())),'listing_entry':(np.ones(L)/L,float(s.groupby('Code',sort=True).Price.first().mean()))}
 marks=np.array([5.,10.,20.,30.,50.,100.,250.]);hs=np.arange(11);caps=2.**hs;js=np.arange(8);eta=.99*2.**(-js[None,:])/caps[:,None];cp=1/((hs+1)*(hs+2));rp=1/((js+1)*(js+2));Aall=362+int(args.oracle);records=[];summaries=[];start=time.time();edges=np.arange(args.rho_low,args.rho_high+args.step/2,args.step)
 print(json.dumps({'stage':'start','critical':critical,'rho_boxes':len(edges)-1,'cost_grid_points':args.grid_points,'delta':args.delta,'bp_interval':[band.bplo,band.bphi],'minimum_own_slope_lower':float(band.kminlo.min())}),flush=True)
 with threadpool_limits(limits=2):
  for rlo,rhi in zip(edges[:-1],edges[1:]):
   lower=np.maximum(0,rlo*anchor-args.delta);upper=np.minimum(cmax,rhi*anchor+args.delta)
   if np.any(lower>upper):continue
   cl=lower[idx];ch=upper[idx];ownlo,ownhi=band.own_regret_box(cl,ch,marks,P,args.grid_points);spill,rangelo,rangehi=band.peer_and_range_box(cl,ch,marks,P,n,groups);total=ownlo+spill
   if args.oracle:total=np.maximum(0,total.max(axis=1))[:,None]
   logs=[]
   for z,cap in enumerate(caps):
    flo=np.minimum(1,cap/np.maximum(rangehi,1e-300));fhi=np.minimum(1,cap/np.maximum(rangelo,1e-300));weighted=np.where(total>=0,total*flo[:,None],total*fhi[:,None])*active[:,None];weighted=np.maximum(weighted,-cap)
    logs.append(np.log1p(weighted[:,:,None]*eta[z,None,None,:]).sum(axis=0))
   logs=np.array(logs)
   for copies in [1,10]:
    threshold=copies*(np.log(Aall/cp[:,None]/rp[None,:])+np.log(1000000)+np.log(copies/.005));violation=(logs-threshold[:,None,:])/eta[:,None,:];row={'rho_low':float(rlo),'rho_high':float(rhi),'delta':args.delta,'critical':critical,'copies':copies,'passes_outer_screen':bool(violation.max()<=1e-8),'max_violation_dollars':float(violation.max())}
    for name,(w,price) in targets.items():row[name+'_cost_low']=float(w@lower);row[name+'_cost_high']=float(w@upper)
    records.append(row)
   if len(records)%20==0:print(json.dumps({'boxes_checked':len(records)//2,'seconds':time.time()-start}),flush=True)
 for copies in [1,10]:
  relevant=[r for r in records if r['copies']==copies]
  for kind,rows in [('prior',relevant),('behavior',[r for r in relevant if r['passes_outer_screen']])]:
   for name,(w,price) in targets.items():
    lo=min([r[name+'_cost_low'] for r in rows],default=None);hi=max([r[name+'_cost_high'] for r in rows],default=None);rec={'state_aware_oracle':args.oracle,'estimand':name,'kind':kind,'copies':copies,'delta':args.delta,'critical':critical,'rho_search_low':args.rho_low,'rho_search_high':args.rho_high,'rho_step':args.step,'cost_grid_points':args.grid_points,'cost_low':lo,'cost_high':hi,'gross_margin_low':price-hi if hi is not None else None,'gross_margin_high':price-lo if lo is not None else None,'mean_observed_price':price,'accepted_boxes':len(rows),'seconds':time.time()-start};summaries.append(rec);print(json.dumps(rec),flush=True)
 stem=f'robust_specialist_oracle{int(args.oracle)}_delta{args.delta:g}_crit{critical:.4f}_rho{args.rho_low:g}_{args.rho_high:g}_step{args.step:g}_g{args.grid_points}'
 pd.DataFrame(records).to_csv(OUT/f'{stem}_boxes.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/f'{stem}_summary.csv',index=False)
 print('ROBUST_SPECIALIST_PROFILE_COMPLETE',flush=True)
if __name__=='__main__':main()
