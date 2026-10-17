"""Certified outer bounds with one unknown cost center per SKU.

Profile each group's center over covering intervals before combining groups.
Lagrange relaxation gives a valid bound for every nonnegative multiplier;
numerical maximization only chooses among valid bounds. Cost cells and center
intervals are outer approximations, never interpolated exclusion evidence.
"""
import argparse,hashlib,json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from audit_panel import OUT
from ask_price_grid import recover_receipts
from demand_timing_sensitivity import terminal_status_data


def group_enclosures(edges,coeff,groups,weights,delta,center_step,sign):
 result=[]
 for group in np.unique(groups):
  ix=np.flatnonzero(groups==group);cap=edges[-1,ix];end=cap.min()+delta
  centers=np.linspace(0,end,max(1,int(np.ceil(end/center_step)))+1)
  low=np.maximum(edges[:-1,ix][None,:,:],np.maximum(0,centers[:-1]-delta)[:,None,None])
  high=np.minimum(edges[1:,ix][None,:,:],(centers[1:]+delta)[:,None,None])
  valid=low<=high+1e-10
  obj=sign*(low if sign>0 else high)*weights[ix][None,None,:]
  obj=np.where(valid,obj,np.inf)
  result.append((obj,coeff[:,ix][None,:,:]))
 return result


def dual_bound(enclosures,threshold):
 evaluations=[]
 def value(multiplier):
  answer=sum(np.min(np.sum(np.min(obj+multiplier*coeff,axis=1),axis=1)) for obj,coeff in enclosures)-multiplier*threshold
  evaluations.append((float(answer),float(multiplier)));return float(answer)
 last=value(0);right=1.
 for _ in range(24):
  current=value(right)
  if current<=last:break
  last=current;right*=2
 minimize_scalar(lambda x:-value(x),bounds=(0,right),method='bounded',options={'xatol':1e-7,'maxiter':160})
 best,multiplier=max(evaluations)
 return best,multiplier,len(evaluations)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--coefficients',required=True);ap.add_argument('--delta',type=float,default=20);ap.add_argument('--center-step',type=float,default=5);ap.add_argument('--alpha-regret',type=float,default=.00025);ap.add_argument('--copies',type=int,nargs='+',default=[1,2,5,10]);ap.add_argument('--oracle-error',type=float,nargs='+',default=[0.]);ap.add_argument('--tag',default='sku_profile');args=ap.parse_args();assert args.delta>=0 and args.center_step>0 and all(0<=e<1 for e in args.oracle_error);start=time.time()
 data=np.load(args.coefficients,allow_pickle=True);coeff=data['coefficients'];edges=data['edges'];codes=data['codes'];G,L=coeff.shape;assert edges.shape==(G+1,L) and (coeff>=0).all()
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);entry=s.drop_duplicates('Code').set_index('Code').reindex(codes);groups=entry.groupby(['device_id','Storage','Condition'],sort=True).ngroup().to_numpy();index=pd.Index(codes).get_indexer(s.Code);assert (index>=0).all();p=recover_receipts(s.Price.to_numpy());active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy()*active;weights=np.bincount(index,weights=events,minlength=L)/events.sum();price=float(events@p/events.sum());cap=edges[-1];priorcap=np.minimum(cap,np.array([cap[groups==k].min()+2*args.delta for k in groups]));records=[]
 print(json.dumps({'stage':'start','delta':args.delta,'center_step':args.center_step,'groups':len(np.unique(groups)),'cells':G,'continuation_sales':int(events.sum()),'coefficient_file':args.coefficients,'coefficient_sha256':hashlib.sha256(open(args.coefficients,'rb').read()).hexdigest()}),flush=True)
 for direction,sign in [('low',1),('high',-1)]:
  enclosure=group_enclosures(edges,coeff,groups,weights,args.delta,args.center_step,sign)
  for copies in args.copies:
   for error in args.oracle_error:
    threshold=copies*(np.log(4*351)+np.log(1e6)+np.log(copies/args.alpha_regret))-active.sum()*np.log1p(-.99*error)
    bound,multiplier,evals=dual_bound(enclosure,threshold);cost=sign*bound
    rec={'model':'sku_center_profile_dual','estimand':'continuation_sold','delta':args.delta,'center_step':args.center_step,'copies':copies,'oracle_error':error,'direction':direction,'cost_bound':cost,'margin_bound':price-cost,'mean_receipt':price,'prior_cost_low':0.,'prior_cost_high':float(weights@priorcap),'alpha_R':args.alpha_regret,'threshold':float(threshold),'dual_multiplier':multiplier,'evaluations':evals,'seconds':time.time()-start};records.append(rec);print(json.dumps(rec),flush=True)
  del enclosure
 pd.DataFrame(records).to_csv(OUT/f'{args.tag}_d{args.delta:g}_h{args.center_step:g}_c{G}.csv',index=False);print('SKU_PROFILE_DUAL_COMPLETE',flush=True)

if __name__=='__main__':main()
