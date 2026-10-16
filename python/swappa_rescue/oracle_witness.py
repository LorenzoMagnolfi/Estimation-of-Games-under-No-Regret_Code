"""Exact expected-demand checks for the state-aware policy class.

FOCs only initialize costs. Passing is a numerical necessary-event witness,
not evidence of the actual seller algorithm or causal demand identification.
"""
import argparse,json
import joblib,numpy as np,pandas as pd
from scipy.special import expit,lambertw
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from coordinate_profit import peer_spillover_bounds


def maximum_own_profit(c,offset,b,gp,gn,rival,cap):
 """Exact maximum of (price-cost)*piecewise-logit probability."""
 values=[]
 for gap,left,right in [(gn,c,np.maximum(c,np.minimum(rival,cap))),(gp,np.minimum(cap,np.maximum(c,rival)),np.full_like(c,cap))]:
  k=-(b+gap);intercept=offset-gap*rival
  optimum=c+(1+lambertw(np.exp(intercept-k*c-1)).real)/k
  a=np.clip(optimum,left,right)
  values.append((a-c)*expit(offset+b*a+gp*np.maximum(a-rival,0)+gn*np.minimum(a-rival,0)))
 return np.maximum.reduce(values)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cost-file');ap.add_argument('--reference-loading',type=float,default=0.);ap.add_argument('--seller',default='UPGRADE SOLUTION INC');ap.add_argument('--single-inventory',action='store_true');ap.add_argument('--price-cap',type=float,default=350.);args=ap.parse_args();suffix='' if args.seller=='UPGRADE SOLUTION INC' and not args.single_inventory else '_'+''.join(x if x.isalnum() else '_' for x in args.seller)+'_single'+str(int(args.single_inventory))
 s=terminal_status_data();s=s[s.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib')
 s['cost_factor']=(1-args.reference_loading)+args.reference_loading*s.Ref_Price/s.initial_reference
 assert 0<=args.reference_loading<=1
 assert s.cost_factor.min()>0
 all_minimum=(s.Price/s.cost_factor).groupby(s.Code).min()
 if args.reference_loading:suffix+=f'_ref{args.reference_loading:g}'
 if args.single_inventory:s=s[s.own_n.eq(1)].reset_index(drop=True)
 factor=s.cost_factor.to_numpy();codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);z=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();cmax=all_minimum.reindex(codes).to_numpy();anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events*factor,minlength=L)/events.sum()
 ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;k=-(b+np.where(p>=rival,gp,gn));offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);P=args.price_cap
 diagonal=np.bincount(idx,weights=active*k*qa*(1-qa)*factor,minlength=L);rhs=np.bincount(idx,weights=active*(k*qa*(1-qa)*p-qa),minlength=L);rr=[];cc=[];vv=[]
 for ids in groups:
  ni=int(n[ids[0]]);m=len(ids)
  if ni<=1:continue
  coef=active[ids,None]*bp/(ni-1)*(1-np.eye(m))*(qa[ids]*(1-qa[ids]))[None,:]
  rr.extend(np.repeat(idx[ids],m));cc.extend(np.tile(idx[ids],m));vv.extend((-coef*factor[ids][None,:]).ravel());np.add.at(rhs,idx[ids],-coef@p[ids])
 absent=diagonal<=1e-14;diagonal[absent]=1;rhs[absent]=.7*anchor[absent]
 foc=spsolve(diags(diagonal)+coo_matrix((vv,(rr,cc)),shape=(L,L)).tocsr(),rhs);foc=np.clip(foc,0,cmax)
 candidates=[('unrestricted_foc_initialization',foc)]
 if args.cost_file:
  candidate=pd.read_csv(args.cost_file).set_index('Code').cost.reindex(codes).to_numpy();assert np.all((candidate>=0)&(candidate<=cmax+1e-8));candidates=[('supplied_costs',candidate)];suffix+='_supplied'
 for rho,delta in ([((.69),5),(.7,10),(.75,15),(.75,25)] if args.seller=='UPGRADE SOLUTION INC' and not args.cost_file else []):
  low=np.maximum(0,rho*anchor-delta);high=np.minimum(cmax,rho*anchor+delta)
  if np.all(low<=high):candidates.append((f'rho{rho}_delta{delta}',np.clip(foc,low,high)))
 records=[]
 with threadpool_limits(limits=2):
  for name,cost in candidates:
   c=cost[idx]*factor;marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);proposal=np.column_stack([np.maximum(np.arange(int(P)+1)[None,:],c[:,None]),np.minimum(P,c[:,None]+marks)])
   logits=offset[:,None]+b[:,None]*proposal+gp*np.maximum(proposal-rival[:,None],0)+gn*np.minimum(proposal-rival[:,None],0)
   own=expit(logits)*(proposal-c[:,None])-qa[:,None]*(p[:,None]-c[:,None]);spl,sph=peer_spillover_bounds(proposal,proposal,p,z,n,groups,p-c,p-c,bp,P);assert np.max(abs(spl-sph))<1e-8
   rstar=np.maximum(0,(own+spl).max(1));rang=maximum_own_profit(c,offset,b,gp,gn,rival,P)
   for ids in groups:
    ni=int(n[ids[0]]);m=len(ids)
    if ni<=1:continue
    diff=expit(z[ids][None,:]+bp*(P-p[ids,None])/(ni-1))-expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/(ni-1));rang[ids]+=(diff*(1-np.eye(m)))@(p[ids]-c[ids])
   for certificate,S in [('max1',np.maximum(1,rang)),('plus1',1+rang),('fully_normalized',np.maximum(rang,1e-300))]:
    logwealth=float(np.log1p(.99*rstar[active]/S[active]).sum());threshold=float(np.log(4*(int(P)+13))+np.log(1000000)+np.log(1/.005));rec={'candidate':name,'seller':args.seller,'single_inventory':args.single_inventory,'reference_loading':args.reference_loading,'certificate':certificate,'logwealth':logwealth,'threshold':threshold,'passes_dominating_oracle_constraint':logwealth<=threshold,'observed_sale_weighted_cost':float(sw@cost),'gross_margin':float(events@p/events.sum()-sw@cost),'max_normalized_regret':float((rstar/S).max())};records.append(rec);print(json.dumps(rec),flush=True)
   pd.DataFrame({'Code':codes,'cost':cost}).to_csv(OUT/f'oracle_witness_{name}_costs{suffix}.csv',index=False)
 pd.DataFrame(records).to_csv(OUT/f'oracle_witness_audit{suffix}.csv',index=False)
 print('ORACLE_WITNESS_COMPLETE',flush=True)
if __name__=='__main__':main()
