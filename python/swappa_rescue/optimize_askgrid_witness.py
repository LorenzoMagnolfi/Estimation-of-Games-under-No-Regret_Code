"""Joint cost-witness search with analytical total-profit gradients.

Optimization uses the integer asking-price grid, locally or in full.
The resulting cost vector MUST be audited against the full menu separately.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import expit,lambertw
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from ask_price_grid import recover_receipts
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data


def prepare(fitname="multiproduct_initial_terminal.joblib",full_grid=False):
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/fitname);codes,idx=np.unique(s.Code,return_inverse=True);p=s.Price.to_numpy();x=recover_receipts(p);ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();q=fit.predict(s);z=fit.logit(s);ef=fit.effects();b=(ef['own']+ef.get('own_seller::UPGRADE SOLUTION INC',0)+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);active=s.since_seen.gt(0).to_numpy();P=350.;groups=[]
 for ids in s.groupby('market_id',sort=False).indices.values():
  m=len(ids);a=np.tile(np.ceil(1.03*np.arange(1,int(np.floor(P/1.03))+1)-1e-9),(m,1)) if full_grid else np.column_stack([np.clip(p[ids,None]+np.arange(-75,76,5)[None,:],0,P),np.full(m,P)]);ax=1.03*np.maximum(1,np.floor(a/1.03+1e-9));a=np.ceil(ax-1e-9);ownq=expit(offset[ids,None]+b[ids,None]*a+gp*np.maximum(a-rival[ids,None],0)+gn*np.minimum(a-rival[ids,None],0));mask=1-np.eye(m)
  diff=(expit(z[ids][None,None,:]+bp*(a-p[ids,None])[:,:,None]/max(1,m-1))-q[ids][None,None,:])*mask[:,None,:]
  for j in range(m):diff[j,:,j]=ownq[j]-q[ids[j]]
  const=(ax-x[ids,None])*ownq+diff@x[ids];qhigh=expit(z[ids][None,:]+bp*(P-p[ids,None])/max(1,m-1));groups.append((ids,diff,const,mask,qhigh))
 def loss_gradient(cost):
  c=cost[idx];candidates=[];probabilities=[]
  for gap,left,right in [(gn,c,np.maximum(c,np.minimum(rival,P))),(gp,np.minimum(P,np.maximum(c,rival)),np.full_like(c,P))]:
   k=-(b+gap);intercept=offset-gap*rival;a=np.clip(c+(1+lambertw(np.exp(intercept-k*c-1)).real)/k,left,right);qa=expit(offset+b*a+gp*np.maximum(a-rival,0)+gn*np.minimum(a-rival,0));candidates.append((a-c)*qa);probabilities.append(qa)
  first=candidates[0]>=candidates[1];U=np.where(first,candidates[0],candidates[1]);qbest=np.where(first,probabilities[0],probabilities[1]);value=0.;gradient=np.zeros(len(codes))
  for ids,diff,const,mask,qhigh in groups:
   m=len(ids);margin=x[ids]-c[ids];gains=const-diff@c[ids];choice=gains.argmax(axis=1);r=np.maximum(0,gains[np.arange(m),choice]);dr=-diff[np.arange(m),choice,:]
   qlow=expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/max(1,m-1));change=(qhigh-qlow)*mask;C=U[ids]+change@margin;dc=-change
   dc[np.arange(m),np.arange(m)]=-qbest[ids]-bp/max(1,m-1)*((qlow*(1-qlow)*mask)@margin)
   use=active[ids];value+=float(np.log1p(.99*r[use]/C[use]).sum());weight=.99/(C+.99*r)*use;local=(weight[:,None]*(dr-r[:,None]/C[:,None]*dc)).sum(0);np.add.at(gradient,idx[ids],local)
  return value,gradient
 return s,codes,loss_gradient


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fit",default="multiproduct_initial_terminal.joblib");ap.add_argument("--full-grid",action="store_true");args=ap.parse_args();tag=("_full" if args.full_grid else "")+("_hetero" if args.fit!="multiproduct_initial_terminal.joblib" else "");s,codes,fun=prepare(args.fit,args.full_grid);upper=s.assign(receipt=recover_receipts(s.Price.to_numpy())).groupby('Code',sort=True).receipt.min().to_numpy();initial=pd.read_csv(OUT/'ask_grid_witness_multiproduct_initial_terminal_costs.csv').set_index('Code').cost.reindex(codes).to_numpy();rng=np.random.default_rng(61017);c=np.clip(initial,1e-3,upper-1e-3);value,grad=fun(c);errors=[]
 for j in rng.choice(len(c),20,replace=False):
  step=1e-5;left=c.copy();right=c.copy();left[j]-=step;right[j]+=step;num=(fun(right)[0]-fun(left)[0])/(2*step);errors.append(abs(num-grad[j]))
 assert max(errors)<1e-4, max(errors)
 print(json.dumps({'initial_local_logwealth':value,'gradient_max_error':max(errors)}),flush=True);start=time.time();iteration=0
 def callback(x):
  nonlocal iteration
  iteration+=1
  if iteration%10==0:print(json.dumps({'iteration':iteration,'local_logwealth':fun(x)[0],'seconds':time.time()-start}),flush=True)
 with threadpool_limits(limits=2):result=minimize(fun,initial,jac=True,bounds=list(zip(np.zeros(len(upper)),upper)),method='L-BFGS-B',callback=callback,options={'maxiter':250,'ftol':1e-11,'gtol':1e-7,'maxls':30})
 report={'success':bool(result.success),'message':str(result.message),'local_menu_logwealth':float(result.fun),'iterations':int(result.nit),'seconds':time.time()-start,'audit_required':True,'fit':args.fit,'full_grid':args.full_grid};print(json.dumps(report),flush=True);pd.DataFrame({'Code':codes,'cost':result.x}).to_csv(OUT/f'upgrade_askgrid_optimized{tag}_costs.csv',index=False);(OUT/f'upgrade_askgrid_optimized{tag}_search.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
