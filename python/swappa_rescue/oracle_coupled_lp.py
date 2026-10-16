"""Coupled continuous costs under a state-aware optimal-price policy.

The learner uses certificate 1+U_i(c_i)+sum_j margin_j*[q_j(P)-q_j(c_i)],
where U_i is the continuous-price maximum own profit. Convexity of U_i in cost
and positive peer substitution yield an affine upper certificate preserving
ALL peer costs. Local price comparisons are only lower bounds on the oracle
policy's gain; they are not action-dependent external-regret comparators.

A McCormick outer relaxation retains the common cost vector in the normalized
oracle-loss restriction. Demand intervals retain paired actual/counterfactual
logits. No parametric cost function, cost distribution, or cost grid is imposed.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import lambertw
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack,diags
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from demand_band_bounds import LogitBand,own_gain_z_extrema,logit_difference_bounds


def own_gain_chord(band, proposed, cmax, cmin=None):
 """Affine lower bound for fixed-price gain on [0,cmax]."""
 proposed=np.asarray(proposed).reshape(-1,1);center,error,_=band.displacement(proposed)
 def endpoint(cost):
  new=proposed-cost[:,None];old=band.p[:,None]-cost[:,None]
  shift=np.where(new>=0,center-error,center+error)
  lo,_=own_gain_z_extrema(band.zlo[:,None],band.zhi[:,None],shift,new,old)
  return lo[:,0]
 cmin=np.zeros(band.N) if cmin is None else cmin
 left=endpoint(cmin);right=endpoint(cmax);slope=np.divide(right-left,cmax-cmin,out=np.zeros(band.N),where=cmax>cmin)
 return left-slope*cmin,slope


def own_range_chord(band,cmax,price_cap,cmin=None):
 """Affine upper bound for maximum own profit on a specified cost box."""
 cmin=np.zeros(band.N) if cmin is None else cmin
 _,zlohi=band.logits(cmin[:,None]);_,zhihi=band.logits(cmax[:,None])
 ulo=np.minimum(price_cap-cmin,lambertw(np.exp(zlohi[:,0]-1)).real/band.kminlo)
 uhi=np.minimum(ulo,np.minimum(price_cap-cmax,lambertw(np.exp(zhihi[:,0]-1)).real/band.kminlo))
 slope=np.divide(uhi-ulo,cmax-cmin,out=np.zeros(band.N),where=cmax>cmin)
 return ulo-slope*cmin,slope


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--critical',type=float);ap.add_argument('--copies',type=int,nargs='+',default=[1,10]);ap.add_argument('--estimand',choices=['sold','listing_entry','both'],default='both');ap.add_argument('--time-limit',type=float,default=180);ap.add_argument('--method',default='highs-ds');ap.add_argument('--cut-rounds',type=int,default=8);ap.add_argument('--rho-low',type=float);ap.add_argument('--rho-high',type=float);ap.add_argument('--delta',type=float,default=25.);args=ap.parse_args();P=350.
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');bd=np.load(OUT/'demand_uniform_initial_p350_b1000.npz');critical=float(bd['critical']) if args.critical is None else args.critical;band=LogitBand(fit,s,bd['covariance'],critical)
 codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);cmax=s.groupby('Code',sort=True).Price.min().to_numpy();p=s.Price.to_numpy();n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());active=s.since_seen.gt(0).to_numpy();activeids=np.flatnonzero(active);M=len(activeids);ai=np.full(N,-1,dtype=int);ai[activeids]=np.arange(M);events=s.sale_terminal_status.to_numpy();sw=np.bincount(idx,weights=events,minlength=L)/events.sum();targets={'sold':(sw,float(events@p/events.sum())),'listing_entry':(np.ones(L)/L,float(s.groupby('Code',sort=True).Price.first().mean()))};assert np.all(p==np.round(p))
 anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();costlow=np.zeros(L);costhigh=cmax.copy()
 if args.rho_low is not None:
  assert args.rho_high is not None
  costlow=np.maximum(0,args.rho_low*anchor-args.delta);costhigh=np.minimum(cmax,args.rho_high*anchor+args.delta)
  assert np.all(costlow<=costhigh), 'Infeasible structural cost box'
 start=time.time();U0,negative_slope=own_range_chord(band,costhigh[idx],P,costlow[idx]);bslope=-negative_slope
 cr=list(np.arange(M));cc=list(idx[activeids]);cv=list(-bslope[activeids]);cintercept=1+U0[activeids]
 for ids in groups:
  ni=int(n[ids[0]]);m=len(ids)
  if ni<=1:continue
  assert ni==m
  h0=band.bphi*(costlow[idx[ids],None]-p[ids,None])/(ni-1);h1=band.bphi*(P-p[ids,None])/(ni-1);_,dq=logit_difference_bounds(band.zlo[ids][None,:],band.zhi[ids][None,:],h0,h1);dq*=1-np.eye(m)
  for local,i in enumerate(ids):
   if not active[i]:continue
   cintercept[ai[i]]+=dq[local]@p[ids];cr.extend([ai[i]]*m);cc.extend(idx[ids]);cv.extend(-dq[local])
 C=coo_matrix((cv,(cr,cc)),shape=(M,L)).tocsr();Clo=cintercept+C@costhigh;Chi=cintercept+C@costlow;assert Clo.min()>=1-1e-8
 rr=[];rc=[];rv=[];rhs=[];row_index=0
 for delta in [-100.,-75.,-50.,-30.,-20.,-10.,-5.,5.,10.,20.,30.,50.,75.,100.]:
  proposed=np.clip(p+delta,0,P)[:,None];intercept,slope=own_gain_chord(band,proposed,costhigh[idx],costlow[idx]);peer_rows={}
  for ids in groups:
   ni=int(n[ids[0]]);m=len(ids)
   if ni<=1:continue
   dprice=(proposed[ids,0]-p[ids])/(ni-1);shift=np.where(dprice>=0,band.bplo*dprice,band.bphi*dprice)[:,None];dq,_=logit_difference_bounds(band.zlo[ids][None,:],band.zhi[ids][None,:],0.,shift);dq*=1-np.eye(m)
   for local,i in enumerate(ids):
    if active[i]:peer_rows[i]=(ids,dq[local])
  for i in activeids:
   r=row_index;row_index+=1;rr.extend([r,r]);rc.extend([idx[i],L+ai[i]]);rv.extend([slope[i],-1.]);constant=intercept[i]
   if i in peer_rows:
    ids,dq=peer_rows[i];constant+=dq@p[ids];rr.extend([r]*len(ids));rc.extend(idx[ids]);rv.extend(-dq)
   rhs.append(-constant)
 rows=coo_matrix((rv,(rr,rc)),shape=(row_index,L+2*M)).tocsr();rhs=np.array(rhs)
 # Eliminate u: every gain must obey both McCormick upper bounds.
 G=rows[:,:L];repeat=np.tile(np.arange(M),14)
 V1=coo_matrix((-Chi[repeat],(np.arange(row_index),repeat)),shape=(row_index,M)).tocsr()
 V2=coo_matrix((-Clo[repeat],(np.arange(row_index),repeat)),shape=(row_index,M)).tocsr()
 cuts=vstack([hstack([G,V1]),hstack([G-C[repeat],V2])],format='csr')
 cutrhs=np.r_[rhs,rhs+cintercept[repeat]-Clo[repeat]]
 cutbudget=hstack([csr_matrix((1,L)),csr_matrix(np.ones((1,M)))],format='csr')
 cutbounds=[(float(a),float(b)) for a,b in zip(costlow,costhigh)]+[(0,1)]*M
 def solve_cuts(objective,threshold):
  selected=np.zeros(len(cutrhs),dtype=bool);last=None;complete=False
  for iteration in range(args.cut_rounds):
   mat=vstack([cuts[selected],cutbudget],format='csr');right=np.r_[cutrhs[selected],threshold]
   sol=linprog(objective,A_ub=mat,b_ub=right,bounds=cutbounds,method=args.method,options={'time_limit':args.time_limit,'threads':2,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
   if not sol.success:break
   last=sol;viol=cuts@sol.x-cutrhs;complete=bool(viol.max()<=1e-7)
   print(json.dumps({'cut_round':iteration,'rows':int(selected.sum()),'objective':sol.fun,'max_omitted_violation':float(viol.max()),'complete':complete}),flush=True)
   if complete:break
   # Add the most violated comparator of each type for each observation.
   v=viol.reshape(2,14,M)
   for kind in range(2):
    arg=v[kind].argmax(axis=0);chosen=kind*row_index+arg*M+np.arange(M);selected[chosen[viol[chosen]>1e-7]]=True
  return last,complete,float(viol.max()) if last is not None else None

 Z=csr_matrix((M,L));I=diags(np.ones(M));upper1=hstack([Z,I,-diags(Chi)],format='csr');upper2=hstack([-C,I,-diags(Clo)],format='csr');budget=hstack([csr_matrix((1,L+M)),csr_matrix(np.ones((1,M)))],format='csr');matrix=vstack([rows,upper1,upper2,budget],format='csr');base_rhs=np.r_[rhs,np.zeros(M),cintercept-Clo];bounds=[(0,float(x)) for x in cmax]+[(0,float(x)) for x in Chi]+[(0,1)]*M;records=[]
 print(json.dumps({'stage':'LP_ready','N':N,'L':L,'active_rows':M,'variables':L+2*M,'constraints':matrix.shape[0],'nonzeros':matrix.nnz,'critical':critical,'construction_seconds':time.time()-start}),flush=True)
 with threadpool_limits(limits=2):
  for copies in args.copies:
   threshold=copies*(np.log(4*363)+np.log(1000000)+np.log(copies/.005))/np.log(1.99);right=np.r_[base_rhs,threshold]
   for estimand,(w,price) in targets.items():
    if args.estimand not in ['both',estimand]:continue
    obj=np.r_[w,np.zeros(M)]
    for direction,sign in [('low',1.),('high',-1.)]:
     st=time.time();sol,complete,violation=solve_cuts(sign*obj,threshold);cost=float(sign*sol.fun) if sol is not None else None
     rec={'model':'state_aware_oracle_coupled_continuous_costs','estimand':estimand,'critical':critical,'copies':copies,'direction':direction,'success':sol is not None,'complete_constraints':complete,'status':sol.message if sol is not None else 'No successful subset LP','cost_bound':cost,'gross_margin_bound':price-cost if cost is not None else None,'mean_observed_price':price,'rho_low':args.rho_low,'rho_high':args.rho_high,'delta':args.delta,'prior_cost_low':float(w@costlow),'prior_cost_high':float(w@costhigh),'alpha_R':.005,'lifetime_clock_max':1000000,'seconds':time.time()-st,'max_violation':violation};records.append(rec);print(json.dumps(rec),flush=True);pd.DataFrame(records).to_csv(OUT/f'oracle_coupled_cuts_crit{critical:.4f}_rho{args.rho_low}_{args.rho_high}_delta{args.delta:g}.csv',index=False)
 pd.DataFrame(records).to_csv(OUT/f'oracle_coupled_cuts_crit{critical:.4f}_rho{args.rho_low}_{args.rho_high}_delta{args.delta:g}.csv',index=False)
 print('ORACLE_COUPLED_LP_COMPLETE',flush=True)
if __name__=='__main__':main()
