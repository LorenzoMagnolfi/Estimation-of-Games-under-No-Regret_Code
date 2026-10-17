"""Gaussian multiplier bands over all focal counterfactual prices.

These are asymptotic first-stage diagnostics, conditional on validity of the
cluster covariance/Gaussian approximation for the target design. They are not
finite-sample coverage claims. Each price path is piecewise affine in the logit
coefficients; its studentized Gaussian maximum is computed analytically over
the full continuous price interval, not on a price grid. Monte Carlo quantile
error gets a separate binomial order-statistic allowance.
"""
import argparse,json,time,hashlib
import joblib,numpy as np
from scipy.special import expit
from scipy.stats import binom,chi2
from scipy.sparse import csr_matrix,hstack
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data


def max_standardized_line(a,b,A,B,C):
    """max_{0<=t<=1} |a+b*t|/sqrt(A+2*B*t+C*t*t), columnwise draws."""
    A=A[:,None];B=B[:,None];C=C[:,None]
    def ratio(numerator,variance):
        return np.divide(np.abs(numerator),np.sqrt(np.maximum(variance,0)),out=np.zeros_like(numerator),where=variance>1e-20)
    result=np.maximum(ratio(a,A),ratio(a+b,A+2*B+C))
    den=b*B-a*C
    t=np.divide(a*B-b*A,den,out=np.full_like(a,-1.),where=np.abs(den)>1e-25)
    candidate=ratio(a+b*t,A+2*B*t+C*t*t)
    return np.maximum(result,np.where((t>=0)&(t<=1),candidate,0)).max(axis=0)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--alpha-band',type=float);ap.add_argument('--alpha-mc',type=float);ap.add_argument('--family-size',type=int,default=1);ap.add_argument('--seller',default='UPGRADE SOLUTION INC');ap.add_argument('--pairs',action='store_true');ap.add_argument('--draws',type=int,default=1000);ap.add_argument('--price-cap',type=float,default=350.);ap.add_argument('--seed',type=int,default=161026);args=ap.parse_args();assert args.family_size>=1;Pmax=args.price_cap
 s=terminal_status_data();s=s[s.Seller.eq(args.seller)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cv=np.load(OUT/'demand_cluster_initial_covariance.npz');cov=cv['covariance'];theta=cv['theta']
 vals,vecs=np.linalg.eigh((cov+cov.T)/2);keep=vals>max(vals.max()*1e-12,1e-16);factor=vecs[:,keep]*np.sqrt(vals[keep]);rank=int(keep.sum());N=len(s)
 def design(price=None):
  action=None if price is None else price-s.Ref_Price.to_numpy()
  return hstack([csr_matrix(np.ones((N,1))),csr_matrix(fit.scaler.transform(fit.continuous(s,action))),fit.encoder.transform(fit.cat(s))],format='csr').toarray()
 Xa=design();X0=design(np.zeros(N));Xk=design(np.clip(s.rival_min.to_numpy(),0,Pmax));XP=design(np.full(N,Pmax))
 assert np.max(abs(Xa@theta-fit.logit(s)))<1e-8
 va=Xa@factor;v0=X0@factor;vk=Xk@factor;vp=XP@factor;curves=[]
 groups=list(s.groupby('market_id',sort=False).indices.values()) if args.pairs else [];pair_design=[]
 for ids in groups:
  if len(ids)<2:continue
  i,j=np.triu_indices(len(ids),1);left=ids[i];right=ids[j];se=np.sqrt(np.sum((va[left]-va[right])**2,axis=1));keep=se>1e-10
  pair_design.append((left[keep],right[keep],se[keep]))
 for left,right in [(v0,vk),(vk,vp)]:
  delta=right-left
  for start in [left,left-va]:curves.append((start,delta,(start*start).sum(1),(start*delta).sum(1),(delta*delta).sum(1)))
 j=fit.names.index('own_peer_price');peer=factor[j+1]/fit.scaler.scale_[j]/100;peer/=np.linalg.norm(peer)
 rng=np.random.default_rng(args.seed);maxima=[];starttime=time.time()
 with threadpool_limits(limits=2):
  for start in range(0,args.draws,64):
   Z=rng.standard_normal((rank,min(64,args.draws-start)));mx=np.abs(peer@Z)
   actual_draw=va@Z
   for left,right,se in pair_design:
    if len(se):mx=np.maximum(mx,(abs(actual_draw[left]-actual_draw[right])/se[:,None]).max(axis=0))
   for left,delta,A,B,C in curves:
    a=left@Z;b=delta@Z;mx=np.maximum(mx,max_standardized_line(a,b,A,B,C))
    # Directional slopes are needed even when the observed price is at a kink.
    Croot=np.sqrt(C);slope=np.divide(np.abs(b),Croot[:,None],out=np.zeros_like(b),where=Croot[:,None]>1e-10).max(axis=0);mx=np.maximum(mx,slope)
   maxima.extend(mx.tolist())
 alpha_band=.04/args.family_size if args.alpha_band is None else args.alpha_band;alpha_mc=.005/args.family_size if args.alpha_mc is None else args.alpha_mc;assert 0<alpha_band<1 and 0<alpha_mc<1;order=int(binom.ppf(1-alpha_mc,args.draws,1-alpha_band))+1
 assert order<=args.draws,'More draws are required for the Monte Carlo quantile allowance.'
 critical=float(np.sort(maxima)[order-1]);baseline_se=np.sqrt(np.sum(va*va,axis=1));report={'family_size':args.family_size,'seller':args.seller,'draws':args.draws,'seed':args.seed,'focal_rows':N,'coefficient_rank':rank,'within_market_pair_directions':sum(len(x[2]) for x in pair_design),'price_cap':Pmax,'alpha_band':alpha_band,'alpha_MC':alpha_mc,'upper_order_statistic':order,'critical_uniform':critical,'ordinary_empirical_quantile':float(np.quantile(maxima,1-alpha_band)),'ellipsoid_radius':float(np.sqrt(chi2.ppf(1-alpha_band,rank))),'baseline_logit_se_quantiles':np.quantile(baseline_se,[0,.25,.5,.75,1]).tolist(),'seconds':time.time()-starttime,'interpretation':'Asymptotic Gaussian diagnostic; conditional target-design approximation, cluster covariance, structural logit specification, and causal counterfactual interpretation remain maintained assumptions.'}
 prefix='pairs_' if args.pairs else ''
 suffix='' if args.seller=='UPGRADE SOLUTION INC' else '_'+''.join(x if x.isalnum() else '_' for x in args.seller)
 stem=f'demand_uniform_{prefix}initial_p{Pmax:g}_b{args.draws}{suffix}'+(f'_family{args.family_size}' if args.family_size>1 else '')
 (OUT/f'{stem}.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/f'{stem}.npz',maxima=maxima,critical=critical,alpha_band=alpha_band,alpha_mc=alpha_mc,order=order,covariance=factor@factor.T,theta=theta,X_actual=Xa,X_zero=X0,X_kink=Xk,X_cap=XP)
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
