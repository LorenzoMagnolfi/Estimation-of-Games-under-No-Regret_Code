"""Diagnostic Gaussian perturbations of the ENTIRE shared demand model.

One daily product-group learner has a best common posted-price advisor,
with a valid total-inventory profit-range certificate. Cost is constant
within the selected SKU. Grid endpoints and perturbation quantiles are
NOT confidence intervals; this diagnoses functional uncertainty before
choosing and validating a formal inference construction.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from demand_band_bounds import LogitBand
from ask_price_grid import recover_receipts
from daily_sku_bounds import peer_changes
from oracle_witness import maximum_own_profit


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=200);ap.add_argument('--cost-step',type=float,default=.5);args=ap.parse_args();start=time.time();s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')&s.device_id.eq('apple-iphone-se-2nd-gen')&s.Storage.eq('64 GB')&s.Condition.eq('Good')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/'demand_cluster_initial_covariance.npz');theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];band=LogitBand(fit,s,data['covariance'],0.);eig,vec=np.linalg.eigh(data['covariance']);keep=eig>max(eig.max()*1e-12,1e-16);factor=vec[:,keep]*np.sqrt(eig[keep]);rng=np.random.default_rng(17027);perturb=np.column_stack([np.zeros(len(theta)),factor@rng.normal(size=(int(keep.sum()),args.draws))]);cols=np.r_[band.cols,fit.names.index('own_peer_price')+1];scales=fit.scaler.scale_[cols-1];p=s.Price.to_numpy();x=recover_receipts(p);active=s.since_seen.gt(0).to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());dates,day=np.unique(s.date,return_inverse=True);costs=np.linspace(0,x.min(),int(np.ceil(x.min()/args.cost_step))+1);c=np.broadcast_to(costs,(len(s),len(costs)));P=350.;rival=s.rival_min.to_numpy();logref=np.log(s.Ref_Price.to_numpy()/500);receipt=1.03*np.arange(1,int(np.floor(P/1.03))+1);newx=np.where(active[:,None],receipt[None,:],x[:,None]);newp=np.ceil(newx-1e-9);change=newp-p[:,None];peer=peer_changes(change,groups);gap_plus=np.maximum(newp-rival[:,None],0)-np.maximum(p-rival,0)[:,None];gap_minus=np.minimum(newp-rival[:,None],0)-np.minimum(p-rival,0)[:,None];up=peer_changes((active*(P-p))[:,None],groups)[:,0];down_slope=peer_changes(active[:,None].astype(float),groups)[:,0];down_intercept=peer_changes((-(active*p))[:,None],groups)[:,0];threshold=float(np.log(4*27)+np.log(1e6)+np.log(1/.00025));curves=[];records=[]
 def curve(th):
  beta=th[cols]/(100*scales);b=beta[0]+beta[3]*logref;gp,gn,bp=beta[1],beta[2],beta[4]
  if bp<0 or np.any(b+gp>=0) or np.any(b+gn>=0):return None
  z=band.X@th;qa=expit(z);qnew=expit(z[:,None]+b[:,None]*change+gp*gap_plus+gn*gap_minus+bp*peer);A=np.zeros((len(dates),len(receipt)));B=A.copy();np.add.at(A,day,newx*qnew);np.add.at(B,day,qnew);advisor=np.max(A[:,:,None]-B[:,:,None]*costs[None,None,:],axis=1);obsA=np.bincount(day,weights=x*qa,minlength=len(dates));obsB=np.bincount(day,weights=qa,minlength=len(dates));observed=obsA[:,None]-obsB[:,None]*costs[None,:]
  offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);owncap=maximum_own_profit(c,(offset+bp*up)[:,None],b[:,None],gp,gn,rival[:,None],P);entry=(x[:,None]-c)*(expit(z+bp*up)[:,None]-expit(z[:,None]+bp*(down_intercept[:,None]+down_slope[:,None]*costs[None,:])));range_rows=np.where(active[:,None],owncap,entry);C=np.zeros((len(dates),len(costs)));np.add.at(C,day,range_rows);gain=advisor-observed;frac=np.divide(gain,C,out=np.zeros_like(gain),where=C>1e-10);assert np.all(np.abs(frac)<=1+1e-7);return np.log1p(.99*frac).sum(axis=0)
 with threadpool_limits(limits=2):
  for draw in range(args.draws+1):
   wealth=curve(theta+perturb[:,draw])
   if wealth is None:records.append({'draw':draw,'valid_structure':False});continue
   curves.append((draw,wealth));accepted=costs[wealth<=threshold];rec={'draw':draw,'valid_structure':True,'grid_cost_low':float(accepted.min()) if len(accepted) else None,'grid_cost_high':float(accepted.max()) if len(accepted) else None,'minimum_wealth':float(wealth.min())};records.append(rec)
   if draw%25==0:print(json.dumps({**rec,'seconds':time.time()-start}),flush=True)
 table=pd.DataFrame(records);table.to_csv(OUT/'daily_functional_draw_endpoints.csv',index=False);np.savez_compressed(OUT/'daily_functional_curves.npz',costs=costs,draws=np.array([d for d,w in curves]),wealth=np.array([w for d,w in curves]),threshold=threshold);valid=table[(table.draw>0)&table.valid_structure];report={'draws':args.draws,'valid_draws':len(valid),'cost_step':args.cost_step,'threshold':threshold,'point_fit':records[0],'perturbation_endpoint_quantiles':valid[['grid_cost_low','grid_cost_high']].quantile([.01,.05,.5,.95,.99]).to_dict(),'seconds':time.time()-start,'interpretation':'Gaussian perturbation diagnostic conditional on constant SKU cost, causal logit demand, daily joint decisions, and expected-payoff common-price advice. Quantiles are not confidence bounds.'};(OUT/'daily_functional_uncertainty.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);print('DAILY_FUNCTIONAL_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
