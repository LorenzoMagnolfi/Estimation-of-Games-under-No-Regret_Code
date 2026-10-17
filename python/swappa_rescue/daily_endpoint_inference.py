"""Continuous point-set certification and asymptotic endpoint inference.

Conditional on a regular scalar-cost model and the clustered demand Gaussian
approximation. The endpoint band is an implicit-function delta method, not a
finite-sample confidence claim and not a quantile of sampled feasible costs.
"""
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--range-scale',type=float,default=1.);ap.add_argument('--fit-name',default='multiproduct_initial_terminal.joblib');ap.add_argument('--cov-name',default='demand_cluster_initial_covariance.npz');ap.add_argument('--delta',type=float,default=0);ap.add_argument('--index',choices=['constant','reference'],default='constant');ap.add_argument('--lifetime',type=int,default=1500);ap.add_argument('--tolerance',type=float,default=1e-5);ap.add_argument('--gradient-step',type=float,default=.001);ap.add_argument('--tag');args=ap.parse_args();start=time.time();s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')&s.device_id.eq('apple-iphone-se-2nd-gen')&s.Storage.eq('64 GB')&s.Condition.eq('Good')].sort_values(['Code','date']).reset_index(drop=True);fitpath=OUT/args.fit_name;covpath=OUT/args.cov_name;fit=joblib.load(fitpath);data=np.load(covpath);cov=data['covariance'];np.testing.assert_allclose(data['theta'],np.r_[fit.model.intercept_[0],fit.model.coef_[0]],rtol=0,atol=1e-12);threshold=float(np.log(4*27)+np.log(args.lifetime)+np.log(1/.00025));stem=args.tag or f'daily_endpoints_{args.index}_delta{args.delta:g}_T{args.lifetime}';eig,vec=np.linalg.eigh(cov);keep=eig>max(eig.max()*1e-12,1e-16);factor=vec[:,keep]*np.sqrt(eig[keep]);source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_cost_model.py'),Path(__file__).with_name('daily_heterogeneous_cost.py'),Path(__file__).with_name('daily_sku_bounds.py'),fitpath,covpath,OUT/'panel.parquet']}
 with threadpool_limits(limits=2):
  assert args.delta>=0
  if args.delta:
   assert args.index=='constant'
   from daily_heterogeneous_cost import DailyHeterogeneousCostModel
   model=DailyHeterogeneousCostModel(s,fit,cov,args.delta)
  else:model=DailyCostModel(s,fit,cov,args.index)
  assert args.range_scale>=1
  model.range_scale=args.range_scale
  cert=model.certify(threshold,args.tolerance);(OUT/f'{stem}_cells.json').write_text(json.dumps(cert,indent=2));assert cert['regular_two_boundary_set'],'Cannot use regular-endpoint inference for this point set';roots=np.array(cert['roots']);regularity=model.regularity(roots,[r for r in cert['runs'] if r['status']=='boundary']);assert regularity[0]['continuous_root_certificate']['wealth_derivative_interval'][1]<0 and regularity[1]['continuous_root_certificate']['wealth_derivative_interval'][0]>0;assert all(r['continuous_root_certificate']['distinct_own_optimum_ties']==0 for r in regularity);slope=np.array([r['wealth_cost_derivatives']['0.0001'] for r in regularity]);assert np.all(abs(slope)>1e-5);assert all(r['minimum_active_advisor_profit_gap']>1e-10 for r in regularity),'Advisor tie requires directional inference';gradient=np.zeros((2,factor.shape[1]));print(json.dumps({'stage':'point_set','index':args.index,'roots':roots.tolist(),'regularity':regularity,'seconds':time.time()-start}),flush=True)
  for j in range(factor.shape[1]):
   model.prepare(model.theta+args.gradient_step*factor[:,j]);plus=model.values(roots);model.prepare(model.theta-args.gradient_step*factor[:,j]);minus=model.values(roots);gradient[:,j]=-(plus-minus)/(2*args.gradient_step*slope)
   if (j+1)%25==0:print(json.dumps({'stage':'gradient','completed':j+1,'total':factor.shape[1],'seconds':time.time()-start}),flush=True)
  model.prepare(model.theta);se=np.sqrt((gradient*gradient).sum(axis=1));critical=float(norm.ppf(1-.045/(2*20*27)));lower=max(0,float(roots[0]-critical*se[0]-args.tolerance));upper=min(model.cap,float(roots[1]+critical*se[1]+args.tolerance));report={'model':'daily_common_price_advisor_bounded_cost' if args.delta else 'daily_common_price_advisor_scalar_cost','index':args.index,'range_scale':args.range_scale,'demand_fit':args.fit_name,'covariance_input':args.cov_name,'heterogeneity_delta':args.delta,'lifetime_daily_rounds':args.lifetime,'threshold':threshold,'cost_parameter_roots':roots.tolist(),'mean_cost_multiplier':model.target_factor,'point_cost_set':(roots*model.target_factor).tolist(),'point_margin_set':[model.mean_receipt-float(roots[1])*model.target_factor,model.mean_receipt-float(roots[0])*model.target_factor],'asymptotic_cost_outer_interval':[lower*model.target_factor,upper*model.target_factor],'asymptotic_margin_outer_interval':[model.mean_receipt-upper*model.target_factor,model.mean_receipt-lower*model.target_factor],'endpoint_standard_errors':(se*model.target_factor).tolist(),'endpoint_correlation':float(gradient[0]@gradient[1]/np.prod(se)),'normal_critical':critical,'firststage_failure_allocation':.045,'learning_failure_allocation':.005,'seller_family':20,'sku_family':27,'mean_receipt':model.mean_receipt,'prior_cost_set':[0,model.cap*model.target_factor],'continuation_sales':int(model.events.sum()),'daily_pricing_rounds':int(model.live.sum()),'rows':len(s),'listings':int(s.Code.nunique()),'cost_factor_quantiles':np.quantile(model.factor,[0,.1,.5,.9,1]).tolist(),'gradient_step':args.gradient_step,'cost_cell_tolerance':args.tolerance,'regularity':regularity,'source_hashes':source_hashes,'seconds':time.time()-start,'interpretation':'Asymptotic endpoint delta-method outer interval conditional on the specified replacement-cost schedule or bounded-heterogeneity restriction, causal demand and its clustered Gaussian approximation, regular consistency-set boundaries, daily shared-prior expected-payoff learning, and initialization/history restrictions. Model selection and cost misspecification are not covered.'};np.savez_compressed(OUT/f'{stem}_gradients.npz',gradient=gradient,factor=factor,roots=roots,cost_slope=slope,target_factor=model.target_factor);(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 print('DAILY_ENDPOINT_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
