"""Asymptotic inference for a certified cell-LP outer-bound functional.

A unique nondegenerate LP basis and regular active wealth coefficients give
an envelope derivative. This is inference for the conservative LP bound,
not a claim that a fractional LP solution is a realizable cost vector.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel
from daily_heterogeneous_cost import DailyHeterogeneousCostModel
from daily_cell_regularity import audit_active_cell


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--cov-name',default='demand_cluster_initial_covariance.npz');ap.add_argument('--gradient-step',type=float,default=1e-5);ap.add_argument('--tag');args=ap.parse_args();start=time.time()
    profilepath=OUT/args.profile;profile=json.loads(profilepath.read_text());fitpath=OUT/profile['fit'];covpath=OUT/args.cov_name;fit=joblib.load(fitpath);data=np.load(covpath);cov=data['covariance'];theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];np.testing.assert_allclose(theta,data['theta'],rtol=0,atol=1e-12)
    for name in ['daily_cost_model.py','daily_heterogeneous_cost.py','daily_sku_bounds.py']:
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==profile['source_hashes'][name], 'Profile model source changed'
    assert hashlib.sha256(fitpath.read_bytes()).hexdigest()==profile['source_hashes'][fitpath.name]
    s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);models=[];groups=profile['groups'];weights=np.array([g['sales'] for g in groups]);weights/=weights.sum();delta=profile['cost_delta'];step=profile['cell_width'];threshold=profile['threshold'];active=profile['low_and_high_solutions'];signs=[1,-1];multipliers=[];certificates=[];all_cells=[];cell_certificates=[]
    eig,vec=np.linalg.eigh(cov);keep=eig>max(eig.max()*1e-12,1e-16);factor=vec[:,keep]*np.sqrt(eig[keep]);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_cell_regularity.py'),profilepath,fitpath,covpath]}
    with threadpool_limits(limits=2):
        for number,(key,sub) in enumerate(s.groupby(['device_id','Storage','Condition'],sort=True)):
            assert list(key)==groups[number]['key'];model=DailyCostModel(sub,fit,cov) if delta==0 else DailyHeterogeneousCostModel(sub,fit,cov,delta);models.append(model);edges=np.linspace(0,model.cap,int(np.ceil(model.cap/step))+1);local=[]
            for i in range(0,len(edges)-1,32):
                left=edges[i:min(i+32,len(edges)-1)];right=edges[i+1:min(i+33,len(edges))];low,_=model.interval_bounds(left,right);local.extend([{'sku':number,'low':float(l),'high':float(h),'wealth_low':float(w)} for l,h,w in zip(left,right,low)])
            all_cells.extend(local)
            for side,solution in enumerate(active):
                for row in solution:
                    if row['sku']==number:cell_certificates.append(dict(sku=number,side=side,**audit_active_cell(model,row['low'],row['high'])))
        for side,(sign,solution) in enumerate(zip(signs,active)):
            bysku=[[r for r in solution if r['sku']==k] for k in range(len(groups))];mixed=[k for k,rows in enumerate(bysku) if len(rows)>1];assert len(mixed)==1,'Needs one nondegenerate mixed SKU';k=mixed[0];assert len(bysku[k])==2;r0,r1=bysku[k];axis='low' if sign==1 else 'high';lam=-sign*weights[k]*(r0[axis]-r1[axis])/(r0['wealth_low']-r1['wealth_low']);assert lam>0
            nu=np.array([sign*weights[k]*rows[0][axis]+lam*rows[0]['wealth_low'] for k,rows in enumerate(bysku)])
            min_reduced=np.inf;max_basic_error=0.;basic={(r['sku'],r['low'],r['high']) for r in solution}
            for r in all_cells:
                reduced=sign*weights[r['sku']]*r[axis]+lam*r['wealth_low']-nu[r['sku']]
                if (r['sku'],r['low'],r['high']) in basic:max_basic_error=max(max_basic_error,abs(reduced))
                else:min_reduced=min(min_reduced,reduced)
            assert min_reduced>1e-10 and max_basic_error<1e-7,'Degenerate LP needs directional inference'
            mass=np.array([sum(r['weight'] for r in rows) for rows in bysku]);np.testing.assert_allclose(mass,1,atol=1e-8)
            totalwealth=sum(r['weight']*r['wealth_low'] for r in solution);assert abs(totalwealth-threshold)<1e-7
            primal=sum(weights[r['sku']]*r[axis]*r['weight'] for r in solution);assert abs(primal-profile['cost_outer_interval'][side])<1e-7
            minimum_weight=min(r['weight'] for r in solution);assert minimum_weight>1e-8
            multipliers.append(lam);certificates.append({'wealth_multiplier':float(lam),'minimum_nonbasic_reduced_cost':float(min_reduced),'maximum_basic_reduced_cost_error':float(max_basic_error),'minimum_positive_basic_weight':float(minimum_weight),'wealth_constraint_residual':float(totalwealth-threshold),'mixed_sku':int(k)})
        def fixed_basis_wealth(th):
            result=np.zeros(2)
            for k,model in enumerate(models):
                model.prepare(th)
                for side,solution in enumerate(active):
                    rows=[r for r in solution if r['sku']==k];left=np.array([r['low'] for r in rows]);right=np.array([r['high'] for r in rows]);low,_=model.interval_bounds(left,right);result[side]+=sum(r['weight']*w for r,w in zip(rows,low))
            return result
        baseline=fixed_basis_wealth(theta);np.testing.assert_allclose(baseline,threshold,atol=1e-7)
        gradient=np.zeros((2,factor.shape[1]))
        for j in range(factor.shape[1]):
            plus=fixed_basis_wealth(theta+args.gradient_step*factor[:,j]);minus=fixed_basis_wealth(theta-args.gradient_step*factor[:,j]);gradient[:,j]=np.array(signs)*multipliers*(plus-minus)/(2*args.gradient_step)
            if (j+1)%25==0:print(json.dumps({'stage':'gradient','completed':j+1,'total':factor.shape[1],'seconds':time.time()-start}),flush=True)
        se=np.sqrt((gradient*gradient).sum(axis=1));critical=float(norm.ppf(1-.045/(2*20)));bound=profile['cost_outer_interval'];prior=profile['prior_cost_interval'];interval=[max(prior[0],bound[0]-critical*se[0]),min(prior[1],bound[1]+critical*se[1])];price=profile['mean_receipt'];report={'profile':args.profile,'demand_fit':profile['fit'],'covariance':args.cov_name,'cost_delta':delta,'point_cost_outer_interval':bound,'asymptotic_cost_outer_interval':interval,'asymptotic_margin_outer_interval':[price-interval[1],price-interval[0]],'endpoint_standard_errors':se.tolist(),'normal_critical':critical,'seller_family':20,'firststage_failure_allocation':.045,'learning_failure_allocation':.005,'lp_regularity':certificates,'active_cell_regularity':cell_certificates,'gradient_step':args.gradient_step,'source_hashes':hashes,'seconds':time.time()-start,'interpretation':'Asymptotic conditional inference on a conservative continuous-cell LP outer bound. Requires correct structural demand, clustered Gaussian approximation, smooth active cell bounds, nondegenerate LP basis, and the shared expected-payoff SKU-round learner. Does not establish finite-sample coverage, sharpness, or cost-vector existence.'};stem=args.tag or Path(args.profile).stem+'_inference';np.savez_compressed(OUT/f'{stem}_gradients.npz',gradient=gradient,factor=factor);(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    print('DAILY_MULTISKU_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
