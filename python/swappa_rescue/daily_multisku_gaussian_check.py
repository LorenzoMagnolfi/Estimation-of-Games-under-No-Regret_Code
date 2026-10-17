"""Nonlinear coherent-Gaussian diagnostic for the full cell-LP functional.

Draws are not confidence intervals. Reprofiling checks the local envelope
approximation rather than holding the LP basis fixed. Invalid slope draws
are recorded with prior bounds, never silently discarded.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel
from daily_heterogeneous_cost import DailyHeterogeneousCostModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--draws',type=int,default=60);ap.add_argument('--seed',type=int,default=171029);args=ap.parse_args();start=time.time();profilepath=OUT/args.profile;profile=json.loads(profilepath.read_text());stem=Path(args.profile).stem;gradpath=OUT/f'{stem}_inference_gradients.npz';g=np.load(gradpath);factor=g['factor'];gradient=g['gradient'];fitpath=OUT/profile['fit'];fit=joblib.load(fitpath);theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);models=[];lefts=[];rights=[];indices=[];weights=np.array([r['sales'] for r in profile['groups']]);weights/=weights.sum();delta=profile['cost_delta'];cov=np.zeros((len(theta),len(theta)));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_cost_model.py'),Path(__file__).with_name('daily_heterogeneous_cost.py'),Path(__file__).with_name('daily_sku_bounds.py'),profilepath,gradpath,fitpath]};commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    with threadpool_limits(limits=2):
        for k,(key,sub) in enumerate(s.groupby(['device_id','Storage','Condition'],sort=True)):
            model=DailyCostModel(sub,fit,cov) if delta==0 else DailyHeterogeneousCostModel(sub,fit,cov,delta);models.append(model);edges=np.linspace(0,model.cap,int(np.ceil(model.cap/profile['cell_width']))+1);lefts.append(edges[:-1]);rights.append(edges[1:]);indices.extend([k]*(len(edges)-1))
        which=np.array(indices);left=np.concatenate(lefts);right=np.concatenate(rights);Aeq=csr_matrix((np.ones(len(which)),(which,np.arange(len(which)))),shape=(len(models),len(which)))
        def evaluate(th):
            low=[]
            for model,l,r in zip(models,lefts,rights):
                model.prepare(th)
                for i in range(0,len(l),32):low.extend(model.interval_bounds(l[i:i+32],r[i:i+32])[0])
            result=[]
            for sign,axis in [(1,left),(-1,right)]:
                opt=linprog(sign*weights[which]*axis,A_ub=csr_matrix(np.array(low)[None,:]),b_ub=[profile['threshold']],A_eq=Aeq,b_eq=np.ones(len(models)),bounds=(0,None),method='highs')
                if not opt.success:raise ValueError('Perturbed profile has no feasible cost vector')
                result.append(float(sign*opt.fun))
            return result
        baseline=np.array(evaluate(theta));np.testing.assert_allclose(baseline,profile['cost_outer_interval'],atol=1e-7);rng=np.random.default_rng(args.seed);Z=rng.normal(size=(args.draws,factor.shape[1]));rows=[]
        print(json.dumps({'stage':'start','code_commit':commit,'profile':args.profile,'baseline':baseline.tolist(),'draws':args.draws,'seconds':time.time()-start}),flush=True)
        for j,z in enumerate(Z):
            reason=None
            try:value=np.array(evaluate(theta+factor@z))
            except ValueError as error:value=np.array(profile['prior_cost_interval']);reason=str(error)
            linear=baseline+gradient@z;row={'draw':j,'valid_structure':reason is None,'failure_reason':reason,'reprofiled_interval':value.tolist(),'linearized_interval':linear.tolist(),'nonlinear_residual':(value-linear).tolist(),'seconds':time.time()-start};rows.append(row)
            (OUT/f'{stem}_gaussian_progress.json').write_text(json.dumps({'profile':args.profile,'requested_draws':args.draws,'completed_draws':len(rows),'rows':rows,'source_hashes':hashes,'code_commit':commit},indent=2))
            if (j+1)%5==0:print(json.dumps(row),flush=True)
        residual=np.array([r['nonlinear_residual'] for r in rows]);report={'profile':args.profile,'draws':args.draws,'seed':args.seed,'invalid_draws':sum(not r['valid_structure'] for r in rows),'residual_mean':residual.mean(axis=0).tolist(),'residual_rmse':np.sqrt((residual*residual).mean(axis=0)).tolist(),'residual_quantiles':{str(q):np.quantile(residual,q,axis=0).tolist() for q in [.01,.05,.5,.95,.99]},'rows':rows,'source_hashes':hashes,'code_commit':commit,'seconds':time.time()-start,'interpretation':'Nonlinear Gaussian diagnostic of the entire conservative LP endpoint functional. These draws or their quantiles are not a calibrated confidence interval.'};(OUT/f'{stem}_gaussian_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_hashes']},indent=2),flush=True)
    print('DAILY_MULTISKU_GAUSSIAN_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
