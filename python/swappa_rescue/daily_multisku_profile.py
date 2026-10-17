"""Continuous-cost LP outer bounds for one learner across all SKU rounds.

Every observed target SKU enters the advisor event; costs have separate SKU
levels. A cell-LP relaxes the separable wealth constraint. Fractional mixtures
are conservative outer bounds, not actual cost vectors or existence proofs.
"""
import argparse
import hashlib
import json
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
from daily_scope import configure_model
from daily_proxy_error import proxy_wealth_allowance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--payoff-error',type=float,default=0.);parser.add_argument('--seller-family',type=int,default=20);parser.add_argument('--range-scale',type=float,default=1.);parser.add_argument('--rival-band',type=float,nargs=2);parser.add_argument('--delta',type=float,default=0);parser.add_argument('--step',type=float,default=.5);parser.add_argument('--lifetime',type=int,default=1000000);parser.add_argument('--fit-name',default='multiproduct_initial_terminal.joblib');parser.add_argument('--cov-name',default='demand_cluster_initial_covariance.npz');parser.add_argument('--tag');args=parser.parse_args();start=time.time()
    s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True)
    assert set(s.device_id)<=set(['apple-iphone-8','apple-iphone-se-2nd-gen','apple-iphone-se-3rd-gen-2022'])
    assert set(s.Storage)<=set(['64 GB','128 GB','256 GB']) and set(s.Condition)<=set(['Fair','Good','Mint'])
    fitpath=OUT/args.fit_name;covpath=OUT/args.cov_name;fit=joblib.load(fitpath);cov=np.load(covpath)['covariance'];models=[];groups=[];cells=[];minimum_grid=[]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_cost_model.py'),Path(__file__).with_name('daily_heterogeneous_cost.py'),Path(__file__).with_name('daily_sku_bounds.py'),Path(__file__).with_name('daily_scope.py'),Path(__file__).with_name('daily_proxy_error.py'),fitpath,covpath]}
    assert args.seller_family>=1
    threshold=float(np.log(4)+np.log(args.lifetime)+np.log(args.seller_family/.005))
    with threadpool_limits(limits=2):
        for number,(key,sub) in enumerate(s.groupby(['device_id','Storage','Condition'],sort=True)):
            model=DailyCostModel(sub,fit,cov) if args.delta==0 else DailyHeterogeneousCostModel(sub,fit,cov,args.delta)
            configure_model(model,args.range_scale,args.rival_band);models.append(model);edges=np.linspace(0,model.cap,int(np.ceil(model.cap/args.step))+1);local=[]
            for startcell in range(0,len(edges)-1,32):
                left=edges[startcell:min(startcell+32,len(edges)-1)];right=edges[startcell+1:min(startcell+33,len(edges))];lo,hi=model.interval_bounds(left,right)
                local.extend([{'sku':number,'low':float(l),'high':float(h),'wealth_low':float(a),'wealth_high':float(b)} for l,h,a,b in zip(left,right,lo,hi)])
            centers=(edges[:-1]+edges[1:])/2;wealth=[]
            for i in range(0,len(centers),32):wealth.extend(model.values(centers[i:i+32]))
            best=int(np.argmin(wealth));minimum_grid.append({'cost':float(centers[best]),'wealth':float(wealth[best])});cells.extend(local)
            groups.append({'number':number,'key':list(key),'sales':float(model.events.sum()),'mean_receipt':model.mean_receipt,'prior_cap':model.cap,'rounds':int(model.live.sum()),'minimum_grid_cost':float(centers[best]),'minimum_grid_wealth':float(wealth[best])})
            print(json.dumps({'stage':'sku','group':groups[-1],'seconds':time.time()-start}),flush=True)
        base_threshold=threshold;payoff_allowance=proxy_wealth_allowance(sum(g['rounds'] for g in groups),args.payoff_error);threshold+=payoff_allowance
        weights=np.array([g['sales'] for g in groups]);weights/=weights.sum();which=np.array([r['sku'] for r in cells]);lower=np.array([r['low'] for r in cells]);upper=np.array([r['high'] for r in cells]);wlow=np.array([r['wealth_low'] for r in cells]);Aeq=csr_matrix((np.ones(len(cells)),(which,np.arange(len(cells)))),shape=(len(groups),len(cells)))
        endpoints=[];solutions=[]
        for sign,axis in [(1,lower),(-1,upper)]:
            result=linprog(sign*weights[which]*axis,A_ub=csr_matrix(wlow[None,:]),b_ub=[threshold],A_eq=Aeq,b_eq=np.ones(len(groups)),bounds=(0,None),method='highs');assert result.success,result.message;endpoints.append(float(sign*result.fun));active=np.flatnonzero(result.x>1e-8);solutions.append([dict(cells[i],weight=float(result.x[i])) for i in active])
        witness_costs=np.array([g['cost'] for g in minimum_grid]);witness_wealth=float(sum(g['wealth'] for g in minimum_grid));price=float(sum(w*g['mean_receipt'] for w,g in zip(weights,groups)));report={'model':'one_expected_payoff_learner_across_daily_sku_rounds','cost_delta':args.delta,'seller_family':args.seller_family,'range_scale':args.range_scale,'rival_band':args.rival_band,'fit':args.fit_name,'lifetime_sku_rounds':args.lifetime,'threshold':threshold,'base_threshold':base_threshold,'normalized_payoff_error':args.payoff_error,'proxy_oracle_wealth_allowance':payoff_allowance,'requires_proxy_full_oracle':args.payoff_error>0,'cell_width':args.step,'cost_outer_interval':endpoints,'margin_outer_interval':[price-endpoints[1],price-endpoints[0]],'mean_receipt':price,'prior_cost_interval':[0,float(sum(w*g['prior_cap'] for w,g in zip(weights,groups)))],'groups':groups,'total_rounds':sum(g['rounds'] for g in groups),'low_and_high_solutions':solutions,'grid_minimum_wealth':witness_wealth,'grid_cost_vector':witness_costs.tolist(),'grid_mean_cost':float(weights@witness_costs),'grid_vector_passes_outer_screen':witness_wealth<=threshold,'source_hashes':hashes,'seconds':time.time()-start,'interpretation':'Fitted-demand continuous-cost LP outer bounds conditional on one shared pricing learner across all daily SKU rounds. A stronger cross-product learning restriction than separate SKU learners. No first-stage confidence claim. For positive delta, accepted costs and the grid minimizer are not feasibility witnesses.'}
        stem=args.tag or f'daily_multisku_d{args.delta:g}_h{args.step:g}_T{args.lifetime}'
        (OUT/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['groups','low_and_high_solutions','source_hashes']},indent=2),flush=True)
    print('DAILY_MULTISKU_PROFILE_COMPLETE',flush=True)

if __name__=='__main__':main()
