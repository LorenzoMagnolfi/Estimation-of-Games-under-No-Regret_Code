"""Continuous mean-cost outer screens under bounded listing heterogeneity.

These are fitted-demand outer sets. No first-stage confidence claim is made.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_heterogeneous_cost import DailyHeterogeneousCostModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--delta',type=float,nargs='+',default=[0,2,5,10,20])
    parser.add_argument('--lifetime',type=int,default=1500)
    parser.add_argument('--tolerance',type=float,default=1e-4)
    args=parser.parse_args();start=time.time()
    s=terminal_status_data()
    s=s[s.Seller.eq('UPGRADE SOLUTION INC')&s.device_id.eq('apple-iphone-se-2nd-gen')&s.Storage.eq('64 GB')&s.Condition.eq('Good')].sort_values(['Code','date']).reset_index(drop=True)
    fitpath=OUT/'multiproduct_initial_terminal.joblib';covpath=OUT/'demand_cluster_initial_covariance.npz'
    fit=joblib.load(fitpath);cov=np.load(covpath)['covariance']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_heterogeneous_cost.py'),Path(__file__).with_name('daily_cost_model.py'),Path(__file__).with_name('daily_sku_bounds.py'),fitpath,covpath]}
    threshold=float(np.log(4*27)+np.log(args.lifetime)+np.log(1/.00025));records=[]
    with threadpool_limits(limits=2):
        for delta in args.delta:
            model=DailyHeterogeneousCostModel(s,fit,cov,delta)
            cert=model.certify(threshold,args.tolerance)
            survivors=[r for r in cert['cells'] if r['status']!='rejected']
            interval=[min(r['low'] for r in survivors),max(r['high'] for r in survivors)] if survivors else None
            row={'delta':delta,'lifetime_daily_rounds':args.lifetime,'point_cost_outer_interval':interval,'point_margin_outer_interval':[model.mean_receipt-interval[1],model.mean_receipt-interval[0]] if interval else None,'prior_cost_interval':[0,model.cap],'threshold':threshold,'runs':cert['runs'],'roots':cert['roots'],'regular':cert['regular_two_boundary_set'],'mean_receipt':model.mean_receipt,'seconds':time.time()-start}
            records.append(row)
            (OUT/f'daily_heterogeneous_delta{delta:g}_T{args.lifetime}_cells.json').write_text(json.dumps(cert,indent=2))
            print(json.dumps(row),flush=True)
    (OUT/f'daily_heterogeneous_T{args.lifetime}.json').write_text(json.dumps({'results':records,'source_hashes':hashes,'interpretation':'Fitted-demand outer bounds under persistent listing costs within delta of the empirical sale-weighted mean. Costwise gain minimization relaxes the mean constraint. Surviving values are not existence certificates or confidence intervals.'},indent=2))
    print('DAILY_HETEROGENEITY_COMPLETE',flush=True)

if __name__=='__main__':main()
