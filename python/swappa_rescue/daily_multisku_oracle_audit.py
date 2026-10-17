"""Independent full-joint oracle audit of a constant-SKU-cost event witness."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel
from daily_joint_oracle import group_profit_upper


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='daily_multisku_common_d0_h02.json');ap.add_argument('--mean-step',type=float,default=5.);ap.add_argument('--tag');args=ap.parse_args();start=time.time();profilepath=OUT/args.profile;profile=json.loads(profilepath.read_text());assert profile['cost_delta']==0
    s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fitpath=OUT/profile['fit'];fit=joblib.load(fitpath);cov=np.zeros((len(fit.model.coef_[0])+1,)*2);records=[];costrows=[];totalwealth=0
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('daily_joint_oracle.py'),Path(__file__).with_name('daily_cost_model.py'),profilepath,fitpath]}
    with threadpool_limits(limits=2):
        for k,(key,sub) in enumerate(s.groupby(['device_id','Storage','Condition'],sort=True)):
            model=DailyCostModel(sub,fit,cov);cost=float(profile['grid_cost_vector'][k]);assert 0<=cost<=model.cap;c=np.full(model.N,cost);C=model.ranges([cost])[:,0];wealth=0.;actual=expit(model.z)*(model.x-c)
            for ids in model.groups:
                day=model.day[ids[0]]
                if not model.live[day]:continue
                upper=group_profit_upper(c[ids],model.p[ids],model.z[ids],model.b[ids],model.gp,model.gn,model.rival[ids],model.bp,model.active[ids],mean_step=args.mean_step,P=model.P)
                gain=max(0.,upper-actual[ids].sum());ratio=min(1.,gain/C[day]);wealth+=float(np.log1p(.99*ratio))
            common=float(model.values([cost])[0]);assert wealth>=common-1e-7;totalwealth+=wealth
            records.append({'key':list(key),'cost':cost,'common_advisor_wealth':common,'full_oracle_wealth_upper':wealth,'rounds':int(model.live.sum())})
            costrows.extend({'Code':code,'cost':cost} for code in sub.Code.unique());print(json.dumps({'stage':'sku','sku':k,'record':records[-1],'cumulative_wealth_upper':totalwealth,'seconds':time.time()-start}),flush=True)
    stem=args.tag or Path(args.profile).stem+f'_oracle_h{args.mean_step:g}';costpath=OUT/f'{stem}_costs.csv';pd.DataFrame(costrows).to_csv(costpath,index=False)
    result={'profile':args.profile,'full_oracle_wealth_upper':totalwealth,'threshold':profile['threshold'],'passes_full_oracle_event':bool(totalwealth<=profile['threshold']),'witness_mean_cost':profile['grid_mean_cost'],'witness_mean_margin':profile['mean_receipt']-profile['grid_mean_cost'],'mean_step':args.mean_step,'sku_audits':records,'cost_file':costpath.name,'cost_file_sha256':hashlib.sha256(costpath.read_bytes()).hexdigest(),'source_hashes':hashes,'seconds':time.time()-start,'interpretation':'A specific persistent cost vector passes the fitted-demand full-joint advisor event only if its certified wealth upper bound is below threshold. A witness, not an estimate, uncertainty interval, or proof of reproduction by a named algorithm.'};(OUT/f'{stem}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    print('DAILY_MULTISKU_ORACLE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
