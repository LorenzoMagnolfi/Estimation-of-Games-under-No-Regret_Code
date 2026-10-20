"""Attribute the frozen LP results to individual comparisons and target weights."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from empirical_pricing_rules import RUN,DESIGN,ROOT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts
from policy_cost_frontier import endpoint

def main():
    spec=json.loads(DESIGN.read_text());data=terminal_status_data();out=[]
    for seller,stem in [('GadgetPickup','gp'),('UPGRADE SOLUTION INC','up')]:
        s=data[data.Seller.eq(seller)&data.date.ge(spec['test_start'])].copy()
        f=pd.read_parquet(RUN/(seller.replace(' ','_').replace(',','').replace('.','')+'_forecasts.parquet'))
        s=s.merge(f[f.date.ge(spec['test_start'])][['Code','date','eligible']],on=['Code','date'],validate='one_to_one').reset_index(drop=True)
        codes,idx=np.unique(s.Code,return_inverse=True);x=recover_receipts(s.Price.to_numpy());N=int(s.eligible.sum())
        for fit in ['seller','common']:
            z=np.load(RUN/(stem+'_'+fit+'_coef.npz'),allow_pickle=True);q=z['q_observed'];mass=np.bincount(idx,weights=q,minlength=len(codes));v=mass/mass.sum();aw=-float(x@q/N);bw=-mass/N
            keep=np.array([not str(p).startswith('ask_shift_') for p in z['policies']]);A=np.r_[z['A'][keep],aw];B=np.vstack([z['B'][keep],bw]);bounds=[(0,None)]*len(codes)
            rr=[dict(policy=str(p),zero_cost_gain=float(a),cost_loading_max_ratio=float(np.max(b/v)),single_inequality_cost_lower=max(0,float(a/np.max(b/v))) if np.max(b/v)>0 else None) for p,a,b in zip(z['policies'],z['A'],z['B'])]
            out.append(dict(seller=seller,fit=fit,all_retained_terminal_sales=int(s.sale_terminal_status.sum()),expected_sales=float(q.sum()),expected_receipt=float(x@q/q.sum()),actual_all_retained_mean_receipt=float(x@s.sale_terminal_status/s.sale_terminal_status.sum()),fitted_rules_plus_withdrawal_zero_budget_lower=endpoint(v,A,B,0,bounds,1),policy_certificate=rr))
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=out)
    (ROOT/'docs/evidence/gadgetpickup_cost_attribution_20261020.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ATTRIBUTION_COMPLETE')

if __name__=='__main__':main()
