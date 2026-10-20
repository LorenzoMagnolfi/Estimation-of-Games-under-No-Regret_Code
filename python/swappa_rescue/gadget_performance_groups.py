"""Separate product-level performance from cost homogeneity at fixed demand."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from ask_price_grid import recover_receipts
from audit_panel import ROOT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_performance_frontier import solve

DESIGN=ROOT/'docs/evidence/gadgetpickup_performance_group_design_20261020.json'


def main():
    design=json.loads(DESIGN.read_text());p=RUN/'performance_predictions.parquet'
    s=pd.read_parquet(p);s=s[s.date.ge('2023-11-01')].copy()
    d=terminal_status_data();s=s.merge(d[['Code','date','device_id']],on=['Code','date'],validate='one_to_one')
    x=recover_receipts(s.Price.to_numpy());q=s.q.to_numpy();y=s.sale_terminal_status.to_numpy();shifts=design['comparison_family']
    R0=float(x@q);Q0=q.sum();results=[]
    for scale in design['response_scales']:
        qc=np.column_stack([expit(logit(q)+scale*(logit(s['q_shift_'+str(k)].to_numpy())-logit(q))) for k in shifts])
        xc=s[['receipt_shift_'+str(k) for k in shifts]].to_numpy()
        for spec in design['specifications']:
            labels,ix=np.unique(s[spec['cost_group']],return_inverse=True);J=len(labels)
            mass=np.bincount(ix,weights=q,minlength=J);w=mass/Q0;wy=np.bincount(ix,weights=y,minlength=J)/y.sum()
            def masks(key):
                return [np.ones(len(s),bool)] if key is None else [s[key].eq(g).to_numpy() for g in sorted(s[key].unique())]
            pm=masks(spec['participation_group']);gm=masks(spec['performance_group'])
            for rho in design['rho_grid']:
                aa=[];bb=[]
                for m in pm:
                    aa.append(-float(x[m]@q[m]));bb.append(-np.bincount(ix[m],weights=q[m],minlength=J))
                for m in gm:
                    for k in range(len(shifts)):
                        aa.append(float((1-rho)*(xc[m,k]@qc[m,k])-x[m]@q[m]))
                        bb.append(np.bincount(ix[m],weights=(1-rho)*qc[m,k]-q[m],minlength=J))
                a=np.array(aa);b=np.array(bb)
                for target,weights in [('expected',w),('realized',wy)]:
                    lo=solve(weights,a,b,1);hi=solve(weights,a,b,-1)
                    results.append(dict(specification=spec['name'],response_scale=scale,rho=rho,target=target,cost_parameters=J,participation_groups=len(pm),performance_groups=len(gm),lower=lo,upper=hi))
            print('GROUP COMPLETE',scale,spec['name'],flush=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),expected_mean_receipt=R0/Q0,realized_mean_receipt=float(x@y/y.sum()),profiles=results,interpretation=design['qualification'])
    for p in [RUN/'performance_groups.json',ROOT/'docs/evidence/gadgetpickup_performance_group_results_20261020.json']:
        p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PERFORMANCE_GROUPS_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
