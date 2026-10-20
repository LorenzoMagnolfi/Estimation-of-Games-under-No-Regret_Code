"""Invert the performance frontier in the other direction; independent checks."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.special import expit,logit
from scipy.sparse import csr_matrix
from ask_price_grid import recover_receipts
from audit_panel import ROOT
from demand_timing_sensitivity import terminal_status_data
from gadget_demand_audit import RUN
from gadget_performance_frontier import solve


def main():
    # Analytic two-product example, not a fitted-data check.
    q=np.array([.2,.4]);qd=np.array([.3,.5]);x=np.array([100.,200.]);xd=np.array([90.,190.]);rho=.1
    a=np.array([-x@q,(1-rho)*(xd@qd)-x@q]);b=np.vstack([-q,(1-rho)*qd-q]);w=q/q.sum()
    lo=solve(w,a,b,1);assert abs(lo['value']-46.6666666666667)<1e-8
    for c in [np.zeros(2),np.array([140.,0.]),np.array([50.,180.])]:
        pi0=(x-c)@q;pid=(xd-c)@qd
        assert abs((a-b@c)[1]-((1-rho)*pid-pi0))<1e-10
    s=pd.read_parquet(RUN/'performance_predictions.parquet');s=s[s.date.ge('2023-11-01')].copy()
    d=terminal_status_data();s=s.merge(d[['Code','date','device_id']],on=['Code','date'],validate='one_to_one')
    ix=pd.factorize(s.Code,sort=True)[0];J=ix.max()+1;x=recover_receipts(s.Price.to_numpy());q=s.q.to_numpy();w=np.bincount(ix,weights=q,minlength=J)/q.sum();shifts=[-10,-5,5,10]
    xc=s[['receipt_shift_'+str(k) for k in shifts]].to_numpy();answers=[]
    for scale in [.5,.75,1.]:
        qc=np.column_stack([expit(logit(q)+scale*(logit(s['q_shift_'+str(k)].to_numpy())-logit(q))) for k in shifts])
        for level in ['pooled','device','category']:
            key={'device':'device_id','category':'category'}.get(level)
            masks=[np.ones(len(s),bool)] if key is None else [s[key].eq(g).to_numpy() for g in sorted(s[key].unique())]
            R=[];Q=[];Rc=[];Qc=[]
            for m in masks:
                R.append(float(x[m]@q[m]));Q.append(np.bincount(ix[m],weights=q[m],minlength=J))
                Rc.append((xc[m]*qc[m]).sum(axis=0));Qc.append(np.column_stack([np.bincount(ix[m],weights=qc[m,k],minlength=J) for k in range(4)]).T)
            R=np.array(R);Q=np.array(Q);Rc=np.array(Rc);Qc=np.array(Qc)
            zero_threshold=max(0.,float(np.max(1-R/np.max(Rc,axis=1))))
            for ceiling in [0.,50.,100.,150.,200.]:
                def feasible(rho):
                    a=np.r_[-R,((1-rho)*Rc-R[:,None]).ravel()]
                    b=np.vstack([-Q,((1-rho)*Qc-Q[:,None,:]).reshape(-1,J)])
                    res=linprog(np.zeros(J),A_ub=csr_matrix(np.vstack([-b,w])),b_ub=np.r_[-a,ceiling],bounds=(0,None),method='highs')
                    assert res.status in [0,2],res.message
                    return res.status==0
                if feasible(0.):crit=0.
                else:
                    low=0.;high=1.;assert feasible(high)
                    for _ in range(38):
                        mid=(low+high)/2
                        if feasible(mid):high=mid
                        else:low=mid
                    crit=high
                if ceiling==0:assert abs(crit-zero_threshold)<1e-6
                answers.append(dict(response_scale=scale,level=level,mean_cost_ceiling=ceiling,minimum_shortfall_allowance=crit,interpretation='Mean cost at or below ceiling is excluded when rho is strictly below this threshold.'))
            print('THRESHOLDS',scale,level,flush=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256((RUN/'performance_predictions.parquet').read_bytes()).hexdigest(),checks=['Analytic two-product endpoint','Direct profit versus constraint sign','Numerical zero-cost thresholds equal analytic revenue ratios'],thresholds=answers)
    (ROOT/'docs/evidence/gadgetpickup_performance_thresholds_20261020.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('THRESHOLDS_COMPLETE',flush=True)


if __name__=='__main__':main()
