"""Independent profit-identity and feasibility checks for the allowance audit."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from audit_panel import ROOT
from gadget_demand_audit import RUN
from ask_price_grid import recover_receipts


def main():
    result=json.loads((ROOT/'docs/evidence/gadgetpickup_adjustment_cost_results_20261020.json').read_text())
    p=pd.read_parquet(RUN/'performance_predictions.parquet')
    r=pd.read_parquet(RUN/'repricing_performance_predictions.parquet')
    s=p.merge(r[['Code','date','device_id','previous_date','previous_price','undo_all_q','undo_all_receipt']],on=['Code','date'],validate='one_to_one')
    s=s[s.date.ge('2023-11-01')].copy()
    s['valid']=(s.date-s.previous_date).dt.days.eq(1)&s.previous_date.ne(pd.Timestamp('2023-11-19'))
    s['x']=recover_receipts(s.Price.to_numpy())
    dates=set(pd.to_datetime(result['batch_dates']))
    reference={a['model']:a for a in result['models'] if a['window']=='novdec' and a['schedule']=='batch_dates' and not a['complete_menu']}
    checks=[];largest_error=0.
    for model,g in s.groupby('device_id'):
        codes=sorted(g.Code.unique());lookup={code:i for i,code in enumerate(codes)};J=len(codes)
        mass=np.array([g.loc[g.Code.eq(code),'q'].sum() for code in codes]);revenue=float((g.x*g.q).sum())
        aa=[np.r_[mass,0.]];bb=[revenue];labels=['participation']
        test_c=np.linspace(37.,413.,J);test_K=2.31
        for date,z in g.groupby('date'):
            if date not in dates or not z.valid.all():continue
            indexes=z.Code.map(lookup).to_numpy();q=z.q.to_numpy();x=z.x.to_numpy()
            actual=float((q*(x-test_c[indexes])).sum())
            if z.Price.ne(z.previous_price).any():
                qalt=z.undo_all_q.to_numpy();xalt=z.undo_all_receipt.to_numpy()
                coeff=np.zeros(J)
                for i,dd in zip(indexes,q-qalt):coeff[i]+=dd
                row=np.r_[coeff,1.];rhs=float((q*x-qalt*xalt).sum())
                direct=test_K-actual+float((qalt*(xalt-test_c[indexes])).sum())
                largest_error=max(largest_error,abs(row@np.r_[test_c,test_K]-rhs-direct))
                aa.append(row);bb.append(rhs);labels.append('change:'+str(date.date()))
            else:
                for shift in [-10,-5,5,10]:
                    qalt=z['q_shift_'+str(shift)].to_numpy();xalt=z['receipt_shift_'+str(shift)].to_numpy()
                    coeff=np.zeros(J)
                    for i,dd in zip(indexes,q-qalt):coeff[i]+=dd
                    row=np.r_[coeff,-1.];rhs=float((q*x-qalt*xalt).sum())
                    direct=float((qalt*(xalt-test_c[indexes])).sum())-actual-test_K
                    largest_error=max(largest_error,abs(row@np.r_[test_c,test_K]-rhs-direct))
                    aa.append(row);bb.append(rhs);labels.append('inaction:'+str(date.date())+':'+str(shift))
        A=np.array(aa);b=np.array(bb);sol=linprog(np.zeros(J+1),A_ub=csr_matrix(A),b_ub=b,bounds=(0,None),method='highs-ds')
        assert sol.status in [0,2],sol.message
        assert (sol.status==0)==reference[model]['feasible']
        rec=dict(model=str(model),feasible=sol.status==0,independent_solver_status=int(sol.status))
        if sol.success:rec['maximum_primal_violation']=float(max(0.,np.max(A@sol.x-b)))
        else:
            subset=[labels.index(x) for x in reference[model]['infeasible_subset']]
            AA=A[subset];bbb=b[subset];scale=np.maximum(1.,np.maximum(abs(bbb),np.max(abs(AA),axis=1)))
            AA=AA/scale[:,None];bbb=bbb/scale
            dual=linprog(bbb,A_ub=-AA.T,b_ub=np.zeros(J+1),A_eq=np.ones((1,len(bbb))),b_eq=[1.],bounds=(0,None),method='highs-ds')
            assert dual.success and dual.fun < -1e-8 and np.min(AA.T@dual.x)>-1e-7
            rec['certificate_margin']=float(-dual.fun)
            rec['certificate_rows']=reference[model]['infeasible_subset']
            rec['certificate_weights']=dual.x.tolist()
            # Diagnostic attribution only: not a new specification selected for cost reporting.
            nopart=linprog(np.zeros(J+1),A_ub=csr_matrix(A[1:]),b_ub=b[1:],bounds=(0,None),method='highs-ds')
            assert nopart.status in [0,2],nopart.message
            rec['infeasible_without_participation']=nopart.status==2
        checks.append(rec)
    assert largest_error<1e-9
    payload=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profit_identity_max_error=largest_error,independent_model_checks=checks,interpretation='Numerical verification at fixed estimated demand, not statistical rejection. Dropping participation is a diagnostic attribution check only.')
    (ROOT/'docs/evidence/gadgetpickup_allowance_audit_checks_20261020.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
