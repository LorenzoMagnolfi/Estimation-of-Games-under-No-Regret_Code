"""Independent persistence checks, bound attribution, and diagnostic figure."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel import ROOT
from gadget_demand_audit import RUN


def main():
    run_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    review=json.loads((ROOT/'docs/evidence/gadgetpickup_persistent_review_results_20261020.json').read_text());event=json.loads((ROOT/'docs/evidence/gadgetpickup_batch_demand_results_20261020.json').read_text());seller=json.loads((ROOT/'docs/evidence/gadgetpickup_seller_aggregate_review_results_20261020.json').read_text())
    dates=pd.to_datetime(event['batch_dates']);pre=dates[dates<pd.Timestamp('2023-11-01')];post=dates[dates>=pd.Timestamp('2023-11-01')]
    cadence={name:dict(intervals=len(d)-1,mean=float(np.mean(np.diff(d.to_numpy()).astype('timedelta64[D]').astype(int))),median=float(np.median(np.diff(d.to_numpy()).astype('timedelta64[D]').astype(int)))) for name,d in [('preperiod',pre),('novdec',post)]}
    q=.2;H=7;P=sum(q*(1-q)**h for h in range(H));assert abs(P-(1-(1-q)**H))<1e-14
    assert abs(sum((100.-63.)*q*(1-q)**h for h in range(H))-37.*P)<1e-12
    g=pd.read_parquet(RUN/'persistent_review_predictions.parquet');g=g[g.spec.eq('cached_corrected_additive')&g.date.ge('2023-11-01')].copy();g=g[g.groupby(['device_id','date']).valid.transform('all')&g.date.isin(dates)].reset_index(drop=True)
    ix=pd.factorize(g.Code,sort=True)[0];J=ix.max()+1;models=sorted(g.device_id.unique());M=len(models);changed=g.groupby(['device_id','date']).changed.transform('any').to_numpy();days=g[['device_id','date']].copy();days['changed']=changed;days=days.drop_duplicates();C=days[days.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy();I=days[~days.changed].groupby('device_id').size().reindex(models,fill_value=0).to_numpy()
    hs=review['gap_distribution']
    def prob(x):return np.mean([1-(1-np.asarray(x))**h for h in hs],axis=0)
    q=prob(g.q);qo=prob(g.qold);x=g.x.to_numpy();xo=g.undo_all_receipt.to_numpy();mass=np.bincount(ix,weights=q,minlength=J);w=mass/mass.sum();A=[np.r_[mass,np.zeros(M)]];b=[float(x@q)];labels=['participation']
    m=changed;A.append(np.r_[np.bincount(ix[m],weights=q[m]-qo[m],minlength=J),C]);b.append(float((x[m]*q[m]-xo[m]*qo[m]).sum()));labels.append('implemented_gain')
    for on_change in [False,True]:
        m=changed if on_change else ~changed
        for shift in [-10,-5,5,10]:
            qa=prob(g['q_shift_'+str(shift)]);xa=g['receipt_shift_'+str(shift)].to_numpy();A.append(np.r_[np.bincount(ix[m],weights=q[m]-qa[m],minlength=J),np.zeros(M) if on_change else -I]);b.append(float((x[m]*q[m]-xa[m]*qa[m]).sum()));labels.append(('change:' if on_change else 'inaction:')+str(shift))
    A=np.array(A);b=np.array(b);obj=np.r_[w,np.zeros(M)];attribution=[]
    for name,keep in [('full',list(range(10))),('participation_and_changed_menu',[0,6,7,8,9]),('participation_rollback_changed_menu',[0,1,6,7,8,9])]:
        sol=linprog(obj,A_ub=csr_matrix(A[keep]),b_ub=b[keep],bounds=(0,None),method='highs-ds');assert sol.success
        hi=linprog(-obj,A_ub=csr_matrix(A[keep]),b_ub=b[keep],bounds=(0,None),method='highs-ds');assert hi.success
        attribution.append(dict(restrictions=name,cost_lower=float(sol.fun),cost_upper=float(-hi.fun),dual_nonzero=[dict(row=labels[j],multiplier=float(v)) for j,v in zip(keep,sol.ineqlin.marginals) if abs(v)>1e-9],max_primal_violation=float(max(0.,np.max(A[keep]@sol.x-b[keep])))))
    expected=next(a for a in seller['results'] if a['spec']=='cached_corrected_additive' and a['horizon']=='preperiod_gap_distribution' and a['complete_menu'])
    assert abs(attribution[0]['cost_lower']-expected['cost_lower'])<1e-7
    stacks=[]
    for window in ['full','novdec']:
        d=[a for a in event['stack_details'] if window=='full' or a['anchor']>='2023-11-01']
        stacks.append(dict(window=window,anchors=len(d),missing_category_days=sum(a['missing_category_days'] for a in d),potential_category_days=sum(a['potential_category_days'] for a in d),treated_category_episodes=sum(a['treated_categories'] for a in d),control_category_episodes=sum(a['control_categories'] for a in d),treated_episodes_repriced_within_week=sum(a['treated_categories_with_subsequent_changes'] for a in d),control_episodes_repriced_within_week=sum(a['control_categories_with_subsequent_changes'] for a in d)))
    with plt.rc_context({'text.usetex':False,'font.family':'DejaVu Sans','font.size':11}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,outcome,factor,label in [(axes[0],'ask',1.,'Change in mean asking price ($)'),(axes[1],'rate',100.,'Change in daily sale rate (percentage points)')]:
            rec=next(a for a in event['events'] if a['window']=='full' and a['outcome']==outcome)
            xx=np.arange(-7,8);yy=[];ee=[]
            for k in xx:
                v=rec['coefficients'].get(str(k),{'estimate':0.,'se':0.});yy.append(v['estimate']*factor);ee.append(v['se']*factor*1.96 if v['se'] is not None else np.nan)
            ax.axhline(0,color='#777777',lw=.8);ax.axvline(-.5,color='#999999',ls=':',lw=1);ax.errorbar(xx,yy,yerr=ee,color='#164B73',fmt='o-',lw=1.4,ms=4,capsize=2);ax.set_xlabel('Days relative to the coordinated markdown');ax.set_ylabel(label);ax.set_xticks([-7,-5,-3,-1,1,3,5,7]);ax.grid(axis='y',alpha=.15)
        axes[0].set_title('Prices remain lower after the markdown');axes[1].set_title('The sales response is less precise')
        fig.suptitle('Coordinated markdowns: comparison to the day before the cut',fontsize=14)
        fig.text(.5,.025,'29 batch dates; category and date effects within each event. Descriptive 95% clustered intervals; no causal claim.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(RUN/'persistent_review_demand_diagnostics.png',dpi=170);fig.savefig(RUN/'persistent_review_demand_diagnostics.pdf');plt.close(fig)
    out=dict(code_commit=run_commit,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cadence=cadence,probability_fixture=dict(q=.2,H=7,sale_probability=P,incorrect_linear_count=1.4),bound_attribution=attribution,event_support=stacks,interpretation='Independent primary-model LP reconstruction and attribution. Conditional lower bound source distinguished from empirically restricted adjustment allowance. Figure intervals are descriptive.')
    (ROOT/'docs/evidence/gadgetpickup_persistence_audit_summary_20261020.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
