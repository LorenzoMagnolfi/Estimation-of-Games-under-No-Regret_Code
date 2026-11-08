"""Independent design-matrix and numerical checks for the demand repair pilot."""
import json
import hashlib
from pathlib import Path
import subprocess
import types

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from demand_repair_pilot_20261106 import ROOT, OUT, continuous, design


def main():
    paper=ROOT.parent/'Estimation-of-Games-under-No-Regret'
    rev='16be52968a0776f0aec086a2bbeabf5b8f6abfcb'
    source=subprocess.check_output(['git','show',rev+':JPE Revision/swappa-revival-2026-11-03/code/demand.py'],cwd=paper,text=True)
    original=types.ModuleType('original_revival_demand')
    exec(compile(source,'revival_demand.py','exec'),original.__dict__)
    s=pd.read_parquet(OUT/'pilot_panel.parquet').reset_index(drop=True)
    z=np.load(OUT/'d0_revival_like_coef.npz')
    b=z['beta']; nd=len(z['names'])
    ss=s.rename(columns={'ref':'ref_fill','ref_external':'ref_from_decluttr','initial_res':'init_res'})
    with threadpool_limits(limits=4):
        xo,names,_=original.build_design(ss)
        xp,_,mean,sd=design(s)
        bo=b.copy();bo[0]-=np.dot(b[1:nd+1],mean/sd);bo[1:nd+1]/=sd
        discrepancy=float(np.max(np.abs(xo@bo-xp@b)))
        assert discrepancy<1e-8,discrepancy
        eta=xo@bo;q=1/(1+np.exp(-eta))
        penalty=np.r_[np.zeros(nd+1),np.ones(len(b)-nd-1)]
        score=float(np.max(np.abs(xo.T@(q-s.sale.to_numpy())+penalty*bo)))
        assert score<1e-5,score
        a,_=continuous(s,rich=True);a0,_=continuous(s,rich=True,shift=0)
        assert np.array_equal(a,a0)
        records=json.loads((OUT/'all_results.json').read_text())+json.loads((OUT/'raw_reference_results.json').read_text())
        assert len(records)==11
        assert all(r['convergence']['score_max']<1e-5 and r['holdout_convergence']['score_max']<1e-5 for r in records)
        current=next(r for r in records if r['tag']=='d4_current_matched')
        prior=next(r for r in records if r['tag']=='d5_previous_matched')
        assert current['rows']==prior['rows'] and current['sales']==prior['sales']
        # Local signs for joint seller price changes, accounting for owned-peer changes.
        shapes={}
        for tag,kw in [('d0_revival_like',{}),('d1_rich_competition',dict(rich=True)),('d6_flexible_gap',dict(rich=True,flexible=True)),('raw_d6',dict(rich=True,flexible=True))]:
            active=pd.read_parquet(OUT/'raw_reference_panel.parquet') if tag.startswith('raw_') else s
            saved=np.load(OUT/(tag+'_coef.npz'))
            x0,_=continuous(active,**kw);x1,_=continuous(active,shift=1e-5,**kw)
            deriv=((x1-x0)/1e-5/saved['sd'])@saved['beta'][1:len(saved['sd'])+1]
            shapes[tag]=dict(positive_logodds_response_share=float((deriv>1e-7).mean()),derivative_quantiles=np.quantile(deriv,[0,.1,.5,.9,1]).tolist())
    report=dict(status='passed',independent_source=rev+':JPE Revision/swappa-revival-2026-11-03/code/demand.py',design_predictor_max_difference=discrepancy,original_design_score_max=score,zero_shift_identity=True,separate_listing_holdout=True,all_22_fits_converged=True,matched_timing_same_rows_and_sales=True,shape_checks=shapes,limitation='Numerical checks do not establish causal identification, correct observation process, or confidence coverage.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    evidence=dict(status='exploratory demand diagnostics, no cost inversion or confidence intervals',verification=report,risk_set=json.loads((OUT/'risk_set_audit.json').read_text()),reference=json.loads((OUT/'reference_reconstruction_audit.json').read_text()),models=records,script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*20261106.py')},source_manifest=json.loads((OUT/'manifest.json').read_text()))
    evidence['risk_set']['sample_matches_revival']=bool(evidence['risk_set']['risk_rows']==457189 and evidence['risk_set']['sales']==22792)
    (ROOT/'docs/evidence/swappa_demand_repair_results_20261106.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
