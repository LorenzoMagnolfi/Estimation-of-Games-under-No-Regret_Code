"""Independent derivative and result checks for the downstream robustness run."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from demand_repair_pilot_20261106 import OUT, FOCAL, continuous, design

with threadpool_limits(limits=4):
    s=pd.read_parquet(OUT/'raw_reference_panel.parquet')
    checks=[]
    rng=np.random.default_rng(61106)
    for tag,kw in [('raw_d0',{}),('raw_d1',dict(rich=True)),('raw_d6',dict(rich=True,flexible=True))]:
        X,names,mean,sd=design(s,**kw); beta=np.load(OUT/(tag+'_coef.npz'))['beta']; nd=len(names)+1
        for seller in FOCAL:
            ix=s.Seller.eq(seller).to_numpy(); g=s.loc[ix].copy(); xx=X[ix].toarray()
            base,_=continuous(g,**kw)
            for col in ['rival_min','rival_mean','rival_median']: g[col]*=1.03
            alt,_=continuous(g,shift=.047,**kw); xx[:,1:nd]+=(alt-base)/sd
            q=expit(xx@beta); payoff=g.Price.to_numpy()*1.047*.97-.8*g.ref.to_numpy()
            grad=xx.T@(payoff*q*(1-q))
            v=rng.normal(size=len(beta)); v/=np.linalg.norm(v); h=1e-5
            fd=(payoff@expit(xx@(beta+h*v))-payoff@expit(xx@(beta-h*v)))/(2*h)
            analytic=float(grad@v)
            assert np.isclose(fd,analytic,rtol=1e-6,atol=1e-4)
            checks.append(dict(model=tag,seller=seller,finite_difference=fd,analytic=analytic,error=abs(fd-analytic)))
    result=json.loads((OUT/'downstream_results.json').read_text())
    boundary=[]
    for m in result['results']:
        for seller,r in m['sellers'].items():
            assert r['direct_profit_error']<1e-7
            assert .2<r['theta_provisional'][0]<=r['theta_provisional'][1]<1.2
            for mode,sc in r['scenarios'].items():
                for f,a in sc.items():
                    if a['boundary_hit']: boundary.append([m['model'],seller,mode,f])
                    if f=='0.03': assert a['missing_roots_in_cost_set']==0
    out=dict(passed=True,derivatives=checks,grid_boundary_censoring=boundary,scope='Numerical validation, not statistical coverage validation')
    (OUT/'downstream_checks.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
