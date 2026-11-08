"""Check all cost values when flexible-demand first-order roots leave the initial grid."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from threadpoolctl import threadpool_limits
from demand_repair_pilot_20261106 import OUT, continuous, design

with threadpool_limits(limits=4):
    file=OUT/'downstream_results.json'; result=json.loads(file.read_text())
    m=next(m for m in result['results'] if m['model']=='raw_d6')
    s=pd.read_parquet(OUT/'raw_reference_panel.parquet'); kw=dict(rich=True,flexible=True)
    X,names,mean,sd=design(s,**kw); beta=np.load(OUT/'raw_d6_coef.npz')['beta']; nd=len(names)+1
    for seller,row in m['sellers'].items():
        ix=s.Seller.eq(seller).to_numpy(); g=s.loc[ix].copy(); z0=X[ix]@beta; base,_=continuous(g,**kw)
        price=g.Price.to_numpy(); ref=g.ref.to_numpy()
        def value(own,rival):
            gg=g.copy()
            for col in ['rival_min','rival_mean','rival_median']: gg[col]=g[col]*rival
            z,_=continuous(gg,shift=own-1,**kw); q=expit(z0+((z-base)/sd)@beta[1:nd])
            return np.array([np.sum(price*own*q),np.sum(ref*q),q.sum()])
        th=np.round(np.arange(row['theta_provisional'][0],row['theta_provisional'][1]+.001,.0025),4)
        for mode,sc in row['scenarios'].items():
            cache={}
            def difference(k):
                if k not in cache:
                    sr=1 if mode=='fixed' else 1+k
                    cache[k]=value((1+k)*1.01,sr)-value((1+k)*.99,sr)
                return cache[k]
            grid=np.arange(-.8,2.0001,.01); dif=np.array([difference(float(k)) for k in grid]); roots={}; counts={}
            for fee in [0.,.03]:
                roots[fee]=[]; counts[fee]=[]
                for theta in th:
                    der=(1-fee)*dif[:,0]-theta*dif[:,1]
                    cross=np.flatnonzero((der[:-1]>0)&(der[1:]<=0)); counts[fee].append(len(cross))
                    rr=[brentq(lambda k:(1-fee)*difference(k)[0]-theta*difference(k)[1],grid[j],grid[j+1],xtol=1e-9) for j in cross]
                    # Preserve original first-positive-to-negative-root convention; flag multiplicity.
                    roots[fee].append(rr[0] if rr else np.nan)
            shift=np.array(roots[.03])-roots[0.]; valid=np.isfinite(shift)
            def span(a): return [float(np.min(a)),float(np.max(a))] if len(a) else None
            a=sc['0.03']; a['initial_grid_optimal_level_shift']=a['optimal_level_shift']
            a['root_extension']=dict(search=[-.8,2.,.01],theta=th.tolist(),roots_no_fee=roots[0.],roots_fee=roots[.03],counts_no_fee=counts[0.],counts_fee=counts[.03])
            a['optimal_level_shift']=span(shift[valid]); a['buyer_share']=span(shift[valid]/(.03*(1+shift[valid])))
            a['missing_roots_in_cost_set']=int((~valid).sum())
            sales0=value(1,1)[2]; a['sales_change']=span(np.array([value(1+k,1 if mode=='fixed' else 1+k)[2]/sales0-1 for k in shift[valid]]))
            print(seller,mode,a['optimal_level_shift'],'missing',a['missing_roots_in_cost_set'],'multiple',max(counts[0.]+counts[.03]),flush=True)
    result['meta']['root_extension_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    file.write_text(json.dumps(result,indent=2))
