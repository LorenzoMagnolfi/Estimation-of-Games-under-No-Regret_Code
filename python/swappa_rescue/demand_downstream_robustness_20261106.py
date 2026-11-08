"""Comparable demand robustness; legacy separate-test inference is provisional."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from demand_repair_pilot_20261106 import OUT, FOCAL, continuous, design

TH = np.round(np.arange(.2, 1.20001, .0025), 4)
KP = np.round(np.arange(-.05, .10001, .0025), 4)
SH = np.array([-.10, -.05, -.025, -.01, -.005, .005, .01, .025, .05, .10])


def span(x):
    return [float(np.min(x)), float(np.max(x))] if len(x) else None


def hac(a, b):
    a, b = a-a.mean(axis=-1, keepdims=True), b-b.mean(axis=-1, keepdims=True)
    v = np.mean(a*b, axis=-1)
    for h in range(1, 8):
        v += (1-h/8)*(np.sum(a[...,h:]*b[...,:-h], axis=-1)+np.sum(b[...,h:]*a[...,:-h], axis=-1))/a.shape[-1]
    return v


def covariance(X, s, beta, nd):
    q = expit(X@beta)
    lam = np.r_[np.zeros(nd), np.ones(X.shape[1]-nd)]
    H = (X.T@X.multiply((q*(1-q))[:,None])).toarray()+np.diag(lam+1e-9)
    inv = np.linalg.inv(H)
    score = X.multiply((s.sale.to_numpy()-q)[:,None]).tocsr()
    meat = -(score.T@score).toarray()
    for col in ['Code', 'date_str']:
        codes = pd.Categorical(s[col]).codes
        G = sp.csr_matrix((np.ones(len(s)), (codes,np.arange(len(s)))))
        a = (G@score).toarray()
        meat += a.T@a
    V = inv@meat@inv
    w,u = np.linalg.eigh((V+V.T)/2)
    return (u*np.maximum(w,0))@u.T, dict(negative_eigenvalues=int((w<0).sum()), clipped_mass=float(-w[w<0].sum()), score_max=float(np.abs(X.T@(q-s.sale.to_numpy())+lam*beta).max()))


def run(tag, kw):
    s = pd.read_parquet(OUT/'raw_reference_panel.parquet').reset_index(drop=True)
    X,names,mean,sd = design(s, **kw)
    saved = np.load(OUT/(tag+'_coef.npz'))
    assert np.allclose(mean,saved['mean']) and np.allclose(sd,saved['sd'])
    assert list(saved['names']) == names
    beta=saved['beta']; nd=len(names)+1
    V, checks = covariance(X,s,beta,nd)
    result = dict(model=tag, checks=checks, kappa_grid=[float(KP[0]),float(KP[-1]),.0025], sellers={})
    for seller in FOCAL:
        ix=s.Seller.eq(seller).to_numpy(); g=s.loc[ix].reset_index(drop=True)
        XX=X[ix]; z0=XX@beta; dense,_=continuous(g,**kw)
        dc=pd.Categorical(g.date_str).codes; T=int(dc.max()+1)
        D=sp.csr_matrix((np.ones(len(g)),(dc,np.arange(len(g)))),shape=(T,len(g)))
        price=g.Price.to_numpy(); ref=g.ref.to_numpy(); cache={}
        def ev(own, rival=1.):
            key=(round(own,10),round(rival,10))
            if key in cache: return cache[key]
            gg=g.copy()
            for col in ['rival_min','rival_mean','rival_median']: gg[col]=g[col]*rival
            zz,_=continuous(gg,shift=own-1,**kw)
            delta=(zz-dense)/sd
            q=expit(z0+delta@beta[1:nd]); w=q*(1-q)
            a={k:np.asarray(D@v) for k,v in [('R',price*own*q),('CR',ref*q),('S',q)]}
            for key2,v in [('gR',price*own*w),('gCR',ref*w)]:
                gr=np.asarray(XX.T@v); gr[1:nd]+=delta.T@v; a[key2]=gr
            cache[key]=a
            return a
        def moments(base, alts, fee):
            A=(1-fee)*np.array([v['R']-base['R'] for v in alts]); B=np.array([v['CR']-base['CR'] for v in alts])
            GA=(1-fee)*np.array([v['gR']-base['gR'] for v in alts])/T
            GB=np.array([v['gCR']-base['gCR'] for v in alts])/T
            ma=A.mean(1); mb=B.mean(1)
            va=hac(A,A)/T+np.sum((GA@V)*GA,axis=1)
            vb=hac(B,B)/T+np.sum((GB@V)*GB,axis=1)
            ab=hac(A,B)/T+np.sum((GA@V)*GB,axis=1)
            gain=ma[:,None]-mb[:,None]*TH
            var=va[:,None]-2*ab[:,None]*TH+vb[:,None]*TH**2
            assert var.min()>-1e-5
            return gain, gain/np.sqrt(np.maximum(var,1e-300))
        base=ev(1); alt=[ev(1+k) for k in SH]
        gains,t=moments(base,alt,0)
        fit=(gains<=0).all(0); keep=(t<=norm.ppf(1-.05/len(SH))).all(0)
        joint=(t<=norm.ppf(1-.05/(2*len(SH)))).all(0)
        sold=g.sale.eq(1); mp=float(g.loc[sold,'Price'].mean()); mr=float(g.loc[sold,'ref'].mean())
        # Direct row-profit versus aggregated moment identity at a non-grid theta.
        q0=expit(z0); zz,_=continuous(g,shift=.01,**kw)
        q1=expit(z0+((zz-dense)/sd)@beta[1:nd])
        direct=np.sum((price*1.01-.8*ref)*q1-(price-.8*ref)*q0)
        agg=np.sum(ev(1.01)['R']-base['R']-.8*(ev(1.01)['CR']-base['CR']))
        assert np.isclose(direct,agg,atol=1e-7)
        row=dict(days=T,mean_price_sold=mp,mean_ref_sold=mr,theta_fit=span(TH[fit]),theta_provisional=span(TH[keep]),margin_fit=span(1-TH[fit]*mr/mp),margin_provisional=span(1-TH[keep]*mr/mp),cost_provisional=span(TH[keep]*mr),direct_profit_error=float(abs(direct-agg)),scenarios={})
        for mode in ['fixed','follow']:
            bases=[]; alternatives=[]
            for kap in KP:
                r=1 if mode=='fixed' else 1+kap
                bases.append(ev(1+kap,r)); alternatives.append([ev((1+kap)*(1+k),r) for k in SH])
            roots={}; scenario={}
            for fee in [0.,.03]:
                ok=[]; plug=[]; derivatives=[]
                for b,aa in zip(bases,alternatives):
                    gain,tt=moments(b,aa,fee)
                    ok.append(joint & (tt<=norm.ppf(1-.05/20)).all(0))
                    plug.append(fit & (gain<=0).all(0))
                    up, down = int(np.flatnonzero(SH==.01)[0]), int(np.flatnonzero(SH==-.01)[0])
                    derivatives.append(((1-fee)*(aa[up]['R']-aa[down]['R']).mean()-TH*(aa[up]['CR']-aa[down]['CR']).mean()))
                ok=np.array(ok); plug=np.array(plug); der=np.array(derivatives)
                rr=np.full(len(TH),np.nan); mult=0
                for j in range(len(TH)):
                    cross=np.flatnonzero((der[:-1,j]>0)&(der[1:,j]<=0))
                    if len(cross)>1: mult+=1
                    if len(cross):
                        a=cross[0]; rr[j]=KP[a]+(KP[a+1]-KP[a])*der[a,j]/(der[a,j]-der[a+1,j])
                roots[fee]=rr
                kval=KP[ok.any(1)]
                scenario[str(fee)]=dict(kappa_fit=span(KP[plug.any(1)]),kappa_provisional_separate=span(kval),boundary_hit=bool(ok[[0,-1]].any()),multiple_foc_roots=mult,accepted_pairs=int(ok.sum()))
                if fee==0:
                    assert np.array_equal(ok[np.flatnonzero(KP==0)[0]],joint)
                else:
                    sh=rr-roots[0.]; valid=keep & np.isfinite(sh)
                    shifts=sh[valid]; sales=np.array([b['S'].mean() for b in bases])
                    scenario[str(fee)].update(optimal_level_shift=span(shifts),buyer_share=span(shifts/(fee*(1+shifts))),sales_change=span(np.interp(shifts,KP,sales)/base['S'].mean()-1),missing_roots_in_cost_set=int((keep & ~np.isfinite(sh)).sum()))
            row['scenarios'][mode]=scenario
            print(tag,seller,mode,json.dumps(scenario),flush=True)
        result['sellers'][seller]=row
        print('COST',tag,seller,row['theta_provisional'],row['margin_provisional'],flush=True)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=4):
        results=[]; started=time.time()
        for tag,kw in [('raw_d0',{}),('raw_d1',dict(rich=True)),('raw_d6',dict(rich=True,flexible=True))]:
            KP = np.round(np.arange(-.15, .25001, .0025), 4) if tag=='raw_d6' else np.round(np.arange(-.05, .10001, .0025), 4)
            cached=OUT/(tag+'_downstream.json')
            r=json.loads(cached.read_text()) if '--resume' in sys.argv and cached.exists() else run(tag,kw)
            r.setdefault('kappa_grid',[float(KP[0]),float(KP[-1]),.0025])
            results.append(r)
            (OUT/(tag+'_downstream.json')).write_text(json.dumps(r,indent=2))
        hashes={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'raw_reference_panel.parquet',*[OUT/(t+'_coef.npz') for t in ['raw_d0','raw_d1','raw_d6']]]}
        final=dict(results=results,meta=dict(seconds_this_invocation=time.time()-started,hashes=hashes,theta_grid=[.2,1.2,.0025],kappa_grid='See each model; flexible model expanded to avoid missing roots',shifts=SH.tolist(),fees=[0,.03],inference='PROVISIONAL: HAC7 plus two-way-clustered demand delta; omitted cross covariance; separate Bonferroni 20 moments, not matched-combination joint test; no seller-selection correction'))
        (OUT/'downstream_results.json').write_text(json.dumps(final,indent=2))
