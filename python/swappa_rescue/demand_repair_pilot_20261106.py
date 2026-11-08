"""Exploratory demand repair on the public Swappa panel; no causal or coverage claim.

Reconstructs the revival risk-set convention from the existing full-source panel,
then compares exposure restrictions, richer competition, product-week effects,
and contemporaneous/previous quotes on identical rows. Outputs are separate from
the revival branch and all October artifacts. Run from any directory with Python.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import expit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/swappa_demand_repair_20261106'
PANEL = ROOT / 'output/swappa_rescue_20261016/fullsource/reference_screened/panel.parquet'
FOCAL = ['GadgetPickup', 'We Buy Mobile Corp']
SEED = 20261106


def save(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def prepare():
    d = pd.read_parquet(PANEL).sort_values(['Code', 'date']).reset_index(drop=True)
    d = d.loc[d.date.ne('2023-11-19')].copy()
    last = d.groupby('Code').date.transform('max')
    d['is_last'] = d.date.eq(last)
    d['sale'] = (d.Real_Sold.eq(1) & d.is_last).astype(float)
    d['upd_gap'] = (d.Updated - last).dt.days
    d['since_seen'] = (d.date - d.groupby('Code').date.transform('min')).dt.days
    d['next_gap'] = (d.groupby('Code').date.shift(-1) - d.date).dt.days
    d['previous_gap'] = (d.date - d.groupby('Code').date.shift()).dt.days
    d['previous_price'] = d.groupby('Code').Price.shift()
    d['sku'] = d[['device_id', 'Storage', 'Condition']].astype(str).agg('|'.join, axis=1)
    d['peer_mean'] = (d.own_total - d.Price) / (d.own_n - 1).replace(0, np.nan)
    # Market summaries use all active offers, before risk-set exclusions.
    parts = []
    for mid, g in d.groupby('market_id', sort=False):
        p, sel = g.Price.to_numpy(), g.Seller.to_numpy()
        for seller in np.unique(sel):
            rival = p[sel != seller]
            if len(rival):
                parts.append((mid, seller, np.median(rival), np.std(np.log(rival))))
            else:
                parts.append((mid, seller, np.median(p), 0.0))
    med = pd.DataFrame(parts, columns=['market_id', 'Seller', 'rival_median', 'rival_log_sd'])
    d = d.merge(med, on=['market_id', 'Seller'], validate='many_to_one')
    sd = d.groupby(['sku', 'Seller', 'date']).rival_median.first().reset_index().sort_values(['sku', 'Seller', 'date'])
    sd['internal_centered'] = sd.groupby(['sku', 'Seller']).rival_median.transform(lambda x: x.rolling(7, center=True, min_periods=1).median())
    sd['internal_trailing'] = sd.groupby(['sku', 'Seller']).rival_median.transform(lambda x: x.rolling(7, min_periods=1).median())
    d = d.merge(sd, on=['sku', 'Seller', 'date', 'rival_median'], validate='many_to_one')
    ratio = np.median(d.loc[d.Ref_Price.notna(), 'Ref_Price'] / d.loc[d.Ref_Price.notna(), 'internal_centered'])
    d['ref'] = d.Ref_Price.fillna(d.internal_centered * ratio)
    d['ref_external'] = d.Ref_Price.notna().astype(float)
    d['battery'] = pd.to_numeric(d.Battery_Life.str.extract(r'(\d+)')[0], errors='coerce')
    final = d.date.max()
    late = d.is_last & d.Real_Sold.eq(1) & ~d.upd_gap.between(0, 2)
    unknown = d.is_last & d.Real_Sold.isna()
    s = d.loc[d.date.lt(final) & ~late & ~unknown & d.ref.notna()].copy()
    first = s.sort_values('date').drop_duplicates('Code').set_index('Code')
    s['initial_res'] = np.log(s.Code.map(first.Price) / s.Code.map(first.ref))
    s['interval'] = np.where(s.date.eq('2023-11-18'), 2, 1)
    s['date_str'] = s.date.dt.strftime('%Y-%m-%d')
    s['week_str'] = ((s.date - pd.Timestamp('2023-07-11')).dt.days // 7).astype(str)
    s['sku_week'] = s.sku + ':' + s.week_str
    top = s.groupby('Seller').size().nlargest(30).index
    s['seller_group'] = s.Seller.where(s.Seller.isin(top), 'other')
    # No listing can return after a terminal event. Intermediate gaps are observation gaps.
    # Conservative sample removes all histories with a missing daily snapshot except Nov19.
    gap_ok = d.next_gap.isna() | d.next_gap.eq(1) | (d.date.eq('2023-11-18') & d.next_gap.eq(2))
    complete_codes = set(d.loc[~gap_ok, 'Code'])
    s['strict_day'] = ~s.Code.isin(complete_codes) & s.date.ne('2023-11-18')
    s['fresh_previous'] = s.previous_gap.eq(1)
    s['unchanged_previous'] = s.fresh_previous & s.Price.eq(s.previous_price)
    # Previous peer quote held to the contemporaneous inventory; sensitivity, not an instrument.
    s['previous_peer'] = (s.own_total - s.Price) / (s.own_n - 1).replace(0, np.nan)
    pri = d[['Code', 'date', 'previous_price']].copy()
    totals = pri.assign(seller=d.Seller, market_id=d.market_id).groupby(['market_id', 'seller']).previous_price.agg(['sum', 'count'])
    own = s.set_index(['market_id', 'Seller']).index
    ps = totals.reindex(own)
    s['previous_peer'] = (ps['sum'].to_numpy() - s.previous_price) / (ps['count'].to_numpy() - 1)
    s['lag_peer_complete'] = (ps['count'].to_numpy() == s.own_n.to_numpy()) | s.own_n.eq(1)
    rng = np.random.default_rng(SEED)
    codes = sorted(s.Code.unique())
    held = set(np.asarray(codes)[rng.random(len(codes)) < .2])
    s['holdout'] = s.Code.isin(held)
    audit = dict(raw_panel_rows=len(d), risk_rows=len(s), sales=int(s.sale.sum()), listings=s.Code.nunique(), dates=s.date.nunique(), reference_ratio=float(ratio), external_reference_share=float(s.ref_external.mean()), intermediate_gap_histories=len(complete_codes), intermediate_gap_rows=int((~gap_ok).sum()), strict_rows=int(s.strict_day.sum()), strict_sales=int(s.loc[s.strict_day, 'sale'].sum()), fresh_previous_rows=int(s.fresh_previous.sum()), unchanged_previous_rows=int(s.unchanged_previous.sum()), unknown_terminal_excluded=int(unknown.sum()), late_terminal_excluded=int(late.sum()), revival_expected_rows=457189, revival_expected_sales=22792)
    audit['sample_matches_revival'] = bool(len(s) == 457189 and s.sale.sum() == 22792)
    audit['timing'] = s.assign(change=np.select([~s.fresh_previous, s.Price.lt(s.previous_price), s.Price.gt(s.previous_price)], ['entry_gap', 'cut', 'increase'], default='unchanged')).groupby('change').agg(rows=('sale', 'size'), sales=('sale', 'sum'), rate=('sale', 'mean')).reset_index().to_dict('records')
    audit['focal'] = s.loc[s.Seller.isin(FOCAL)].groupby('Seller').agg(rows=('sale', 'size'), sales=('sale', 'sum'), strict=('strict_day', 'sum'), fresh=('fresh_previous', 'sum')).reset_index().to_dict('records')
    s.to_parquet(OUT / 'pilot_panel.parquet', index=False)
    save(audit, 'risk_set_audit.json')
    print('RISK_SET', json.dumps(audit, default=str), flush=True)
    return s.reset_index(drop=True)


def continuous(s, rich=False, lag=False, shift=0.0, rounded=False, flexible=False):
    p = s.previous_price.to_numpy() if lag else s.Price.to_numpy(dtype=float)
    peer = s.previous_peer.to_numpy() if lag else s.peer_mean.to_numpy()
    if rounded:
        raise NotImplementedError('Rounded peer updates require the full preselection inventory and verified ask/display mapping.')
    else:
        p, peer = p * (1+shift), peer * (1+shift)
    ref = s.ref.to_numpy()
    hasr, hasp = s.rival_n.to_numpy() > 0, s.own_n.to_numpy() > 1
    lp = np.log(p/ref)
    gap = np.where(hasr, np.log(p / s.rival_min.fillna(1).to_numpy()), 0)
    X = dict(lp_ref=lp, gap_up=np.maximum(gap, 0), gap_dn=np.minimum(gap, 0), peer_rel=np.where(hasp, np.log(np.where(hasp, peer, 1)/ref), 0))
    age = np.log1p(s.since_seen.to_numpy())
    X['age'] = age
    for i,k in enumerate(np.log([2,4,8,16,32])):
        X[f'age_k{i}'] = np.maximum(age-k, 0)
    b = s.battery.to_numpy()
    X.update(batt=np.nan_to_num(b, nan=90)/100, batt_missing=np.isnan(b).astype(float), batt_full=(b==100).astype(float), log_rival_n=np.log1p(s.rival_n), no_rival=(~hasr).astype(float), log_own_n=np.log(s.own_n), has_peer=hasp.astype(float), log_interval=np.log(s.interval), ref_external=s.ref_external, initial_res=s.initial_res)
    if rich:
        # These summaries of rival offers remain fixed under a unilateral seller deviation.
        X.update(rival_mean_ref=np.where(hasr, np.log(s.rival_mean.fillna(1)/ref), 0), rival_sd=s.rival_log_sd, rival_median_ref=np.where(hasr, np.log(s.rival_median/ref), 0))
    if flexible:
        for i,k in enumerate([-.2,-.1,.1,.2]):
            X[f'gap_hinge{i}'] = np.maximum(gap-k,0)
    z = pd.DataFrame(X).to_numpy(dtype=float)
    assert np.isfinite(z).all(), 'nonfinite covariates'
    return z, list(X)


def design(s, product_week=False, **kwargs):
    z,names = continuous(s, **kwargs)
    sd = z.std(0)
    sd[sd < 1e-8] = 1
    mean = z.mean(0)
    F = []
    for col in (['sku_week', 'date_str', 'seller_group'] if product_week else ['sku','date_str','seller_group']):
        c = pd.Categorical(s[col])
        valid = c.codes > 0
        F.append(sp.csr_matrix((np.ones(valid.sum()), (np.flatnonzero(valid), c.codes[valid]-1)), shape=(len(s), len(c.categories)-1)))
    X = sp.hstack([np.ones((len(s),1)), sp.csr_matrix((z-mean)/sd), *F], format='csr')
    return X,names,mean,sd


def fit(X,y,nd):
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean()/(1-y.mean()))
    lam = np.r_[np.zeros(nd), np.ones(X.shape[1]-nd)]
    def loss(v):
        zz=X@v
        return np.logaddexp(0,zz).sum()-y@zz+.5*np.sum(lam*v*v)
    old=loss(b)
    for it in range(40):
        q=expit(X@b)
        grad=X.T@(q-y)+lam*b
        H=X.T@X.multiply((q*(1-q))[:,None])+sp.diags(lam+1e-9)
        step=spsolve(H.tocsc(),grad)
        rate=1.
        while loss(b-rate*step)>old+1e-8 and rate>1e-7:
            rate*=.5
        b-=rate*step
        new=loss(b)
        if np.max(np.abs(rate*step))<1e-7 or abs(new-old)<1e-8:
            break
        old=new
    score=float(np.max(np.abs(X.T@(expit(X@b)-y)+lam*b)))
    return b,dict(iterations=it+1,score_max=score,objective=float(loss(b)))


def effects(s,X,b,names,mean,sd,kw):
    q=expit(X@b)
    base,_=continuous(s,**kw)
    res=[]
    for name in FOCAL:
        ix=s.Seller.eq(name).to_numpy()
        if not ix.any(): continue
        refpred=float(q[ix].sum())
        for shift in [-.05,-.025,-.01,-.005,.005,.01,.025,.05]:
            alt,_=continuous(s,shift=shift,**kw)
            q1=expit(X@b+((alt-base)/sd)@b[1:len(names)+1])
            res.append(dict(seller=name,shift=shift,predicted_base_sales=refpred,change_sales=float((q1[ix]-q[ix]).sum()),change_pct=float(100*(q1[ix].sum()/refpred-1)),arc_elasticity=float(np.log(q1[ix].sum()/refpred)/np.log(1+shift))))
    return res


def run_one(s,tag,rich=False,product_week=False,lag=False,flexible=False):
    s=s.reset_index(drop=True)
    kw=dict(rich=rich,lag=lag,flexible=flexible)
    X,names,mean,sd=design(s,product_week=product_week,**kw)
    y=s.sale.to_numpy()
    t=time.time()
    beta,conv=fit(X,y,len(names)+1)
    train=~s.holdout.to_numpy()
    bc,cvc=fit(X[train],y[train],len(names)+1)
    pred=expit(X[~train]@bc)
    record=dict(tag=tag,rows=len(s),sales=int(y.sum()),parameters=X.shape[1],seconds=time.time()-t,convergence=conv,holdout_convergence=cvc,coefficients=dict(zip(names,(beta[1:len(names)+1]/sd).tolist())),holdout_logloss=float(log_loss(y[~train],pred)),holdout_actual_sales=float(y[~train].sum()),holdout_predicted_sales=float(pred.sum()),effects=effects(s,X,beta,names,mean,sd,kw))
    # Calibration is independent of structural inversion; do not select on inferred margins.
    hold=s.loc[~train,['Seller','sale']].copy(); hold['pred']=pred
    record['focal_holdout']=hold.loc[hold.Seller.isin(FOCAL)].groupby('Seller').agg(rows=('sale','size'),actual=('sale','sum'),predicted=('pred','sum')).reset_index().to_dict('records')
    save(record,tag+'.json')
    np.savez_compressed(OUT/(tag+'_coef.npz'),beta=beta,names=np.asarray(names),mean=mean,sd=sd)
    print('FIT',json.dumps({k:v for k,v in record.items() if k not in ['effects','coefficients']}),flush=True)
    print('PRICE',tag,{n:record['coefficients'][n] for n in names[:4]},flush=True)
    print('SHIFT',tag,[v for v in record['effects'] if v['shift']==-.05],flush=True)
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save(dict(status='exploratory; no causal identification or structural inference',source_panel=str(PANEL.relative_to(ROOT)),panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),code_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),revival_commit='16be52968a0776f0aec086a2bbeabf5b8f6abfcb',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,holdout_seed=SEED), 'manifest.json')
    with threadpool_limits(limits=4):
        s=pd.read_parquet(OUT/'pilot_panel.parquet').reset_index(drop=True) if '--reuse-panel' in sys.argv else prepare()
        results=[]
        for tag,sub,kwargs in [
            ('d0_revival_like',s,{}),
            ('d1_rich_competition',s,dict(rich=True)),
            ('d2_product_week',s,dict(rich=True,product_week=True)),
            ('d3_complete_daily_histories',s.loc[s.strict_day],dict(rich=True)),
            ('d4_current_matched',s.loc[s.fresh_previous & s.lag_peer_complete],dict(rich=True)),
            ('d5_previous_matched',s.loc[s.fresh_previous & s.lag_peer_complete],dict(rich=True,lag=True)),
            ('d6_flexible_gap',s,dict(rich=True,flexible=True))]:
            results.append(run_one(sub,tag,**kwargs))
            save(results,'all_results.json')
    print('DEMAND_REPAIR_PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
