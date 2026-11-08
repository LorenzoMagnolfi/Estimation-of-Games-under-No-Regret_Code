"""Audit reference-price reconstruction against original Decluttr workbooks.

Uses named workbook columns, not inferred positions in extracted text. Keeps all
prior pilot results. Fits the same demand comparisons with the reconstructed r.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from demand_repair_pilot_20261106 import ROOT, OUT, run_one, save

PAPER = ROOT.parent / 'Estimation-of-Games-under-No-Regret'
REV = '16be52968a0776f0aec086a2bbeabf5b8f6abfcb'


def read_one(p):
    x=pd.read_excel(p,usecols=lambda c:c in ['name','price','date','Product Line','Grade','Network'])
    x['file']=p.name
    return x


def main():
    files=sorted((ROOT/'data/Decluttr_Daily_Data').glob('data_*.xlsx'))
    if (OUT/'decluttr_named_columns.parquet').exists():
        r=pd.read_parquet(OUT/'decluttr_named_columns.parquet')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            r=pd.concat(pool.map(read_one,files),ignore_index=True)
        r.to_parquet(OUT/'decluttr_named_columns.parquet',index=False)
    mapping=subprocess.check_output(['git','show',REV+':JPE Revision/swappa-revival-2026-11-03/code/device_match.tsv'],cwd=PAPER,text=True)
    mt=pd.read_csv(io.StringIO(mapping),sep='\t')
    r['device_id']=r['Product Line'].str.strip().map(dict(zip(mt.ProductLine,mt.device_id)))
    nm=r['name'].str.strip()
    storage=nm.str.extract(r'(\d+)\s?(GB|TB)')
    r['Storage']=storage[0]+' '+storage[1]
    r['Condition']=r.Grade.map({'Good':'Fair','Very Good':'Good','Pristine':'Mint'})
    r['date']=pd.to_datetime(r.date)
    r['price']=pd.to_numeric(r.price,errors='coerce')
    valid=r.device_id.notna() & r.Storage.notna() & r.Condition.notna() & r.price.gt(0)
    names=nm.str.endswith('UNLOCKED').fillna(False)
    networks=r.Network.eq('UNLOCKED')
    diag=dict(workbooks=len(files),raw_rows=len(r),valid_rows=int(valid.sum()),name_unlocked=int((valid&names).sum()),network_unlocked=int((valid&networks).sum()),disagreement=int((valid&(names!=networks)).sum()),revival_reported_unlocked=46867)
    # Mirror the branch's name-based restriction, but use the named Grade column.
    z=r.loc[valid&names].copy()
    keys=['device_id','Storage','Condition']
    obs=z.groupby(keys+['date']).price.mean().rename('fresh').reset_index().sort_values(keys+['date']).reset_index(drop=True)
    rejected=0
    for _,idx in obs.groupby(keys,sort=False).groups.items():
        past=[]
        for i in idx:
            val=obs.at[i,'fresh']
            med=np.median(past) if len(past)>=7 else None
            if med is not None and not .5*med<=val<=2*med:
                obs.at[i,'fresh']=np.nan; rejected+=1
            else: past.append(val)
    s=pd.read_parquet(OUT/'pilot_panel.parquet')
    cal=s[keys].drop_duplicates().merge(pd.DataFrame({'date':pd.date_range('2023-07-11','2023-12-23')}),how='cross')
    cal=cal.merge(obs,on=keys+['date'],how='left').sort_values(keys+['date'])
    cal['new_external']=cal.groupby(keys).fresh.ffill()
    s=s.merge(cal[keys+['date','new_external']],on=keys+['date'],how='left',validate='many_to_one')
    old=s.ref.copy()
    ratio=float(np.median(s.loc[s.new_external.notna(),'new_external']/s.loc[s.new_external.notna(),'internal_centered']))
    s['ref']=s.new_external.fillna(s.internal_centered*ratio)
    s['ref_external']=s.new_external.notna().astype(float)
    first=s.sort_values('date').drop_duplicates('Code').set_index('Code')
    s['initial_res']=np.log(s.Code.map(first.Price)/s.Code.map(first.ref))
    diag.update(rejected_quotes=rejected,sku_dates=int(obs.fresh.notna().sum()),external_share=float(s.ref_external.mean()),fallback_ratio=ratio,reference_changed_share=float((abs(s.ref-old)>1e-7).mean()),reference_ratio_quantiles=(s.ref/old).quantile([0,.1,.5,.9,1]).to_dict(),price_reference_old_sources='existing Stata-derived full-source panel',new_sources='named workbook columns, branch name-unlocked restriction',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    save(diag,'reference_reconstruction_audit.json')
    print('REFERENCE',json.dumps({k:v for k,v in diag.items() if k!='source_sha256'}),flush=True)
    s.to_parquet(OUT/'raw_reference_panel.parquet',index=False)
    results=[]
    with threadpool_limits(limits=4):
        for tag,kwargs in [('raw_d0',{}),('raw_d1',dict(rich=True)),('raw_d2',dict(rich=True,product_week=True)),('raw_d6',dict(rich=True,flexible=True))]:
            results.append(run_one(s,tag,**kwargs));save(results,'raw_reference_results.json')
    print('REFERENCE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
