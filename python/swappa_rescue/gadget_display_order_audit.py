"""Recover historical table order and audit demand-identifying variation."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from audit_panel import ROOT
from gadget_demand_audit import RUN

DESIGN=ROOT/'docs/evidence/gadgetpickup_display_order_design_20261020.json'

def extract():
    records=[]; sources=[]
    paths=sorted((ROOT/'data/Swappa_Daily_Data').glob('*.xlsx'))
    for k,p in enumerate(paths):
        book=load_workbook(p,read_only=True,data_only=True); rows=book.worksheets[0].iter_rows(values_only=True)
        headers=list(next(rows)); columns={str(x):i for i,x in enumerate(headers)}
        required=['Code','device_id','date','Price']
        if not all(x in columns for x in required):
            sources.append(dict(file=p.name,error='missing columns',headers=[str(v) for v in headers]));book.close();continue
        rankname='#' if '#' in columns else None
        count=Counter(); n=0
        for row in rows:
            code=row[columns['Code']];device=row[columns['device_id']]
            if not isinstance(code,str) or not code.startswith('L'):continue
            count[device]+=1;n+=1
            records.append(dict(Code=code,device_id=device,date=row[columns['date']],file=p.name,table_position=count[device],display_number=row[columns[rankname]] if rankname else None,source_price=row[columns['Price']]))
        book.close();sources.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=n,rank_column=rankname))
        if (k+1)%20==0: print(f'EXTRACTED {k+1}/{len(paths)} files',flush=True)
    a=pd.DataFrame(records);a['date']=pd.to_datetime(a.date);a['display_number_numeric']=pd.to_numeric(a.display_number,errors='coerce');a['source_price_numeric']=pd.to_numeric(a.source_price.astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False),errors='coerce')
    a.to_parquet(RUN/'historical_display_order.parquet',index=False)
    return a,sources

def summarize(a,sources):
    d=pd.read_parquet(ROOT/'output/swappa_rescue_20261016/fullsource/reference_screened/panel.parquet')
    duplicate=a.duplicated(['Code','date'],keep=False)
    dup=a.loc[duplicate].groupby(['Code','date']).agg(first_position=('table_position','min'),last_position=('table_position','max'),copies=('Code','size'),prices=('source_price_numeric','nunique'))
    crossing_pages=((dup.last_position-1)//50)>((dup.first_position-1)//50)
    gadget_codes=set(d.loc[d.Seller.eq('GadgetPickup'),'Code'])
    b=a.drop_duplicates(['Code','date']).merge(d,on=['Code','date','device_id'],how='inner',validate='one_to_one')
    b=b.sort_values(['date','device_id','table_position'])
    b['price_drop_from_previous']=b.groupby(['date','device_id']).Price.diff().lt(0)
    gp=b[b.Seller.eq('GadgetPickup')].copy()
    gp['cheapest_rival']=gp.rival_min.ge(gp.Price)
    gp['device_price_rank']=gp.groupby(['device_id','date']).Price.rank(method='min')
    ties=[];pair_total=0;pair_reversed=0;created_order=[0,0,0]
    previous={}
    for (date,device),z in gp.groupby(['date','device_id'],sort=True):
        for price,t in z.groupby('Price'):
            t=t.sort_values('table_position'); codes=t.Code.tolist(); loc=dict(zip(codes,t.table_position)); created=dict(zip(codes,t.Created))
            for ca,cb in itertools.combinations(codes,2):
                if pd.notna(created[ca]) and pd.notna(created[cb]):
                    created_order[0]+=created[ca]<created[cb];created_order[1]+=created[ca]>created[cb];created_order[2]+=created[ca]==created[cb]
            key=(device,price)
            if key in previous and (date-previous[key][0]).days==1:
                old=previous[key][1];common=[c for c in codes if c in old]
                for ca,cb in itertools.combinations(common,2):
                    pair_total+=1;pair_reversed+=old[ca]>old[cb]
            previous[key]=(date,loc)
            if len(t)>1:ties.append(dict(date=str(date.date()),device=device,price=float(price),listings=len(t),span=int(t.table_position.max()-t.table_position.min())))
    s=d[d.Seller.eq('GadgetPickup')].sort_values(['Code','date']).copy(); lag=s.groupby('Code')[['date','Price','rival_min']].shift(1)
    consecutive=s.date.sub(lag.date).dt.days.eq(1)&s.rival_min.notna()&lag.rival_min.notna()
    crossing=consecutive&(s.Price.le(s.rival_min)!=lag.Price.le(lag.rival_min))
    c=pd.DataFrame(dict(own_price_changed=s.Price.ne(lag.Price),rival_min_changed=s.rival_min.ne(lag.rival_min))).loc[crossing]
    last=s.groupby('Code').agg(date=('date','max'),updated=('Updated','first'),sold=('Real_Sold','first'))
    last=last[last.date.lt(d.date.max())].copy();last['update_gap']=(last.updated-last.date).dt.days
    gap=last.groupby(['sold','update_gap'],dropna=False).size().reset_index(name='listings')
    gap=gap.astype(object).where(gap.notna(),None)
    summaries=[]
    for label,z in [('all',gp),('at_or_below_same_category_rival_min',gp[gp.cheapest_rival]),('above_same_category_rival_min',gp[~gp.cheapest_rival&gp.rival_min.notna()])]:
        summaries.append(dict(group=label,rows=len(z),sales=int(z.sale_terminal_status.sum()),position_median=float(z.table_position.median()),position_q90=float(z.table_position.quantile(.9)),position_first_share=float(z.table_position.eq(1).mean()),position_top10_share=float(z.table_position.le(10).mean())))
    b[['Code','date','device_id','table_position','display_number_numeric','Price','price_drop_from_previous']].to_parquet(RUN/'display_order_matched.parquet',index=False)
    return dict(files=len(sources),file_errors=[z for z in sources if 'error' in z],extracted_rows=len(a),dates=int(a.date.nunique()),duplicate_rows=int(duplicate.sum()),duplicate_listing_days=len(dup),duplicate_copy_counts=dup.copies.value_counts().to_dict(),duplicate_same_price_fraction=float(dup.prices.eq(1).mean()),duplicate_cross_50_row_boundary_fraction=float(crossing_pages.mean()),gadget_duplicate_listing_days=int(dup.reset_index().Code.isin(gadget_codes).sum()),matched_rows=len(b),unmatched_unique_rows=int(len(a.drop_duplicates(['Code','date']))-len(b)),numeric_display_fraction=float(a.display_number_numeric.notna().mean()),display_equals_position_fraction=float(a.display_number_numeric.eq(a.table_position).mean()),source_price_mismatch_rows=int(b.source_price_numeric.ne(b.Price).sum()),adjacent_price_inversions=int(b.price_drop_from_previous.sum()),device_dates=int(b.groupby(['device_id','date']).ngroups),device_dates_with_inversion=int(b.loc[b.price_drop_from_previous,['device_id','date']].drop_duplicates().shape[0]),gadget_position=summaries,gadget_tied_groups=len(ties),gadget_same_price_pair_transitions=pair_total,gadget_same_price_pair_order_reversals=int(pair_reversed),gadget_tie_creation_order=dict(older_first=int(created_order[0]),newer_first=int(created_order[1]),same_created_date=int(created_order[2])),rival_min_crossings=int(crossing.sum()),crossing_sources=c.value_counts().reset_index(name='rows').to_dict('records'),updated_gap_counts=gap.to_dict('records'))

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--from-cache',action='store_true');args=parser.parse_args()
    extraction_commit=None
    if args.from_cache:
        prior=json.loads((RUN/'display_order_audit.json').read_text());sources=prior['source_files']
        for r in sources:
            assert 'error' not in r and hashlib.sha256((ROOT/'data/Swappa_Daily_Data'/r['file']).read_bytes()).hexdigest()==r['sha256']
        a=pd.read_parquet(RUN/'historical_display_order.parquet');extraction_commit=prior.get('extraction_commit') or prior['code_commit']
    else:a,sources=extract()
    result=dict(extraction_commit=extraction_commit,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),summary=summarize(a,sources),source_files=sources,interpretation='Displayed table order is observed; featured flags and causal/random ordering are not established. Updated is not verified transaction time. No causal demand coefficient, instrument, regret bound or cost estimate.')
    for p in [RUN/'display_order_audit.json',ROOT/'docs/evidence/gadgetpickup_display_order_results_20261020.json']:p.write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2,default=str),flush=True);print('DISPLAY_ORDER_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
