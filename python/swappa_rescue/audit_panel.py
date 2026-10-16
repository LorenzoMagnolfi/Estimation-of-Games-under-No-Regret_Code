"""Audit listing timing and construct full-market, seller-excluded competition.

Outputs are exploratory data diagnostics, not structural estimates.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'swappa_rescue_20261016'
KEYS = ['device_id', 'Storage', 'Condition', 'date']

def build_panel():
    source = ROOT / 'data/intermediate/raw_data_cleaned.dta'
    d = pd.read_stata(source, convert_categoricals=False).sort_values(['Code','date']).reset_index(drop=True)
    assert not d.duplicated(['Code','date']).any()
    d['sale_legacy'] = d.Sold_Today.fillna(0).astype(int)
    d['sale_forward'] = ((d.Real_Sold == 1) & (d.Updated > d.date) & (d.Updated <= d.date + pd.Timedelta(days=1))).astype(int)
    d['sale_last_seen'] = ((d.Real_Sold == 1) & (d.date == d.groupby('Code').date.transform('max')) & ((d.Updated-d.date).dt.days.between(0,1))).astype(int)
    d['age'] = (d.date-d.Created).dt.days
    d['since_seen'] = (d.date-d.groupby('Code').date.transform('min')).dt.days
    d['residual'] = d.Price-d.Ref_Price
    d['week'] = (d.date-d.date.min()).dt.days//7
    d['market_id'] = d.groupby(KEYS, sort=True).ngroup()
    market = d.groupby('market_id').Price.agg(['sum','count','min']).rename(columns={'sum':'market_total','count':'market_n','min':'market_min'})
    seller = d.groupby(['market_id','Seller']).Price.agg(['sum','count','min']).rename(columns={'sum':'own_total','count':'own_n','min':'own_min'}).reset_index()
    seller = seller.sort_values(['market_id','own_min','Seller']).reset_index(drop=True)
    seller['rank'] = seller.groupby('market_id').cumcount()
    first_two = seller[seller['rank'] < 2].pivot(index='market_id', columns='rank', values='own_min').rename(columns={0:'min1',1:'min2'})
    seller = seller.merge(first_two, on='market_id', validate='many_to_one')
    seller['rival_min'] = np.where(seller['rank']==0,seller.min2,seller.min1)
    d = d.merge(market, on='market_id', validate='many_to_one').merge(seller[['market_id','Seller','own_total','own_n','rival_min']],on=['market_id','Seller'],validate='many_to_one')
    d['rival_n'] = d.market_n-d.own_n
    d['rival_mean'] = (d.market_total-d.own_total)/d.rival_n.replace(0,np.nan)
    d['gap_min'] = d.Price-d.rival_min
    d['gap_mean'] = d.Price-d.rival_mean
    d['rival_min_residual'] = d.rival_min-d.Ref_Price
    d['rival_mean_residual'] = d.rival_mean-d.Ref_Price
    listing = d.groupby('Code').agg(first=('date','min'),last=('date','max'),created=('Created','first'),updated=('Updated','first'),sold=('Real_Sold','max'),legacy=('sale_legacy','sum'),forward=('sale_forward','sum'),last_seen=('sale_last_seen','sum'),n=('date','size'),prices=('Price','nunique'),seller=('Seller','first'))
    listing['last_updated_gap'] = (listing.updated-listing['last']).dt.days
    summary = {'rows':len(d),'listings':len(listing),'sales_legacy':int(d.sale_legacy.sum()),'sales_forward':int(d.sale_forward.sum()),'sales_last_seen':int(d.sale_last_seen.sum()),'known_sold_listings':int((listing.sold==1).sum()),'known_sold_gap_counts':listing.loc[listing.sold==1,'last_updated_gap'].value_counts().sort_index().to_dict(),'same_seller_competition_rows':int((d.own_n>1).sum()),'no_other_seller_rows':int((d.rival_n==0).sum()),'own_share_median':float((d.own_n/d.market_n).median()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'interpretation':'sale_forward uses Updated in (date,date+1]; sale_last_seen moves verified sold statuses to last observed exposure only when Updated is 0 or 1 days later. Neither date assignment is certified without source audit.'}
    byseller = d.groupby('Seller').agg(rows=('Code','size'),listings=('Code','nunique'),legacy=('sale_legacy','sum'),forward=('sale_forward','sum'),last_seen=('sale_last_seen','sum'),price=('Price','mean'),ref=('Ref_Price','mean'),rivals=('rival_n','mean'),own_inventory=('own_n','mean')).sort_values('rows',ascending=False)
    OUT.mkdir(parents=True,exist_ok=True)
    d.to_parquet(OUT/'panel.parquet',index=False)
    listing.to_parquet(OUT/'listings.parquet')
    byseller.to_csv(OUT/'seller_audit.csv')
    (OUT/'data_audit.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    print(byseller.head(15).to_string())
    return d

if __name__ == '__main__':
    build_panel()
