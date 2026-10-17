"""Rebuild competition and sale endpoints BEFORE reference/price selection.
No original source file or earlier analysis output is overwritten. Reference
quotes are carried forward only, as a public quality index; availability and
quote age remain explicit demand covariates. A carried quote is not evidence
that Decluttr still offers the device at that price.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from audit_panel import ROOT,KEYS
OUT_FULL=ROOT/'output/swappa_rescue_20261016/fullsource'

def attach_competition(d):
 d=d.copy();d['market_id']=d.groupby(KEYS,sort=True).ngroup()
 market=d.groupby('market_id').Price.agg(['sum','count','min']).rename(columns={'sum':'market_total','count':'market_n','min':'market_min'})
 seller=d.groupby(['market_id','Seller']).Price.agg(['sum','count','min']).rename(columns={'sum':'own_total','count':'own_n','min':'own_min'}).reset_index().sort_values(['market_id','own_min','Seller']).reset_index(drop=True)
 seller['rank']=seller.groupby('market_id').cumcount();two=seller[seller['rank']<2].pivot(index='market_id',columns='rank',values='own_min').rename(columns={0:'min1',1:'min2'});seller=seller.merge(two,on='market_id',validate='many_to_one');seller['rival_min']=np.where(seller['rank']==0,seller.min2,seller.min1)
 d=d.merge(market,on='market_id',validate='many_to_one').merge(seller[['market_id','Seller','own_total','own_n','rival_min']],on=['market_id','Seller'],validate='many_to_one');d['rival_n']=d.market_n-d.own_n;d['rival_mean']=(d.market_total-d.own_total)/d.rival_n.replace(0,np.nan);return d

def main():
 global OUT_FULL
 ap=argparse.ArgumentParser();ap.add_argument('--screen-reference',action='store_true');args=ap.parse_args()
 if args.screen_reference:OUT_FULL=OUT_FULL/'reference_screened'
 OUT_FULL.mkdir(parents=True,exist_ok=True);rawpath=ROOT/'data/intermediate/raw_swappa.dta';refpath=ROOT/'data/intermediate/raw_decluttr.dta';d=pd.read_stata(rawpath,convert_categoricals=False);raw_rows=len(d);duplicates=d[d.duplicated(['Code','date'],keep=False)];conflicts={c:int(duplicates.groupby(['Code','date'])[c].nunique(dropna=False).gt(1).sum()) for c in ['Price','Seller','device_id','Storage','Condition','Real_Sold']};assert not any(conflicts.values()),conflicts
 d=d.drop_duplicates(['Code','date']).sort_values(['Code','date']).reset_index(drop=True);assert d[['Code','date','Price','Seller','device_id','Storage','Condition']].notna().all().all();assert (d.Price>0).all()
 # Endpoint labels use the complete raw listing histories, before any matching.
 last=d.groupby('Code').date.transform('max');first=d.groupby('Code').date.transform('min');d['sale_terminal_status']=(d.Real_Sold.eq(1)&d.date.eq(last)).astype(int);d['since_seen']=(d.date-first).dt.days;d['age']=(d.date-d.Created).dt.days;d['week']=(d.date-d.date.min()).dt.days//7
 refs=pd.read_stata(refpath,convert_categoricals=False);refs['price']=pd.to_numeric(refs.price,errors='coerce');keys=['device_id','Storage','Condition'];observed=refs.groupby(keys+['date']).price.mean().rename('fresh_reference').reset_index();observed['reference_quote_rejected']=0;rejected=[]
 if args.screen_reference:
  observed=observed.sort_values(keys+['date']).reset_index(drop=True)
  for sku,indices in observed.groupby(keys,sort=False).groups.items():
   past=[]
   for i in indices:
    price=float(observed.at[i,'fresh_reference']);median=float(np.median(past)) if len(past)>=7 else None
    if median is not None and not .5*median<=price<=2*median:
     observed.at[i,'reference_quote_rejected']=1;observed.at[i,'fresh_reference']=np.nan;rejected.append({'sku':list(sku),'date':str(observed.at[i,'date'].date()),'quote':price,'past_accepted_median':median})
    else:past.append(price)
 skus=d[keys].drop_duplicates();calendar=skus.merge(pd.DataFrame({'date':pd.date_range(d.date.min(),d.date.max())}),how='cross').merge(observed,on=keys+['date'],how='left',validate='one_to_one').sort_values(keys+['date']);calendar['reference_quote_rejected']=calendar.reference_quote_rejected.fillna(0).astype(int);calendar['reference_observed']=calendar.fresh_reference.notna().astype(int);calendar['quote_date']=calendar.date.where(calendar.reference_observed.eq(1));calendar['Ref_Price']=calendar.groupby(keys).fresh_reference.ffill();calendar['quote_date']=calendar.groupby(keys).quote_date.ffill();calendar['reference_age_days']=(calendar.date-calendar.quote_date).dt.days
 d=d.merge(calendar[keys+['date','Ref_Price','reference_observed','reference_age_days','reference_quote_rejected']],on=keys+['date'],how='left',validate='many_to_one').sort_values(['Code','date']).reset_index(drop=True);entry=d.drop_duplicates('Code').set_index('Code');d['initial_reference']=d.Code.map(entry.Ref_Price).fillna(0);d['initial_reference_missing']=d.initial_reference.eq(0).astype(int);d['initial_price']=d.Code.map(entry.Price);d['initial_residual']=d.initial_price-d.initial_reference
 d=attach_competition(d);d['residual']=d.Price-d.Ref_Price;d['gap_min']=d.Price-d.rival_min;d['gap_mean']=d.Price-d.rival_mean;d['rival_min_residual']=d.rival_min-d.Ref_Price;d['rival_mean_residual']=d.rival_mean-d.Ref_Price;d['fullsource_prepared']=True
 # Brute-force verification includes competitors omitted from the old clean file.
 rng=np.random.default_rng(901017)
 for pos in rng.choice(len(d),500,replace=False):
  row=d.iloc[pos];market=d[d.market_id.eq(row.market_id)];peer=market[market.Seller.ne(row.Seller)];own=market[market.Seller.eq(row.Seller)];assert row.own_n==len(own) and row.rival_n==len(peer)
  if len(peer):assert abs(row.rival_min-peer.Price.min())<1e-9 and abs(row.rival_mean-peer.Price.mean())<1e-9
  else:assert pd.isna(row.rival_min)
 old=pd.read_parquet(ROOT/'output/swappa_rescue_20261016/panel.parquet');compare=old[['Code','date','rival_n','rival_min','own_n']].merge(d[['Code','date','rival_n','rival_min','own_n']],on=['Code','date'],suffixes=('_old','_full'),validate='one_to_one');comparison=[]
 for label,codes in [('all',None),('UPGRADE SOLUTION INC',set(d.loc[d.Seller.eq('UPGRADE SOLUTION INC'),'Code']))]:
  sub=compare if codes is None else compare[compare.Code.isin(codes)];comparison.append({'seller':label,'matched_rows':len(sub),'rival_count_changed_share':float(sub.rival_n_old.ne(sub.rival_n_full).mean()),'rival_min_changed_share':float((sub.rival_min_old.fillna(-1)!=sub.rival_min_full.fillna(-1)).mean()),'own_count_changed_share':float(sub.own_n_old.ne(sub.own_n_full).mean())})
 oldlast=old.groupby('Code').date.max();newlast=d.groupby('Code').date.max();gap=(newlast.reindex(oldlast.index)-oldlast).dt.days;upgrade=d[d.Seller.eq('UPGRADE SOLUTION INC')];firststage=d[d.Ref_Price.notna()&d.rival_n.gt(0)&d.date.lt(d.date.max())];u=firststage[firststage.Seller.eq('UPGRADE SOLUTION INC')]
 report={'reference_screened':args.screen_reference,'rejected_reference_quotes':len(rejected),'raw_rows':raw_rows,'deduplicated_rows':len(d),'duplicates_removed':raw_rows-len(d),'duplicate_conflicts':conflicts,'raw_listings':int(d.Code.nunique()),'current_reference_observed_rows':int(d.reference_observed.sum()),'reference_available_after_forward_fill':int(d.Ref_Price.notna().sum()),'firststage_rows':len(firststage),'firststage_sales':int(firststage.sale_terminal_status.sum()),'clean_listing_endpoints_earlier_than_raw':int(gap.gt(0).sum()),'upgrade_clean_endpoints_earlier_than_raw':int(gap.reindex(upgrade.Code.unique()).gt(0).sum()),'upgrade':{'full_rows':len(upgrade),'firststage_rows':len(u),'listings':int(u.Code.nunique()),'sales':int(u.sale_terminal_status.sum()),'continuation_sales':int((u.sale_terminal_status*u.since_seen.gt(0)).sum()),'missing_entry_reference_rows':int(u.initial_reference_missing.sum()),'reference_age_quantiles':u.reference_age_days.quantile([.5,.9,1]).to_dict(),'price_min':float(u.Price.min()),'price_max':float(u.Price.max())},'old_competition_comparison':comparison,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [rawpath,refpath]},'brute_force_market_checks':500,'interpretation':'Full raw inventory precedes matching. Forward-filled reference prices are quality indices with observed availability and quote-age controls, not verified contemporaneous outside offers. Terminal sale labeling remains an observation-model assumption.'}
 (OUT_FULL/'rejected_reference_quotes.json').write_text(json.dumps(rejected,indent=2));d.to_parquet(OUT_FULL/'panel.parquet',index=False);(OUT_FULL/'source_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
