"""Descriptive replenishment and repricing evidence; not a procurement-cost test."""
import json
import numpy as np
import pandas as pd
from quality_demand import data
from audit_panel import OUT

def main():
    d=data().sort_values(['Code','date']); records=[]; product_records=[]
    sellers=['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']
    date0=d.date.min(); date1=d.date.max(); interior=pd.date_range(date0+pd.Timedelta(days=7),date1-pd.Timedelta(days=7))
    for seller in sellers:
        s=d[d.Seller.eq(seller)].copy(); prev=s.groupby('Code').Price.shift(); prevdate=s.groupby('Code').date.shift()
        adjacent=(s.date-prevdate).dt.days.eq(1); change=adjacent&s.Price.ne(prev)
        entries=s.drop_duplicates('Code'); exits=s.drop_duplicates('Code',keep='last')
        sales=exits[exits.sale_last_seen.eq(1)&exits.date.isin(interior)]
        in_entries=entries[entries.date.isin(interior)]
        active=s.groupby('date').size().reindex(interior,fill_value=0)
        new=in_entries.groupby('date').size().reindex(interior,fill_value=0)
        sold=sales.groupby('date').size().reindex(interior,fill_value=0)
        replacements=[]; baseline=[]
        for key,g in s.groupby(['device_id','Storage','Condition']):
            ge=g.drop_duplicates('Code'); gs=g.drop_duplicates('Code',keep='last'); gs=gs[gs.sale_last_seen.eq(1)&gs.date.isin(interior)]
            counts=ge.groupby('date').size().reindex(pd.date_range(date0,date1),fill_value=0)
            # Subsequent seven days, excluding the exit date; descriptive only.
            future=counts.iloc[::-1].rolling(7,min_periods=7).sum().iloc[::-1].shift(-1)
            replacements.extend(future.reindex(gs.date).ge(1).tolist())
            baseline.extend(future.reindex(g[g.date.isin(interior)].date.drop_duplicates()).ge(1).tolist())
            product_records.append({'seller':seller,'device_id':key[0],'storage':key[1],'condition':key[2],'listings':ge.Code.nunique(),'interior_new':ge.date.isin(interior).sum(),'interior_sales':len(gs),'sale_followed_by_entry_7d':float(future.reindex(gs.date).ge(1).mean()) if len(gs) else None})
        rec={'seller':seller,'listings':s.Code.nunique(),'listing_days':len(s),'adjacent_listing_days':int(adjacent.sum()),'price_change_days':int(change.sum()),'price_change_fraction':float(change.sum()/adjacent.sum()),'listings_ever_repriced_fraction':float(s.loc[change,'Code'].nunique()/s.Code.nunique()),'median_abs_price_change':float((s.Price-prev).abs()[change].median()),'dates_any_repricing':s.loc[change,'date'].nunique(),'interior_calendar_days':len(interior),'mean_active_inventory':float(active.mean()),'inventory_cv':float(active.std()/active.mean()),'interior_new_listings':len(in_entries),'interior_terminal_sales':len(sales),'new_per_sale':len(in_entries)/len(sales),'sale_followed_by_same_product_entry_7d':float(np.mean(replacements)),'active_product_day_followed_by_entry_7d':float(np.mean(baseline)),'new_sale_daily_correlation':float(new.corr(sold))}
        records.append(rec); print(json.dumps(rec,default=float),flush=True)
    pd.DataFrame(records).to_csv(OUT/'inventory_audit.csv',index=False)
    pd.DataFrame(product_records).to_csv(OUT/'inventory_product_audit.csv',index=False)
if __name__=='__main__':main()
