"""Flag copied snapshots without imputing missing prices or sale times."""
import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd
from audit_panel import ROOT
from gadget_demand_audit import RUN
from demand_timing_sensitivity import terminal_status_data

def main():
    a=pd.read_parquet(RUN/'historical_display_order.parquet').drop_duplicates(['Code','date'])
    dates=sorted(a.date.unique()); rows=[]; copied=[]
    for prev,now in zip(dates,dates[1:]):
        p=a[a.date.eq(prev)];n=a[a.date.eq(now)]
        m=p.merge(n,on=['Code','device_id'],suffixes=('_p','_n'))
        same_price=int(m.source_price_numeric_p.eq(m.source_price_numeric_n).sum());same_position=int(m.table_position_p.eq(m.table_position_n).sum())
        record=dict(previous=str(prev)[:10],date=str(now)[:10],previous_rows=len(p),current_rows=len(n),shared=len(m),same_price=same_price,same_position=same_position)
        candidate=len(p)==len(n)==len(m)==same_price==same_position
        record['candidate_copy']=candidate;record['all_nondate_cells_equal']=False
        if candidate:
            pf=ROOT/'data/Swappa_Daily_Data'/p.file.iloc[0];nf=ROOT/'data/Swappa_Daily_Data'/n.file.iloc[0]
            x=pd.read_excel(pf);y=pd.read_excel(nf);cols=[v for v in x.columns if v!='date']
            confirmed=list(x)==list(y) and x[cols].equals(y[cols]);record['all_nondate_cells_equal']=confirmed
            if confirmed:copied.append(now)
        rows.append(record)
    pd.DataFrame(rows).to_csv(RUN/'snapshot_transition_audit.csv',index=False)
    cal=pd.DataFrame({'date':dates});cal['copied_previous_snapshot']=cal.date.isin(copied)
    fresh=cal.loc[~cal.copied_previous_snapshot,'date'];nextdate=dict(zip(fresh.iloc[:-1],fresh.iloc[1:]))
    cal['next_fresh_date']=cal.date.map(nextdate);cal['days_to_next_fresh_snapshot']=(cal.next_fresh_date-cal.date).dt.days
    cal['one_day_observation_interval']=~cal.copied_previous_snapshot & cal.days_to_next_fresh_snapshot.eq(1)
    cal.to_csv(RUN/'snapshot_calendar.csv',index=False)
    d=pd.read_parquet(ROOT/'output/swappa_rescue_20261016/fullsource/reference_screened/panel.parquet');s=d[d.Seller.eq('GadgetPickup')];bad=s[s.date.isin(copied)]
    firststage=terminal_status_data();firststage=firststage[firststage.Seller.eq('GadgetPickup')]
    safe=firststage.date.isin(cal.loc[cal.one_day_observation_interval,'date'])
    one_day=dict(rows=int(safe.sum()),sales=int(firststage.loc[safe,'sale_terminal_status'].sum()),dates=int(firststage.loc[safe,'date'].nunique()),excluded_rows=int((~safe).sum()),excluded_sales=int(firststage.loc[~safe,'sale_terminal_status'].sum()))
    results=dict(gadget_one_day_interval_sample=one_day,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),copied_dates=[str(x)[:10] for x in copied],copy_checks=[r for r in rows if r['candidate_copy']],gadget_copied_rows=len(bad),gadget_terminal_sales_on_copied_dates=int(bad.sale_terminal_status.sum()),calendar_flags=cal.loc[~cal.one_day_observation_interval].astype(object).where(cal.notna(),None).to_dict('records'),interpretation='Copied dates are not new observations. Removing them leaves a two-day interval; do not treat it as one daily sale exposure, silently move all sales to the earlier day, or claim observed seller feedback. The final date is also censored. No raw file or previous model overwritten.')
    (ROOT/'docs/evidence/gadgetpickup_snapshot_calendar_results_20261020.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
    print(json.dumps(results,indent=2,default=str))

if __name__=='__main__':main()
