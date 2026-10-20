"""Run the existing feedback diagnostic on the valid one-day calendar."""
import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd
import gadget_inventory_feedback_extension as core
from audit_panel import ROOT
from gadget_demand_audit import RUN
from demand_timing_sensitivity import terminal_status_data

def main():
    calendar=pd.read_csv(RUN/'snapshot_calendar.csv',parse_dates=['date'])
    data=terminal_status_data();data=data[data.date.isin(calendar.loc[calendar.one_day_observation_interval,'date'])].copy()
    run=RUN/'observed_intervals';run.mkdir(exist_ok=True)
    # Supply corrected input and a separate output directory to the unchanged
    # calculation. The original function and files are not edited or rerun.
    old_input,old_output=core.terminal_status_data,core.RUN
    try:
        core.terminal_status_data=lambda:data
        core.RUN=run
        records=core.feedback()
    finally:
        core.terminal_status_data,core.RUN=old_input,old_output
    frame=pd.DataFrame(records); totals=frame.groupby(['unit','restriction'])[['days','empty_days','zero_cost_dominated_days']].sum().reset_index().to_dict('records')
    out=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),calculation_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),categories=int(frame.category.nunique()),totals=totals,selected=frame[frame.selected].to_dict('records'),interpretation='One-day-calendar version of the previous conditional information diagnostic. Constant conditional arm means, timing, coverage and optional monotonicity remain maintained assumptions. No verified seller behavior or new-method cost set.')
    (ROOT/'docs/evidence/gadgetpickup_observation_feedback_results_20261020.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2));print('OBSERVATION_FEEDBACK_COMPLETE')
if __name__=='__main__':main()
