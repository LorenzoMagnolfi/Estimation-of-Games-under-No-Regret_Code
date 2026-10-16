"""Check a cost-aware undominated-pricing domain before assuming it.

If q(c)<=1/2 and each log-odds price slope is <=-k_min<0, prices
above c+2/k_min are dominated by a lower price, in every such state.
This is an extra restriction on the learner's permitted actions, not ANR alone.
"""
import json
import joblib
import numpy as np
import pandas as pd
from quality_demand import data,QualityDemand
from audit_panel import OUT

def main():
    d=data();fit=joblib.load(OUT/'quality_demand_quality.joblib');ef=fit.effects();records=[]
    for seller in ['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']:
        s=d[d.Seller.eq(seller)].copy();p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy()
        b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100
        lower_slope=np.minimum(-(b+gp),-(b+gn));assert lower_slope.min()>0
        offset=fit.logit(s)-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
        # Unique price at which fitted probability equals one half.
        threshold_below=-(offset-gn*rival)/(b+gn);threshold_above=-(offset-gp*rival)/(b+gp)
        p50=np.where(threshold_below<=rival,threshold_below,threshold_above)
        low=np.maximum.reduce([np.zeros(len(s)),p-2/lower_slope,p50]);high=p
        for indexing in ['fixed_dollars','current_reference_residual']:
            anchor=np.zeros(len(s)) if indexing=='fixed_dollars' else ref
            a=s[['Code']].copy();a['lower']=low-anchor;a['upper']=high-anchor
            z=a.groupby('Code').agg(lower=('lower','max'),upper=('upper','min'));feasible=z.lower<=z.upper+1e-9
            rec={'seller':seller,'indexing':indexing,'listings':len(z),'infeasible_listings':int((~feasible).sum()),'infeasible_share':float((~feasible).mean()),'median_feasible_cost_width':float((z.upper-z.lower)[feasible].median()),'max_required_markup_cap':float((2/lower_slope).max()),'min_slope_magnitude':float(lower_slope.min()),'interpretation':'Compatibility audit only. q(c)<=0.5 must hold on the model support, not only observed states; no dominated pricing is additional to ANR.'}
            records.append(rec);print(json.dumps(rec),flush=True)
    pd.DataFrame(records).to_csv(OUT/'dominance_audit.csv',index=False)
if __name__=='__main__':main()
