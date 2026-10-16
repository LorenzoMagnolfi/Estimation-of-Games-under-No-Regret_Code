"""Profile observed-state regret for persistent affine inventory costs."""
import json
import joblib
import numpy as np
import pandas as pd
from demand_pilot import Demand
from audit_panel import OUT

def main():
    d=pd.read_parquet(OUT/'panel.parquet'); d=d[d.rival_n.gt(0)&d.date.lt(d.date.max())].copy(); d=d[~(d.Real_Sold.eq(1)&d.date.gt(d.Updated))].copy()
    top=d.groupby('Seller').Code.nunique().nlargest(20).index; d['seller_group']=np.where(d.Seller.isin(top),d.Seller,'Other')
    fit=joblib.load(OUT/'demand_rich_sale_last_seen.joblib')
    betas=np.arange(.3,1.201,.002); offsets=np.arange(-150.,151.,2.)
    theta_b,theta_d=np.meshgrid(betas,offsets,indexing='ij'); beta=theta_b.ravel(); delta=theta_d.ravel()
    records=[]
    for seller in ['AMS Traders Inc.','GadgetPickup','UPGRADE SOLUTION INC','Certified Cell, Inc','SaveGadget']:
      base=d[d.Seller.eq(seller)].sort_values(['Code','date']).copy()
      base['anchor']=base.groupby('Code').Ref_Price.transform('first')
      for unit in ['listing_day','listing_entry']:
        s=base if unit=='listing_day' else base.drop_duplicates('Code')
        qa=fit.predict(s); ref=s.Ref_Price.to_numpy(); ref0=s.anchor.to_numpy(); price=s.Price.to_numpy()
        Ps=[]; Qs=[]
        for a in np.arange(-250,251,10)+.01:
            p=np.maximum(1,ref+a); x=s.copy(); x['residual']=p-ref
            Ps.append(p); Qs.append(fit.predict(x))
        for a in np.arange(.5,1.501,.025):
            p=ref*a; x=s.copy(); x['residual']=p-ref
            Ps.append(p); Qs.append(fit.predict(x))
        # The option not to offer the device is a zero-payoff comparator.
        Ps.append(np.zeros(len(s))); Qs.append(np.zeros(len(s)))
        P=np.column_stack(Ps); Q=np.column_stack(Qs)
        a=(Q*P).mean(axis=0)-(qa*price).mean()
        b=((Q-qa[:,None])*ref0[:,None]).mean(axis=0)
        c=(Q-qa[:,None]).mean(axis=0)
        regret=np.maximum(0,(a[:,None]-b[:,None]*beta-c[:,None]*delta).max(axis=0))
        feasible_cost=(beta*ref0.min()+delta>=0)&(beta*ref0.max()+delta>=0)
        regret[~feasible_cost]=np.inf
        for model in ['proportional','affine']:
          mask=delta==0 if model=='proportional' else np.ones(len(beta),bool)
          best=np.where(mask,regret,np.inf).argmin()
          for eps in [0,.1,.25,.5,1.,2.,5.]:
            ok=mask&(regret<=eps)
            mc=beta[ok]*ref0.mean()+delta[ok]
            row={'seller':seller,'unit':unit,'model':model,'N':len(s),'listings':s.Code.nunique(),'R_min':regret[best],'best_beta':beta[best],'best_offset':delta[best],'best_mean_cost':beta[best]*ref0.mean()+delta[best],'best_gross_markup':price.mean()-(beta[best]*ref0.mean()+delta[best]),'epsilon':eps,'cost_low':mc.min() if len(mc) else None,'cost_high':mc.max() if len(mc) else None,'mean_price':price.mean(),'boundary':bool(ok.any() and np.any(ok&((beta==betas.min())|(beta==betas.max())|(delta==offsets.min())|(delta==offsets.max()))))}
            records.append(row)
          print(json.dumps(records[-4],default=float),flush=True)
        pd.DataFrame(records).to_csv(OUT/'affine_cost_pilot.csv',index=False)
if __name__=='__main__':main()
