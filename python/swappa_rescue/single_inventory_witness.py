"""Search actual cost witnesses using the stronger continuous-price oracle."""
import argparse,json
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from oracle_witness import maximum_own_profit


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seller',default='SaveGadget');ap.add_argument('--price-cap',type=float,default=1500.);ap.add_argument('--reference-loading',type=float,default=0.);args=ap.parse_args();P=args.price_cap
 allrows=terminal_status_data();allrows=allrows[allrows.Seller.eq(args.seller)].copy();allrows['factor']=(1-args.reference_loading)+args.reference_loading*allrows.Ref_Price/allrows.initial_reference;cmax=(allrows.Price/allrows.factor).groupby(allrows.Code).min()
 s=allrows[allrows.own_n.eq(1)].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');ef=fit.effects();p=s.Price.to_numpy();q=fit.predict(s);z=fit.logit(s);ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();factor=s.factor.to_numpy();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);active=s.since_seen.gt(0).to_numpy();records=[]
 for code,rawids in s.groupby('Code',sort=True).indices.items():
  ids=rawids[active[rawids]];upper=float(cmax.loc[code])
  def loss(cbase):
   c=cbase*factor[ids];U=maximum_own_profit(c,offset[ids],b[ids],gp,gn,rival[ids],P);r=np.maximum(0,1-(p[ids]-c)*q[ids]/np.maximum(U,1e-300));return float(np.log1p(.99*r).sum())
  if len(ids):
   grid=np.linspace(0,upper,101);values=np.array([loss(c) for c in grid]);k=int(values.argmin());a=grid[max(0,k-1)];bb=grid[min(100,k+1)];res=minimize_scalar(loss,bounds=(a,bb),method='bounded',options={'xatol':1e-8});cbase=float(res.x) if res.fun<values[k] else float(grid[k]);val=loss(cbase)
  else:cbase=.7*upper;val=0.
  records.append({'Code':code,'base_cost':cbase,'logwealth':val,'active_rows':len(ids),'price_min':float(p[rawids].min()),'price_max':float(p[rawids].max()),'cost_cap':upper,'ref_min':float(ref[rawids].min()),'ref_max':float(ref[rawids].max()),'q_min':float(q[rawids].min()),'q_max':float(q[rawids].max())})
 result=pd.DataFrame(records);mapping=result.set_index('Code').base_cost;c=s.Code.map(mapping).to_numpy()*factor;events=s.sale_terminal_status.to_numpy();cevents=events*active;threshold=np.log(4*(int(P)+13))+np.log(1000000)+np.log(1/.005)
 report={'seller':args.seller,'reference_loading':args.reference_loading,'continuous_oracle_logwealth':float(result.logwealth.sum()),'D1_threshold':float(threshold),'passes_D1':bool(result.logwealth.sum()<=threshold),'cost_sold':float(events@c/events.sum()),'cost_continuation_sold':float(cevents@c/cevents.sum()),'margin_continuation_sold':float(cevents@(p-c)/cevents.sum()),'top_ten_loss_share':float(result.nlargest(10,'logwealth').logwealth.sum()/result.logwealth.sum())};print(json.dumps(report,indent=2),flush=True);print(result.nlargest(10,'logwealth').to_string(index=False),flush=True);safe=''.join(x if x.isalnum() else '_' for x in args.seller);stem=f'single_inventory_optimized_witness_{safe}_ref{args.reference_loading:g}';result.to_csv(OUT/f'{stem}.csv',index=False);(OUT/f'{stem}.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
