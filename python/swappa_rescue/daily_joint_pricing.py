"""Fitted-demand diagnostic for one joint inventory-pricing decision per day.

A deterministic advisor chooses a common cost-plus markup separately within
SKU/date, from a fixed menu, and reprices continuation items simultaneously.
Entry prices stay fixed; their profit spillovers remain in firm payoffs.
This is a different learner class, not a reinterpretation of listing rounds.
No cost confidence interval or causal identification is claimed here.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts
from oracle_witness import maximum_own_profit


def joint_logits(z,p,proposed,b,gp,gn,rival,bp,groups):
 delta=proposed-p[:,None];peer=np.zeros_like(delta)
 for ids in groups:
  if len(ids)>1:peer[ids]=(delta[ids].sum(axis=0)[None,:]-delta[ids])/(len(ids)-1)
 return z[:,None]+b[:,None]*delta+gp*(np.maximum(proposed-rival[:,None],0)-np.maximum(p-rival,0)[:,None])+gn*(np.minimum(proposed-rival[:,None],0)-np.minimum(p-rival,0)[:,None])+bp*peer


def evaluate(s,fit,cost,marks):
 codes,idx=np.unique(s.Code,return_inverse=True);c=cost[idx];p=s.Price.to_numpy();x=recover_receipts(p);active=s.since_seen.gt(0).to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());dates,day=np.unique(s.date,return_inverse=True);z=fit.logit(s);qa=expit(z);ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;assert bp>=0;rival=s.rival_min.to_numpy();P=350.
 px=1.03*np.minimum(np.floor(P/1.03),np.maximum(1,np.ceil((c[:,None]+marks)/1.03-1e-10)));px=np.where(active[:,None],px,x[:,None]);proposed=np.ceil(px-1e-9);q=expit(joint_logits(z,p,proposed,b,gp,gn,rival,bp,groups));profits=(px-c[:,None])*q;baseline=(x-c)*qa;advisor=np.zeros(len(dates));observed=np.bincount(day,weights=baseline,minlength=len(dates));hmax=np.zeros(len(s));hmin=np.zeros(len(s))
 for ids in groups:
  advisor[day[ids[0]]]+=profits[ids].sum(axis=0).max()
  if len(ids)>1:
   up=active[ids]*(P-p[ids]);down=active[ids]*(c[ids]-p[ids]);hmax[ids]=bp*(up.sum()-up)/(len(ids)-1);hmin[ids]=bp*(down.sum()-down)/(len(ids)-1)
 offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
 owncap=maximum_own_profit(c,offset+hmax,b,np.full(len(s),gp),np.full(len(s),gn),rival,P)
 rang=np.where(active,owncap,(x-c)*(expit(z+hmax)-expit(z+hmin)));C=np.bincount(day,weights=rang,minlength=len(dates));gain=advisor-observed;has=C>1e-10;assert np.all(np.abs(gain[has])<=C[has]+1e-7);ratio=gain[has]/C[has];wealth=float(np.log1p(.99*ratio).sum());threshold=float(np.log(4)+np.log(1e6)+np.log(1/.00025));events=s.sale_terminal_status.to_numpy()*active;return {'daily_rounds':int(has.sum()),'logwealth':wealth,'threshold':threshold,'passes':wealth<=threshold,'continuation_cost':float(events@c/events.sum()),'continuation_margin':float(events@(x-c)/events.sum()),'mean_normalized_gain':float(ratio.mean()),'max_normalized_gain':float(ratio.max()),'min_normalized_gain':float(ratio.min())}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cost-file',required=True);args=ap.parse_args();s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);codes=np.unique(s.Code);base=pd.read_csv(args.cost_file).set_index('Code').cost.reindex(codes).to_numpy();s['receipt']=recover_receipts(s.Price.to_numpy());cap=s.groupby('Code').receipt.min().reindex(codes).to_numpy();fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');marks=np.arange(0,201,5,dtype=float);records=[]
 for scale in np.arange(0,1.401,.05):
  rec={'cost_scale':float(scale),**evaluate(s,fit,np.minimum(scale*base,cap),marks)};records.append(rec);print(json.dumps(rec),flush=True)
 pd.DataFrame(records).to_csv(OUT/'daily_joint_pricing_diagnostic.csv',index=False);print('DAILY_JOINT_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
