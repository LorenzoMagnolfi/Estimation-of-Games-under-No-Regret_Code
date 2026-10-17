"""Certified fitted-demand upper bounds on joint inventory-pricing profit.

Partition the mean displayed price. At its interval upper endpoint, positive
peer-price effects give a separable profit upper bound. A scalar Lagrange
multiplier enforces the mean-price interval. Every multiplier gives an upper
bound; bisection only tightens it. The true integer ask grid is enumerated.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import recover_receipts
from oracle_witness import maximum_own_profit


def group_profit_upper(c,p,z,b,gp,gn,rival,bp,active,mean_step=10.,iterations=24,P=350.):
 n=len(c);x=recover_receipts(p);asks=np.arange(1,int(np.floor(P/1.03))+1);receipts=1.03*asks;display=np.ceil(receipts-1e-9);allowed=np.where(active[:,None],receipts[None,:]>=c[:,None]-1e-10,display[None,:]==p[:,None]);assert allowed.any(axis=1).all();offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
 if n==1:
  q=expit(offset[:,None]+b[:,None]*display+gp*np.maximum(display-rival[:,None],0)+gn*np.minimum(display-rival[:,None],0));return float(np.where(allowed,(receipts-c[:,None])*q,-np.inf).max())
 if not active.any():return float(((x-c)*expit(z)).sum())
 peers=(p.sum()-p)/(n-1);base=offset[:,None]-bp*peers[:,None]+(b[:,None]-bp/(n-1))*display+gp*np.maximum(display-rival[:,None],0)+gn*np.minimum(display-rival[:,None],0)
 minp=np.where(allowed,display,np.inf).min(axis=1);maxp=np.where(allowed,display,-np.inf).max(axis=1);lower=minp.mean();upper=maxp.mean();edges=np.linspace(lower,upper,max(1,int(np.ceil((upper-lower)/mean_step)))+1);best=-np.inf
 for lo,hi in zip(edges[:-1],edges[1:]):
  profits=np.where(allowed,(receipts[None,:]-c[:,None])*expit(base+bp*n*hi/(n-1)),-np.inf)
  def dual(lam):
   tilted=profits-lam*display[None,:];choice=np.argmax(tilted,axis=1);value=tilted[np.arange(n),choice].sum()+lam*n*(hi if lam>=0 else lo);return float(value),float(display[choice].sum())
  value,total=dual(0.);candidate=value
  if total<n*lo-1e-9 or total>n*hi+1e-9:
   target=n*(lo if total<n*lo else hi);left,right=(-1.,0.) if total<target else (0.,1.)
   for _ in range(30):
    lv,ls=dual(left);rv,rs=dual(right);candidate=min(candidate,lv,rv)
    if ls>=target-1e-9 and rs<=target+1e-9:break
    if ls<target:left*=2
    if rs>target:right*=2
   for _ in range(iterations):
    mid=(left+right)/2;v,s=dual(mid);candidate=min(candidate,v)
    if s>target:left=mid
    else:right=mid
  best=max(best,candidate)
 return float(best)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cost-file',required=True);ap.add_argument('--mean-step',type=float,default=10);args=ap.parse_args();start=time.time();s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cost=pd.read_csv(args.cost_file).set_index('Code').cost;c=s.Code.map(cost).to_numpy();p=s.Price.to_numpy();x=recover_receipts(p);assert ((c>=0)&(c<=x+1e-9)).all();z=fit.logit(s);active=s.since_seen.gt(0).to_numpy();rival=s.rival_min.to_numpy();ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;assert bp>=0;groups=list(s.groupby('market_id',sort=False).indices.values());dates,day=np.unique(s.date,return_inverse=True);upper=np.zeros(len(dates));baseline=np.bincount(day,weights=(x-c)*expit(z),minlength=len(dates));C=np.zeros(len(dates));records=[]
 for g,ids in enumerate(groups):
  m=len(ids);profit=group_profit_upper(c[ids],p[ids],z[ids],b[ids],gp,gn,rival[ids],bp,active[ids],args.mean_step);actual=float(((x[ids]-c[ids])*expit(z[ids])).sum());assert profit>=actual-1e-7
  hmax=np.zeros(m);hmin=np.zeros(m)
  if m>1:
   up=active[ids]*(350-p[ids]);down=active[ids]*(c[ids]-p[ids]);hmax=bp*(up.sum()-up)/(m-1);hmin=bp*(down.sum()-down)/(m-1)
  offset=z[ids]-b[ids]*p[ids]-gp*np.maximum(p[ids]-rival[ids],0)-gn*np.minimum(p[ids]-rival[ids],0);owncap=maximum_own_profit(c[ids],offset+hmax,b[ids],np.full(m,gp),np.full(m,gn),rival[ids],350.);range_value=float(np.where(active[ids],owncap,(x[ids]-c[ids])*(expit(z[ids]+hmax)-expit(z[ids]+hmin))).sum());upper[day[ids[0]]]+=min(profit,actual+range_value);C[day[ids[0]]]+=range_value;records.append({'market_id':int(s.market_id.iloc[ids[0]]),'upper_profit':profit,'observed_profit':actual,'range_certificate':range_value})
  if (g+1)%100==0:print(json.dumps({'stage':'groups','completed':g+1,'total':len(groups),'seconds':time.time()-start}),flush=True)
 mask=C>1e-10;gain=np.maximum(0,upper-baseline);wealth=float(np.log1p(.99*gain[mask]/C[mask]).sum());threshold=float(np.log(4)+np.log(1e6)+np.log(1/.00025));report={'mean_step':args.mean_step,'daily_rounds':int(mask.sum()),'oracle_logwealth_upper':wealth,'threshold':threshold,'certified_daily_oracle_event_witness':wealth<=threshold,'seconds':time.time()-start,'interpretation':'Fitted-demand event feasibility only, not a confidence interval. Daily joint oracle, fixed entry prices, current exogenous market state, and full expected-payoff learning remain maintained.'};print(json.dumps(report,indent=2),flush=True);(OUT/f'daily_oracle_h{args.mean_step:g}.json').write_text(json.dumps(report,indent=2));pd.DataFrame(records).to_csv(OUT/f'daily_oracle_h{args.mean_step:g}_groups.csv',index=False);print('DAILY_ORACLE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
