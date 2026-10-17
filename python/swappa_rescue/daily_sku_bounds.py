"""Continuous-cost outer screens for daily joint pricing within a SKU.

A separate shared-prior learner prices each product group once per day.
All twenty-seven target SKU events receive an explicit prior/union allowance. Demand
uses a common five-coefficient slope ellipsoid, not independent point slopes.
Only the center is common: arbitrary persistent listing deviations up to
specified delta are allowed. Accepted boxes are not existence certificates.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import expit,lambertw
from scipy.stats import chi2
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from demand_band_bounds import LogitBand,own_gain_z_extrema,logit_difference_bounds
from ask_price_grid import recover_receipts


def peer_changes(delta,groups):
 result=np.zeros_like(delta)
 for ids in groups:
  if len(ids)>1:result[ids]=(delta[ids].sum(axis=0,keepdims=True)-delta[ids])/(len(ids)-1)
 return result


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--normalization',choices=['range','maximum'],default='range');ap.add_argument('--scale-factor',type=float,nargs='+',default=[1.,1.25,2.]);ap.add_argument('--device',default='apple-iphone-se-2nd-gen');ap.add_argument('--storage',default='64 GB');ap.add_argument('--condition',default='Good');ap.add_argument('--delta',nargs='+',type=float,default=[0,10,20]);ap.add_argument('--center-step',type=float,default=1);args=ap.parse_args();start=time.time();d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')];s=d[d.device_id.eq(args.device)&d.Storage.eq(args.storage)&d.Condition.eq(args.condition)].sort_values(['Code','date']).reset_index(drop=True);assert len(s);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/'demand_uniform_pairs_initial_p350_b50000_family20.npz');assert float(data['alpha_band'])<=.00175 and float(data['alpha_mc'])<=.0002;critical5=float(np.sqrt(chi2.ppf(.994,5)));band=LogitBand(fit,s,data['covariance'],float(data['critical']),slope_critical=critical5,peer_critical=critical5);codes,idx=np.unique(s.Code,return_inverse=True);p=s.Price.to_numpy();x=recover_receipts(p);cap=pd.Series(x,index=s.Code).groupby(level=0).min().reindex(codes).to_numpy();active=s.since_seen.gt(0).to_numpy();events=s.sale_terminal_status.to_numpy()*active;weights=np.bincount(idx,weights=events,minlength=len(codes))/events.sum();price=float(events@x/events.sum());groups=list(s.groupby('market_id',sort=False).indices.values());dates,day=np.unique(s.date,return_inverse=True);cols=np.r_[band.cols,fit.names.index('own_peer_price')+1];scales=fit.scaler.scale_[cols-1];theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]][cols];cov=data['covariance'][np.ix_(cols,cols)];P=350.;marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);records=[];summaries=[];threshold=float(np.log(4*27)+np.log(1e6)+np.log(1/.00025))
 def contrast(proposed,peer):
  change=proposed-p[:,None];rival=s.rival_min.to_numpy()[:,None];D=np.stack([change,np.maximum(proposed-rival,0)-np.maximum(p[:,None]-rival,0),np.minimum(proposed-rival,0)-np.minimum(p[:,None]-rival,0),change*np.log(s.Ref_Price.to_numpy()[:,None]/500),peer],axis=2)/(100*scales)
  center=D@theta;error=critical5*np.sqrt(np.maximum(np.einsum('nai,ij,naj->na',D,cov,D,optimize=True),0));return center,error
 with threadpool_limits(limits=2):
  for delta in args.delta:
   centers=np.linspace(0,cap.min()+delta,max(1,int(np.ceil((cap.min()+delta)/args.center_step)))+1);local=[]
   for a,b in zip(centers[:-1],centers[1:]):
    low=np.full(len(codes),max(0,a-delta));high=np.minimum(cap,b+delta);cl=low[idx];ch=high[idx];px=1.03*np.minimum(np.floor(P/1.03),np.maximum(1,np.ceil((ch[:,None]+marks)/1.03-1e-10)));px=np.where(active[:,None],px,x[:,None]);proposed=np.ceil(px-1e-9);center,error=contrast(proposed,peer_changes(proposed-p[:,None],groups));gains=[]
    for cost in [cl,ch]:
     lower,_=own_gain_z_extrema(band.zlo[:,None],band.zhi[:,None],center-error,px-cost[:,None],(x-cost)[:,None]);gains.append(lower)
    lower=np.minimum(*gains);daily=np.zeros((len(dates),len(marks)));np.add.at(daily,day,lower);regret=np.maximum(0,daily.max(axis=1))
    if args.normalization=='maximum':
     from joint_profit_ratio import daily_ratio_lower
     frac=daily_ratio_lower(band.zlo,band.zhi,center-error,px,x,cl,ch,day,len(dates))
    else:
     highpeer=peer_changes((active*(P-p))[:,None],groups);lowpeer=peer_changes((active*(cl-p))[:,None],groups);atcost,errcost=contrast(cl[:,None],highpeer);zupper=band.zhi+atcost[:,0]+errcost[:,0];ownrange=np.minimum(P-cl,lambertw(np.exp(zupper-1)).real/band.kminlo);_,entryrange=logit_difference_bounds(band.zlo,band.zhi,band.bphi*lowpeer[:,0],band.bphi*highpeer[:,0]);range_row=np.where(active,ownrange,(x-cl)*entryrange);range_day=np.bincount(day,weights=range_row,minlength=len(dates));frac=np.divide(regret,range_day,out=np.zeros_like(regret),where=range_day>1e-10)
    for scale in args.scale_factor:
     assert scale>=1
     wealth=float(np.log1p(.99*frac/scale).sum());rec={'normalization':args.normalization,'scale_factor':scale,'delta':delta,'center_low':float(a),'center_high':float(b),'logwealth_lower':wealth,'threshold':threshold,'accepted_outer_box':wealth<=threshold,'cost_low':float(weights@low),'cost_high':float(weights@high)};records.append(rec);local.append(rec)
   for scale in args.scale_factor:
    surviving=[r for r in local if r['accepted_outer_box'] and r['scale_factor']==scale];priorcap=np.minimum(cap,cap.min()+2*delta);summary={'normalization':args.normalization,'scale_factor':scale,'target_sku_family':27,'device':args.device,'storage':args.storage,'condition':args.condition,'delta':delta,'days':len(dates),'listings':len(codes),'continuation_sales':int(events.sum()),'mean_receipt':price,'prior_cost_low':0.,'prior_cost_high':float(weights@priorcap),'cost_lower':min(r['cost_low'] for r in surviving) if surviving else None,'cost_upper':max(r['cost_high'] for r in surviving) if surviving else None,'accepted_boxes':len(surviving),'total_boxes':len(centers)-1,'threshold':threshold,'critical5':critical5,'seconds':time.time()-start};summaries.append(summary);print(json.dumps(summary),flush=True)
 stem='daily_'+args.normalization+'_sku_'+args.device.replace('apple-iphone-','')+'_'+args.storage.replace(' ','')+'_'+args.condition;pd.DataFrame(records).to_csv(OUT/f'{stem}_boxes.csv',index=False);(OUT/f'{stem}_summary.json').write_text(json.dumps(summaries,indent=2));print('DAILY_SKU_OUTER_COMPLETE',flush=True)

if __name__=='__main__':main()
