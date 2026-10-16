"""Replacement-cost profiles with persistent listing-specific residuals.

c_l,t = rho * [(1-loading)*entry_reference_l+loading*current_reference_t] + e_l.
The e_l are fixed, bounded, and otherwise unrestricted. This explicitly changes
the cost interpretation from a fixed dollar inventory value to a market-linked
replacement/opportunity value.

Continuous proportional-cost boxes with bounded listing heterogeneity.

Outer screening: c_l = rho * initial_reference_l + e_l, |e_l|<=delta.
No density/independence assumption for e_l. Within each rho interval all costs
are enclosed, and every regret/range bound is outer. No accepted box proves
existence; rejection excludes every cost vector in that box.
"""
import argparse,json,time
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from demand_timing_sensitivity import terminal_status_data
from coordinate_profit import peer_spillover_bounds
from profit_specialists import coordinate_range,sleeping_cell_bounds,confidence_cell_bounds,markup_regret_interval


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference-loading',type=float,default=1.);ap.add_argument('--outcome',default='sale_terminal_status');ap.add_argument('--step',type=float,default=.02);ap.add_argument('--price-cap',type=float,default=350.);ap.add_argument('--demand',choices=['baseline','initial_control'],default='baseline');ap.add_argument('--continuation-only',action='store_true');ap.add_argument('--min-scale',type=int,default=0);ap.add_argument('--rates',type=int,default=8);ap.add_argument('--envelope',choices=['second_order','exact_log'],default='second_order');ap.add_argument('--rate-scale',type=float,default=.5);ap.add_argument('--activation',choices=['sleeping','confidence'],default='sleeping');ap.add_argument('--dispersion',type=float,nargs='+',default=[0.,5.,10.,25.]);args=ap.parse_args()
 d=terminal_status_data() if args.outcome=='sale_terminal_status' else data();s=d[d.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True)
 assert args.demand!='initial_control' or (args.continuation_only and args.outcome=='sale_terminal_status')
 fitname='multiproduct_initial_terminal.joblib' if args.demand=='initial_control' else ('multiproduct_demand.joblib' if args.outcome=='sale_last_seen' else f'multiproduct_{args.outcome}.joblib')
 fit=joblib.load(OUT/fitname)
 active=s.since_seen.gt(0).to_numpy() if args.continuation_only else np.ones(len(s),bool)
 suffix='' if args.demand=='baseline' and not args.continuation_only else f'_{args.demand}_continuation' 
 codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();cmax=s.groupby('Code',sort=True).Price.min().to_numpy()
 p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);actual_logit=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values())
 ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100
 offset=actual_logit-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);kmin=np.minimum(-(b+gp),-(b+gn));kmax=np.maximum(-(b+gp),-(b+gn));Pmax=args.price_cap
 assert bp>=0 and kmin.min()>0 and p.max()<=Pmax and 0<args.rate_scale<1 and 0<=args.reference_loading<=1
 cost_anchor=anchor[idx]+args.reference_loading*(ref-anchor[idx])
 assert args.envelope=='exact_log' or args.rate_scale<=.5
 def logits(price):return offset[:,None]+b[:,None]*price+gp*np.maximum(price-rival[:,None],0)+gn*np.minimum(price-rival[:,None],0)
 def prob(price):return expit(logits(price))
 marks=np.array([5.,10.,20.,30.,50.,100.,250.]);A=len(marks);Aall=int(Pmax)+1+11
 hs=np.arange(args.min_scale,11);caps=2.**hs;js=np.arange(args.rates);eta=args.rate_scale*2.**(-js[None,:])/caps[:,None];cp=1/((hs+1)*(hs+2));rp=1/((js+1)*(js+2));alpha=.005;Tmax=1000000
 events=s[args.outcome].to_numpy();sw=np.bincount(idx,weights=events,minlength=L)/events.sum();qw=np.bincount(idx,weights=qa,minlength=L)/qa.sum();prices={'sold':float(events@p/events.sum()),'expected_sales':float(qa@p/qa.sum())};weights={'sold':events/events.sum(),'expected_sales':qa/qa.sum()}
 if args.continuation_only:
  cevents=events*active;cq=qa*active
  weights['continuation_sold']=cevents/cevents.sum();weights['continuation_expected_sales']=cq/cq.sum();prices['continuation_sold']=float(cevents@p/cevents.sum());prices['continuation_expected_sales']=float(cq@p/cq.sum())
 records=[];summaries=[];rho_edges=np.arange(0,1.2000001,args.step)
 for delta in args.dispersion:
  start=time.time();local=[]
  for rlo,rhi in zip(rho_edges[:-1],rho_edges[1:]):
   elow=np.full(L,-delta);ehigh=np.full(L,delta)
   np.maximum.at(elow,idx,-rhi*cost_anchor);np.minimum.at(ehigh,idx,p-rlo*cost_anchor)
   if np.any(elow>ehigh):continue
   cl=np.maximum(0,rlo*cost_anchor+elow[idx]);ch=np.minimum(p,rhi*cost_anchor+ehigh[idx]);c=(cl+ch)/2;half=(ch-cl)/2
   pm=np.minimum(Pmax,c[:,None]+marks);qm=prob(pm);r=qm*(pm-c[:,None])-qa[:,None]*(p[:,None]-c[:,None]);plo=np.minimum(Pmax,cl[:,None]+marks);phi=np.minimum(Pmax,ch[:,None]+marks);qlo=prob(plo);qhi=prob(phi)
   vf=np.where((qhi<=.5)&(qlo>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)));deriv=np.maximum(qa[:,None],np.abs(qa[:,None]-marks*kmax[:,None]*vf));dcap=np.abs(qa[:,None]-prob(np.full((N,1),Pmax)));deriv=np.where(cl[:,None]+marks>=Pmax,dcap,np.where(ch[:,None]+marks<=Pmax,deriv,np.maximum(deriv,dcap)))
   spl,sph=peer_spillover_bounds(plo,phi,p,actual_logit,n,groups,p-ch,p-cl,bp,Pmax);ownlo,ownhi=markup_regret_interval(cl,ch,marks,p,qa,offset,b,gp,gn,rival,Pmax);lo=ownlo+spl;hi=ownhi+sph
   range_lo=coordinate_range(cl,p,actual_logit,n,groups,bp,Pmax,logits(cl[:,None]).ravel(),kmin,p-cl);range_hi=coordinate_range(ch,p,actual_logit,n,groups,bp,Pmax,logits(ch[:,None]).ravel(),kmin,p-ch)
   assert np.all(range_lo>=range_hi-1e-8)
   R=[];V=[];LOG=[]
   for cap in caps:
    rr,vv=(sleeping_cell_bounds if args.activation=='sleeping' else confidence_cell_bounds)(lo,hi,range_lo,range_hi,cap);rr=rr*active[:,None];vv=vv*active[:,None];R.append(rr.sum(axis=0));V.append(np.minimum(vv,cap*cap).sum(axis=0))
    e=args.rate_scale*2.**(-js)/cap
    LOG.append(np.log1p(np.maximum(rr,-cap)[:,:,None]*e[None,None,:]).sum(axis=0))
   R=np.array(R);V=np.array(V);LOG=np.array(LOG)
   for copies in [1,10]:
    rhs=copies*(np.log(Aall/cp[:,None]/rp[None,:])+np.log(Tmax)+np.log(copies/alpha))/eta
    violation=R[:,:,None]-eta[:,None,:]*V[:,:,None]-rhs[:,None,:] if args.envelope=='second_order' else (LOG-rhs[:,None,:]*eta[:,None,:])/eta[:,None,:];worst=np.unravel_index(np.argmax(violation),violation.shape);ok=violation.max()<=1e-8
    rec={'delta':delta,'rho_low':float(rlo),'rho_high':float(rhi),'copies':copies,'passes_outer_screen':bool(ok),'max_violation_dollars':float(violation.max()),'binding_cap':float(caps[worst[0]]),'binding_markup':float(marks[worst[1]]),'binding_rate':int(worst[2])}
    for name,w in weights.items():rec[name+'_cost_low']=float(w@cl);rec[name+'_cost_high']=float(w@ch)
    records.append(rec);local.append(rec)
  for copies in [1,10]:
   allrows=[r for r in local if r['copies']==copies];valid=[r for r in allrows if r['passes_outer_screen']]
   for name in weights:
    for kind,rows in [('prior',allrows),('behavior',valid)]:
     low=min([r[name+'_cost_low'] for r in rows],default=None);high=max([r[name+'_cost_high'] for r in rows],default=None)
     rec={'outcome':args.outcome,'reference_loading':args.reference_loading,'activation':args.activation,'envelope':args.envelope,'rate_scale':args.rate_scale,'min_scale':args.min_scale,'rates_tested':args.rates,'demand':args.demand,'continuation_only':args.continuation_only,'active_rows':int(active.sum()),'delta':delta,'copies':copies,'estimand':name,'kind':kind,'cost_low':low,'cost_high':high,'markup_low':prices[name]-high if high is not None else None,'markup_high':prices[name]-low if low is not None else None,'boxes':len(rows),'N':N,'L':L,'price_cap':Pmax,'rho_step':args.step,'alpha_R':alpha,'lifetime_clock_max':Tmax}
     summaries.append(rec);print(json.dumps(rec),flush=True)
  pd.DataFrame(records).to_csv(OUT/f'replacement_cost_boxes_load{args.reference_loading:g}_h{args.min_scale}_j{args.rates}_{args.activation}_{args.envelope}_r{args.rate_scale:g}_{args.outcome}_p{Pmax:g}_step{args.step:g}{suffix}.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/f'replacement_cost_profile_load{args.reference_loading:g}_h{args.min_scale}_j{args.rates}_{args.activation}_{args.envelope}_r{args.rate_scale:g}_{args.outcome}_p{Pmax:g}_step{args.step:g}{suffix}.csv',index=False)
  print(json.dumps({'stage':'dispersion_complete','delta':delta,'seconds':time.time()-start}),flush=True)
 print('REPLACEMENT_COST_PROFILE_COMPLETE',flush=True)
if __name__=='__main__':main()
