"""Scalar replacement-cost model for daily joint pricing within one SKU.

The common-price advisor maximizes finitely many affine-in-cost profits.
The total-profit range certificate decreases with the cost parameter. These
properties give continuous-cost enclosures without interpolation exclusions.
"""
import numpy as np,pandas as pd
from scipy.special import expit,lambertw
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, hstack
from ask_price_grid import recover_receipts
from daily_sku_bounds import peer_changes
from oracle_witness import maximum_own_profit

class DailyCostModel:
 def __init__(self,s,fit,covariance,index='constant'):
  self.range_scale=1.;self.s=s.reset_index(drop=True);s=self.s;self.fit=fit;self.covariance=covariance;self.theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];self.X=hstack([csr_matrix(np.ones((len(s),1))),csr_matrix(fit.scaler.transform(fit.continuous(s))),fit.encoder.transform(fit.cat(s))],format='csr').toarray();self.cols=np.array([fit.names.index(name)+1 for name in ['own','gap_positive','gap_negative','price_ref_interaction','own_peer_price']]);self.seller_slopes=[(j+1,s.Seller.eq(name.split('::',1)[1]).to_numpy(),100*fit.scaler.scale_[j]) for j,name in enumerate(fit.names) if name.startswith('own_seller::')];self.scales=fit.scaler.scale_[self.cols-1];self.p=s.Price.to_numpy();self.x=recover_receipts(self.p);self.active=s.since_seen.gt(0).to_numpy();self.groups=list(s.groupby('market_id',sort=False).indices.values());self.dates,self.day=np.unique(s.date,return_inverse=True);self.D=len(self.dates);self.N=len(s);self.events=s.sale_terminal_status.to_numpy()*self.active;self.mean_receipt=float(self.events@self.x/self.events.sum());self.index=index
  if index=='constant':self.factor=np.ones(self.N)
  elif index=='reference':self.factor=s.Ref_Price.to_numpy()/float(s.loc[s.date.eq(s.date.min()),'Ref_Price'].iloc[0])
  else:raise ValueError(index)
  for ids in self.groups:assert np.ptp(self.factor[ids])<1e-10
  self.target_factor=float(self.events@self.factor/self.events.sum())
  assert self.target_factor>0
  self.cap=float(np.min(self.x/self.factor));self.P=350.;self.rival=s.rival_min.to_numpy();self.logref=np.log(s.Ref_Price.to_numpy()/500);self.receipt=1.03*np.arange(1,int(np.floor(self.P/1.03))+1);self.newx=np.where(self.active[:,None],self.receipt[None,:],self.x[:,None]);self.newp=np.ceil(self.newx-1e-9);self.change=self.newp-self.p[:,None];self.peer=peer_changes(self.change,self.groups);self.gap_plus=np.maximum(self.newp-self.rival[:,None],0)-np.maximum(self.p-self.rival,0)[:,None];self.gap_minus=np.minimum(self.newp-self.rival[:,None],0)-np.minimum(self.p-self.rival,0)[:,None];self.up=peer_changes((self.active*(self.P-self.p))[:,None],self.groups)[:,0];self.down_slope=peer_changes((self.active*self.factor)[:,None],self.groups)[:,0];self.down_intercept=peer_changes((-(self.active*self.p))[:,None],self.groups)[:,0];self.live=np.bincount(self.day,weights=self.active,minlength=self.D)>0;self.prepare(self.theta)
 def prepare(self,theta):
  beta=theta[self.cols]/(100*self.scales);self.b=beta[0]+beta[3]*self.logref;self.gp,self.gn,self.bp=beta[1],beta[2],beta[4]
  for j,mask,scale in self.seller_slopes:self.b+=theta[j]*mask/scale
  if self.bp<0 or np.any(self.b+self.gp>=0) or np.any(self.b+self.gn>=0):raise ValueError('Demand violates maintained slope signs')
  self.z=self.X@theta;qa=expit(self.z);qnew=expit(self.z[:,None]+self.b[:,None]*self.change+self.gp*self.gap_plus+self.gn*self.gap_minus+self.bp*self.peer);self.A=np.zeros((self.D,len(self.receipt)));self.B=self.A.copy();np.add.at(self.A,self.day,self.newx*qnew);np.add.at(self.B,self.day,self.factor[:,None]*qnew);self.obsA=np.bincount(self.day,weights=self.x*qa,minlength=self.D);self.obsB=np.bincount(self.day,weights=self.factor*qa,minlength=self.D);self.rA=self.A-self.obsA[:,None];self.rB=self.B-self.obsB[:,None];self.offset=self.z-self.b*self.p-self.gp*np.maximum(self.p-self.rival,0)-self.gn*np.minimum(self.p-self.rival,0);return self
 def ranges(self,costs):
  costs=np.atleast_1d(costs);c=self.factor[:,None]*costs[None,:];owncap=maximum_own_profit(c,(self.offset+self.bp*self.up)[:,None],self.b[:,None],self.gp,self.gn,self.rival[:,None],self.P);entry=(self.x[:,None]-c)*(expit(self.z+self.bp*self.up)[:,None]-expit(self.z[:,None]+self.bp*(self.down_intercept[:,None]+self.down_slope[:,None]*costs[None,:])));rows=np.where(self.active[:,None],owncap,entry);C=np.zeros((self.D,len(costs)));np.add.at(C,self.day,rows);assert np.all(C[self.live]>0);return self.range_scale*C
 def values(self,costs):
  costs=np.atleast_1d(costs);r=np.max(self.rA[:,:,None]-self.rB[:,:,None]*costs[None,None,:],axis=1);C=self.ranges(costs);ratio=r[self.live]/C[self.live];assert np.all(abs(ratio)<=1+1e-7);return np.log1p(.99*ratio).sum(axis=0)
 def interval_bounds(self,left,right):
  left=np.atleast_1d(left);right=np.atleast_1d(right);assert (right>=left).all();rl=self.rA[:,:,None]-self.rB[:,:,None]*left[None,None,:];rh=self.rA[:,:,None]-self.rB[:,:,None]*right[None,None,:];low=np.minimum(rl,rh).max(axis=1)[self.live];high=np.maximum(rl.max(axis=1),rh.max(axis=1))[self.live];Cl=self.ranges(left)[self.live];Ch=self.ranges(right)[self.live];assert (Cl>=Ch-1e-8).all();lo=low/np.where(low>=0,Cl,Ch);hi=high/np.where(high>=0,Ch,Cl);assert lo.max()<=1+1e-7 and hi.min()>=-1-1e-7;lo=np.maximum(-1,lo);hi=np.minimum(1,hi);return np.log1p(.99*lo).sum(axis=0),np.log1p(.99*hi).sum(axis=0)
 def certify(self,threshold,tolerance=1e-5,initial_step=2.):
  edges=np.linspace(0,self.cap,int(np.ceil(self.cap/initial_step))+1);pending=list(zip(edges[:-1],edges[1:]));done=[];rounds=0
  while pending:
   rounds+=1;assert rounds<40;next_cells=[]
   for start in range(0,len(pending),64):
    cells=pending[start:start+64];left=np.array([x[0] for x in cells]);right=np.array([x[1] for x in cells]);lower,upper=self.interval_bounds(left,right)
    for l,h,lo,hi in zip(left,right,lower,upper):
     label='accepted' if hi<=threshold-1e-10 else 'rejected' if lo>threshold+1e-10 else 'boundary' if h-l<=tolerance else None
     if label:done.append({'low':float(l),'high':float(h),'wealth_low':float(lo),'wealth_high':float(hi),'status':label})
     else:mid=(l+h)/2;next_cells.extend([(l,mid),(mid,h)])
   pending=next_cells
  done=sorted(done,key=lambda x:x['low']);assert abs(done[0]['low'])<1e-10 and abs(done[-1]['high']-self.cap)<1e-10;assert all(abs(a['high']-b['low'])<1e-10 for a,b in zip(done[:-1],done[1:]));runs=[]
  for row in done:
   if runs and runs[-1]['status']==row['status']:runs[-1]['high']=row['high']
   else:runs.append({'status':row['status'],'low':row['low'],'high':row['high']})
  regular=[r['status'] for r in runs]==['rejected','boundary','accepted','boundary','rejected'];roots=[]
  if regular:
   for r in [runs[1],runs[3]]:roots.append(float(brentq(lambda c:self.values([c])[0]-threshold,r['low'],r['high'],xtol=1e-10)))
  if roots and max(abs(self.values(roots)-threshold))>1e-6:regular=False;roots=[]
  return {'regular_two_boundary_set':regular,'roots':roots,'runs':runs,'cells':done,'rounds':rounds,'cost_cap':self.cap}
 def own_optimum_probabilities(self,mu):
  c=self.factor*mu;offset=self.offset+self.bp*self.up;values=[];probabilities=[];prices=[]
  for gap,left,right in [(self.gn,c,np.maximum(c,np.minimum(self.rival,self.P))),(self.gp,np.minimum(self.P,np.maximum(c,self.rival)),np.full_like(c,self.P))]:
   k=-(self.b+gap);intercept=offset-gap*self.rival;optimum=c+(1+lambertw(np.exp(intercept-k*c-1)).real)/k;a=np.clip(optimum,left,right);q=expit(offset+self.b*a+self.gp*np.maximum(a-self.rival,0)+self.gn*np.minimum(a-self.rival,0));values.append((a-c)*q);probabilities.append(q);prices.append(a)
  values=np.array(values);probabilities=np.array(probabilities);prices=np.array(prices);maximum=values.max(axis=0);plausible=values>=maximum[None,:]-1e-10;qmin=np.where(plausible,probabilities,np.inf).min(axis=0);qmax=np.where(plausible,probabilities,-np.inf).max(axis=0);different=np.where(plausible,prices,-np.inf).max(axis=0)-np.where(plausible,prices,np.inf).min(axis=0)>1e-6
  return maximum,qmin,qmax,different
 def range_derivative_bounds(self,left,right):
  _,_,qleft,_=self.own_optimum_probabilities(left);_,qright,_,_=self.own_optimum_probabilities(right);ownlo=-self.factor*qleft;ownhi=-self.factor*qright;qlo=expit(self.z+self.bp*(self.down_intercept+self.down_slope*left));qhi=expit(self.z+self.bp*(self.down_intercept+self.down_slope*right));maximum=expit(self.z+self.bp*self.up);vmin=np.minimum(qlo*(1-qlo),qhi*(1-qhi));vmax=np.where((qlo<=.5)&(qhi>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)));k=self.bp*self.down_slope;entrylo=-self.factor*(maximum-qlo)-(self.x-self.factor*left)*k*vmax;entryhi=-self.factor*(maximum-qhi)-(self.x-self.factor*right)*k*vmin;lo=np.bincount(self.day,weights=np.where(self.active,ownlo,entrylo),minlength=self.D);hi=np.bincount(self.day,weights=np.where(self.active,ownhi,entryhi),minlength=self.D);assert (lo<=hi+1e-8).all();return self.range_scale*lo,self.range_scale*hi
 def root_derivative_certificate(self,root,left,right):
  choice=np.argmax(self.A-self.B*root,axis=1);selected=np.arange(self.D);gaps=[]
  for c in [left,right]:
   profits=self.A-self.B*c;best=profits[selected,choice].copy();profits[selected,choice]=-np.inf;gaps.append(best-profits.max(axis=1))
  minimum_gap=float(np.minimum(*gaps)[self.live].min());assert minimum_gap>0,'Advisor changes inside root enclosure'
  rl=(self.rA-self.rB*left)[selected,choice][self.live];rh=(self.rA-self.rB*right)[selected,choice][self.live];slope=-self.rB[selected,choice][self.live];C=self.ranges([left,right])[self.live];Cl,Ch=C[:,0],C[:,1];dlo,dhi=self.range_derivative_bounds(left,right);vlow,vhigh=-dhi[self.live],-dlo[self.live];ratiocorners=np.array([r/cc for r in [rl,rh] for cc in [Cl,Ch]]);rmin=np.maximum(-1,ratiocorners.min(axis=0));rmax=np.minimum(1,ratiocorners.max(axis=0));first=np.array([slope/Cl,slope/Ch]);second=np.array([r*v/(cc*cc) for r in [rl,rh] for v in [vlow,vhigh] for cc in [Cl,Ch]]);dlow=first.min(axis=0)+second.min(axis=0);dhigh=first.max(axis=0)+second.max(axis=0);derivatives=np.array([.99*d/(1+.99*r) for d in [dlow,dhigh] for r in [rmin,rmax]]);_,_,_,distinct=self.own_optimum_probabilities(root)
  return {'cost_bracket':[left,right],'minimum_advisor_gap_over_bracket':minimum_gap,'wealth_derivative_interval':[float(derivatives.min(axis=0).sum()),float(derivatives.max(axis=0).sum())],'distinct_own_optimum_ties':int((distinct&self.active).sum())}
 def regularity(self,roots,brackets=None):
  result=[]
  for j,root in enumerate(roots):
   profits=self.A-self.B*root;sort=np.sort(profits[self.live],axis=1);gaps=sort[:,-1]-sort[:,-2];derivatives={str(h):float((self.values([root+h])[0]-self.values([root-h])[0])/(2*h)) for h in [1e-3,1e-4,1e-5]};row={'root':root,'wealth_cost_derivatives':derivatives,'minimum_active_advisor_profit_gap':float(gaps.min()),'advisor_profit_gap_quantiles':np.quantile(gaps,[0,.1,.5,1]).tolist()}
   if brackets is not None:row['continuous_root_certificate']=self.root_derivative_certificate(root,brackets[j]['low'],brackets[j]['high'])
   result.append(row)
  return result
