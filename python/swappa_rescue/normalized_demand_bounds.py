"""Correlated demand bounds for profit-normalized learning certificates."""
import numpy as np
from scipy.special import expit,lambertw
from demand_band_bounds import LogitBand


def normalized_difference_bounds(zlo,zhi,dlo,dhi,left,right):
 """Extrema of [q(z+d+right)-q(z+d+left)]/q(z) on a rectangle.

There is no interior stationary point when left != right. Each edge has at
most one interior stationary point. All eight edge candidates are checked.
"""
 def value(z,d):
  u=np.exp(z);a=np.exp(d+left);b=np.exp(d+right)
  return (b-a)*(1+u)/((1+a*u)*(1+b*u))
 values=[]
 for d in [dlo,dhi]:
  a=np.exp(d+left);b=np.exp(d+right);u=-1+np.sqrt(np.maximum(0,1+(1-a-b)/(a*b)))
  stationary=np.clip(np.log(np.maximum(u,1e-300)),zlo,zhi)
  for z in [zlo,zhi,stationary]:values.append(value(z,d))
 for z in [zlo,zhi]:
  d=np.clip(-z-(left+right)/2,dlo,dhi);values.append(value(z,d))
 return np.minimum.reduce(values),np.maximum.reduce(values)


def probability_ratio_bounds(zlo,zhi,dlo,dhi):
 vals=[expit(z+d)/expit(z) for z in [zlo,zhi] for d in [dlo,dhi]]
 return np.minimum.reduce(vals),np.maximum.reduce(vals)


class NormalizedBand(LogitBand):
 def prepare_peers(self,groups):
  self.pair_data=[]
  for ids in groups:
   if len(ids)<=1:continue
   X=self.X[ids];V=X@self.cov@X.T;var=np.maximum(np.diag(V)[:,None]+np.diag(V)[None,:]-2*V,0)
   delta=self.z[ids][None,:]-self.z[ids][:,None];error=self.critical*np.sqrt(var)
   self.pair_data.append((ids,delta-error,delta+error))
 def bounds(self,cl,ch,marks,cap,n,grid_points=17,peer_cl=None,peer_ch=None):
  peer_cl=cl if peer_cl is None else peer_cl;peer_ch=ch if peer_ch is None else peer_ch
  ownlo=np.full((self.N,len(marks)),np.inf)
  for t in np.linspace(0,1,grid_points):
   c=cl+t*(ch-cl);a=np.minimum(cap,c[:,None]+marks);d,err,_=self.displacement(a)
   ratio,_=probability_ratio_bounds(self.zlo[:,None],self.zhi[:,None],d-err,d+err)
   ownlo=np.minimum(ownlo,(a-c[:,None])*ratio-(self.p[:,None]-c[:,None]))
  plo=np.minimum(cap,cl[:,None]+marks);d,err,_=self.displacement(plo);_,rhi=probability_ratio_bounds(self.zlo[:,None],self.zhi[:,None],d-err,d+err)
  lipschitz=np.maximum(1,np.maximum(rhi,marks*self.kmaxhi[:,None]*rhi));ownlo-=(ch-cl)[:,None]*lipschitz/(2*(grid_points-1))
  dc,err,_=self.displacement(cl[:,None]);dc=(dc+err)[:,0]
  # Lambert-W own-profit cap divided by the SAME baseline probability.
  stationary=np.clip(np.log(np.maximum(dc-1,1e-300)),self.zlo,self.zhi)
  vals=[lambertw(np.exp(z+dc-1)).real/self.kminlo/expit(z) for z in [self.zlo,self.zhi,stationary]]
  ranhi=np.minimum((cap-cl)/self.qlo,np.maximum.reduce(vals));spill=np.zeros_like(ownlo)
  for ids,dlo,dhi in self.pair_data:
   m=len(ids);ni=int(n[ids[0]]);assert ni==m
   mask=1-np.eye(m);zflo=self.zlo[ids][:,None];zfhi=self.zhi[ids][:,None]
   for a in range(len(marks)):
    change=(plo[ids,a]-self.p[ids])/(ni-1);h=np.where(change>=0,self.bplo*change,self.bphi*change)[:,None]
    lower,_=normalized_difference_bounds(zflo,zfhi,dlo,dhi,0.,h)
    margins=np.where(lower>=0,(self.p[ids]-peer_ch[ids])[None,:],(self.p[ids]-peer_cl[ids])[None,:]);spill[ids,a]=(lower*margins*mask).sum(1)
   left=self.bphi*(cl[ids]-self.p[ids])[:,None]/(ni-1);right=self.bphi*(cap-self.p[ids])[:,None]/(ni-1)
   _,upper=normalized_difference_bounds(zflo,zfhi,dlo,dhi,left,right)
   ranhi[ids]+=(upper*(self.p[ids]-peer_cl[ids])[None,:]*mask).sum(1)
  return ownlo+spill,ranhi
