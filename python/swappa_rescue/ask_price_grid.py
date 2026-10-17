"""Swappa integer asking-price sensitivity, with unrounded payment units.
Maintained historical rule: display=ceil(1.03*integer ask); transaction price
before sales tax is 1.03*ask. Observed-grid compatibility is tested explicitly.
"""
import numpy as np
from scipy.special import lambertw,expit
from normalized_demand_bounds import NormalizedBand,probability_ratio_bounds,normalized_difference_bounds

def recover_receipts(display):
    ask=np.floor(np.asarray(display,dtype=float)/1.03+1e-9)
    receipt=1.03*ask
    if not np.all(np.ceil(receipt-1e-9)==display):raise ValueError('Observed price is outside maintained ask-price grid')
    return receipt

def rounded_proposal(cost,marks,cap):
    maxask=np.floor(cap/1.03+1e-9)
    ask=np.minimum(maxask,np.maximum(1,np.ceil((cost[:,None]+marks)/1.03-1e-10)))
    receipt=1.03*ask
    return receipt,np.ceil(receipt-1e-9)

class AskGridBand(NormalizedBand):
    def bounds(self,cl,ch,marks,cap,n,grid_points=17,peer_cl=None,peer_ch=None):
        peer_cl=cl if peer_cl is None else peer_cl;peer_ch=ch if peer_ch is None else peer_ch
        receipt=recover_receipts(self.p)
        # A fixed feasible grid action for every cost in this cell. The oracle
        # dominates these comparisons; no cost-grid smoothness assumption is used.
        x,proposal=rounded_proposal((cl+ch)/2,marks,cap)
        minimum=1.03*np.ceil(ch/1.03-1e-10)
        x=np.maximum(x,minimum[:,None]);proposal=np.ceil(x-1e-9)
        assert np.all(x>=ch[:,None]-1e-8) and np.all(proposal<=cap)
        d,err,_=self.displacement(proposal)
        ratio,_=probability_ratio_bounds(self.zlo[:,None],self.zhi[:,None],d-err,d+err)
        ownlo=np.minimum((x-cl[:,None])*ratio-(receipt-cl)[:,None],(x-ch[:,None])*ratio-(receipt-ch)[:,None])
        dc,err,_=self.displacement(cl[:,None]);dc=(dc+err)[:,0]
        stationary=np.clip(np.log(np.maximum(dc-1,1e-300)),self.zlo,self.zhi)
        vals=[lambertw(np.exp(z+dc-1)).real/self.kminlo/ expit(z) for z in [self.zlo,self.zhi,stationary]]
        ranhi=np.minimum((cap-cl)/self.qlo,np.maximum.reduce(vals));spill=np.zeros_like(ownlo)
        for ids,dlo,dhi in self.pair_data:
            m=len(ids);ni=int(n[ids[0]]);assert ni==m
            mask=1-np.eye(m);zflo=self.zlo[ids][:,None];zfhi=self.zhi[ids][:,None]
            for a in range(len(marks)):
                change=(proposal[ids,a]-self.p[ids])/(ni-1);h=np.where(change>=0,self.bplo*change,self.bphi*change)[:,None]
                lower,_=normalized_difference_bounds(zflo,zfhi,dlo,dhi,0.,h)
                margins=np.where(lower>=0,(receipt[ids]-peer_ch[ids])[None,:],(receipt[ids]-peer_cl[ids])[None,:]);spill[ids,a]=(lower*margins*mask).sum(1)
            left=self.bphi*(cl[ids]-self.p[ids])[:,None]/(ni-1);right=self.bphi*(cap-self.p[ids])[:,None]/(ni-1)
            _,upper=normalized_difference_bounds(zflo,zfhi,dlo,dhi,left,right)
            ranhi[ids]+=(upper*(receipt[ids]-peer_cl[ids])[None,:]*mask).sum(1)
        return ownlo+spill,ranhi
