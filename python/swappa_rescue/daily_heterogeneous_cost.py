"""Conservative daily-wealth screen with persistent heterogeneous SKU costs.

The target mu is the empirical continuation-sale-weighted mean cost. Each
listing cost is within delta of mu, is nonnegative, and is no larger than
any observed receipt on that listing. We relax the mean constraint when
minimizing gains, so the surviving set is an OUTER set, not a feasibility
certificate. Delta zero exactly recovers DailyCostModel.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, lambertw
from daily_cost_model import DailyCostModel
from daily_sku_bounds import peer_changes
from oracle_witness import maximum_own_profit


class DailyHeterogeneousCostModel(DailyCostModel):
    def __init__(self, s, fit, covariance, delta):
        self.delta = float(delta)
        assert self.delta >= 0
        super().__init__(s, fit, covariance, 'constant')
        caps = self.s.assign(receipt=self.x).groupby('Code').receipt.min()
        self.row_cap = self.s.Code.map(caps).to_numpy()
        weights = pd.Series(self.events, index=self.s.Code).groupby(level=0).sum()
        # A listing with the smallest observed receipt requires mu <= mincap+delta.
        # Every true cost is also <= min(cap_i,mincap+2delta).
        self.cap = min(float(caps.min()+delta), float(weights @ np.minimum(caps, caps.min()+2*delta)/weights.sum()))
        self.cap_values, self.cap_index = np.unique(self.row_cap, return_inverse=True)
        self.day_active_cap = np.zeros(self.D)
        np.maximum.at(self.day_active_cap, self.day, np.where(self.active,self.row_cap,0))
        self.prepare(self.theta)

    def prepare(self, theta):
        super().prepare(theta)
        if not hasattr(self,'row_cap'):
            return self
        qa = expit(self.z)
        qnew = expit(self.z[:,None]+self.b[:,None]*self.change+self.gp*self.gap_plus+self.gn*self.gap_minus+self.bp*self.peer)
        dq = qnew-qa[:,None]
        self.negative = np.zeros_like(self.rA)
        np.add.at(self.negative,self.day,np.minimum(dq,0))
        self.positive = np.zeros((len(self.cap_values),self.D,len(self.receipt)))
        np.add.at(self.positive,(self.cap_index,self.day),np.maximum(dq,0))
        return self

    def cost_box(self, mu):
        return np.full(self.N,max(0,float(mu)-self.delta)), np.minimum(self.row_cap,float(mu)+self.delta)

    def vector_range(self, c):
        own = maximum_own_profit(c,self.offset+self.bp*self.up,self.b,self.gp,self.gn,self.rival,self.P)
        down = peer_changes((self.active*(c-self.p))[:,None],self.groups)[:,0]
        entry = (self.x-c)*(expit(self.z+self.bp*self.up)-expit(self.z+self.bp*down))
        result = np.bincount(self.day,weights=np.where(self.active,own,entry),minlength=self.D)
        assert np.all(result[self.live]>0)
        return self.range_scale*result

    def gain(self, low_mu, high_mu):
        # This lower bound holds for EVERY mu in [low_mu,high_mu].
        return self.rA-self.negative*max(0,low_mu-self.delta)-np.einsum('k,kda->da',np.minimum(self.cap_values,high_mu+self.delta),self.positive)

    def allowed(self, mu):
        required = np.minimum(self.day_active_cap,mu+self.delta)
        return self.receipt[None,:] >= required[:,None]-1e-10

    def values(self, costs):
        result=[]
        for mu in np.atleast_1d(costs):
            low,high=self.cost_box(mu)
            gain=np.where(self.allowed(mu),self.gain(mu,mu),-np.inf).max(axis=1)[self.live]
            cup=self.vector_range(low)[self.live]
            clo=self.vector_range(high)[self.live]
            ratio=gain/np.where(gain>=0,cup,clo)
            # A relaxed costwise lower gain can be below -C_lower.
            ratio=np.maximum(-1,ratio)
            assert np.all(ratio<=1+1e-7)
            result.append(float(np.log1p(.99*ratio).sum()))
        return np.array(result)

    def interval_bounds(self,left,right):
        lows=[];highs=[]
        for l,h in zip(np.atleast_1d(left),np.atleast_1d(right)):
            assert h>=l
            gainlo=np.where(self.allowed(h),self.gain(l,h),-np.inf).max(axis=1)[self.live]
            gainhi=np.where(self.allowed(l),self.gain(h,l),-np.inf).max(axis=1)[self.live]
            ll,lh=self.cost_box(l);hl,hh=self.cost_box(h)
            # C(low cost) and C(high cost) decrease in mu. The denominator
            # switches between them when the lower gain changes sign.
            cup_l=self.vector_range(ll)[self.live];cup_h=self.vector_range(hl)[self.live]
            clo_l=self.vector_range(lh)[self.live];clo_h=self.vector_range(hh)[self.live]
            low=gainlo/np.where(gainlo>=0,cup_l,clo_h)
            high=gainhi/np.where(gainhi>=0,cup_h,clo_l)
            lows.append(float(np.log1p(.99*np.maximum(-1,low)).sum()))
            highs.append(float(np.log1p(.99*np.maximum(-1,high)).sum()))
        return np.array(lows),np.array(highs)

    def vector_optima(self,c):
        offset=self.offset+self.bp*self.up;values=[];qs=[];prices=[]
        for gap,left,right in [(self.gn,c,np.maximum(c,np.minimum(self.rival,self.P))),(self.gp,np.minimum(self.P,np.maximum(c,self.rival)),np.full_like(c,self.P))]:
            k=-(self.b+gap);intercept=offset-gap*self.rival
            price=np.clip(c+(1+lambertw(np.exp(intercept-k*c-1)).real)/k,left,right)
            q=expit(offset+self.b*price+self.gp*np.maximum(price-self.rival,0)+self.gn*np.minimum(price-self.rival,0))
            values.append((price-c)*q);qs.append(q);prices.append(price)
        values=np.array(values);qs=np.array(qs);prices=np.array(prices);best=values.max(axis=0);possible=values>=best[None,:]-1e-10
        different=np.where(possible,prices,-np.inf).max(axis=0)-np.where(possible,prices,np.inf).min(axis=0)>1e-6
        return np.where(possible,qs,np.inf).min(axis=0),np.where(possible,qs,-np.inf).max(axis=0),different

    def vector_range_derivative_bounds(self,c_left,c_right,slope):
        _,qleft,_=self.vector_optima(c_left);qright,_,_=self.vector_optima(c_right)
        ownlo=-slope*qleft;ownhi=-slope*qright
        shiftleft=peer_changes((self.active*(c_left-self.p))[:,None],self.groups)[:,0]
        shiftright=peer_changes((self.active*(c_right-self.p))[:,None],self.groups)[:,0]
        k=self.bp*peer_changes((self.active*slope)[:,None],self.groups)[:,0]
        qlo=expit(self.z+self.bp*shiftleft);qhi=expit(self.z+self.bp*shiftright);maximum=expit(self.z+self.bp*self.up)
        vmin=np.minimum(qlo*(1-qlo),qhi*(1-qhi));vmax=np.where((qlo<=.5)&(qhi>=.5),.25,np.maximum(qlo*(1-qlo),qhi*(1-qhi)))
        entrylo=-slope*(maximum-qlo)-(self.x-c_left)*k*vmax
        entryhi=-slope*(maximum-qhi)-(self.x-c_right)*k*vmin
        lo=np.bincount(self.day,weights=np.where(self.active,ownlo,entrylo),minlength=self.D)
        hi=np.bincount(self.day,weights=np.where(self.active,ownhi,entryhi),minlength=self.D)
        assert np.all(lo<=hi+1e-8)
        return self.range_scale*lo,self.range_scale*hi

    def root_derivative_certificate(self,root,left,right):
        knots=np.r_[self.delta,self.cap_values-self.delta,self.receipt-self.delta]
        assert not np.any((knots>=left)&(knots<=right)), 'Cost or menu kink inside root bracket'
        assert np.array_equal(self.allowed(left),self.allowed(right))
        gains=np.where(self.allowed(root),self.gain(root,root),-np.inf);choice=np.argmax(gains,axis=1);selected=np.arange(self.D);gaps=[]
        for mu in [left,right]:
            profits=np.where(self.allowed(mu),self.gain(mu,mu),-np.inf);best=profits[selected,choice].copy();profits[selected,choice]=-np.inf;gaps.append(best-profits.max(axis=1))
        minimum_gap=float(np.minimum(*gaps)[self.live].min());assert minimum_gap>0
        rleft=self.gain(left,left)[selected,choice];rright=self.gain(right,right)[selected,choice]
        rprime=(-self.negative*float(root>self.delta)-np.einsum('k,kda->da',(root+self.delta<self.cap_values).astype(float),self.positive))[selected,choice]
        zero=(abs(rleft)<1e-10)&(abs(rright)<1e-10)&(abs(rprime)<1e-10)
        assert np.all(((rleft*rright)>0)|zero|~self.live), 'Regret sign kink inside root bracket'
        ll,lh=self.cost_box(left);hl,hh=self.cost_box(right)
        ls=np.full(self.N,float(root>self.delta));hs=(root+self.delta<self.row_cap).astype(float)
        dlo1,dhi1=self.vector_range_derivative_bounds(ll,hl,ls);dlo2,dhi2=self.vector_range_derivative_bounds(lh,hh,hs)
        positive=rleft>=0
        Cleft=np.where(positive,self.vector_range(ll),self.vector_range(lh))[self.live]
        Cright=np.where(positive,self.vector_range(hl),self.vector_range(hh))[self.live]
        vlo=-np.where(positive,dhi1,dhi2)[self.live];vhi=-np.where(positive,dlo1,dlo2)[self.live]
        rl=rleft[self.live];rh=rright[self.live];rs=rprime[self.live]
        corners=np.array([r/c for r in [rl,rh] for c in [Cleft,Cright]])
        rmin=corners.min(axis=0);rmax=corners.max(axis=0);clipped=rmax<-1
        assert np.all((rmin>-1)|clipped), 'Clipping kink inside root bracket'
        rmin=np.maximum(-1,rmin);rmax=np.maximum(-1,rmax)
        first=np.array([rs/Cleft,rs/Cright]);second=np.array([r*v/(c*c) for r in [rl,rh] for v in [vlo,vhi] for c in [Cleft,Cright]])
        dlow=first.min(axis=0)+second.min(axis=0);dhigh=first.max(axis=0)+second.max(axis=0)
        derivatives=np.array([.99*d/(1+.99*r) for d in [dlow,dhigh] for r in [rmin,rmax]])
        derivatives[:,clipped]=0
        lowcost,highcost=self.cost_box(root);ties=self.vector_optima(lowcost)[2]|self.vector_optima(highcost)[2]
        chosen=choice[self.day];ix=np.arange(self.N);shift=self.b*self.change[ix,chosen]+self.gp*self.gap_plus[ix,chosen]+self.gn*self.gap_minus[ix,chosen]+self.bp*self.peer[ix,chosen]
        structural=(self.change[ix,chosen]==0)&(self.gap_plus[ix,chosen]==0)&(self.gap_minus[ix,chosen]==0)&(self.peer[ix,chosen]==0)
        nonstructural_ties=(abs(shift)<1e-10)&~structural
        assert not np.any(nonstructural_ties), 'Demand-sign kink needs directional inference'
        return {'cost_bracket':[left,right],'minimum_advisor_gap_over_bracket':minimum_gap,'wealth_derivative_interval':[float(derivatives.min(axis=0).sum()),float(derivatives.max(axis=0).sum())],'distinct_own_optimum_ties':int((ties&self.active).sum()),'minimum_nonstructural_logit_shift':float(np.min(abs(shift[~structural]))),'cost_menu_kink_distance':float(np.min(abs(knots-root))),'clipped_days':int(clipped.sum())}

    def regularity(self,roots,brackets=None):
        result=[]
        for j,root in enumerate(roots):
            profits=np.where(self.allowed(root),self.gain(root,root),-np.inf);order=np.sort(profits[self.live],axis=1);gaps=order[:,-1]-order[:,-2]
            row={'root':root,'wealth_cost_derivatives':{str(h):float((self.values([root+h])[0]-self.values([root-h])[0])/(2*h)) for h in [1e-3,1e-4,1e-5]},'minimum_active_advisor_profit_gap':float(gaps.min()),'advisor_profit_gap_quantiles':np.quantile(gaps,[0,.1,.5,1]).tolist()}
            if brackets is not None:row['continuous_root_certificate']=self.root_derivative_certificate(root,brackets[j]['low'],brackets[j]['high'])
            result.append(row)
        return result
