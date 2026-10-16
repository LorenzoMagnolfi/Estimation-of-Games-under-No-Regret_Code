"""Outer payoff/range bounds under a simultaneous structural-logit band.

The supplied critical value and covariance must cover all absolute logits,
price-contrast logits, own slopes, and the peer-price slope jointly. Numerical
helpers below do not themselves establish validity of that statistical band.
"""
import numpy as np
from scipy.special import expit,lambertw


def logit_difference_bounds(zlo,zhi,left_shift,right_shift):
    """Exact extrema in z of expit(z+right)-expit(z+left)."""
    middle=np.clip(-(left_shift+right_shift)/2,zlo,zhi)
    vals=[expit(z+right_shift)-expit(z+left_shift) for z in [zlo,zhi,middle]]
    return np.minimum.reduce(vals),np.maximum.reduce(vals)


def own_gain_z_extrema(zlo,zhi,shift,new_margin,old_margin):
    """Exact extrema of new_margin*q(z+shift)-old_margin*q(z)."""
    a=np.exp(shift);u=np.sqrt(np.maximum(new_margin*a,0));v=np.sqrt(np.maximum(old_margin,0));den=u-a*v
    odds=np.divide(v-u,den,out=np.full(np.broadcast_shapes(u.shape,v.shape),-1.),where=np.abs(den)>1e-20)
    stationary=np.where(odds>0,np.log(np.maximum(odds,1e-300)),zlo);stationary=np.clip(stationary,zlo,zhi)
    vals=[new_margin*expit(z+shift)-old_margin*expit(z) for z in [zlo,zhi,stationary]]
    return np.minimum.reduce(vals),np.maximum.reduce(vals)


class LogitBand:
    def __init__(self,fit,d,covariance,critical):
        from scipy.sparse import csr_matrix,hstack
        self.p=d.Price.to_numpy();self.rival=d.rival_min.to_numpy();self.ref=d.Ref_Price.to_numpy();self.fit=fit;self.cov=covariance;self.critical=critical;self.N=len(d)
        X=hstack([csr_matrix(np.ones((self.N,1))),csr_matrix(fit.scaler.transform(fit.continuous(d))),fit.encoder.transform(fit.cat(d))],format='csr').toarray()
        self.X=X
        theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];self.z=X@theta;self.var=np.einsum('ij,jk,ik->i',X,covariance,X,optimize=True)
        names=['own','gap_positive','gap_negative','price_ref_interaction'];self.cols=np.array([fit.names.index(x)+1 for x in names]);self.scales=fit.scaler.scale_[self.cols-1];self.C=covariance[np.ix_(self.cols,self.cols)];self.basecross=X@covariance[:,self.cols];self.theta=theta[self.cols]
        self.zlo=self.z-critical*np.sqrt(np.maximum(self.var,0));self.zhi=self.z+critical*np.sqrt(np.maximum(self.var,0));self.qlo=expit(self.zlo);self.qhi=expit(self.zhi)
        pricevec=np.zeros((self.N,4));pricevec[:,0]=1;pricevec[:,3]=np.log(self.ref/500)
        limits=[]
        for j in [1,2]:
            v=pricevec.copy();v[:,j]=1;v=v/(100*self.scales);point=-(v@self.theta);se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',v,self.C,v,optimize=True),0));limits.append((point-critical*se,point+critical*se))
        self.kminlo=np.minimum(limits[0][0],limits[1][0]);self.kminhi=np.minimum(limits[0][1],limits[1][1]);self.kmaxhi=np.maximum(limits[0][1],limits[1][1]);assert self.kminlo.min()>0
        j=fit.names.index('own_peer_price')+1;scale=100*fit.scaler.scale_[j-1];point=theta[j]/scale;se=np.sqrt(max(0,covariance[j,j]))/scale
        assert point+critical*se>=0
        self.bplo=max(0,point-critical*se);self.bphi=point+critical*se
    def displacement(self,price):
        delta=price-self.p[:,None];rival=self.rival[:,None];actual=self.p[:,None]
        D=np.stack([delta,np.maximum(price-rival,0)-np.maximum(actual-rival,0),np.minimum(price-rival,0)-np.minimum(actual-rival,0),delta*np.log(self.ref[:,None]/500)],axis=2)/(100*self.scales)
        center=D@self.theta;var=np.maximum(np.einsum('nai,ij,naj->na',D,self.C,D,optimize=True),0);se=np.sqrt(var)
        varfull=np.maximum(self.var[:,None]+2*np.einsum('nai,ni->na',D,self.basecross)+var,0)
        return center,self.critical*se,self.critical*np.sqrt(varfull)
    def logits(self,price):
        delta,err,abs_err=self.displacement(price);center=self.z[:,None]+delta
        low=np.maximum(center-abs_err,self.zlo[:,None]+delta-err);high=np.minimum(center+abs_err,self.zhi[:,None]+delta+err)
        return low,high
    def own_regret_box(self,cl,ch,marks,price_cap,grid_points=33):
        # Grid only computes a lower envelope; a uniform analytic Lipschitz
        # allowance covers every cost between nodes. This is not a grid CI.
        lower=np.full((self.N,len(marks)),np.inf);upper=-lower.copy()
        for t in np.linspace(0,1,grid_points):
            c=cl+t*(ch-cl);proposed=np.minimum(price_cap,c[:,None]+marks);margin=proposed-c[:,None];old=self.p[:,None]-c[:,None];center,error,_=self.displacement(proposed)
            lo,_=own_gain_z_extrema(self.zlo[:,None],self.zhi[:,None],center-error,margin,old)
            _,hi=own_gain_z_extrema(self.zlo[:,None],self.zhi[:,None],center+error,margin,old)
            lower=np.minimum(lower,lo);upper=np.maximum(upper,hi)
        pleft=np.minimum(price_cap,cl[:,None]+marks);pright=np.minimum(price_cap,ch[:,None]+marks);zleftlo,zlefthi=self.logits(pleft);zrightlo,zrighthi=self.logits(pright)
        qlow=expit(zrightlo);qhigh=expit(zlefthi);vmax=np.where((qlow<=.5)&(qhigh>=.5),.25,np.maximum(qlow*(1-qlow),qhigh*(1-qhigh)))
        lipschitz=np.maximum(self.qhi[:,None],marks*self.kmaxhi[:,None]*vmax)
        error=(ch-cl)[:,None]*lipschitz/(2*(grid_points-1))
        return lower-error,upper+error
    def peer_and_range_box(self,cl,ch,marks,price_cap,n,groups,peer_cl=None,peer_ch=None):
        peer_cl=cl if peer_cl is None else peer_cl;peer_ch=ch if peer_ch is None else peer_ch
        plo=np.minimum(price_cap,cl[:,None]+marks);phi=np.minimum(price_cap,ch[:,None]+marks);spill=np.zeros_like(plo)
        zcllo,zclhi=self.logits(cl[:,None]);zchlo,zchhi=self.logits(ch[:,None])
        rangehi=np.minimum(price_cap-cl,lambertw(np.exp(zclhi[:,0]-1)).real/self.kminlo)
        rangelo=np.minimum(price_cap-ch,lambertw(np.exp(zchlo[:,0]-1)).real/self.kminhi)
        for ids in groups:
            ni=int(n[ids[0]]);m=len(ids)
            if ni<=1:continue
            assert ni==m,'Robust prototype requires complete own-inventory groups.'
            mask=(1-np.eye(m));marginlo=(self.p[ids]-peer_ch[ids])[None,None,:];marginhi=(self.p[ids]-peer_cl[ids])[None,None,:]
            delta=(plo[ids]-self.p[ids,None])/(ni-1);shift=np.where(delta>=0,self.bplo*delta,self.bphi*delta)[:,:,None]
            dqlo,_=logit_difference_bounds(self.zlo[ids][None,None,:],self.zhi[ids][None,None,:],0.,shift)
            spill[ids]=(np.where(dqlo>=0,dqlo*marginlo,dqlo*marginhi)*mask[:,None,:]).sum(2)
            for c,coefficient,margin,target,which in [(cl,self.bphi,self.p[ids]-peer_cl[ids],rangehi,1),(ch,self.bplo,self.p[ids]-peer_ch[ids],rangelo,0)]:
                h0=coefficient*(c[ids]-self.p[ids])[:,None]/(ni-1);h1=coefficient*(price_cap-self.p[ids])[:,None]/(ni-1)
                bounds=logit_difference_bounds(self.zlo[ids][None,:],self.zhi[ids][None,:],h0,h1)
                target[ids]+=(bounds[which]*margin[None,:]*mask).sum(1)
        assert np.all(rangehi>=rangelo-1e-8)
        return spill,rangelo,rangehi
