"""Independent checks of own-inventory profit corrections."""
import unittest
import numpy as np
from scipy.special import expit,logit
from coordinate_profit import peer_spillover_bounds
from economic_envelope import profit_cap

class CoordinateProfitTests(unittest.TestCase):
    def test_exact_peer_changes(self):
        p=np.array([100.,120.,135.]);q=np.array([.1,.2,.3]);m=np.array([20.,30.,40.]);n=np.full(3,3);bp=.002
        cf=np.array([[80.,160.],[100.,145.],[110.,170.]])
        lo,hi=peer_spillover_bounds(cf,cf,p,logit(q),n,[np.arange(3)],m,m,bp,100.)
        exact=np.zeros_like(cf)
        for i in range(3):
          for a in range(2):
            for j in range(3):
                if j!=i:exact[i,a]+=m[j]*(expit(logit(q[j])+bp*(cf[i,a]-p[i])/2)-q[j])
        np.testing.assert_allclose(lo,exact,atol=1e-12);np.testing.assert_allclose(hi,exact,atol=1e-12)

    def test_interval_spillover_bounds(self):
        rng=np.random.default_rng(161016);p=np.array([100.,120.,135.,160.]);q=np.array([.1,.2,.3,.15]);ml=np.array([10.,15.,20.,5.]);mh=ml+30;nl=np.full(4,4);bp=.003
        pl=np.tile(p[:,None],(1,3))+np.array([-50.,-10.,20.]);ph=pl+25
        low,high=peer_spillover_bounds(pl,ph,p,logit(q),nl,[np.arange(4)],ml,mh,bp,100.)
        for _ in range(100):
            margins=ml+rng.random(4)*(mh-ml);cf=pl+rng.random(pl.shape)*(ph-pl);exact=np.zeros_like(cf)
            for i in range(4):
              for a in range(3):
                for j in range(4):
                    if j!=i:exact[i,a]+=margins[j]*(expit(logit(q[j])+bp*(cf[i,a]-p[i])/3)-q[j])
            self.assertTrue(np.all(exact>=low-1e-12));self.assertTrue(np.all(exact<=high+1e-12))

    def test_missing_peer_bound(self):
        p=np.array([100.,120.]);q=np.array([.1,.2]);m=np.array([20.,30.]);n=np.full(2,3);bp=.003;cf=np.array([[40.,200.],[50.,250.]])
        lo,hi=peer_spillover_bounds(cf,cf,p,logit(q),n,[np.arange(2)],m,m,bp,10.)
        for i in range(2):
          for a in range(2):
            shift=bp*(cf[i,a]-p[i])/2;j=1-i
            observed=m[j]*(expit(logit(q[j])+shift)-q[j])
            qm=.4;qcf=expit(logit(qm)+shift);mm=10/max(qm,qcf)
            actual=observed+mm*(qcf-qm)
            self.assertGreaterEqual(actual,lo[i,a]-1e-12);self.assertLessEqual(actual,hi[i,a]+1e-12)

    def test_total_profit_range(self):
        qb=.9;k=.03;bp=.003;Pmax=650.;K=profit_cap(qb,k);grid=np.linspace(100,Pmax,10001)
        for n in [1,2,5,20,100]:
            own=(grid-100)*expit(logit(qb)-k*(grid-100))
            if n>1:
                peermean=(grid+(n-2)*140)/(n-1)
                peers=(n-1)*40*expit(logit(qb)-k*40+bp*(peermean-Pmax))
            else:peers=np.zeros_like(grid)
            u=own+peers
            self.assertLessEqual(u.max()-u.min(),K*(1+bp*Pmax)+1e-10)

if __name__=='__main__':unittest.main()
