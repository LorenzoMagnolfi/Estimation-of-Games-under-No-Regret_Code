"""Independent checks for the proposed randomized second-order envelope."""
import unittest
import numpy as np
from second_order_diagnostic import second_order_bound, max_profit
import pandas as pd
from scipy.special import expit, logit
from economic_envelope import profit_cap

class SecondOrderTests(unittest.TestCase):
    def test_piecewise_profit_maximum(self):
        class Fit:
            def effects(self):return {'own':-2.,'price_ref_interaction':0.,'gap_positive':-1.,'gap_negative':-.5}
            def logit(self,s):
                gap=s.Price.to_numpy()-s.rival_min.to_numpy()
                return 10.-.02*s.Price.to_numpy()-.01*np.maximum(gap,0)-.005*np.minimum(gap,0)
        fit=Fit();s=pd.DataFrame({'Price':[400.,450.,550.,600.],'Ref_Price':[500.]*4,'rival_min':[450.]*4})
        cost=np.array([0.,350.,480.,800.]);analytic=max_profit(s,fit,cost)
        grid=np.linspace(250,750,20001)
        logits=10-.02*grid-.01*np.maximum(grid-450,0)-.005*np.minimum(grid-450,0)
        brute=np.maximum(0,(expit(logits)[None,:]*(grid[None,:]-cost[:,None])).max(axis=1))
        self.assertTrue(np.all(analytic>=brute-1e-9))
        np.testing.assert_allclose(analytic,brute,atol=1e-5)

    def test_economic_profit_envelope(self):
        # Exact smooth envelope versus a dense grid; steeper piecewise demand
        # also stays below it, as required by the log-odds slope restriction.
        for qb in [.5,.75,.9,.99]:
            k=.03;m=np.linspace(0,1000,100001);cap=profit_cap(qb,k)
            profit=m*expit(logit(qb)-k*m)
            self.assertLessEqual(profit.max(),cap+1e-10)
            self.assertAlmostEqual(profit.max(),cap,places=6)
            kinkprofit=m*expit(logit(qb)-k*m-.04*np.maximum(m-20,0))
            self.assertLessEqual(kinkprofit.max(),cap+1e-10)

    def test_log_inequality(self):
        x=np.r_[np.linspace(-.5,0,1001),np.geomspace(1e-8,1000,1001)]
        self.assertTrue(np.all(np.log1p(x)>=x-x*x-1e-14))

    def test_wealth_and_exact_tail(self):
        # Enumerate all 2^9 action histories. State adapts to the last action.
        # Each branch carries its exact probability, so no Monte Carlo tolerance.
        A,J,T,alpha=2,3,9,.25
        eta=np.tile(np.array([.5,.25,.125]),(A,1))
        w0=np.ones((A,J))/(A*J)
        states=[(1.,w0,np.zeros(A),np.zeros(A),1.,0)]
        L=np.log(A*J/alpha); crossing_prob=0.; rejection_prob=0.
        for t in range(T):
            new=[]
            for prob,w,R,V,Wmax,last in states:
                u=np.array([.1,.9]) if last==0 else np.array([.85,.15])
                expertprob=(eta*w).sum(axis=1)/(eta*w).sum()
                expectedW=0.
                for action in range(A):
                    r=u-u[action]; wn=w*(1+eta*r[:,None]); rr=R+r; vv=V+r*r
                    expectedW+=expertprob[action]*wn.sum()
                    new.append((prob*expertprob[action],wn,rr,vv,max(Wmax,wn.sum()),action))
                    bound=(L/eta+eta*vv[:,None]).min(axis=1)
                    if np.any(rr>bound+1e-12): self.assertGreaterEqual(wn.sum(),1/alpha-1e-12)
                    np.testing.assert_array_less(np.log(w0)+eta*rr[:,None]-eta**2*vv[:,None]-1e-12,np.log(wn))
                self.assertAlmostEqual(expectedW,w.sum(),places=12)
            states=new
        self.assertAlmostEqual(sum(x[0] for x in states),1.,places=12)
        crossing_prob=sum(x[0] for x in states if x[4]>=1/alpha)
        self.assertLessEqual(crossing_prob,alpha+1e-12)

    def test_rate_grid_and_feedback_monotonicity(self):
        V=np.array([10.,30.,0.]); K=np.array([1.,2.,5.]); L=np.log(5020)
        b1=second_order_bound(V,K,1000,L,1)
        b10=second_order_bound(V,K,1000,L,10)
        self.assertTrue(np.all(b10>=b1))
        self.assertTrue(np.all(second_order_bound(V,K*2,1000,L)>=b1-1e-12))
        self.assertGreater(b1[2],0.)

if __name__=='__main__':unittest.main()
