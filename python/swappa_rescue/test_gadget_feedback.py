"""Focused numerical checks of the sequential demand confidence construction."""
import unittest
import numpy as np
from scipy.special import betaln,xlogy
from gadget_feedback_audit import interval

class FeedbackBounds(unittest.TestCase):
    def test_one_observation_closed_form(self):
        alpha=.05/(5*1*2)
        self.assertTrue(np.allclose(interval(1,1,1,'mixture'),[alpha/2,1],atol=1e-12))
        self.assertTrue(np.allclose(interval(1,0,1,'mixture'),[0,1-alpha/2],atol=1e-12))
        self.assertEqual(interval(0,0,1,'mixture'),(0.,1.))

    def test_exact_finite_horizon_crossing_probability(self):
        # Enumerate all success-count states, dropping paths at first crossing.
        # This checks the implementation against the proved Ville bound.
        alpha=.05/(5*1*2)
        for q in [.05,.25,.5,.9]:
            alive=np.array([1.]);crossed=0.
            for n in range(1,51):
                new=np.r_[alive*(1-q),0.]+np.r_[0.,alive*q]
                s=np.arange(n+1)
                log_m=betaln(s+.5,n-s+.5)-betaln(.5,.5)-xlogy(s,q)-xlogy(n-s,1-q)
                hit=log_m>=np.log(1/alpha);crossed+=new[hit].sum();new[hit]=0.;alive=new
            self.assertLessEqual(crossed,alpha+1e-12)
            self.assertAlmostEqual(crossed+alive.sum(),1.,places=11)

    def test_fractional_bounds_and_convexity(self):
        for n,s in [(10,.25),(10,2.5),(20,17.75)]:
            lo,hi=interval(n,s,3,'mixture');self.assertLessEqual(lo,s/n);self.assertGreaterEqual(hi,s/n)
        y=np.linspace(0,1,101)
        for q in [.05,.3,.8]:
            for theta in [.01,.4,.99]:
                value=(theta/q)**y*((1-theta)/(1-q))**(1-y)
                chord=y*theta/q+(1-y)*(1-theta)/(1-q)
                self.assertTrue(np.all(value<=chord+1e-12))

if __name__=='__main__':unittest.main()
