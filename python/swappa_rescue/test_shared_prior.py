"""Exact checks for the shared-prior arbitrary-window bound."""
import unittest
import numpy as np

class SharedPriorTests(unittest.TestCase):
    def test_prior_and_retention(self):
        j=np.arange(1000);self.assertAlmostEqual(np.sum(1/((j+1)*(j+2)))+1/1001,1.)
        s,N=137,900;t=np.arange(s+1,s+N+1,dtype=float)
        self.assertAlmostEqual(np.log(t/(t+1)).sum(),np.log((s+1)/(s+N+1)),places=12)

    def test_arbitrary_window_wealth(self):
        # Enumerate all prehistory and window action paths; no simulation error.
        eta=np.tile(np.array([.5,.25,.125]),(2,1));pi=np.tile(np.array([.5,1/6,1/3])/2,(2,1))
        alpha=.25;T=8
        for start in [0,2,5]:
            states=[(1.,pi.copy(),np.zeros(2),np.zeros(2),1.,1.,0)]
            for t in range(1,T+1):
                new=[]
                for prob,w,R,V,Ws,maxratio,last in states:
                    if t==start+1:Ws=w.sum();maxratio=1.
                    u=np.array([.05,.95]) if last==0 else np.array([.85,.15]);ap=(eta*w).sum(1)/(eta*w).sum();expectedW=0.
                    for a in range(2):
                        r=u-u[a];raw=w*(1+eta*r[:,None]);wn=t/(t+1)*raw+pi*raw.sum()/(t+1)
                        expectedW+=ap[a]*wn.sum();rr=R+r if t>start else R;vv=V+r*r if t>start else V
                        self.assertTrue(np.all(wn/wn.sum()>=pi/(t+1)-1e-14))
                        ratio=max(maxratio,wn.sum()/Ws) if t>start else 1.
                        if t>start:
                            lower=np.log(pi)-np.log(t+1)+eta*rr[:,None]-eta**2*vv[:,None]
                            self.assertTrue(np.all(np.log(wn/Ws)>=lower-1e-12))
                            bound=((np.log(1/pi)+np.log(t+1)+np.log(1/alpha))/eta+eta*vv[:,None]).min(1)
                            if np.any(rr>bound+1e-12):self.assertGreaterEqual(wn.sum()/Ws,1/alpha-1e-12)
                        new.append((prob*ap[a],wn,rr,vv,Ws,ratio,a))
                    self.assertAlmostEqual(expectedW,w.sum(),places=12)
                states=new
            self.assertLessEqual(sum(z[0] for z in states if z[5]>=1/alpha),alpha+1e-12)

if __name__=='__main__':unittest.main()
