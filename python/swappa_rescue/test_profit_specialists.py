"""Range certificates, cost-cell activation, and sleeping-Prod identities."""
import unittest
import numpy as np
from scipy.special import expit
from profit_specialists import coordinate_range,sleeping_cell_bounds,confidence_cell_bounds,markup_regret_interval

class SpecialistTests(unittest.TestCase):
 def test_total_profit_range_and_monotonicity(self):
  rng=np.random.default_rng(161026)
  for n in [1,2,5,9]:
   for rep in range(20):
    P=350.;p=rng.uniform(70,300,n);cost=rng.uniform(0,p);k=rng.uniform(.015,.09,n);bp=.004;odds=rng.normal(-1,2,n);groups=[np.arange(n)];nn=np.full(n,n)
    costodds=odds+k*(p-cost)
    bound=coordinate_range(cost,p,odds,nn,groups,bp,P,costodds,k,p-cost)
    for i in range(n):
     prices=np.linspace(cost[i],P,151);q=expit(odds[i]-k[i]*(prices-p[i]));profit=q*(prices-cost[i])
     if n>1:
      for j in range(n):
       if i!=j:profit+=expit(odds[j]+bp*(prices-p[i])/(n-1))*(p[j]-cost[j])
     self.assertLessEqual(profit.max()-profit.min(),bound[i]+1e-10)
    lower=np.maximum(0,cost-10);upper=np.minimum(p,cost+10)
    bl=coordinate_range(lower,p,odds,nn,groups,bp,P,odds+k*(p-lower),k,p-lower)
    bh=coordinate_range(upper,p,odds,nn,groups,bp,P,odds+k*(p-upper),k,p-upper)
    self.assertTrue(np.all(bl>=bh-1e-10))
 def test_activation_outer_bounds(self):
  rng=np.random.default_rng(941)
  for trial in range(100):
   ra=rng.uniform(1,30,10);rb=ra*rng.random(10);lo=rng.normal(size=(10,4));hi=lo+rng.uniform(0,3,(10,4));cap=10.
   rl,vh=sleeping_cell_bounds(lo,hi,ra,rb,cap)
   for z in np.linspace(0,1,11):
    actual_r=lo+z*(hi-lo);active=(ra-z*(ra-rb)<=cap)[:,None];value=actual_r*active
    self.assertTrue(np.all(value>=rl-1e-12));self.assertTrue(np.all(value**2<=vh+1e-12))
 def test_sleeping_shared_wealth(self):
  # Exact action-tree check, including adaptive awake sets and unknown history.
  caps=np.array([.5,2.]);eta=np.broadcast_to((1/(2*caps))[:,None],(2,2));pi=np.full((2,2),.25);T=8;start=2
  states=[(1.,pi.copy(),np.zeros((2,2)),np.zeros((2,2)),1.,0)]
  for t in range(1,T+1):
   new=[]
   for pathprob,w,R,V,Ws,last in states:
    if t==start+1:Ws=w.sum()
    u=np.array([.1,.4]) if last==0 else np.array([.9,.1]);awake=np.broadcast_to((np.ptp(u)<=caps)[:,None],(2,2));mass=eta*w*awake;ap=mass.sum(0)/mass.sum();EW=0.
    for a in range(2):
     r=awake*(u-u[a])[None,:];raw=w*(1+eta*r);wn=t/(t+1)*raw+pi*raw.sum()/(t+1);rr=R+r if t>start else R;vv=V+r*r if t>start else V;EW+=ap[a]*wn.sum()
     self.assertTrue(np.all(wn/wn.sum()>=pi/(t+1)-1e-13))
     if t>start:
      lower=np.log(pi)-np.log(t+1)+eta*rr-eta**2*vv
      self.assertTrue(np.all(np.log(wn/Ws)>=lower-1e-12))
     new.append((pathprob*ap[a],wn,rr,vv,Ws,a))
    self.assertAlmostEqual(EW,w.sum(),places=12)
   states=new


 def test_confidence_interval_products(self):
  rng=np.random.default_rng(411)
  for trial in range(30):
   ra=rng.uniform(1,30,10);rb=ra*rng.random(10);lo=rng.uniform(-rb[:,None],0,(10,4));hi=rng.uniform(0,rb[:,None],(10,4));cap=5.
   rl,vh=confidence_cell_bounds(lo,hi,ra,rb,cap)
   for z in np.linspace(0,1,21):
    rr=lo+z*(hi-lo);K=ra-z*(ra-rb);f=np.minimum(1,cap/K)[:,None];v=f*rr
    self.assertTrue(np.all(v>=rl-1e-12));self.assertTrue(np.all(v*v<=vh+1e-12))
 def test_exact_cost_profit_extrema(self):
  rng=np.random.default_rng(9012);N=15;marks=np.array([5.,20.,80.,200.]);P=350.
  for rep in range(15):
   p=rng.uniform(80,320,N);cl=rng.uniform(0,p);ch=rng.uniform(cl,p);offset=rng.normal(3,2,N);b=-rng.uniform(.01,.06,N);gp=-.02;gn=-.005;rival=rng.uniform(50,300,N)
   qa=expit(offset+b*p+gp*np.maximum(p-rival,0)+gn*np.minimum(p-rival,0))
   lo,hi=markup_regret_interval(cl,ch,marks,p,qa,offset,b,gp,gn,rival,P)
   for x in np.linspace(0,1,101):
    c=cl+x*(ch-cl);pp=np.minimum(P,c[:,None]+marks);q=expit(offset[:,None]+b[:,None]*pp+gp*np.maximum(pp-rival[:,None],0)+gn*np.minimum(pp-rival[:,None],0));r=q*(pp-c[:,None])-qa[:,None]*(p[:,None]-c[:,None])
    self.assertTrue(np.all(r>=lo-1e-10));self.assertTrue(np.all(r<=hi+1e-10))
 def test_confidence_exact_wealth(self):
  caps=np.array([.25,1.]);eta=np.broadcast_to((.99/caps)[:,None],(2,2));pi=np.full((2,2),.25);T=8;start=2
  states=[(pi.copy(),np.zeros((2,2)),1.,0)]
  for t in range(1,T+1):
   new=[]
   for w,S,Ws,last in states:
    if t==start+1:Ws=w.sum()
    u=np.array([.1,.6]) if last==0 else np.array([.9,.1]);f=np.broadcast_to(np.minimum(1,caps/np.ptp(u))[:,None],(2,2));mass=eta*w*f;ap=mass.sum(0)/mass.sum();EW=0.
    for a in range(2):
     g=f*(u-u[a])[None,:];raw=w*(1+eta*g);wn=t/(t+1)*raw+pi*raw.sum()/(t+1);ss=S+np.log1p(eta*g) if t>start else S;EW+=ap[a]*wn.sum()
     if t>start:self.assertTrue(np.all(np.log(wn/Ws)>=np.log(pi)-np.log(t+1)+ss-1e-12))
     new.append((wn,ss,Ws,a))
    self.assertAlmostEqual(EW,w.sum(),places=12)
   states=new

if __name__=='__main__':unittest.main()
