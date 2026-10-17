import itertools,unittest
import numpy as np
from sku_profile_dual import group_enclosures,dual_bound

class SkuDualTests(unittest.TestCase):
 def test_bound_contains_enumerated_feasible_cost_vectors(self):
  rng=np.random.default_rng(16017);edges=np.broadcast_to(np.arange(0,5)[:,None],(5,4)).copy();groups=np.array([0,0,1,1]);weights=np.array([.1,.2,.3,.4]);delta=.6
  for trial in range(5):
   coeff=rng.uniform(0,4,(4,4));threshold=7.
   costs=np.array(list(itertools.product(np.linspace(0,4,9),repeat=4)))
   cells=np.minimum((costs).astype(int),3);gain=coeff[cells,np.arange(4)].sum(axis=1)
   # On cell boundaries either adjacent cell can contain the true cost.
   # Using one valid cell per candidate suffices for a conservative test.
   keep=(gain<=threshold)&(np.abs(costs[:,0]-costs[:,1])<=2*delta)&(np.abs(costs[:,2]-costs[:,3])<=2*delta)
   feasible=costs[keep]@weights
   if not len(feasible):continue
   for sign in [1,-1]:
    enclosure=group_enclosures(edges,coeff,groups,weights,delta,.5,sign);bound,_,_=dual_bound(enclosure,threshold)
    self.assertLessEqual(bound,float(np.min(sign*feasible))+1e-8)

class ConditionalGroupDualTests(unittest.TestCase):
 def test_center_specific_peer_coefficients(self):
  rng=np.random.default_rng(18);enclosure=[];alternatives=[]
  for group in range(2):
   obj=rng.uniform(0,20,(3,3,2));coeff=rng.uniform(0,5,(3,3,2));enclosure.append((obj,coeff));choices=[]
   for center in range(3):
    for cells in itertools.product(range(3),repeat=2):choices.append((sum(obj[center,cells[i],i] for i in range(2)),sum(coeff[center,cells[i],i] for i in range(2))))
   alternatives.append(choices)
  for threshold in [5,10,15]:
   feasible=[x[0]+y[0] for x in alternatives[0] for y in alternatives[1] if x[1]+y[1]<=threshold]
   if feasible:
    bound,_,_=dual_bound(enclosure,threshold);self.assertLessEqual(bound,min(feasible)+1e-8)
 def test_approximate_advice_log_allowance(self):
  r=np.linspace(0,1,101)[:,None];e=np.array([0,.001,.005,.05,.5])[None,:]
  self.assertTrue(np.all(np.log1p(.99*(r-e))>=np.log1p(.99*r)+np.log1p(-.99*e)-1e-14))

if __name__=='__main__':unittest.main()
