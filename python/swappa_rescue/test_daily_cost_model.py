import unittest
import joblib,numpy as np
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel
from oracle_witness import maximum_own_profit

@unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires demand fit')
class DailyCostTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')&s.device_id.eq('apple-iphone-se-2nd-gen')&s.Storage.eq('64 GB')&s.Condition.eq('Good')].sort_values(['Code','date']).reset_index(drop=True);cls.s=s;cls.fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cls.cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance']
 def test_seller_slopes_match_full_counterfactual_design(self):
  path=OUT/'continuation_seller_full_1.joblib'
  if not path.exists():self.skipTest('Requires seller-specific fit')
  fit=joblib.load(path);cov=np.zeros((len(fit.model.coef_[0])+1,)*2);model=DailyCostModel(self.s,fit,cov)
  for j in [80,120,180,280]:
   changed=self.s.copy();changed['Price']=model.newp[:,j];changed['residual']=changed.Price-changed.Ref_Price;changed['own_total']=changed.groupby('market_id').Price.transform('sum')
   direct=fit.logit(changed);calculated=model.z+model.b*model.change[:,j]+model.gp*model.gap_plus[:,j]+model.gn*model.gap_minus[:,j]+model.bp*model.peer[:,j]
   np.testing.assert_allclose(calculated,direct,atol=1e-10)
 def test_interval_bounds_cover_interior_costs_and_range_derivatives(self):
  rng=np.random.default_rng(729)
  for index in ['constant','reference']:
   model=DailyCostModel(self.s,self.fit,self.cov,index);left=rng.uniform(0,model.cap-20,5);right=np.minimum(model.cap,left+rng.uniform(2,20,5));low,high=model.interval_bounds(left,right)
   for t in [0,.1,.33,.5,.87,1]:
    value=model.values(left+t*(right-left));self.assertTrue(np.all(value>=low-1e-8));self.assertTrue(np.all(value<=high+1e-8))
   for l,h in zip(left,right):
    lo,hi=model.range_derivative_bounds(l,h);mid=(l+h)/2;derivative=(model.ranges([mid+1e-4])[:,0]-model.ranges([mid-1e-4])[:,0])/2e-4;self.assertTrue(np.all(derivative>=lo-1e-7));self.assertTrue(np.all(derivative<=hi+1e-7))
 def test_index_uses_first_observed_reference(self):
  model=DailyCostModel(self.s,self.fit,self.cov,'reference');first=self.s.date.eq(self.s.date.min());self.assertTrue(np.all(model.factor[first]==1));np.testing.assert_allclose(model.factor,self.s.Ref_Price.to_numpy()/self.s.loc[first,'Ref_Price'].iloc[0])
 def test_new_kernel_matches_saved_independent_diagnostic(self):
  path=OUT/'daily_functional_curves.npz'
  if not path.exists():self.skipTest('Requires independent perturbation diagnostic')
  saved=np.load(path);take=np.arange(0,len(saved['costs']),13);model=DailyCostModel(self.s,self.fit,self.cov);np.testing.assert_allclose(model.values(saved['costs'][take]),saved['wealth'][0,take],atol=1e-9)
 def test_optimum_derivative_helper_matches_profit_maximum(self):
  model=DailyCostModel(self.s,self.fit,self.cov)
  for cost in [0,60,110]:
   actual,*_=model.own_optimum_probabilities(cost);expected=maximum_own_profit(np.full(model.N,cost),model.offset+model.bp*model.up,model.b,model.gp,model.gn,model.rival,model.P);np.testing.assert_allclose(actual,expected,atol=1e-10)

class HeterogeneousCostTests(unittest.TestCase):
 setUpClass=classmethod(DailyCostTests.setUpClass.__func__)
 def test_integer_peer_inputs_retain_fractional_means(self):
  from daily_sku_bounds import peer_changes
  np.testing.assert_allclose(peer_changes(np.array([[0],[1],[2]]),[np.arange(3)]).ravel(),[1.5,1.,.5])
 def test_zero_heterogeneity_recovers_scalar_model(self):
  from daily_heterogeneous_cost import DailyHeterogeneousCostModel
  base=DailyCostModel(self.s,self.fit,self.cov);other=DailyHeterogeneousCostModel(self.s,self.fit,self.cov,0)
  grid=np.array([0,40,65,100,114,123.5]);np.testing.assert_allclose(other.values(grid),base.values(grid),atol=1e-8)
 def test_heterogeneous_interval_enclosures(self):
  from daily_heterogeneous_cost import DailyHeterogeneousCostModel
  for delta in [2,10]:
   model=DailyHeterogeneousCostModel(self.s,self.fit,self.cov,delta);left=np.array([0,3,60,100,120]);right=np.minimum(left+3,model.cap);lo,hi=model.interval_bounds(left,right)
   for frac in [0,.1,.4,.7,1]:
    value=model.values(left+frac*(right-left));self.assertTrue(np.all(value>=lo-1e-8));self.assertTrue(np.all(value<=hi+1e-8))
 def test_heterogeneous_root_derivative_certificates(self):
  from daily_heterogeneous_cost import DailyHeterogeneousCostModel
  threshold=float(np.log(4*27)+np.log(1500)+np.log(1/.00025))
  for delta in [2,5,10]:
   model=DailyHeterogeneousCostModel(self.s,self.fit,self.cov,delta);cert=model.certify(threshold);self.assertTrue(cert['regular_two_boundary_set'])
   regular=model.regularity(cert['roots'],[r for r in cert['runs'] if r['status']=='boundary'])
   for item in regular:
    low,high=item['continuous_root_certificate']['wealth_derivative_interval']
    for derivative in item['wealth_cost_derivatives'].values():self.assertLessEqual(low,derivative+1e-7);self.assertGreaterEqual(high,derivative-1e-7)
 def test_conservative_wealth_for_persistent_cost_witnesses(self):
  from daily_heterogeneous_cost import DailyHeterogeneousCostModel
  from scipy.special import expit
  rng=np.random.default_rng(2711);model=DailyHeterogeneousCostModel(self.s,self.fit,self.cov,10);codes,idx=np.unique(self.s.Code,return_inverse=True)
  qnew=expit(model.z[:,None]+model.b[:,None]*model.change+model.gp*model.gap_plus+model.gn*model.gap_minus+model.bp*model.peer);dq=qnew-expit(model.z)[:,None]
  for center in [10,65,100,116]:
   costs=center+rng.uniform(-5,5,len(codes));c=costs[idx];mu=float(model.events@c/model.events.sum());self.assertTrue(np.all(c<=model.row_cap));self.assertLessEqual(np.max(abs(c-mu)),model.delta)
   cr=np.zeros_like(model.rA);np.add.at(cr,model.day,c[:,None]*dq);required=np.zeros(model.D);np.maximum.at(required,model.day,np.where(model.active,c,0));gain=np.where(model.receipt[None,:]>=required[:,None]-1e-10,model.rA-cr,-np.inf).max(axis=1);actual=np.log1p(.99*gain[model.live]/model.vector_range(c)[model.live]).sum();self.assertLessEqual(model.values([mu])[0],actual+1e-8)

if __name__=='__main__':unittest.main()
