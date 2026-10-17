import unittest
import joblib
import numpy as np
from audit_panel import OUT
from daily_cost_model import DailyCostModel
from daily_heterogeneous_cost import DailyHeterogeneousCostModel
from daily_scope import configure_model
from daily_proxy_error import proxy_wealth_allowance


class ProxyErrorTests(unittest.TestCase):
    def test_wealth_allowance_for_full_proxy_oracle(self):
        rng=np.random.default_rng(193);eta=.99
        for epsilon in [0,.001,.005,.01,.1]:
            utility=rng.uniform(0,1,(200,20));proxy=np.clip(utility+rng.uniform(-epsilon,epsilon,utility.shape),0,1);action=rng.integers(0,20,200);advice=rng.integers(0,20,200);rows=np.arange(200);true_gain=utility[rows,advice]-utility[rows,action];proxy_gain=proxy.max(axis=1)-proxy[rows,action]
            self.assertTrue(np.all(true_gain<=proxy_gain+2*epsilon+1e-12));self.assertLessEqual(float(np.log1p(eta*true_gain).sum()),float(np.log1p(eta*proxy_gain).sum())+proxy_wealth_allowance(200,epsilon,eta)+1e-10)


@unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires fit')
class ScopeTests(unittest.TestCase):
    def test_menu_persists_under_coherent_demand_changes(self):
        from test_daily_cost_model import DailyCostTests
        DailyCostTests.setUpClass();fixture=DailyCostTests
        for delta in [0,5]:
            model=DailyCostModel(fixture.s,fixture.fit,fixture.cov) if delta==0 else DailyHeterogeneousCostModel(fixture.s,fixture.fit,fixture.cov,delta)
            original_range=model.ranges([60])[model.live] if delta==0 else model.vector_range(model.cost_box(60)[0])[model.live]
            configure_model(model,1.25,[.8,2]);self.assertTrue(model.advisor_menu[:,-1].all());self.assertTrue(np.all(np.isneginf(model.rA[~model.advisor_menu])))
            model.prepare(model.theta);self.assertTrue(np.all(np.isneginf(model.rA[~model.advisor_menu])))
            changed_range=model.ranges([60])[model.live] if delta==0 else model.vector_range(model.cost_box(60)[0])[model.live]
            np.testing.assert_allclose(changed_range,1.25*original_range)
            lo,hi=model.interval_bounds([60],[62]);value=model.values([61]);self.assertGreaterEqual(value[0],lo[0]-1e-8);self.assertLessEqual(value[0],hi[0]+1e-8)

if __name__=='__main__':unittest.main()
