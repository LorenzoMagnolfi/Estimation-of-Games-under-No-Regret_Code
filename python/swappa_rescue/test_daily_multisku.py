import json
import unittest
import joblib
import numpy as np
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from daily_cost_model import DailyCostModel
from daily_heterogeneous_cost import DailyHeterogeneousCostModel
from daily_cell_regularity import audit_active_cell


@unittest.skipUnless((OUT/'daily_multisku_common_d0_h02.json').exists(),'Requires joint profile')
class MultiSkuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=terminal_status_data();cls.s=data[data.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True)

    def test_lp_outer_bounds_contain_actual_cost_witnesses(self):
        report=json.loads((OUT/'daily_multisku_common_d0_h02.json').read_text());fit=joblib.load(OUT/report['fit']);cov=np.zeros((len(fit.model.coef_[0])+1,)*2);models=[DailyCostModel(sub,fit,cov) for key,sub in self.s.groupby(['device_id','Storage','Condition'],sort=True)]
        centers=np.array(report['grid_cost_vector']);caps=np.array([m.cap for m in models]);weights=np.array([g['sales'] for g in report['groups']]);weights/=weights.sum();rng=np.random.default_rng(731);vectors=[centers]
        for k in range(len(models)):
            for value in [0,caps[k]/2,caps[k]]:
                trial=centers.copy();trial[k]=value;vectors.append(trial)
        vectors.extend(np.clip(centers+rng.normal(0,4,(40,len(models))),0,caps));vectors=np.array(vectors);wealth=np.zeros(len(vectors))
        for k,m in enumerate(models):
            for start in range(0,len(vectors),12):wealth[start:start+12]+=m.values(vectors[start:start+12,k])
        valid=wealth<=report['threshold'];self.assertGreater(int(valid.sum()),10);target=vectors@weights
        self.assertTrue(np.all(target[valid]>=report['cost_outer_interval'][0]-1e-8));self.assertTrue(np.all(target[valid]<=report['cost_outer_interval'][1]+1e-8))
        hetero=json.loads((OUT/'daily_multisku_common_d5_h02.json').read_text());self.assertTrue(np.all(target[valid]>=hetero['cost_outer_interval'][0]-1e-8));self.assertTrue(np.all(target[valid]<=hetero['cost_outer_interval'][1]+1e-8))

    def test_active_cell_smoothness_for_reported_inference_models(self):
        for name in ['daily_multisku_common_d0_h02.json','daily_multisku_common_d5_h02.json','daily_multisku_seller_d0_h02.json','daily_multisku_seller_d5_h02.json']:
            path=OUT/name
            if not path.exists():continue
            report=json.loads(path.read_text());fit=joblib.load(OUT/report['fit']);cov=np.zeros((len(fit.model.coef_[0])+1,)*2)
            for k,(key,sub) in enumerate(self.s.groupby(['device_id','Storage','Condition'],sort=True)):
                model=DailyCostModel(sub,fit,cov) if report['cost_delta']==0 else DailyHeterogeneousCostModel(sub,fit,cov,report['cost_delta'])
                for solution in report['low_and_high_solutions']:
                    for row in solution:
                        if row['sku']==k:audit_active_cell(model,row['low'],row['high'])

if __name__=='__main__':unittest.main()
