"""Checks for prediction timing and cost inequality signs."""
import json
import unittest
import numpy as np
import pandas as pd
from empirical_pricing_rules import display,features,candidates,DESIGN
from policy_cost_frontier import endpoint

class EmpiricalRuleChecks(unittest.TestCase):
    def test_display_preserves_every_admissible_price(self):
        prices=np.ceil(1.03*np.arange(10,1942)-1e-9)
        np.testing.assert_array_equal(display(prices),prices)

    def test_forecasts_use_only_previous_information(self):
        dates=pd.to_datetime(['2023-09-29','2023-09-30']+list(pd.date_range('2023-10-25','2023-11-04').strftime('%Y-%m-%d')))
        rows=[dict(Code=code,date=date,Price=float(103+code*10),rival_min=100.,device_id='iphone',Storage='128',Condition='Good',sale_terminal_status=0) for code in [0,1] for date in dates]
        original=pd.DataFrame(rows);changed=original.copy()
        affected=changed.date.ge('2023-11-01')
        changed.loc[affected,['Price','rival_min','sale_terminal_status']]=[205.,180.,1]
        left=features(original);right=features(changed);mask=left.eligible&left.date.le('2023-11-01')
        spec=json.loads(DESIGN.read_text())
        for a,b in zip(candidates(left,spec),candidates(right,spec)):
            self.assertEqual(a[:2],b[:2])
            np.testing.assert_array_equal(a[2][mask],b[2][mask])

    def test_profit_comparison_and_withdrawal_signs(self):
        # 6-.2*c <= eps and -20+.5*c <= eps.
        w=np.array([1.]);a=np.array([6.,-20.]);b=np.array([[.2],[-.5]])
        for eps,expected in [(0,(30,40)),(1,(25,42))]:
            lo=endpoint(w,a,b,eps,[(0,None)],1)
            hi=endpoint(w,a,b,eps,[(0,None)],-1)
            self.assertAlmostEqual(lo['value'],expected[0]);self.assertAlmostEqual(hi['value'],expected[1])
        self.assertEqual(endpoint(w,a[:1],b[:1],0,[(0,None)],-1)['status'],3)

if __name__=='__main__':unittest.main()
