"""Source-timing and information-set invariants for the rebuilt panel."""
import unittest
from unittest.mock import patch
import numpy as np,pandas as pd
from audit_panel import ROOT
FULL=ROOT/'output/swappa_rescue_20261016/fullsource'

@unittest.skipUnless((FULL/'panel.parquet').exists(),'Requires rebuilt full raw panel')
class FullSourceTests(unittest.TestCase):
 folder=FULL
 @classmethod
 def setUpClass(cls):cls.d=pd.read_parquet(cls.folder/'panel.parquet').sort_values(['Code','date'])
 def test_raw_endpoints_survive_reference_selection(self):
  import demand_timing_sensitivity as timing
  d=self.d;last=d.groupby('Code').date.transform('max');expected=(d.Real_Sold.eq(1)&d.date.eq(last)).astype(int);np.testing.assert_array_equal(d.sale_terminal_status,expected)
  with patch.object(timing,'OUT',self.folder):selected=timing.terminal_status_data()
  matched=selected[['Code','date','sale_terminal_status']].merge(d[['Code','date','sale_terminal_status']],on=['Code','date'],suffixes=('_selected','_raw'),validate='one_to_one');np.testing.assert_array_equal(matched.sale_terminal_status_selected,matched.sale_terminal_status_raw)
  self.assertTrue(selected.Ref_Price.notna().all())
 def test_entry_controls_do_not_use_future_reference_quotes(self):
  d=self.d;entry=d.drop_duplicates('Code').set_index('Code');expected=d.Code.map(entry.Ref_Price).fillna(0);np.testing.assert_allclose(d.initial_reference,expected)
  np.testing.assert_allclose(d.initial_residual+d.initial_reference,d.Code.map(entry.Price))
  np.testing.assert_array_equal(d.initial_reference_missing,d.initial_reference.eq(0))
  self.assertTrue((d.reference_age_days.dropna()>=0).all());self.assertTrue(d.loc[d.reference_observed.eq(1),'reference_age_days'].eq(0).all())
 def test_complete_inventory_precedes_reference_filter(self):
  d=self.d;counts=d.groupby(['market_id','Seller']).Code.transform('size');np.testing.assert_array_equal(d.own_n,counts)
  self.assertFalse(d.duplicated(['Code','date']).any());self.assertGreater(d.Price.max(),10000)
  self.assertGreater(d.Ref_Price.isna().sum(),0)
@unittest.skipUnless((FULL/'reference_screened/panel.parquet').exists(),'Requires reference-screened panel')
class ReferenceScreenedTests(FullSourceTests):
 folder=FULL/'reference_screened'

if __name__=='__main__':unittest.main()
