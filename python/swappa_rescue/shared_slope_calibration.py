"""One common slope event plus twenty seller-specific level events.
The calibration is conditional on the structural Gaussian approximation.
"""
import numpy as np
from scipy.stats import chi2,norm

def calibration(data,alpha_regret=.00025):
 assert 0<alpha_regret<=.00025
 if 'alpha_band' in data and 'alpha_mc' in data and float(data['alpha_band'])<=.00175 and float(data['alpha_mc'])<=.0002:
  pass
 else:
  # Older archives kept only the maximum order statistic. Its finite Monte
  # Carlo tail probability can be checked directly for the new allocation.
  assert float(data['critical'])==float(data['maxima'].max())
  assert (1-.00175)**len(data['maxima'])<=.0002
 return {'shared_slope_band':True,'slope_critical':float(np.sqrt(chi2.ppf(.995,4))),'peer_critical':float(norm.ppf(.9995)),'global_slope_alpha':.005,'global_peer_alpha':.001,'seller_band_alpha':.00175,'seller_MC_alpha':.0002,'seller_regret_alpha':alpha_regret,'family_size':20,'total_failure_allocation':.005+.001+20*(.00175+.0002+alpha_regret)}
