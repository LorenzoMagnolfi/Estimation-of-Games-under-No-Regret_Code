"""Lower bounds on joint oracle regret divided by maximum joint profit.
The SAME baseline probabilities and costs enter observed and counterfactual
profit. For a trial ratio R, sum the exact lower extrema of profit_new minus
R times profit_observed. Positive sums certify a profit ratio at least R.
"""
import numpy as np
from demand_band_bounds import own_gain_z_extrema

def daily_ratio_lower(zlo,zhi,dlo,new_receipt,receipt,cl,ch,day,days,iterations=16):
 K=new_receipt.shape[1];low=np.zeros((days,K));high=np.ones((days,K))
 for _ in range(iterations):
  middle=(low+high)/2;ratio=1/(1-middle[day]);terms=[]
  for c in [cl,ch]:
   value,_=own_gain_z_extrema(zlo[:,None],zhi[:,None],dlo,new_receipt-c[:,None],ratio*(receipt-c)[:,None]);terms.append(value)
  lower=np.minimum(*terms);total=np.zeros((days,K));np.add.at(total,day,lower);positive=total>=1e-10;low=np.where(positive,middle,low);high=np.where(positive,high,middle)
 return low.max(axis=1)
