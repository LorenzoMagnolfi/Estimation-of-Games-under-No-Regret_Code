"""Diagnostic: does an aggregate sales moment rule out all-zero costs?

Uses a maintained terminal-sale hazard model. The Bernoulli exponential
bound additionally requires sequential conditional means equal to the
modeled marginals (e.g. conditionally independent listing sales within day).
The day-block Hoeffding alternative allows arbitrary within-day dependence.
This script is NOT a marginal-cost confidence interval.
"""
import json,time,sys
import joblib,numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from scipy.stats import chi2,norm
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from ask_price_grid import AskGridBand,recover_receipts
from demand_timing_sensitivity import terminal_status_data

def main():
 start=time.time();s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/'demand_uniform_pairs_initial_p350_b5000_family20.npz');band=AskGridBand(fit,s,data['covariance'],float(data['critical']),slope_critical=np.sqrt(chi2.ppf(.995,4)),peer_critical=norm.ppf(.9995));groups=list(s.groupby('market_id',sort=False).indices.values());band.prepare_peers(groups);active=s.since_seen.gt(0).to_numpy();N=int(active.sum());sales=int((s.sale_terminal_status.to_numpy()*active).sum());original_low=band.zlo.copy();original_high=band.zhi.copy();qb=np.linspace(0,1,51);cost=np.zeros(len(s));marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.]);coeff=[]
 if '--cache' in sys.argv:
  saved=np.load(OUT/'zero_cost_sales_calibration.npz');coeff=saved['regret_lower'];qb=saved['q_edges']
 else:
  with threadpool_limits(limits=2):
   for j in range(len(qb)-1):
    band.zlo=np.maximum(original_low,logit(qb[j]));band.zhi=np.minimum(original_high,logit(qb[j+1]));valid=band.zlo<=band.zhi;band.zlo=np.minimum(band.zlo,band.zhi);band.qlo=expit(band.zlo);band.qhi=expit(band.zhi)
    gain,ran=band.bounds(cost,cost,marks,350.,s.own_n.to_numpy(),peer_cl=cost,peer_ch=cost);frac=np.maximum(np.maximum(0,gain.max(axis=1))/ran,band.last_joint_fraction_lower);coeff.append(np.where(valid,np.log1p(.99*frac),np.inf)[active])
 with threadpool_limits(limits=2):
  coeff=np.array(coeff);alpha=.00005;lam=.25;upper_bernoulli=(lam*sales+np.log(1/alpha))/(1-np.exp(-lam));counts=s.loc[active].groupby('date').size().to_numpy();day_lambda=.01;upper_day=sales+np.log(1/alpha)/day_lambda+day_lambda*(counts*counts).sum()/8;records=[]
  for label,Q in [('sequential_Bernoulli',upper_bernoulli),('arbitrary_within_day_Hoeffding',upper_day)]:
   evaluations=[]
   def dual(multiplier):
    value=float(np.min(coeff+multiplier*qb[:-1,None],axis=0).sum()-multiplier*Q);evaluations.append((value,float(multiplier)));return -value
   dual(0);minimize_scalar(dual,bounds=(0,1000),method='bounded',options={'xatol':1e-8});bound,multiplier=max(evaluations);threshold=float(np.log(4*351)+np.log(1e6)+np.log(1/.0002));record={'model':label,'observed_sales':sales,'active_rows':N,'expected_sales_upper':float(Q),'moment_alpha':alpha,'moment_lambda':lam if label=='sequential_Bernoulli' else day_lambda,'minimum_zero_cost_logwealth_lower':bound,'dual_multiplier':multiplier,'regret_threshold':threshold,'all_zero_costs_excluded':bound>threshold,'q_cells':len(qb)-1,'seconds':time.time()-start};records.append(record);print(json.dumps(record),flush=True)
 np.savez_compressed(OUT/'zero_cost_sales_calibration.npz',q_edges=qb,regret_lower=coeff,active=active);(OUT/'zero_cost_sales_calibration.json').write_text(json.dumps(records,indent=2))
 print('SALES_CALIBRATION_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
