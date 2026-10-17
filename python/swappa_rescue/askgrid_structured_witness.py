"""Search predeclared cost-reference sensitivity classes; no point estimation."""
import argparse,json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from optimize_askgrid_witness import prepare
from ask_price_grid import recover_receipts

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--delta',type=float,default=20);ap.add_argument('--rhos',type=float,nargs='+',default=[.60,.65,.70,.75,.80]);args=ap.parse_args()
 with threadpool_limits(limits=2):
  s,codes,fun=prepare(full_grid=True);anchor=s.groupby('Code').initial_reference.first().reindex(codes).to_numpy();upper=s.assign(receipt=recover_receipts(s.Price.to_numpy())).groupby('Code').receipt.min().reindex(codes).to_numpy();initial=pd.read_csv(OUT/'upgrade_askgrid_optimized_costs.csv').set_index('Code').cost.reindex(codes).to_numpy();records=[]
  for rho in args.rhos:
   lo=np.maximum(0,rho*anchor-args.delta);hi=np.minimum(upper,rho*anchor+args.delta)
   if np.any(lo>hi):print(json.dumps({'rho':rho,'delta':args.delta,'prior_infeasible':True}),flush=True);continue
   start=time.time();result=minimize(fun,np.clip(initial,lo,hi),jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options={'maxiter':250,'ftol':1e-11,'gtol':1e-7,'maxls':30});threshold=float(np.log(4*351)+np.log(1e6)+np.log(1/.005));rec={'rho':rho,'delta':args.delta,'logwealth':float(result.fun),'threshold':threshold,'passes':result.fun<=threshold,'optimizer_converged':bool(result.success),'iterations':int(result.nit),'seconds':time.time()-start};records.append(rec);print(json.dumps(rec),flush=True)
   pd.DataFrame({'Code':codes,'cost':result.x}).to_csv(OUT/f'askgrid_structured_delta{args.delta:g}_rho{rho:g}_costs.csv',index=False)
   (OUT/f'askgrid_structured_delta{args.delta:g}_search.json').write_text(json.dumps(records,indent=2))
 print('ASKGRID_STRUCTURED_WITNESS_COMPLETE',flush=True)
if __name__=='__main__':main()
