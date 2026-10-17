"""Cost-dispersion sensitivity within model/storage/condition groups.
Group cost centers are unrestricted nuisance parameters. Only within-group
absolute dispersion is bounded; no external reference-cost ratio is imposed.
Uses a continuous-cost outer LP with existing certified nuisance coefficients.
"""
import argparse,json,time
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,eye,hstack,vstack
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from ask_price_grid import recover_receipts
from demand_timing_sensitivity import terminal_status_data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--coefficients',required=True);ap.add_argument('--delta',type=float,default=20);ap.add_argument('--alpha-regret',type=float,default=.00025);ap.add_argument('--copies',type=int,nargs='+',default=[1,2,5,10]);args=ap.parse_args();start=time.time();data=np.load(args.coefficients,allow_pickle=True);coeff=data['coefficients'];edges=data['edges'];codes=data['codes'];G,L=coeff.shape;assert edges.shape==(G+1,L)
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);entry=s.drop_duplicates('Code').set_index('Code').reindex(codes);keys=['device_id','Storage','Condition'];group=entry.groupby(keys,sort=True).ngroup().to_numpy();K=int(group.max())+1;index=pd.Index(codes).get_indexer(s.Code);assert (index>=0).all();p=recover_receipts(s.Price.to_numpy());events=s.sale_terminal_status.to_numpy();active=s.since_seen.gt(0).to_numpy();cap=edges[-1];groups=csr_matrix((np.ones(L),(np.arange(L),group)),shape=(L,K));W=hstack([eye(L,format='csr')]*G,format='csr');low=hstack([csr_matrix(np.diag(edges[g])) for g in range(G)],format='csr');high=hstack([csr_matrix(np.diag(edges[g+1])) for g in range(G)],format='csr');I=eye(L,format='csr');zero=csr_matrix((L,K));zw=csr_matrix((L,G*L));gain=csr_matrix(coeff.ravel()[None,:]);eq=hstack([W,csr_matrix((L,L+K))],format='csr');fixed=vstack([hstack([low,-I,zero]),hstack([-high,I,zero]),hstack([zw,I,-groups]),hstack([zw,-I,groups])],format='csr');rhs=np.r_[np.zeros(2*L),np.full(2*L,args.delta)];A=vstack([hstack([gain,csr_matrix((1,L+K))]),fixed],format='csr');bounds=[(0,1)]*(G*L)+[(0,float(x)) for x in cap]+[(0,float(cap[group==k].max())) for k in range(K)];priorcap=np.minimum(cap,np.array([cap[group==k].min()+2*args.delta for k in group]));records=[]
 # Independent candidate construction, subsequently checked by the full oracle.
 candidate=pd.read_csv(OUT/'upgrade_askgrid_optimized_full_costs.csv').set_index('Code').cost.reindex(codes).to_numpy();centers=np.array([min((candidate[group==k].min()+candidate[group==k].max())/2,cap[group==k].min()+args.delta) for k in range(K)]);candidate=np.clip(candidate,np.maximum(0,centers[group]-args.delta),np.minimum(cap,centers[group]+args.delta));pd.DataFrame({'Code':codes,'cost':candidate}).to_csv(OUT/f'sku_cost_witness_delta{args.delta:g}.csv',index=False)
 cell=np.minimum(G-1,np.maximum(0,(candidate/np.maximum(cap,1e-300)*G).astype(int)));weights=np.zeros((G,L));weights[cell,np.arange(L)]=1;witness=np.r_[weights.ravel(),candidate,centers];assert np.max(abs(eq@witness-1))<1e-8;assert np.max(fixed@witness-rhs)<1e-8
 targets={'sold':events,'continuation_sold':events*active};print(json.dumps({'stage':'start','delta':args.delta,'cost_groups':K,'listing_count':L,'cells':G,'candidate_band_logwealth_lower':float((gain@weights.ravel())[0]),'prior_only_upper_mean_continuation':float(np.bincount(index,weights=events*active,minlength=L)@priorcap/(events*active).sum())}),flush=True)
 with threadpool_limits(limits=2):
  for copies in args.copies:
   threshold=copies*(np.log(4*351)+np.log(1e6)+np.log(copies/args.alpha_regret))
   for estimand,event in targets.items():
    weights=np.bincount(index,weights=event,minlength=L)/event.sum();price=float(event@p/event.sum());obj=np.r_[np.zeros(G*L),weights,np.zeros(K)]
    for direction,sign in [('low',1),('high',-1)]:
     result=linprog(sign*obj,A_ub=A,b_ub=np.r_[threshold,rhs],A_eq=eq,b_eq=np.ones(L),bounds=bounds,method='highs');value=float(sign*result.fun) if result.success else None;rec={'model':'sku_cost_dispersion','delta':args.delta,'copies':copies,'estimand':estimand,'direction':direction,'cost_bound':value,'margin_bound':price-value if value is not None else None,'mean_receipt':price,'prior_cost_low':0.,'prior_cost_high':float(weights@priorcap),'groups':K,'alpha_R':args.alpha_regret,'solver_status':int(result.status),'solver_message':result.message,'seconds':time.time()-start};records.append(rec);print(json.dumps(rec),flush=True)
 pd.DataFrame(records).to_csv(OUT/f'sku_cost_bounds_delta{args.delta:g}_c{G}_alpha{args.alpha_regret:g}.csv',index=False);print('SKU_COST_BOUNDS_COMPLETE',flush=True)
if __name__=='__main__':main()
