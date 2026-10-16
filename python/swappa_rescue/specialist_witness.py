"""Find actual cost witnesses for the specialist outer-screen diagnostics.

FOC-implied costs only initialize a candidate. A cost vector is called a
witness only if it satisfies every tested exact wealth inequality. Neither
FOC initialization nor passing this finite subset establishes a full model.
"""
import argparse,json
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from profit_specialists import coordinate_range,markup_regret_interval
from coordinate_profit import peer_spillover_bounds


def main():
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_sale_terminal_status.joblib')
 codes,idx=np.unique(s.Code,return_inverse=True);L=len(codes);N=len(s);anchor=s.groupby('Code',sort=True).initial_reference.first().to_numpy();cmax=s.groupby('Code',sort=True).Price.min().to_numpy();p=s.Price.to_numpy();ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();qa=fit.predict(s);odds=fit.logit(s);n=s.own_n.to_numpy();groups=list(s.groupby('market_id',sort=False).indices.values());ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;bp=ef['own_peer_price']/100;offset=odds-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);kmin=np.minimum(-(b+gp),-(b+gn));k=-(b+np.where(p>=rival,gp,gn));Pmax=350.
 def logits(price):return offset+b*price+gp*np.maximum(price-rival,0)+gn*np.minimum(price-rival,0)
 diag=np.bincount(idx,weights=k*qa*(1-qa),minlength=L);rhs=np.bincount(idx,weights=k*qa*(1-qa)*p-qa,minlength=L);rows=[];cols=[];values=[]
 for ids in groups:
  ni=int(n[ids[0]]);m=len(ids)
  if ni<=1:continue
  coef=bp/(ni-1)*(1-np.eye(m))*(qa[ids]*(1-qa[ids]))[None,:]
  rows.extend(np.repeat(idx[ids],m));cols.extend(np.tile(idx[ids],m));values.extend((-coef).ravel());np.add.at(rhs,idx[ids],-(coef@p[ids]))
 mat=diags(diag)+coo_matrix((values,(rows,cols)),shape=(L,L)).tocsr();foc=spsolve(mat,rhs);qw=np.bincount(idx,weights=qa,minlength=L)/qa.sum();sw=np.bincount(idx,weights=s.sale_terminal_status,minlength=L)/s.sale_terminal_status.sum()
 print(json.dumps({'FOC_only_diagnostic':True,'cost_expected_weighted':float(qw@foc),'cost_sold_weighted':float(sw@foc),'negative_costs':int((foc<0).sum()),'above_minimum_posted_price':int((foc>cmax).sum()),'cost_residual_ratio_quantiles':np.quantile(foc/anchor,[0,.05,.25,.5,.75,.95,1]).tolist()}),flush=True)
 pd.DataFrame({'Code':codes,'initial_reference':anchor,'minimum_price':cmax,'foc_cost':foc}).to_csv(OUT/'specialist_foc_initialization.csv',index=False)
 marks=np.array([5.,10.,20.,30.,50.,100.,250.]);hs=np.arange(0,11);caps=2.**hs;js=np.arange(8);eta=.99*2.**(-js[None,:])/caps[:,None];cp=1/((hs+1)*(hs+2));rp=1/((js+1)*(js+2));term=np.log(362/cp[:,None]/rp[None,:])+np.log(1000000)+np.log(1/.005);records=[]
 for delta in [5.,10.,25.]:
  for rho in np.arange(.45,.901,.01):
   low=np.maximum(0,rho*anchor-delta);high=np.minimum(cmax,rho*anchor+delta)
   if np.any(low>high):continue
   for initialization in ['projected_foc','midpoint']:
    cost=np.clip(foc,low,high) if initialization=='projected_foc' else (low+high)/2;c=cost[idx]
    own,_=markup_regret_interval(c,c,marks,p,qa,offset,b,gp,gn,rival,Pmax);proposal=np.minimum(Pmax,c[:,None]+marks);spl,sph=peer_spillover_bounds(proposal,proposal,p,odds,n,groups,p-c,p-c,bp,Pmax);regret=own+spl
    # Complete-market observations make spillovers exact in this sample.
    assert np.max(abs(spl-sph))<1e-8
    ran=coordinate_range(c,p,odds,n,groups,bp,Pmax,logits(c),kmin,p-c)
    logwealth=[]
    for z,cap in enumerate(caps):
     f=np.minimum(1,cap/ran);logwealth.append(np.log1p(f[:,None,None]*regret[:,:,None]*eta[z,None,None,:]).sum(axis=0))
    violation=(np.array(logwealth)-term[:,None,:])/eta[:,None,:];worst=float(violation.max());rec={'delta':delta,'rho':float(rho),'initialization':initialization,'passes_tested_restrictions':worst<=1e-8,'max_violation_dollars':worst,'cost_expected_weighted':float(qw@cost),'cost_sold_weighted':float(sw@cost)};records.append(rec)
  local=[r for r in records if r['delta']==delta];best=min(local,key=lambda x:x['max_violation_dollars']);print(json.dumps(best),flush=True)

 # Audit the strongest candidate against the ENTIRE finite policy menu.
 best5=min([r for r in records if r['delta']==5.],key=lambda x:x['max_violation_dollars']);rho=best5['rho'];delta=5.;cost=np.clip(foc,np.maximum(0,rho*anchor-delta),np.minimum(cmax,rho*anchor+delta));c=cost[idx]
 allmarks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.])
 proposal=np.column_stack([np.maximum(np.arange(int(Pmax)+1)[None,:],c[:,None]),np.minimum(Pmax,c[:,None]+allmarks)])
 z=offset[:,None]+b[:,None]*proposal+gp*np.maximum(proposal-rival[:,None],0)+gn*np.minimum(proposal-rival[:,None],0)
 own=expit(z)*(proposal-c[:,None])-qa[:,None]*(p[:,None]-c[:,None])
 spl,sph=peer_spillover_bounds(proposal,proposal,p,odds,n,groups,p-c,p-c,bp,Pmax);regret=own+spl
 assert np.max(abs(spl-sph))<1e-8 and proposal.shape[1]==362
 ran=coordinate_range(c,p,odds,n,groups,bp,Pmax,logits(c),kmin,p-c)
 full_hs=np.arange(20);full_js=np.arange(10);full_caps=2.**full_hs;full_eta=.99*2.**(-full_js[None,:])/full_caps[:,None];full_cp=1/((full_hs+1)*(full_hs+2));full_rp=1/((full_js+1)*(full_js+2));full_term=np.log(362/full_cp[:,None]/full_rp[None,:])+np.log(1000000)+np.log(1/.005)
 full_log=[]
 for z,cap in enumerate(full_caps):
  f=np.minimum(1,cap/ran);full_log.append(np.log1p(f[:,None,None]*regret[:,:,None]*full_eta[z,None,None,:]).sum(axis=0))
 violations=(np.array(full_log)-full_term[:,None,:])/full_eta[:,None,:];worst=np.unravel_index(np.argmax(violations),violations.shape)
 audit={'rho':rho,'delta':delta,'policy_count':362,'cap_count':len(full_caps),'rate_count':len(full_js),'max_violation_dollars':float(violations.max()),'passes_all_finite_policies_tested_rates':bool(violations.max()<=1e-8),'worst_cap':float(full_caps[worst[0]]),'worst_policy_index':int(worst[1]),'worst_rate':int(worst[2])}
 global_range=Pmax*(1+bp*Pmax)
 audit['cap_tail_logwealth_upper']=.99*N*global_range/(2**20)
 audit['rate_tail_logwealth_upper']=.99*N/(2**10)
 audit['minimum_log_threshold']=float(full_term.min())
 audit['all_omitted_tails_automatically_pass']=max(audit['cap_tail_logwealth_upper'],audit['rate_tail_logwealth_upper'])<audit['minimum_log_threshold']
 print(json.dumps(audit),flush=True);(OUT/'specialist_witness_full_policy_audit.json').write_text(json.dumps(audit,indent=2))
 pd.DataFrame({'Code':codes,'cost':cost,'initial_reference':anchor,'rho':rho,'delta':delta}).to_csv(OUT/'specialist_witness_costs.csv',index=False)
 pd.DataFrame(records).to_csv(OUT/'specialist_cost_witnesses.csv',index=False)
 print('SPECIALIST_WITNESS_COMPLETE',flush=True)
if __name__=='__main__':main()
