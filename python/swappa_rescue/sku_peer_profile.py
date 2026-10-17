"""Profile SKU centers while jointly restricting all owned peer costs.

Conditional center intervals constrain every peer before each listing's cost
cells are evaluated. Each saved group supplies conservative cost/wealth
alternatives; the dual combines groups under one aggregate wealth budget.
"""
import argparse,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import chi2,norm
from threadpoolctl import threadpool_limits
from audit_panel import OUT
from demand_timing_sensitivity import terminal_status_data
from ask_price_grid import AskGridBand,recover_receipts
from sku_profile_dual import dual_bound


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--delta',type=float,default=20);ap.add_argument('--center-step',type=float,default=5);ap.add_argument('--cells',type=int,default=20);ap.add_argument('--copies',type=int,nargs='+',default=[1,2,5,10]);ap.add_argument('--oracle-error',type=float,nargs='+',default=[0.,.001,.005]);args=ap.parse_args();assert args.delta>=0 and args.center_step>0 and args.cells>=1;start=time.time();P=350.;alpha=.00025
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);s['receipt']=recover_receipts(s.Price.to_numpy());fitpath=OUT/'multiproduct_initial_terminal.joblib';bandpath=OUT/'demand_uniform_pairs_initial_p350_b5000_family20.npz';fit=joblib.load(fitpath);data=np.load(bandpath);assert len(data['maxima'])>=5000 and float(data['critical'])==float(data['maxima'].max());assert (1-.00175)**len(data['maxima'])<=.0002
 keys=['device_id','Storage','Condition'];assert not s.groupby('Code')[keys].nunique().gt(1).any().any();active=s.since_seen.gt(0);events=s.sale_terminal_status*active;total_sales=int(events.sum());price=float(events@s.receipt/total_sales);N=int(active.sum());prefix=f'sku_peer_d{args.delta:g}_h{args.center_step:g}_g{args.cells}';manifest={'arguments':vars(args),'critical':float(data['critical']),'slope_critical':float(np.sqrt(chi2.ppf(.995,4))),'peer_critical':float(norm.ppf(.9995)),'failure_allocation':.05,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('ask_price_grid.py'),Path(__file__).with_name('demand_band_bounds.py'),fitpath,bandpath,OUT/'panel.parquet']},'sales':total_sales,'active_rows':N,'mean_receipt':price};manifest_path=OUT/f'{prefix}.json'
 if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest,'Existing outputs have different provenance'
 else:manifest_path.write_text(json.dumps(manifest,indent=2))
 group_files=[];marks=np.array([5.,10.,15.,20.,30.,40.,50.,70.,100.,150.,250.])
 with threadpool_limits(limits=2):
  for group,(key,subset) in enumerate(s.groupby(keys,sort=True)):
   path=OUT/f'{prefix}_group{group:02d}.npz';group_files.append(path)
   if path.exists():print(json.dumps({'stage':'reuse_group','group':group,'sku':key}),flush=True);continue
   sub=subset.reset_index(drop=True);codes,idx=np.unique(sub.Code,return_inverse=True);L=len(codes);cap=sub.groupby('Code',sort=True).receipt.min().to_numpy();centers=np.linspace(0,cap.min()+args.delta,max(1,int(np.ceil((cap.min()+args.delta)/args.center_step)))+1);J=len(centers)-1;edges=np.zeros((J,args.cells+1,L));coeff=np.zeros((J,args.cells,L));band=AskGridBand(fit,sub,data['covariance'],float(data['critical']),slope_critical=manifest['slope_critical'],peer_critical=manifest['peer_critical']);band.prepare_peers(list(sub.groupby('market_id',sort=False).indices.values()));act=sub.since_seen.gt(0).to_numpy();w=np.bincount(idx,weights=sub.sale_terminal_status.to_numpy()*act,minlength=L)/total_sales
   print(json.dumps({'stage':'group_start','group':group,'sku':key,'rows':len(sub),'listings':L,'center_intervals':J,'elapsed':time.time()-start}),flush=True)
   for j in range(J):
    lo=np.full(L,max(0,centers[j]-args.delta));hi=np.minimum(cap,centers[j+1]+args.delta);assert (lo<=hi+1e-9).all();edges[j]=lo[None,:]+np.linspace(0,1,args.cells+1)[:,None]*(hi-lo)[None,:]
    for g in range(args.cells):
     cl=edges[j,g,idx];ch=edges[j,g+1,idx];gain,ran=band.bounds(cl,ch,marks,P,sub.own_n.to_numpy(),peer_cl=lo[idx],peer_ch=hi[idx]);frac=np.maximum(np.maximum(0,gain.max(axis=1))/ran,band.last_joint_fraction_lower);np.add.at(coeff[j,g],idx,np.log1p(.99*frac)*act)
    if (j+1)%10==0:print(json.dumps({'stage':'centers','group':group,'completed':j+1,'total':J,'elapsed':time.time()-start}),flush=True)
   np.savez_compressed(path,codes=codes.astype(str),centers=centers,edges=edges,coefficients=coeff,weights=w,sku=np.array(key,dtype=str));print(json.dumps({'stage':'group_complete','group':group,'elapsed':time.time()-start}),flush=True)
  records=[]
  for direction,sign in [('low',1),('high',-1)]:
   enclosure=[]
   for path in group_files:
    item=np.load(path);obj=sign*(item['edges'][:,:-1] if sign>0 else item['edges'][:,1:])*item['weights'][None,None,:];enclosure.append((obj,item['coefficients']))
   for copies in args.copies:
    for error in args.oracle_error:
     threshold=copies*(np.log(4*351)+np.log(1e6)+np.log(copies/alpha))-N*np.log1p(-.99*error);bound,multiplier,evaluations=dual_bound(enclosure,threshold);cost=sign*bound;rec={'model':'sku_center_joint_peer_profile','delta':args.delta,'center_step':args.center_step,'cells':args.cells,'copies':copies,'oracle_error':error,'direction':direction,'cost_bound':cost,'margin_bound':price-cost,'mean_receipt':price,'threshold':float(threshold),'alpha_R':alpha,'dual_multiplier':multiplier,'dual_evaluations':evaluations,'seconds':time.time()-start};records.append(rec);print(json.dumps(rec),flush=True)
  pd.DataFrame(records).to_csv(OUT/f'{prefix}_bounds.csv',index=False)
 print('SKU_PEER_PROFILE_COMPLETE',flush=True)

if __name__=='__main__':main()
