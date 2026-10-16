"""Freeze the fitted demand band and row design for reproducible CPU inference."""
import json,hashlib,subprocess
import joblib,numpy as np
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from normalized_demand_bounds import NormalizedBand


def main():
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/'demand_uniform_pairs_initial_p350_b1000.npz');band=NormalizedBand(fit,s,data['covariance'],float(data['critical']));prefix=OUT/'upgrade_normalized_bundle'
 columns=['Code','date','Price','Ref_Price','initial_reference','own_n','market_id','since_seen','sale_terminal_status'];s[columns].to_csv(str(prefix)+'.csv',index=False);np.savez_compressed(str(prefix)+'.npz',**{k:v for k,v in band.__dict__.items() if k!='fit'})
 upper=s.groupby('Code').Price.min();anchor=s.groupby('Code').initial_reference.first();files=[str(prefix)+x for x in ['.csv','.npz']]
 meta={'seller':'UPGRADE SOLUTION INC','price_cap':350.,'critical':float(data['critical']),'rows':len(s),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rho_max_feasible':{str(delta):float(((upper+delta)/anchor).min()) for delta in [20,35,50]},'sha256':{str(x):hashlib.sha256(open(x,'rb').read()).hexdigest() for x in files}}
 open(str(prefix)+'.json','w').write(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
