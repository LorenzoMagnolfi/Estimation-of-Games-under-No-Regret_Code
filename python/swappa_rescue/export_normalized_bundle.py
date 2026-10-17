"""Freeze the fitted demand band and row design for reproducible CPU inference."""
import argparse,json,hashlib,subprocess
from ask_price_grid import recover_receipts
import joblib,numpy as np
from audit_panel import OUT
from multiproduct_demand import MultiproductDemand
from demand_timing_sensitivity import terminal_status_data
from normalized_demand_bounds import NormalizedBand


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--band-file',default='demand_uniform_pairs_initial_p350_b1000.npz');ap.add_argument('--critical',type=float);ap.add_argument('--prefix',default='upgrade_normalized_bundle');ap.add_argument('--asking-grid',action='store_true');args=ap.parse_args();assert '/' not in args.prefix and chr(92) not in args.prefix
 s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')].sort_values(['Code','date']).reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');data=np.load(OUT/args.band_file);critical=float(data['critical']) if args.critical is None else args.critical;band=NormalizedBand(fit,s,data['covariance'],critical);prefix=OUT/args.prefix
 columns=['Code','date','Price','Ref_Price','initial_reference','own_n','market_id','since_seen','sale_terminal_status'];s[columns].to_csv(str(prefix)+'.csv',index=False);np.savez_compressed(str(prefix)+'.npz',**{k:v for k,v in band.__dict__.items() if k!='fit'})
 upper=s.assign(cost_cap=recover_receipts(s.Price.to_numpy()) if args.asking_grid else s.Price.to_numpy()).groupby('Code').cost_cap.min();anchor=s.groupby('Code').initial_reference.first();files=[str(prefix)+x for x in ['.csv','.npz']]
 meta={'seller':'UPGRADE SOLUTION INC','price_cap':350.,'critical':critical,'asking_grid':args.asking_grid,'demand_band_file':args.band_file,'demand_band_sha256':hashlib.sha256((OUT/args.band_file).read_bytes()).hexdigest(),'rows':len(s),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rho_max_feasible':{str(delta):float(((upper[anchor>0]+delta)/anchor[anchor>0]).min()) for delta in [20,35,50]},'sha256':{str(x):hashlib.sha256(open(x,'rb').read()).hexdigest() for x in files}}
 open(str(prefix)+'.json','w').write(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
