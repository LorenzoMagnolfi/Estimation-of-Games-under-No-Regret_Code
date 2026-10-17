"""Two-way clustered demand uncertainty diagnostics (asymptotic, not final CI).

Listing and calendar-day cluster scores allow repeated-unit and common-date
variation. A PSD projection is recorded explicitly. No application coverage is
claimed from this covariance alone, and causal identification is separate.
"""
import argparse
import hashlib
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack,diags
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from audit_panel import OUT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--time-cluster',choices=['day','week'],default='day');ap.add_argument('--fit-name');ap.add_argument('--stem');ap.add_argument('--variant',choices=['baseline','terminal','initial'],default='baseline');args=ap.parse_args()
    from demand_timing_sensitivity import terminal_status_data
    d=data() if args.variant=='baseline' else terminal_status_data();fitname={'baseline':'multiproduct_demand.joblib','terminal':'multiproduct_sale_terminal_status.joblib','initial':'multiproduct_initial_terminal.joblib'}[args.variant];fitname=args.fit_name or fitname;fit=joblib.load(OUT/fitname);n=len(d);q=fit.predict(d);y=d['sale_last_seen' if args.variant=='baseline' else 'sale_terminal_status'].to_numpy();r=y-q
    stem=args.stem or ('demand_cluster' if args.variant=='baseline' else f'demand_cluster_{args.variant}')
    continuous=fit.scaler.transform(fit.continuous(d));cat=fit.encoder.transform(fit.cat(d))
    X=hstack([csr_matrix(np.ones((n,1))),csr_matrix(continuous),cat],format='csr');p=X.shape[1]
    H=(X.T@diags(q*(1-q))@X).toarray();pen=np.ones(p)/fit.model.C;pen[0]=0;H+=np.diag(pen);bread=np.linalg.inv(H)
    def cluster_meat(groups):
        labels,ix=np.unique(groups,return_inverse=True);g=len(labels);S=(csr_matrix((r,(ix,np.arange(n))),shape=(g,n))@X).toarray();S-=S.mean(axis=0)
        return S.T@S*g/(g-1),g
    ml,gl=cluster_meat(d.Code.to_numpy());time_labels=d.date.to_numpy() if args.time_cluster=='day' else d.date.dt.to_period('W').astype(str).to_numpy();md,gd=cluster_meat(time_labels)
    if args.time_cluster=='day':mr=(X.T@diags(r*r)@X).toarray();gi=n
    else:mr,gi=cluster_meat(d.Code.astype(str).to_numpy()+'::'+time_labels)
    meat=ml+md-mr
    evals,evecs=np.linalg.eigh((meat+meat.T)/2);mpsd=(evecs*np.maximum(evals,0))@evecs.T
    cov=bread@mpsd@bread;cov=(cov+cov.T)/2;covl=bread@ml@bread;covd=bread@md@bread
    theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];rows=[]
    def grad(terms):
        g=np.zeros(p)
        for name,weight in terms.items():
            j=fit.names.index(name);g[j+1]=weight/fit.scaler.scale_[j]
        return g
    specs={'peer_price_per_dollar':{'own_peer_price':.01}}
    for ref in [100,160,310,400]:
      for side in ['gap_positive','gap_negative']:
        specs[f'slope_magnitude_ref{ref}_{side}']={'own':-.01,'price_ref_interaction':-.01*np.log(ref/500),side:-.01}
    for name,terms in specs.items():
        g=grad(terms);estimate=float(g@theta);se=float(np.sqrt(max(0,g@cov@g)))
        rows.append({'parameter':name,'estimate':estimate,'two_way_se':se,'listing_se':float(np.sqrt(max(0,g@covl@g))),'day_se':float(np.sqrt(max(0,g@covd@g))),'normal_95_low':estimate-1.96*se,'normal_95_high':estimate+1.96*se})
    report={'variant':args.variant,'fit':fitname,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/fitname,OUT/'panel.parquet']},'n':n,'parameters':p,'listing_clusters':gl,'date_clusters':gd,'time_cluster_unit':args.time_cluster,'intersection_clusters':gi,'negative_meat_eigenvalues':int((evals<0).sum()),'minimum_meat_eigenvalue':float(evals.min()),'maximum_meat_eigenvalue':float(evals.max()),'hessian_condition':float(np.linalg.cond(H)),'interpretation':'Two-way cluster robust asymptotic covariance, PSD-projected meat, penalized Hessian. These are diagnostic marginal intervals; not a simultaneous demand confidence set or causal validation.','contrasts':rows}
    (OUT/f'{stem}_uncertainty.json').write_text(json.dumps(report,indent=2));pd.DataFrame(rows).to_csv(OUT/f'{stem}_contrasts.csv',index=False);np.savez_compressed(OUT/f'{stem}_covariance.npz',theta=theta,covariance=cov,listing_covariance=covl,day_covariance=covd,hessian=H)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
