"""Two-way clustered demand uncertainty diagnostics (asymptotic, not final CI).

Listing and calendar-day cluster scores allow repeated-unit and common-date
variation. A PSD projection is recorded explicitly. No application coverage is
claimed from this covariance alone, and causal identification is separate.
"""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack,diags
from multiproduct_demand import MultiproductDemand
from quality_demand import data
from audit_panel import OUT


def main():
    d=data();fit=joblib.load(OUT/'multiproduct_demand.joblib');n=len(d);q=fit.predict(d);y=d.sale_last_seen.to_numpy();r=y-q
    continuous=fit.scaler.transform(fit.continuous(d));cat=fit.encoder.transform(fit.cat(d))
    X=hstack([csr_matrix(np.ones((n,1))),csr_matrix(continuous),cat],format='csr');p=X.shape[1]
    H=(X.T@diags(q*(1-q))@X).toarray();pen=np.ones(p)/fit.model.C;pen[0]=0;H+=np.diag(pen);bread=np.linalg.inv(H)
    def cluster_meat(groups):
        labels,ix=np.unique(groups,return_inverse=True);g=len(labels);S=(csr_matrix((r,(ix,np.arange(n))),shape=(g,n))@X).toarray();S-=S.mean(axis=0)
        return S.T@S*g/(g-1),g
    ml,gl=cluster_meat(d.Code.to_numpy());md,gd=cluster_meat(d.date.to_numpy());mr=(X.T@diags(r*r)@X).toarray();meat=ml+md-mr
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
    report={'n':n,'parameters':p,'listing_clusters':gl,'date_clusters':gd,'negative_meat_eigenvalues':int((evals<0).sum()),'minimum_meat_eigenvalue':float(evals.min()),'maximum_meat_eigenvalue':float(evals.max()),'hessian_condition':float(np.linalg.cond(H)),'interpretation':'Two-way cluster robust asymptotic covariance, PSD-projected meat, penalized Hessian. These are diagnostic marginal intervals; not a simultaneous demand confidence set or causal validation.','contrasts':rows}
    (OUT/'demand_cluster_uncertainty.json').write_text(json.dumps(report,indent=2));pd.DataFrame(rows).to_csv(OUT/'demand_cluster_contrasts.csv',index=False);np.savez_compressed(OUT/'demand_cluster_covariance.npz',theta=theta,covariance=cov,listing_covariance=covl,day_covariance=covd,hessian=H)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
