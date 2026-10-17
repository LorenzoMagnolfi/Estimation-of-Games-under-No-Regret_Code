"""Reweight a verified endpoint gradient using an alternative covariance.

Checks that the new covariance lies in the original factor's estimable
subspace. This is algebraically equivalent to recomputing the same gradient
in the new covariance factor, up to the recorded projection residual.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_panel import OUT


def main():
    results=[]
    for demand in ['common','seller']:
        covpath=OUT/f'demand_cluster_{"initial" if demand=="common" else "seller"}_week_covariance.npz';cov=np.load(covpath)['covariance']
        for delta in [0,5]:
            stem=f'daily_multisku_{demand}_d{delta}_h02';source=OUT/f'{stem}_inference.json';report=json.loads(source.read_text());profile=json.loads((OUT/report['profile']).read_text());gradpath=OUT/f'{stem}_inference_gradients.npz';data=np.load(gradpath);L=data['factor'];G=data['gradient']@np.linalg.pinv(L);project=L@np.linalg.pinv(L);residual=float(np.linalg.norm(cov-project@cov@project.T)/np.linalg.norm(cov));assert residual<1e-6,'Alternative covariance adds directions missing from the recorded gradient'
            se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',G,cov,G),0));critical=report['normal_critical'];point=report['point_cost_outer_interval'];prior=profile['prior_cost_interval'];interval=[max(prior[0],point[0]-critical*se[0]),min(prior[1],point[1]+critical*se[1])];mean=profile['mean_receipt'];row={'demand':demand,'delta':delta,'asymptotic_cost_outer_interval':interval,'asymptotic_margin_outer_interval':[mean-interval[1],mean-interval[0]],'endpoint_standard_errors':se.tolist(),'covariance_projection_relative_error':residual,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,gradpath,covpath]},'interpretation':'Weekly-cluster asymptotic sensitivity using the verified cell-LP endpoint gradient. Only 24 time clusters; not a finite-sample coverage guarantee.'};results.append(row);print(json.dumps(row),flush=True)
    (OUT/'daily_multisku_weekly_sensitivity.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
