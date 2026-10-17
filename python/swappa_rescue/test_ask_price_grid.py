"""Independent random-coefficient checks of exact asking-price enclosures."""
import copy,unittest
import numpy as np
from scipy.special import expit
from audit_panel import OUT
from ask_price_grid import recover_receipts

class AskGridTests(unittest.TestCase):
 def test_recover_integer_asking_grid(self):
  asks=np.arange(1,2000);display=np.ceil(1.03*asks-1e-9)
  np.testing.assert_allclose(recover_receipts(display),1.03*asks,atol=1e-10)
  with self.assertRaises(ValueError):recover_receipts(np.array([35.]))
 @unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires generated demand fit')
 def test_normalized_payoff_and_range_enclosures(self):
  import joblib
  from ask_price_grid import AskGridBand as NormalizedBand,recover_receipts,rounded_proposal
  from oracle_witness import maximum_own_profit
  from demand_timing_sensitivity import terminal_status_data
  from coordinate_profit import peer_spillover_bounds
  d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')];counts=d.groupby('market_id').size();chosen=counts[(counts>=2)&(counts<=8)].index[:8];s=d[d.market_id.isin(chosen)].reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance'];critical=2.;band=NormalizedBand(fit,s,cov,critical);rng=np.random.default_rng(717);p=s.Price.to_numpy();x=recover_receipts(p);cl=np.maximum(0,x-45);ch=x-10;marks=np.array([10.,30.,100.]);groups=list(s.groupby('market_id',sort=False).indices.values());n=s.own_n.to_numpy();P=350.;band.prepare_peers(groups)
  lower,upper=band.bounds(cl,ch,marks,P,n,33);vals,vec=np.linalg.eigh(cov);factor=vec*np.sqrt(np.maximum(vals,0));theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]]
  for trial in range(20):
   noise=rng.normal(size=len(theta));noise*=critical*rng.random()/np.linalg.norm(noise);th=theta+factor@noise;actual=copy.deepcopy(fit);actual.model.intercept_=th[:1];actual.model.coef_=th[None,1:];ef=actual.effects();bp=ef['own_peer_price']/100
   if bp<0:continue
   c=rng.uniform(cl,ch);px,proposed=rounded_proposal((cl+ch)/2,marks,P);px=np.maximum(px,(1.03*np.ceil(ch/1.03-1e-10))[:,None]);proposed=np.ceil(px-1e-9);qa=actual.predict(s);z=actual.logit(s);qp=np.column_stack([actual.predict(s,proposed[:,j]-s.Ref_Price.to_numpy()) for j in range(len(marks))]);own=qp*(px-c[:,None])-qa[:,None]*(x[:,None]-c[:,None]);sp,_=peer_spillover_bounds(proposed,proposed,p,z,n,groups,x-c,x-c,bp,P)
   self.assertTrue(np.all((own+sp)/qa[:,None]>=lower-1e-8))
   ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);ran=maximum_own_profit(c,offset,b,gp,gn,rival,P)
   for ids in groups:
    ni=int(n[ids[0]]);m=len(ids)
    if ni<=1:continue
    diff=expit(z[ids][None,:]+bp*(P-p[ids,None])/(ni-1))-expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/(ni-1));ran[ids]+=(diff*(1-np.eye(m)))@(x[ids]-c[ids])
   self.assertTrue(np.all(ran/qa<=upper+1e-8))

if __name__=='__main__':unittest.main()
