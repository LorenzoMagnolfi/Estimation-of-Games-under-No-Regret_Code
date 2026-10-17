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
  d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')];counts=d.groupby('market_id').size();chosen=counts[(counts>=2)&(counts<=8)].index[:8];s=d[d.market_id.isin(chosen)].reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance'];critical=6.;band=NormalizedBand(fit,s,cov,critical,slope_critical=3.85,peer_critical=3.29);rng=np.random.default_rng(717);p=s.Price.to_numpy();x=recover_receipts(p);cl=np.maximum(0,x-45);ch=x-10;marks=np.array([10.,30.,100.]);groups=list(s.groupby('market_id',sort=False).indices.values());n=s.own_n.to_numpy();P=350.;band.prepare_peers(groups)
  lower,upper=band.bounds(cl,ch,marks,P,n,33);vals,vec=np.linalg.eigh(cov);factor=vec*np.sqrt(np.maximum(vals,0));theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]]
  for trial in range(20):
   noise=rng.normal(size=len(theta));noise*=2*rng.random()/np.linalg.norm(noise);th=theta+factor@noise;actual=copy.deepcopy(fit);actual.model.intercept_=th[:1];actual.model.coef_=th[None,1:];ef=actual.effects();bp=ef['own_peer_price']/100
   if bp<0:continue
   c=rng.uniform(cl,ch);px,proposed=rounded_proposal((cl+ch)/2,marks,P);px=np.maximum(px,(1.03*np.ceil(ch/1.03-1e-10))[:,None]);proposed=np.ceil(px-1e-9);qa=actual.predict(s);z=actual.logit(s);qp=np.column_stack([actual.predict(s,proposed[:,j]-s.Ref_Price.to_numpy()) for j in range(len(marks))]);own=qp*(px-c[:,None])-qa[:,None]*(x[:,None]-c[:,None]);sp,_=peer_spillover_bounds(proposed,proposed,p,z,n,groups,x-c,x-c,bp,P)
   self.assertTrue(np.all((own+sp)/qa[:,None]>=lower-1e-8))
   ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);ran=maximum_own_profit(c,offset,b,gp,gn,rival,P)
   for ids in groups:
    ni=int(n[ids[0]]);m=len(ids)
    if ni<=1:continue
    diff=expit(z[ids][None,:]+bp*(P-p[ids,None])/(ni-1))-expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/(ni-1));ran[ids]+=(diff*(1-np.eye(m)))@(x[ids]-c[ids])
   self.assertTrue(np.all(ran/qa<=upper+1e-8))
   # Enumerate the entire feasible integer-ask menu independently of the
   # enclosure's handful of proposals and coupled own-range calculation.
   asks=np.arange(1,int(np.floor(P/1.03))+1);receipts=1.03*asks;display=np.ceil(receipts-1e-9)
   menu=np.broadcast_to(display,(len(s),len(asks)));qmenu=expit(offset[:,None]+b[:,None]*menu+gp*np.maximum(menu-rival[:,None],0)+gn*np.minimum(menu-rival[:,None],0))
   ownmenu=(receipts[None,:]-c[:,None])*qmenu-(x-c)[:,None]*qa[:,None]
   spillmenu,_=peer_spillover_bounds(menu,menu,p,z,n,groups,x-c,x-c,bp,P)
   gain=np.where(receipts[None,:]>=c[:,None],ownmenu+spillmenu,-np.inf)
   exact_regret=np.maximum(0,gain.max(axis=1))
   self.assertTrue(np.all(band.last_joint_fraction_lower<=exact_regret/ran+1e-8))
   if trial==0:
    oldlo,oldhi=band.zlo.copy(),band.zhi.copy();band.zlo=np.maximum(oldlo,z-.1);band.zhi=np.minimum(oldhi,z+.1);band.qlo=expit(band.zlo);band.qhi=expit(band.zhi)
    clipped_gain,clipped_range=band.bounds(cl,ch,marks,P,n,33)
    self.assertTrue(np.all(np.maximum(0,clipped_gain.max(axis=1))/clipped_range<=exact_regret/ran+1e-8))
    self.assertTrue(np.all(band.last_joint_fraction_lower<=exact_regret/ran+1e-8))
    band.zlo,band.zhi=oldlo,oldhi;band.qlo=expit(oldlo);band.qhi=expit(oldhi)
    band.bounds(cl,ch,marks,P,n,33)


 def test_continuous_to_integer_ask_profit_certificate(self):
  from oracle_witness import maximum_own_profit
  rng=np.random.default_rng(20261017);P=350.;N=10000
  c=rng.uniform(0,P,N);rival=rng.uniform(0,P,N)
  kl=np.exp(rng.uniform(np.log(.001),np.log(.5),N));kr=np.exp(rng.uniform(np.log(.001),np.log(.5),N))
  b=np.zeros(N);gp=-kr;gn=-kl;offset=rng.uniform(-15,15,N)
  continuous=maximum_own_profit(c,offset,b,gp,gn,rival,P)
  asks=np.arange(1,int(np.floor(P/1.03))+1);receipts=1.03*asks;display=np.ceil(receipts-1e-9)
  logits=offset[:,None]+gp[:,None]*np.maximum(display[None,:]-rival[:,None],0)+gn[:,None]*np.minimum(display[None,:]-rival[:,None],0)
  feasible=receipts[None,:]>=c[:,None]
  grid=np.maximum(0,np.where(feasible,(receipts[None,:]-c[:,None])*expit(logits),0).max(axis=1))
  kmax=np.maximum(kl,kr);m=np.minimum(1/kmax,P-c);gamma=np.maximum(0,1-1.03/m)*np.exp(-kmax)
  self.assertTrue(np.all(grid+1e-11>=gamma*continuous))

if __name__=='__main__':unittest.main()
