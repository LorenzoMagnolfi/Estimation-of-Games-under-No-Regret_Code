"""Analytic Gaussian maxima and nuisance-band payoff enclosure checks."""
import copy,unittest
import numpy as np
from scipy.special import expit
from demand_uniform_band import max_standardized_line
from demand_band_bounds import own_gain_z_extrema,logit_difference_bounds,LogitBand
from audit_panel import OUT

class DemandBandTests(unittest.TestCase):
 def test_studentized_line_maximum(self):
  rng=np.random.default_rng(810)
  for rep in range(20):
   left=rng.normal(size=(10,4));delta=rng.normal(size=(10,4));z=rng.normal(size=(4,8));a=left@z;b=delta@z;A=(left*left).sum(1);B=(left*delta).sum(1);C=(delta*delta).sum(1);bound=max_standardized_line(a,b,A,B,C)
   brute=np.zeros(8)
   for t in np.linspace(0,1,501):brute=np.maximum(brute,(abs(a+b*t)/np.sqrt((A+2*B*t+C*t*t)[:,None])).max(0))
   self.assertTrue(np.all(bound>=brute-1e-10));self.assertLess(np.max(bound-brute),.001)
 def test_paired_logit_extrema(self):
  rng=np.random.default_rng(123)
  for rep in range(30):
   zl=rng.normal(-2,2,(10,4));zh=zl+rng.random((10,4))*3;shift=rng.normal(0,2,(10,4));new=rng.uniform(0,100,(10,4));old=rng.uniform(0,100,(10,4));lo,hi=own_gain_z_extrema(zl,zh,shift,new,old);dl,dh=logit_difference_bounds(zl,zh,0,shift)
   for t in np.linspace(0,1,101):
    z=zl+t*(zh-zl);v=new*expit(z+shift)-old*expit(z);dv=expit(z+shift)-expit(z)
    self.assertTrue(np.all(v>=lo-1e-10));self.assertTrue(np.all(v<=hi+1e-10));self.assertTrue(np.all(dv>=dl-1e-10));self.assertTrue(np.all(dv<=dh+1e-10))
 @unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists() and (OUT/'demand_cluster_initial_covariance.npz').exists(),'Requires generated demand fit and covariance')
 def test_joint_parameter_cost_and_peer_enclosure(self):
  import joblib
  from demand_timing_sensitivity import terminal_status_data
  from coordinate_profit import peer_spillover_bounds
  from profit_specialists import coordinate_range
  d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')];counts=d.groupby('market_id').size();chosen=counts[(counts>=2)&(counts<=8)].index[:8];s=d[d.market_id.isin(chosen)].reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance'];critical=2.;band=LogitBand(fit,s,cov,critical);rng=np.random.default_rng(313);p=s.Price.to_numpy();cl=np.maximum(0,p-45);ch=p-30;marks=np.array([10.,30.,100.]);groups=list(s.groupby('market_id',sort=False).indices.values());n=s.own_n.to_numpy();P=350.
  lo,hi=band.own_regret_box(cl,ch,marks,P,17);spill,ranlo,ranhi=band.peer_and_range_box(cl,ch,marks,P,n,groups)
  vals,vec=np.linalg.eigh(cov);factor=vec*np.sqrt(np.maximum(vals,0));theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]]
  for trial in range(20):
   noise=rng.normal(size=len(theta));noise*=critical*rng.random()/np.linalg.norm(noise);th=theta+factor@noise;actual=copy.deepcopy(fit);actual.model.intercept_=th[:1];actual.model.coef_=th[None,1:];ef=actual.effects();bp=ef['own_peer_price']/100
   if bp<0:continue
   c=rng.uniform(cl,ch);proposed=np.minimum(P,c[:,None]+marks);qa=actual.predict(s);qp=np.column_stack([actual.predict(s,proposed[:,j]-s.Ref_Price.to_numpy()) for j in range(len(marks))]);r=qp*(proposed-c[:,None])-qa[:,None]*(p[:,None]-c[:,None]);self.assertTrue(np.all(r>=lo-1e-8));self.assertTrue(np.all(r<=hi+1e-8))
   sp,_=peer_spillover_bounds(proposed,proposed,p,actual.logit(s),n,groups,p-c,p-c,bp,P);self.assertTrue(np.all(sp>=spill-1e-8))
   b=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500))/100;kmin=np.minimum(-(b+ef['gap_positive']/100),-(b+ef['gap_negative']/100));ran=coordinate_range(c,p,actual.logit(s),n,groups,bp,P,actual.logit(s,c-s.Ref_Price.to_numpy()),kmin,p-c)
   self.assertTrue(np.all(ran>=ranlo-1e-8));self.assertTrue(np.all(ran<=ranhi+1e-8))

class CoupledOracleTests(unittest.TestCase):
 @unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires generated demand fit')
 def test_continuous_cost_chords_and_coupled_relaxation(self):
  import joblib
  from oracle_coupled_lp import own_gain_chord,own_range_chord
  from oracle_witness import maximum_own_profit
  from demand_timing_sensitivity import terminal_status_data
  d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')].iloc[::71].reset_index(drop=True)
  fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance'];critical=4.3156
  band=LogitBand(fit,d,cov,critical);p=d.Price.to_numpy();cmax=p.copy();P=350.;u0,uslope=own_range_chord(band,cmax,P)
  vals,vec=np.linalg.eigh(cov);factor=vec*np.sqrt(np.maximum(vals,0));theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]];rng=np.random.default_rng(61016)
  for rep in range(15):
   noise=rng.normal(size=len(theta));noise*=critical*rng.random()/np.linalg.norm(noise);th=theta+factor@noise;actual=copy.deepcopy(fit);actual.model.intercept_=th[:1];actual.model.coef_=th[None,1:]
   c=cmax*rng.random(len(d));q=actual.predict(d);upper=u0+uslope*c
   ef=actual.effects();ref=d.Ref_Price.to_numpy();rival=d.rival_min.to_numpy();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;offset=actual.logit(d)-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
   exact=maximum_own_profit(c,offset,b,gp,gn,rival,P)
   self.assertTrue(np.all(exact<=upper+1e-8))
   cl=np.maximum(0,c-10);ch=np.minimum(cmax,c+10);u,slope=own_range_chord(band,ch,P,cl)
   self.assertTrue(np.all(exact<=u+slope*c+1e-8))
   # Dense prices check the continuous analytical upper bound, including p<c.
   for price in np.linspace(0,P,81):
    profit=(price-c)*actual.predict(d,price-d.Ref_Price.to_numpy())
    self.assertTrue(np.all(profit<=upper+1e-8));self.assertTrue(np.all(profit<=exact+1e-8))
   for delta in [-100.,-5.,5.,100.]:
    price=np.clip(p+delta,0,P);a,b=own_gain_chord(band,price,cmax)
    gain=(price-c)*actual.predict(d,price-d.Ref_Price.to_numpy())-(p-c)*q
    self.assertTrue(np.all(a+b*c<=gain+1e-8))
    a,b=own_gain_chord(band,price,ch,cl);self.assertTrue(np.all(a+b*c<=gain+1e-8))
  # McCormick upper faces retain every valid normalized-loss witness.
  for rep in range(100):
   lo=rng.uniform(1,5,50);hi=lo+rng.uniform(0,10,50);S=rng.uniform(lo,hi);u=rng.uniform(0,S);v=u/S
   self.assertTrue(np.all(u<=hi*v+1e-12));self.assertTrue(np.all(u<=S-lo+lo*v+1e-12))

class NormalizedDemandTests(unittest.TestCase):
 def test_relative_logit_rectangle_extrema(self):
  from normalized_demand_bounds import normalized_difference_bounds
  rng=np.random.default_rng(866)
  for rep in range(20):
   zl=rng.uniform(-7,2,(4,5));zh=zl+rng.uniform(0,3,(4,5));dl=rng.uniform(-2,1,(4,5));dh=dl+rng.uniform(0,3,(4,5));left=rng.uniform(-2,2,(4,5));right=rng.uniform(-2,2,(4,5))
   lo,hi=normalized_difference_bounds(zl,zh,dl,dh,left,right)
   for t in np.linspace(0,1,31):
    z=zl+t*(zh-zl)
    for u in np.linspace(0,1,31):
     d=dl+u*(dh-dl);value=(expit(z+d+right)-expit(z+d+left))/expit(z)
     self.assertTrue(np.all(value>=lo-1e-10));self.assertTrue(np.all(value<=hi+1e-10))
 @unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires generated demand fit')
 def test_normalized_payoff_and_range_enclosures(self):
  import joblib
  from normalized_demand_bounds import NormalizedBand
  from oracle_witness import maximum_own_profit
  from demand_timing_sensitivity import terminal_status_data
  from coordinate_profit import peer_spillover_bounds
  d=terminal_status_data();d=d[d.Seller.eq('UPGRADE SOLUTION INC')];counts=d.groupby('market_id').size();chosen=counts[(counts>=2)&(counts<=8)].index[:8];s=d[d.market_id.isin(chosen)].reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');cov=np.load(OUT/'demand_cluster_initial_covariance.npz')['covariance'];critical=2.;band=NormalizedBand(fit,s,cov,critical);rng=np.random.default_rng(717);p=s.Price.to_numpy();cl=np.maximum(0,p-45);ch=p-10;marks=np.array([10.,30.,100.]);groups=list(s.groupby('market_id',sort=False).indices.values());n=s.own_n.to_numpy();P=350.;band.prepare_peers(groups)
  lower,upper=band.bounds(cl,ch,marks,P,n,33);vals,vec=np.linalg.eigh(cov);factor=vec*np.sqrt(np.maximum(vals,0));theta=np.r_[fit.model.intercept_[0],fit.model.coef_[0]]
  for trial in range(20):
   noise=rng.normal(size=len(theta));noise*=critical*rng.random()/np.linalg.norm(noise);th=theta+factor@noise;actual=copy.deepcopy(fit);actual.model.intercept_=th[:1];actual.model.coef_=th[None,1:];ef=actual.effects();bp=ef['own_peer_price']/100
   if bp<0:continue
   c=rng.uniform(cl,ch);proposed=np.minimum(P,c[:,None]+marks);qa=actual.predict(s);z=actual.logit(s);qp=np.column_stack([actual.predict(s,proposed[:,j]-s.Ref_Price.to_numpy()) for j in range(len(marks))]);own=qp*(proposed-c[:,None])-qa[:,None]*(p[:,None]-c[:,None]);sp,_=peer_spillover_bounds(proposed,proposed,p,z,n,groups,p-c,p-c,bp,P)
   self.assertTrue(np.all((own+sp)/qa[:,None]>=lower-1e-8))
   ref=s.Ref_Price.to_numpy();rival=s.rival_min.to_numpy();b=(ef['own']+ef['price_ref_interaction']*np.log(ref/500))/100;gp=ef['gap_positive']/100;gn=ef['gap_negative']/100;offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0);ran=maximum_own_profit(c,offset,b,gp,gn,rival,P)
   for ids in groups:
    ni=int(n[ids[0]]);m=len(ids)
    if ni<=1:continue
    diff=expit(z[ids][None,:]+bp*(P-p[ids,None])/(ni-1))-expit(z[ids][None,:]+bp*(c[ids,None]-p[ids,None])/(ni-1));ran[ids]+=(diff*(1-np.eye(m)))@(p[ids]-c[ids])
   self.assertTrue(np.all(ran/qa<=upper+1e-8))

if __name__=='__main__':unittest.main()
