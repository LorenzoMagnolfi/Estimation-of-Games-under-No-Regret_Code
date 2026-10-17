import unittest
import numpy as np
from audit_panel import OUT
from scipy.special import expit
from daily_joint_oracle import group_profit_upper

class JointOracleTests(unittest.TestCase):
 def test_mean_profile_dominates_full_two_phone_enumeration(self):
  rng=np.random.default_rng(17026);P=60.;asks=np.arange(1,int(np.floor(P/1.03))+1);x=1.03*asks;pgrid=np.ceil(x-1e-9)
  for trial in range(20):
   p=np.array([pgrid[35],pgrid[45]]);c=rng.uniform(0,20,2);b=-rng.uniform(.02,.12,2);gp=-.01;gn=-.005;bp=.01;rival=np.array([35.,40.]);z=rng.uniform(-3,1,2);active=np.array([True,trial%2==0]);a1,a2=np.meshgrid(pgrid,pgrid,indexing='ij');x1,x2=np.meshgrid(x,x,indexing='ij');offset=z-b*p-gp*np.maximum(p-rival,0)-gn*np.minimum(p-rival,0)
   z1=offset[0]+b[0]*a1+gp*np.maximum(a1-rival[0],0)+gn*np.minimum(a1-rival[0],0)+bp*(a2-p[1]);z2=offset[1]+b[1]*a2+gp*np.maximum(a2-rival[1],0)+gn*np.minimum(a2-rival[1],0)+bp*(a1-p[0]);profit=(x1-c[0])*expit(z1)+(x2-c[1])*expit(z2);valid=(x1>=c[0])&(x2>=c[1]);valid&=(a2==p[1]) if not active[1] else True;exact=float(np.where(valid,profit,-np.inf).max());bound=group_profit_upper(c,p,z,b,gp,gn,rival,bp,active,mean_step=3.,P=P);self.assertGreaterEqual(bound+1e-8,exact)

class JointDesignTests(unittest.TestCase):
 @unittest.skipUnless((OUT/'multiproduct_initial_terminal.joblib').exists(),'Requires demand fit')
 def test_simultaneous_price_changes_match_full_model_design(self):
  import joblib
  from demand_timing_sensitivity import terminal_status_data
  from daily_joint_pricing import joint_logits
  from ask_price_grid import recover_receipts
  s=terminal_status_data();s=s[s.Seller.eq('UPGRADE SOLUTION INC')];counts=s.groupby('market_id').size();chosen=counts[(counts>=2)&(counts<=5)].index[:6];s=s[s.market_id.isin(chosen)].reset_index(drop=True);fit=joblib.load(OUT/'multiproduct_initial_terminal.joblib');p=s.Price.to_numpy();rng=np.random.default_rng(718);proposed=np.ceil(1.03*rng.integers(70,220,(len(s),3))-1e-9);groups=list(s.groupby('market_id',sort=False).indices.values());ef=fit.effects();b=(ef['own']+ef['price_ref_interaction']*np.log(s.Ref_Price.to_numpy()/500))/100;actual=joint_logits(fit.logit(s),p,proposed,b,ef['gap_positive']/100,ef['gap_negative']/100,s.rival_min.to_numpy(),ef['own_peer_price']/100,groups)
  for j in range(3):
   changed=s.copy();changed['Price']=proposed[:,j];changed['residual']=changed.Price-changed.Ref_Price;changed['own_total']=changed.groupby('market_id').Price.transform('sum');np.testing.assert_allclose(actual[:,j],fit.logit(changed),atol=1e-10)

class JointRatioTests(unittest.TestCase):
 def test_ratio_bound_against_sampled_joint_profits(self):
  from joint_profit_ratio import daily_ratio_lower
  rng=np.random.default_rng(17);N=12;day=np.repeat(np.arange(3),4);zlo=rng.uniform(-4,-2,N);zhi=zlo+1.5;dlo=rng.uniform(-2,2,(N,4));x=rng.uniform(50,100,N);cl=rng.uniform(5,15,N);ch=cl+10;new=rng.uniform(35,120,(N,4));bound=daily_ratio_lower(zlo,zhi,dlo,new,x,cl,ch,day,3)
  for _ in range(50):
   z=rng.uniform(zlo,zhi);d=dlo+rng.uniform(0,1,(N,4));c=rng.uniform(cl,ch);old=np.bincount(day,weights=(x-c)*expit(z),minlength=3);profit=np.zeros((3,4));np.add.at(profit,day,(new-c[:,None])*expit(z[:,None]+d));exact=np.maximum(0,1-old/np.maximum(old,profit.max(axis=1)));self.assertTrue(np.all(bound<=exact+1e-10))

if __name__=='__main__':unittest.main()
