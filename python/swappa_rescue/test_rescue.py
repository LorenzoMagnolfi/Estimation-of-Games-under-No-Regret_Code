"""Independent analytical and data checks for the new inverse programs."""
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
from continuous_cost_lp import program, endpoint_program
from audit_panel import OUT

class CostBoundsTests(unittest.TestCase):
    def bounds(self,edges,budget):
        C=np.array([[-5.,-3.]])
        B=np.array([[.3,-.08]])
        U,E,bounds,V,K=program(C,B,np.array([500.]),np.array(edges,dtype=float))
        cap=csr_matrix(np.r_[np.zeros(2*V),np.ones(K)][None,:])
        UU=vstack([U,cap]); rhs=np.r_[np.zeros(U.shape[0]),budget]
        obj=np.r_[np.zeros(V),np.ones(V),np.zeros(K)]
        ans=[]
        for sign in [1,-1]:
            r=linprog(sign*obj,A_ub=UU,b_ub=rhs,A_eq=E,b_eq=np.ones(1),bounds=bounds,method='highs')
            self.assertTrue(r.success,r.message)
            self.assertLess(np.abs(E@r.x-1).max(),1e-8)
            self.assertLess((UU@r.x-rhs).max(),1e-8)
            ans.append(sign*r.fun)
        return ans
    def test_endpoint_formulation(self):
        C=np.array([[-5.,-3.]])
        B=np.array([[.3,-.08]])
        U,E,bounds,values,K=endpoint_program(C,B,np.array([500.]),np.array([-1500.,-100.,0.,100.,1500.]))
        obj=np.r_[values.ravel(),np.zeros(K)]
        bounds[-K:]=[(0,0)]*K
        ans=[]
        for sign in [1,-1]:
            r=linprog(sign*obj,A_ub=U,b_ub=np.zeros(U.shape[0]),A_eq=E,b_eq=np.ones(1),bounds=bounds,method='highs')
            self.assertTrue(r.success)
            ans.append(sign*r.fun)
        np.testing.assert_allclose(ans,[-5/.3,3/.08],atol=1e-7)
    def test_analytical_interval(self):
        np.testing.assert_allclose(self.bounds([-100,0,100],0),[-5/.3,3/.08],atol=1e-7)
    def test_regret_budget(self):
        np.testing.assert_allclose(self.bounds([-100,0,100],.1),[-5.1/.3,3.1/.08],atol=1e-7)
    def test_support_not_binding(self):
        np.testing.assert_allclose(self.bounds([-1500,-100,0,100,1500],0),[-5/.3,3/.08],atol=1e-7)
    def test_partition_refinement(self):
        np.testing.assert_allclose(self.bounds([-100,-20,0,20,40,100],0),self.bounds([-100,0,100],0),atol=1e-7)

class CompetitionTests(unittest.TestCase):
    def test_full_market_seller_exclusion(self):
        d=pd.read_parquet(OUT/'panel.parquet')
        groups=dict(tuple(d.groupby('market_id')))
        for _,r in d.sample(500,random_state=16016).iterrows():
            g=groups[r.market_id]
            rival=g.loc[g.Seller!=r.Seller,'Price']
            self.assertEqual(r.rival_n,len(rival))
            if len(rival):
                self.assertEqual(r.rival_min,rival.min())
                self.assertAlmostEqual(r.rival_mean,rival.mean(),places=8)
            else:self.assertTrue(pd.isna(r.rival_min))

if __name__=='__main__': unittest.main()

