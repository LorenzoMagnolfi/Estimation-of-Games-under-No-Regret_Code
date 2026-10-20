"""Verify shape projections against independent linear programs."""
import numpy as np
from scipy.optimize import linprog
from gadget_inventory_feedback_extension import band, shape_bands
from gadget_feedback_audit import interval


def test_mixture_matches_original_at_same_budget():
    for n,s in [(1,0),(1,1),(11,2.5),(50,10)]:
        for j in [1,3,10]:
            np.testing.assert_allclose(band(n,s,.05/(5*j*(j+1))),interval(n,s,j,'mixture'),atol=1e-12)


def test_shape_projection_equals_linear_program():
    rng=np.random.default_rng(20261020)
    for _ in range(12):
        k=6; prices=np.arange(k)
        q=np.sort(rng.uniform(.05,.95,k))[::-1]
        lower=np.maximum(0,q-rng.uniform(0,.5,k)); upper=np.minimum(1,q+rng.uniform(0,.5,k))
        order=rng.permutation(k)
        lo,hi=shape_bands(prices[order],lower[order],upper[order])
        a=np.zeros((k-1,k))
        for j in range(k-1):a[j,j]=-1;a[j,j+1]=1
        for i in range(k):
            objective=np.eye(k)[order[i]]
            minimum=linprog(objective,A_ub=a,b_ub=np.zeros(k-1),bounds=list(zip(lower,upper)),method='highs')
            maximum=linprog(-objective,A_ub=a,b_ub=np.zeros(k-1),bounds=list(zip(lower,upper)),method='highs')
            assert minimum.success and maximum.success
            np.testing.assert_allclose([lo[i],hi[i]],[minimum.fun,-maximum.fun],atol=1e-10)


def test_inconsistent_monotonicity_detected():
    lo,hi=shape_bands(np.array([100,200]),np.array([.1,.8]),np.array([.2,.9]))
    assert np.any(lo>hi)
