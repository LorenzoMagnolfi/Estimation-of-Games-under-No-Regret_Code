"""Explicit learning-scope and contemporaneous rival-price advice sensitivity."""
from types import MethodType
import numpy as np


def _prepare_with_menu(self,theta):
    self._unrestricted_prepare(theta)
    self.A=np.where(self.advisor_menu,self.A,-np.inf)
    self.rA=np.where(self.advisor_menu,self.rA,-np.inf)
    return self


def configure_model(model,range_scale=1.,rival_band=None):
    assert range_scale>=1
    model.range_scale=float(range_scale)
    if rival_band is not None:
        low,high=map(float,rival_band);assert 0<low<high
        rival=np.bincount(model.day,weights=model.rival,minlength=model.D)/np.bincount(model.day,minlength=model.D)
        assert np.max(abs(model.rival-rival[model.day]))<1e-8
        proposed=np.ceil(model.receipt-1e-9)
        model.advisor_menu=(proposed[None,:]>=low*rival[:,None])&(proposed[None,:]<=high*rival[:,None])
        # The highest ask is a predeclared feasible fallback, so negative-
        # margin common-price proposals remain dominated for every cost.
        assert model.x.max()<=model.receipt[-1]+1e-8
        model.advisor_menu[:,-1]=True
        model._unrestricted_prepare=model.prepare
        model.prepare=MethodType(_prepare_with_menu,model)
        model.prepare(model.theta)
    return model
