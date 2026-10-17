"""Local smoothness audit for active lower-wealth cell coefficients."""
import numpy as np


def audit_active_cell(model,left,right):
    heterogeneous=hasattr(model,'delta')
    if heterogeneous:
        profits=np.where(model.allowed(right),model.gain(left,right),-np.inf)
    else:
        pl=model.rA-model.rB*left;ph=model.rA-model.rB*right;profits=np.minimum(pl,ph)
    choice=np.argmax(profits,axis=1);days=np.arange(model.D);best=profits[days,choice];other=profits.copy();other[days,choice]=-np.inf;gaps=best-other.max(axis=1)
    assert np.all(gaps[model.live]>1e-10), 'Active cell has an advisor tie'
    rows=np.arange(model.N);chosen=choice[model.day]
    structural_rows=(model.change[rows,chosen]==0)&(model.gap_plus[rows,chosen]==0)&(model.gap_minus[rows,chosen]==0)&(model.peer[rows,chosen]==0)
    structural_days=np.ones(model.D,dtype=bool);np.logical_and.at(structural_days,model.day,structural_rows)
    nonzero=model.live&~structural_days
    assert np.all(abs(best[nonzero])>1e-10), 'Active lower regret changes range branch at zero'
    report={'minimum_advisor_gap':float(gaps[model.live].min()),'minimum_nonstructural_regret':float(abs(best[nonzero]).min()) if nonzero.any() else None,'structural_zero_days':int((structural_days&model.live).sum())}
    if heterogeneous:
        shift=model.b*model.change[rows,chosen]+model.gp*model.gap_plus[rows,chosen]+model.gn*model.gap_minus[rows,chosen]+model.bp*model.peer[rows,chosen]
        assert np.all(abs(shift[~structural_rows])>1e-10), 'Active lower gain has a demand-sign tie'
        report['minimum_nonstructural_shift']=float(abs(shift[~structural_rows]).min()) if (~structural_rows).any() else None
        low,_=model.cost_box(left);_,high=model.cost_box(right)
        left_ties=model.vector_optima(low)[2];right_ties=model.vector_optima(high)[2]
        used_ties=np.where(best[model.day]>=0,left_ties,right_ties)
        assert not np.any(used_ties&model.active), 'Distinct own-profit optima tied in used range branch'
        denominator=np.where(best>=0,model.vector_range(low),model.vector_range(high))
    else:
        endpoint_gap=abs(pl[days,choice]-ph[days,choice]);assert np.all(endpoint_gap[nonzero]>1e-10), 'Active lower gain ties cost endpoints'
        report['minimum_nonstructural_cost_endpoint_gap']=float(endpoint_gap[nonzero].min()) if nonzero.any() else None
        left_ties=model.own_optimum_probabilities(left)[3];right_ties=model.own_optimum_probabilities(right)[3]
        used_ties=np.where(best[model.day]>=0,left_ties,right_ties)
        assert not np.any(used_ties&model.active), 'Distinct own-profit optima tied in used range branch'
        ranges=model.ranges([left,right]);denominator=np.where(best>=0,ranges[:,0],ranges[:,1])
    ratio=best[model.live]/denominator[model.live]
    assert np.all(abs(ratio+1)>1e-10), 'Lower wealth is at clipping kink'
    report['minimum_clipping_distance']=float(abs(ratio+1).min())
    return report
