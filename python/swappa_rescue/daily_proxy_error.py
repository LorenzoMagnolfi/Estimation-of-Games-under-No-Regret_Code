"""Wealth allowance for uniformly approximate normalized payoff feedback.

If |u_tilde-u|<=epsilon on every feasible joint action and both payoffs are
normalized, proxy full-oracle regret r_tilde is nonnegative and any true
advisor regret r is <=r_tilde+2epsilon. Consequently
log(1+eta*r)<=log(1+eta*r_tilde)+log(1+2eta*epsilon).
Positive epsilon describes an approximate-learning extension; it does not
imply exact asymptotic no regret for true payoffs.
"""
import numpy as np


def proxy_wealth_allowance(rounds,epsilon,eta=.99):
    assert rounds>=0 and epsilon>=0 and 0<eta<1
    return float(rounds*np.log1p(2*eta*epsilon))
