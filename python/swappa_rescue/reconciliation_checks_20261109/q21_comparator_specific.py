"""Comparator-specific data-dependent interval bounds on the day clock, in dollars per window day.

Hypothetical evaluation: the unobserved mixture-based quantities C = sum|r| and V = sum r^2 are replaced by
tau*sigma and tau*sigma^2 with sigma the per-comparator sd of realized daily gains supplied by the coordinator.
This substitution is NOT licensed by any theorem; it shows the size of the data-independent K terms.
"""
import math

K = 1476.64          # common span K_day, dollars
RANGE_MEAN = 326.38  # mean day range, dollars
PROFIT = 591.23      # daily expected profit at theta = 0.8
SIG = [241.7, 87.0, 36.2, 13.2, 11.4, 26.2, 47.3, 84.7]
d = 9
tau = 164
H = 7939 / (1642 / 164)
T = H + tau
t1 = H + 1
delta = 0.025


def anh(sig):
    """AdaNormalHedge.TV: sqrt(3 K C Lambda), C = tau*sig (hypothetical)."""
    lam = math.log(d * t1 ** 2 * math.pi ** 2 / 6) + math.log(2.5 + 1.5 * math.log(1 + T)) + 1
    return math.sqrt(3 * K * tau * sig * lam) / tau, lam


def amlprod(sig, Tbar=T):
    """Adapt-ML-Prod Cor 4 via sleeping reduction with K' = d*Tbar experts, uniform prior."""
    Kp = d * Tbar
    C = 3 * math.log(Kp) + math.log(1 + Kp / (2 * math.e) * (1 + math.log(T + 1)))
    lead = C / math.sqrt(math.log(Kp)) * math.sqrt(K ** 2 + tau * sig ** 2)
    low = 2 * C * K
    return (lead + low) / tau, lead / tau, low / tau, C


def squint(sig, Tbar=T):
    """Squint Thm 4 (improper prior), interval version via reduction (to be verified), pi = 1/(d*Tbar)."""
    pi = 1 / (d * Tbar)
    lead = math.sqrt(2 * tau * sig ** 2 * (1 + 2 * math.log((0.5 + math.log(T + 1)) / pi)))
    low = 5 * K * math.log(1 + (1 + 2 * math.log(T + 1)) / pi)
    return (lead + low) / tau, lead / tau, low / tau


def uma(sig):
    """Zhang, Wang, Tu, Zhou Thm 1 on the simplex: D = sqrt 2, G = sqrt(d)/2 for centred losses."""
    D, G = math.sqrt(2), math.sqrt(d) / 2
    q, p = T, T - tau + 1
    a = 2 * math.log2(2 * q) + 5 * d * math.log(q - p + 2) + 5
    b = 2 * math.ceil(math.log2(q - p + 2))
    low = 10 * D * G * a * b * K
    lead = 3 * math.sqrt(a * b * tau * sig ** 2)
    return (lead + low) / tau, lead / tau, low / tau


print(f'tau={tau}, H_day={H:.1f}, T={T:.1f}, K={K}, profit={PROFIT}')
print('conversion external, common K:', K * math.sqrt(math.log(1 / delta) / (2 * tau)))
print('conversion external, mean range in place of sqrt(mean K_t^2) (lower bound):',
      RANGE_MEAN * math.sqrt(math.log(1 / delta) / (2 * tau)))
print('Phi conversion factor sqrt(2 ln(|Phi|/delta)/tau) with |Phi|=9:', math.sqrt(2 * math.log(9 / delta) / tau))
print('FS-c (Section 3) in dollars: mixed', 0.1590 * K, 'total', 0.2651 * K)
_, lam = anh(1.0)
print('ANH Lambda', lam)
r = amlprod(1.0)
print('Adapt-ML-Prod C', r[3], 'lower-order per day', r[2])
print('Squint lower-order per day', squint(1.0)[2])
print('UMA lower-order per day', uma(1.0)[2])
print('\nsigma | ANH | AdaptMLProd total (lead) | Squint total (lead) | UMA total | Phi conv (sigma as range)')
for s in SIG:
    a_ = anh(s)[0]
    m = amlprod(s)
    q = squint(s)
    u = uma(s)
    pc = s * math.sqrt(2 * math.log(9 / delta) / tau)
    print(f'{s:6.1f} | {a_:7.1f} | {m[0]:8.1f} ({m[1]:6.1f}) | {q[0]:7.1f} ({q[1]:6.1f}) | {u[0]:10.1f} | {pc:6.1f}')
print('\nshares of daily profit for sigma = 11.4 and 241.7, ANH:', anh(11.4)[0] / PROFIT, anh(241.7)[0] / PROFIT)
