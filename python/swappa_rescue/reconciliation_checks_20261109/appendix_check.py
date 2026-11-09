import math, numpy as np
d, Breg, dH, dM = 9, math.log(9), 0.025, 0.025      # Hedge: sigma = c = 1
for H, tau in [(49196, 10120), (792, 164)]:          # SKU-day clock; day clock (floor of 792.93)
    s = np.arange(H + 1, H + tau + 1.0); n = s - 1
    ftrl = math.sqrt(Breg) * np.sqrt(s) / n
    S_fix  = np.minimum(ftrl + np.sqrt(2 * np.log(2 * d * tau / dH) / n), 1).sum()
    S_pi2  = np.minimum(ftrl + np.sqrt(2 * np.log(2 * d * math.pi**2 * n**2 / (6 * dH)) / n), 1).sum()
    S_one  = np.minimum(ftrl + np.sqrt(2 * np.log((d - 1) * tau / dH) / n), 1).sum()
    S_max  = np.minimum(ftrl + np.sqrt(2 * (H + tau - 1) * math.log((d - 1) / dH)) / n, 1).sum()
    E_sum  = np.minimum(ftrl + np.sqrt(2 * math.log(d) / n), 1).sum()
    mart   = math.sqrt(2 * tau * math.log(d / dM))
    split  = math.sqrt(2 * tau * math.log(2 * d / dM)) + math.sqrt(tau * math.log(2 / dM) / 2)
    impl   = 2 * math.sqrt(Breg / H)
    print(H, tau, ftrl.sum(), S_fix, S_pi2, S_one, S_max, mart, split, E_sum, impl * tau)
