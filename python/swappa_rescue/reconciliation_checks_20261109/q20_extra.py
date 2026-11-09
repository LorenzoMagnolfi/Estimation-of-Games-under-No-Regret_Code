import math
import numpy as np
d = 9; Breg = math.log(d)
for (H, tau, lab) in [(49196, 10120, 'cell'), (792, 164, 'day')]:
    impl = 2 * math.sqrt(Breg / H); implw = 2 * math.sqrt(Breg) * 2 * (math.sqrt(H + tau) - math.sqrt(H)) / tau
    print(lab, 'impl sum', round(impl * tau, 2), 'per', round(impl, 5), 'window-mode sum', round(implw * tau, 2), 'per', round(implw, 5))
    s = np.arange(H + 1, H + tau + 1, dtype=float); n = s - 1
    ftrl = math.sqrt(Breg) * np.sqrt(s) / n
    Eb = (ftrl + np.sqrt(2 * Breg / n)).sum()
    markov = (Eb + math.sqrt(2 * tau * math.log(d + 1))) / tau / 0.05
    print('  markov first-moment per round', round(markov, 4), ' E part', round(Eb / tau, 5), ' mart-E part', round(math.sqrt(2 * tau * math.log(d + 1)) / tau, 5))
    for dl in [0.025, 0.02]:
        S = (ftrl + np.sqrt(2 * np.log(2 * d * tau / dl) / n)).sum(); M = math.sqrt(2 * tau * math.log(d / dl))
        print(f'  delta_H=delta_M={dl}: direct per round {(S + M) / tau:.5f}')
    print('  regularization-at-start ratio sqrt(s)/(s-1)*sqrt(H)', ftrl[0] / math.sqrt(Breg / H))
    print('  c-prime at start (E route)', (ftrl[0] + math.sqrt(2 * Breg / H)) / math.sqrt(Breg / H))
print('cells per day', 10120 / 164, 'sqrt', math.sqrt(10120 / 164), 'ratio day/cell direct', 0.48187 / 0.06464)
