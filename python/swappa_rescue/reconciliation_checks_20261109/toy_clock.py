"""Common-shock batched toy: does the cell-clock Q20-W martingale term cover realized regret?"""
import math
import numpy as np
rng = np.random.default_rng(7)
B, D, dM, d = 62, 164, 0.025, 2
reps = 20000
X = rng.integers(0, 2, size=(reps, D))                 # day shock: action 1 pays 1 in every cell iff X=1
# learner: independent uniform draws in each cell (rho_s = 0 for every learner here)
k1 = rng.binomial(B, 0.5, size=(reps, D))              # cells choosing action 1
learner = (k1 * X + (B - k1) * (1 - X)).sum(1)
best = np.maximum(B * X.sum(1), B * (1 - X).sum(1))
reg = np.maximum(best - learner, 0)
cell = math.sqrt(2 * B * D * math.log(d / dM))         # K = 1 per cell, tau = B D
day = B * math.sqrt(2 * D * math.log(d / dM))          # K_day = B, tau_day = D
print('mean realized regret', reg.mean(), ' cell-clock term', round(cell, 1), ' P(fail)', (reg > cell).mean())
print('day-clock term', round(day, 1), ' P(fail)', (reg > day).mean())
print('i.i.d.-cells variant:')
Xc = rng.integers(0, 2, size=(reps, B * D))
A = rng.integers(0, 2, size=(reps, B * D))
learner2 = np.where(A == 1, Xc, 1 - Xc).sum(1)
best2 = np.maximum(Xc.sum(1), (1 - Xc).sum(1))
reg2 = np.maximum(best2 - learner2, 0)
print('mean realized regret', reg2.mean(), ' cell-clock term', round(cell, 1), ' P(fail)', (reg2 > cell).mean())
print('sqrt(H/(H-62))', math.sqrt(49196 / (49196 - 62)))
