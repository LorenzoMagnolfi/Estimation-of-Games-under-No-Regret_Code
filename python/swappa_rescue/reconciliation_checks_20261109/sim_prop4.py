import numpy as np, math, sys
from scipy.stats import norm
rng = np.random.default_rng(12345)
d, B, Hd, T = 3, 20, 40, 164
mu = np.array([0.0, -0.03, -0.06])      # a*=0, small gaps so p_t drifts in window
sc, si = 0.5, 0.5                       # common day shock, idiosyncratic
reps = 2000
alpha = 0.05; z = norm.ppf(1-alpha)
# population kappa^2 = E K^2 (Monte Carlo)
def draw_day(n_days):
    c = rng.uniform(-1,1,(n_days,1,d))
    e = rng.uniform(-1,1,(n_days,B,d))
    return mu + sc*c + si*e
g = draw_day(20000)
K = g.max(-1)-g.min(-1); kappa = math.sqrt((K**2).mean()); Kbar = 2*(sc+si) + (mu.max()-mu.min())
# v_day for a != a*
Y = (g[...,1:] - g[...,[0]]); S = Y.sum(1); vday = S.var(0).max()
DEFF = vday/(B*kappa**2)
print("kappa", kappa, "Kbar", Kbar, "DEFF", DEFF)
rej = np.zeros(d); rej_xi = np.zeros(d); rho_mean = np.zeros(Hd+T); xi_only = np.zeros(d)
cross = []
for rep in range(reps):
    days = draw_day(Hd+T)
    cum = np.zeros(d); n = 0
    Gs = np.zeros((T,d)); pis = np.zeros((T,d))
    for t in range(Hd+T):
        if n == 0: p = np.ones(d)/d
        else:
            s = n+1; eta = math.sqrt(math.log(d)/s)
            w = eta*cum/kappa; w -= w.max(); p = np.exp(w); p /= p.sum()
        rho_mean[t] += (mu.max() - p@mu)/reps
        gd = days[t]
        A = rng.choice(d, size=B, p=p)
        if t >= Hd:
            ga = gd[np.arange(B), A]
            Gs[t-Hd] = (gd - ga[:,None]).sum(0)
            pis[t-Hd] = B*(mu - p@mu)
        cum += gd.sum(0); n += B
    xi = Gs - pis
    Gbar = Gs.mean(0); sd = Gs.std(0)
    xibar = xi.mean(0)
    # test with the xi-part only (allowance replaced by the exact drift): CLT check
    rej_xi += (math.sqrt(T)*xibar > z*sd)
    cross.append(((xi*(pis-pis.mean(0))).mean(0)*2, (pis.var(0)), (xi**2).mean(0)))
print("rejection freq of sqrt(T) xibar > z*lag0 sd(G):", rej_xi/reps)
cr = np.array(cross)
print("mean 2/T sum xi(pi-pibar):", cr[:,0].mean(0), " mean var(pi):", cr[:,1].mean(0), " mean (1/T)sum xi^2:", cr[:,2].mean(0))
