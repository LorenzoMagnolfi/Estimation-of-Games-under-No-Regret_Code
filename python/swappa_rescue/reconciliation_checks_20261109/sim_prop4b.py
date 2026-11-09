import numpy as np, math, sys
from scipy.stats import norm
def run(mu, T, Hd, learn=True, reps=2000, B=20, sc=0.5, si=0.5, seed=1, dH=0.025):
    rng = np.random.default_rng(seed)
    mu = np.array(mu); d = len(mu)
    z = norm.ppf(0.95)
    def draw_day(n):
        return mu + sc*rng.uniform(-1,1,(n,1,d)) + si*rng.uniform(-1,1,(n,B,d))
    g = draw_day(20000); K = g.max(-1)-g.min(-1); kappa = math.sqrt((K**2).mean())
    Kbar = 2*(sc+si) + (mu.max()-mu.min())
    a0 = int(np.argmax(mu))
    Y = g - g[...,[a0]]; vday = Y.sum(1).var(0).max(); DEFF = vday/(B*kappa**2)
    rej_xi = np.zeros(d); rej_full = np.zeros(d); rej_any = 0
    for rep in range(reps):
        days = draw_day(Hd+T); cum = np.zeros(d); n = 0
        Gs = np.zeros((T,d)); pis = np.zeros((T,d)); eps = 0.0
        for t in range(Hd+T):
            if n == 0 or not learn: p = np.ones(d)/d
            else:
                s = n+1; eta = math.sqrt(math.log(d)/s)
                w = eta*cum/kappa; w -= w.max(); p = np.exp(w); p /= p.sum()
            gd = days[t]; A = rng.choice(d, size=B, p=p)
            if t >= Hd:
                ga = gd[np.arange(B), A]; Gs[t-Hd] = (gd - ga[:,None]).sum(0); pis[t-Hd] = B*(mu - p@mu)
                x = math.log((d-1)*T/dH)
                bt = kappa*math.sqrt(math.log(d))*math.sqrt(n+1)/n + kappa*math.sqrt(DEFF)*math.sqrt(2*x/n) + (2*B*Kbar/3)*x/n
                eps += B*bt/T
            cum += gd.sum(0); n += B
        xi = Gs - pis; sd = Gs.std(0)
        r1 = math.sqrt(T)*xi.mean(0) > z*sd
        r2 = Gs.mean(0) - eps > z*sd/math.sqrt(T)
        rej_xi += r1; rej_full += r2
    return kappa, DEFF, eps, rej_xi/reps, rej_full/reps
cases = [("gaps, learning, T=164", [0,-.03,-.06], 164, 40, True),
         ("gaps, frozen uniform p, T=164", [0,-.03,-.06], 164, 40, False),
         ("tie a0=a1, learning, T=164", [0,0,-.06], 164, 40, True),
         ("gaps, learning, T=656", [0,-.03,-.06], 656, 40, True),
         ("tie, learning, T=656", [0,0,-.06], 656, 40, True)]
for name, mu, T, Hd, learn in cases:
    k, D, e, rx, rf = run(mu, T, Hd, learn)
    print(f"{name}: kappa={k:.3f} DEFF={D:.2f} eps_bar(last rep)={e:.2f} | xi-only lag0 test rej={rx} | full test rej={rf}")
    sys.stdout.flush()
