# Two-seller lab probes: when does the equilibrium benchmark go wrong? (9 to 10 November 2026)

Seconds-scale feasibility probes, one seed, 60,000 periods, written after the Swappa horse race
(`docs/swappa-horse-race-equilibrium-benchmark-2026-11-09.md`) found the equilibrium benchmark right under the
stationary assumption. Not certified, not part of any exhibit. Parked by Lorenzo on 10 November ("interesting; we'll
go back to it if necessary").

Run from this folder (the scripts `exec` each other by relative file name), with the research venv:

```
python probe.py 60000                     # paper's logit game; i.i.d. vs persistent costs; informed learners
python probe_edgeworth.py 60000 0.7       # capacity-constrained game (capacity 0.7), decreasing-step Hedge
python probe_adaptive.py 60000 0.7 0.5    # same game, recency-weighted Hedge (discount 0.999), step 0.5
python regret_check.py                    # realized regret by seller and cost type for the three learners
```

Game: two sellers, prices 4..8, costs {0, 1.5, 3, 4.5, 6} with discretized N(3, 1) (the lab's prior). `probe.py`
uses the paper's logit demand (alpha = -1/3); the others use a homogeneous good with unit demand and capacity k per
seller (low price sells k, high price the residual 1 - k, tie 1/2). Benchmark: two-step equilibrium estimator
(beliefs = empirical marginal of rival prices, each cost type best-responds, (mu, sigma) chosen to match own-price
marginals). No-regret side: minimal uniform slack making (mu, sigma) BCCE-feasible (LP), and realized regret.

Results (10 November, office machine):

| Setting | corr(p1, p2) | Equilibrium estimate (truth 3, 1) | Realized regret |
|---|---:|---|---|
| Logit game, i.i.d. costs (the paper's lab) | -0.001 | (2.90, 1.10), learners not converged at 60k | n/a |
| Logit game, persistent costs, learners see the rival's last price | +0.006 | (2.87, 1.20) | n/a |
| Capacity 0.7, decreasing-step Hedge, second half | +0.002 | n/a | +1.2 to +1.4% of profit |
| Capacity 0.7, recency-weighted Hedge (step 0.5 / 2.0), second half | +0.15 / +0.17 | (2.62, 1.60) / (2.60, 1.65), exact fit | -0.45 to -0.55% of profit |

The saved 4M-period lab runs (`AuctionNet/results_unity/lab_v05/artifacts/v05_traj_4M.mat`, outside Dropbox) give
independent prices (total variation from the product of marginals 0.0007) and a type-by-type price map: the logit
game converges to its equilibrium. The benchmark goes wrong only with both strategic tension (capacity) and learners
that keep adapting; the cycling learners have negative realized regret, so the cycles satisfy no regret, and the
truth is BCCE-feasible with zero slack (9 of 15 grid candidates also are). Before any use: many seeds, longer runs,
certified sets (the recency-weighted learner is outside family F; the any-window layer is its certificate),
inference on the equilibrium estimate, and the paper's two-price version of the game.
