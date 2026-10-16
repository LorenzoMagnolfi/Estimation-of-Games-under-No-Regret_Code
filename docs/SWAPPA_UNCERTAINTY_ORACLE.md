# Swappa: first-stage uncertainty and state-aware policies

Internal research checkpoint, 2026-10-16. This is an exploratory code-repository note. No manuscript or response claim is authorized by these results. The research goal remains open.

## Evidence status and returns

The earlier narrow Upgrade bounds condition on estimated demand. A simultaneous Gaussian demand band substantially widens those bounds. They are not a finished confidence interval. The main new tools are correlated demand bounds, continuous-cost outer relaxations, and an explicit stronger class of learning rules containing a state-aware price policy. These tools separate what follows from the learning assumption from what is supplied by cost restrictions.

Return objects are empirical cost and gross posted-price margin bounds, feasible-event diagnostics, first-stage covariance/band diagnostics, and source-preserving tests. Costs remain inventory/replacement values in posted-price units. Procurement costs, fees, and the interpretation of sale timestamps remain unresolved.

## Continuation sample and demand

`continuation_demand.py` estimates the multiproduct logit with initial-price and initial-reference controls. Counterfactuals hold these initial conditions fixed only after the first observed exposure. First-exposure rows have zero learning-comparison weight; their inventory still affects total firm profits. This does not justify an entry-price deviation holding its own initial-price control fixed.

Upgrade has 7,179 terminal-status rows, 897 listings, and 6,282 continuation rows. There are 837 sale-labelled listings, including 669 continuation sales. SaveGadget has 8,026 rows and 511 sale-labelled listings, including 433 continuation sales. Its single-owned-listing contexts have 2,536 rows, 326 listings, 2,285 continuation rows, and 225 sales, including 179 continuation sales. Targets restricted to continuation sales avoid assigning an inferred continuation-pricing cost to entry-only sales. All-sale targets are retained for comparison.

Terminal status remains a maintained observation model: a known-sold listing is labelled at its final observed exposure. The page's Updated field is not a verified transaction timestamp. Current prices, rivals' quotes, inventory, and the expected-profit information used by the learning rule must be available before the focal pricing choice. Daily snapshots do not establish that clock.

The initial-controlled model's two-way clustered sandwich covariance is an asymptotic diagnostic. It does not establish conditional price exogeneity or finite-sample coverage. `demand_uniform_band.py` simulates a Gaussian approximation and analytically maximizes standardized logit errors over continuous counterfactual prices. It covers absolute logits, paired price contrasts, slopes, and the peer-price coefficient. With `--pairs`, it also covers every within-owned-market baseline-logit difference. Initial-price controls help address quality heterogeneity but do not resolve causal identification.

The original Upgrade band critical value is 4.315540723. Adding 79,140 peer-pair directions gives 4.360860472. SaveGadget's pair-inclusive band on [0,1500] gives 4.588027792. Each uses 1,000 Gaussian draws, band tail .04, and a .005 Monte Carlo quantile allowance, with order statistic 976. Combining those allowances with alpha_R=.005 gives an approximate .95 joint statement, conditional on the Gaussian/cluster/design approximation. It is not a finite-sample demand theorem, and the separate seller bands are not a simultaneous cross-seller confidence statement. Adding a further consistency event requires allocating another error allowance; it cannot be silently appended at no cost.

## State-aware policy and its information assumption

For a seller with fixed current inventory, let U_i(c_i) be the maximum own expected profit over prices in [c_i,P]. Let q_j(a) denote peer j's sale probability when focal i is priced at a. A valid total-profit range certificate is

C_i(c) = U_i(c_i) + sum_{j != i} (p_j-c_j) [q_j(P)-q_j(c_i)].

All margins are nonnegative and the peer-price effect is nonnegative. The domain excludes no-offer and holds inventory fixed. Product markets are defined as in the demand model; unmodelled substitution across these markets remains a substantive qualification.

The base menu contains integer prices 0,...,P clipped at own cost and eleven cost-plus policies. Add one policy which chooses the best total-inventory expected-profit price in that menu, using current state and costs. This is a fixed state-dependent function, not a comparator selected after observing the realized focal action. It is available before the action only under the stated information assumption. Since observed prices are integer-valued and at least cost, the best-policy gain r*_t is nonnegative and dominates every base-policy gain. This is a stronger algorithm class than arbitrary external no-regret learning. Its empirical plausibility requires an explicit model-based repricing interpretation.

Post hoc local price comparisons can lower-bound r*_t. They must not be described as valid action-dependent external-regret comparators on their own. A proposed cut below cost is still a lower bound for its cost-clipped menu action: clipping removes negative own profit and increases nonnegative peer profits.

## Learning certificates and coverage logic

The earlier confidence-rated cap/rate experts use weights proportional to their current learning coefficient times wealth and update w by w(1+eta*r). Sharing injects prior weight at rate 1/(t+1). Total wealth is a nonnegative martingale under the learner's randomized action, because the weighted expected excess profits sum to zero. The resulting any-window log-wealth inequality includes log(Tmax), not a reset at the first observed day. Here Tmax=1,000,000 and alpha_R=.005.

For the standalone max(1,C) certificate, the strongest state-aware policy restriction is

sum_t log(1+.99*r*_t/max(1,C_t)) <= log(4*A)+log(Tmax)+log(1/alpha_R).

The analogous plus-one learner replaces max(1,C) by 1+C. These are different learning classes. The plus-one certificate has an explicit dollar scale; its convenient fit must not be used to claim that the other class passed. Higher cap indices and smaller rates give smaller left sides and larger prior penalties, so the nonnegative state-aware policy at the smallest cap and largest rate dominates those constraints.

A profit-normalized extension removes that dollar floor. Allocate half the prior to the old cap/rate family and half to normalized policy/rate copies with current coefficient eta_jt=.99*2^(-j)/C_t. Use policy prior 1/A and rate prior 1/[(j+1)(j+2)]. Choose actions proportional to the sum of current eta_jt times wealth of recommending copies. The same martingale identity applies to predictable, time-varying coefficients. At C_t=0 all gains are zero; normalized copies can be inactive. The normalized oracle at j=0 has prior 1/(4A), giving

sum_t log(1+.99*r*_t/C_t) <= log(4*A)+log(Tmax)+log(1/alpha_R).

The original cap/rate copies remain in the mixture, with their prior penalties increased by log(2). On bounded games they preserve unweighted asymptotic no-regret: a sufficiently large cap is always active, and the countable fixed-rate copies give the usual vanishing average-regret argument. Normalized regret alone would not establish unweighted ANR; retaining the old family matters. For D interleaved learners the necessary aggregate bound is D times [log(4A)+log(Tmax)+log(D/alpha_R)]. D is a substantive feedback/clock sensitivity, not an estimated nuisance parameter.

Single-inventory comparisons use normalized copies that sleep outside n_owned=1 states. The legacy family may remain active elsewhere. Eligibility is determined before the focal price choice. Initial-price controls additionally require sleeping at first exposure. These are contextual learning restrictions and must not be presented as the original unconditional application without explanation.

Coverage uses the new method's witness principle: on the joint regret and demand events, the true cost vector remains in the projected feasible set. Candidate-dependent normalization is an explicit extension of the scalar-envelope implementation, not a verbatim invocation of its theorem. No uniform probability statement over every candidate cost vector is required for this true-witness argument, but all deterministic numerical enclosures must contain that witness.

## Computation and safeguards

`oracle_witness.py` initializes persistent costs from average profit derivatives and checks the full finite menu. First-order conditions only initialize a candidate. Passing the log-wealth event is not proof of the actual seller algorithm or causal demand.

`oracle_coupled_lp.py` retains a common cost for every peer and every comparison. The own-profit maximum is convex in cost; chords between upper-bounded endpoints give a valid affine upper range certificate. Minimum nuisance-adjusted fixed-price gains are concave in cost; chords between lower-bounded endpoints give affine lower gains. A McCormick relaxation of normalized gains gives an outer LP. Eliminating gain auxiliaries and generating violated comparisons makes the small pilots tractable. A successfully solved subset LP is still an outer bound even when omitted comparisons remain violated; it is not a full-constraint optimum or a feasible witness. Time-limited solves with no successful subset give no bound. The broad cost-domain relaxation is loose.

`normalized_demand_bounds.py` divides both total-profit gains and their range by the SAME focal sale probability. It preserves baseline/counterfactual probability ratios and within-market relative logits. For peers, the normalized probability difference is a rational function of exp(z), with at most one stationary point per edge of its logit rectangle and no interior stationary point. Checking edge extrema gives deterministic enclosures. The own-profit range uses a Lambert-W cap divided by the same baseline probability, with its stationary point included. Cost grids carry a uniform Lipschitz allowance; LP cells are continuous intervals and endpoints bound objectives. No cost distribution or discretized true cost is imposed by `normalized_nonparam.py`.

`normalized_cost_profile.py` instead imposes c_l=rho*entry_reference_l+e_l with persistent |e_l|<=delta. It is a sensitivity model, not a default estimate. Wide cost boxes can make this outer screen much looser than a cell-assignment LP. Narrow delta values create a substantial markup floor by themselves.

## Findings retained at this checkpoint

- The original robust Upgrade delta=5 profile under ordinary policy comparisons gives observed-sale costs roughly [5.8,127.7]; adding the state-aware policy gives [77.4,127.7]. The upper-cost endpoint largely comes from the prior restriction. These are wider than fitted-demand bounds and cannot be advertised as the earlier narrow result surviving uncertainty.
- The plus-one Upgrade class has an unrestricted fitted-demand event witness with sale-weighted cost 117.098 and gross margin 24.618. Its log wealth is 20.478 against threshold 26.395. The same initialization fails max(1,C), with 31.938, and full normalization, with 36.043. Failure of one initialization does not reject the class. The rho=.75,delta=25 projected initialization similarly passes plus-one but fails full normalization. Narrower residual-band initializations fail the oracle check.
- The correlated, fully normalized Upgrade nonparametric outer LP with 30 cells gives observed-sale costs [31.059,138.087] and gross margins [3.628,110.657] for D=1. D=10 is nearly uninformative. The delta=25 coarse rho-box screen is even wider; it is an approximation failure, not evidence that low costs are feasible in the joint model.
- SaveGadget single-inventory contexts, P=1500, D=1, 120 cost cells: all-sale costs [157.319,468.418], margins [16.613,327.712]; continuation-sale costs [197.748,477.610], margins [20.882,300.744]. The continuation mean price is 498.492. Prior-only continuation margins start at .196. These are preliminary outer bounds; the 480-cell refinement and independent witness check must be inspected before citing final values. D=10 remains weak.

These findings provide a potentially informative markup lower bound, not yet a credible final marginal-cost interval. Separate seller fits, contexts, clock assumptions, and statistical approximations must remain visible. Do not select a seller or context solely because its interval is narrow.

## Next gates

1. Inspect the SaveGadget 480-cell refinement and independent witness audit. Continue with other professional sellers under the same stated model; obtain joint coverage before making cross-seller claims.
2. Tighten numerical outer bounds before narrowing economic assumptions. Common peer costs and common demand coefficients still matter.
3. Investigate outcome-consistency restrictions with predictable persistent costs. A Bernoulli martingale restriction can limit assigning high costs only to realized sales. With independent pointwise probability bands, however, the allowance is broad: Upgrade qhat sums to 836.96, while qlo/qhi sum to 448.59/1475.00. A crude delta=25 centered-cost moment has roughly a $19.9 per-sale allowance. Shared nuisance uncertainty must be handled to make this useful. No such restriction has yet been imposed.
4. Resolve causal price variation, sale timing, predecision information, feedback opportunities, procurement versus inventory value, and platform fees. Do not substitute the feasible witness for an estimate.

## Resources and provenance

All outputs are under ignored `output/swappa_rescue_20261016/`. Twenty-nine mathematical/data tests passed after the normalized-bound implementation; rerun after further substantive changes. Source snapshots and local commits preserve code; no remote push or manuscript change has occurred.

OSC budget preflight at 2026-10-16 22:45:51 UTC returned ACTIVE and $960. Existing Cardinal commitments in another research campaign were approximately $152 at the CPU-only reference rate, with future driver waves not yet fully reconciled. No Swappa job was submitted. Do not infer available headroom from $960 minus currently queued jobs alone. The local pilots use two threads and finish in seconds to about a minute; larger campaigns require the OSC budget and transfer checks. The temporary keep-awake helper remains scheduled to expire October 19 20:29:23 UTC.

### Refinement and feasibility update

The SaveGadget 480-cell single-inventory run completed. D=1 continuation-sale costs are [305.422,452.186], with gross margins [46.305,193.069]; all-sale costs are [242.980,448.193]. The finer cells materially change endpoints, so 120 cells were not an adequate numerical convergence check. More importantly, no feasible fitted-demand witness has been found: the FOC initialization has normalized log wealth 187.841 versus 27.822, and direct per-listing minimization against the stronger continuous-price oracle yields 170.861. The ten largest contributors account for 48% of that loss; their prices often decline by $100-$200 during a listing. Failure of a candidate is not a formal model rejection, but this large gap prevents promoting the outer bound as a successful application.

A reference-depreciation extension sets c_l,t=c_l*[(1-loading)+loading*Ref_l,t/Ref_l,first], keeping a persistent listing component and using all observed prices to enforce no sales below cost. `--reference-loading` is now available in the witness and normalized nonparametric scripts. The SaveGadget FOC initialization at loading=1 still fails, with normalized wealth 197.974. Do not state that reference-price depreciation fixes the problem. `single_inventory_witness.py` searches each listing's base cost on a grid followed by bounded local minimization and audits a continuous-price oracle; it supplies feasible candidates, not certified global minima. Its targets and losses remain diagnostic.

Upgrade remains worth pursuing because its price paths are much more stable and its fully normalized witness gap is much smaller (36.043 versus 26.395). The next computational check is a finer nonparametric cost-cell run with continuation-sale targets and a direct joint-cost witness search. The original 30-cell result remains broad and cannot be improved in prose without the calculation.
