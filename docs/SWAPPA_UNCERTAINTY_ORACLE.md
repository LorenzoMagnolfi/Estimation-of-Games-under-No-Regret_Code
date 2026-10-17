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

### Feasible Upgrade witness and 120-cell refinement

`optimize_upgrade_witness.py` jointly adjusts all 897 persistent costs using total-profit derivatives. Analytical gradients agree with central differences to 3.71e-10 on the tested coordinates. The optimization used a local integer-price menu and stopped at its 250-iteration limit; no optimum is claimed. Crucially, the resulting costs were then audited against the ENTIRE base menu and its state-aware maximizer in `oracle_witness.py --cost-file ...`. The fully normalized log wealth is 23.6243 against 26.3945, so this is a valid fitted-demand event witness. It also passes the standalone max-one and plus-one classes. Sale-weighted cost is 118.153 and gross margin 23.563. These remain witness values, not point estimates. The cost range is [85.062,264.576]; at rho=.746 the largest absolute residual from entry reference cost is 34.291, so the witness belongs to the delta=35 sensitivity class.

The unrestricted 120-cell Upgrade outer program with the pair-inclusive demand band gives continuation-sale costs [44.869,133.455], gross margins [7.543,96.129], at mean continuation-sale price 140.999. All-sale costs are [35.863,135.686], margins [6.029,105.852]. D=10 remains nearly uninformative. The bounds and witness now refer to the same fully normalized policy class, although the nuisance-band outer program remains conservative. This is progress toward a defensible application, not a final narrow marginal-cost interval.

A local delta=35 canary with rho in [.74,.76], 30 cells, yields D=1 continuation-sale costs [91.771,131.335], against prior-only [87.945,140.200]. D=2 gives [89.906,133.658]. These are conditional-box results, not the result with unknown rho. A full rho profile is necessary. The prior alone therefore explains part, but not all, of the conditional-box precision.

`export_normalized_bundle.py` freezes the Upgrade design and all demand-band arrays. Running the inference path from this bundle reproduces all 24 local canary cost endpoints exactly. The OSC numerical runtime at /users/PAS3582/magnolfi1/envs/pc3/bin/python is used read-only: Python 3.12.14, numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, threadpoolctl 3.6.0, and joblib 1.5.3. Numerical-package versions match the local runtime. No package in the other campaign's environment is modified.


### Asking prices, seller heterogeneity, and selection allowance

The governing price-unit correction is in SWAPPA_PRICE_UNITS.md. The new ask-grid version keeps the demand covariates in displayed-price units and uses unrounded pre-tax receipts in profits. All five audited major sellers satisfy the maintained ceiling-grid rule. A direct full-grid audit of the common-demand cost witness gives normalized log wealth 23.0182 against 26.3609. Continuation-sale effective cost is 117.169 and corresponding margin 23.523, at mean receipt 140.693. These are feasible-event witness values, not point estimates. This vector fits the delta=35 sensitivity class at rho=.746.

continuation_seller_sensitivity.py preserves the initial-price controls and adds separate own-price slopes for five major sellers. In the disjoint-listing holdout, Upgrade's continuation slope changes from -.054378 to -.054322 per dollar; log loss changes from .296413 to .296435. This is predictive stability under a specific relaxation, not evidence of causal exogeneity. The original common-demand witness fails under the full seller-specific fit. A separate full-grid joint-cost optimization finds a passing witness: log wealth 16.4062, continuation cost 118.195, and margin 22.498. Independent full-grid evaluation reproduces the optimized objective. Optimization stopped at 250 iterations, so no minimum is claimed. Its smallest reference residual band is 34.9917 at rho=.747775; the delta=35 class contains it. Source: upgrade_askgrid_optimized_full_hetero_costs.csv and ask_grid_witness_continuation_seller_full_1_supplied.json.

The unrestricted ask-grid outer bounds with 120 continuous cost cells and the original single-seller band are continuation-sale costs [45.900,133.225], margins [7.468,94.793]. At D=2 the margin floor is 4.593, at D=5 .920, and at D=10 only the prior floor .088 remains. Thus feedback-clock assumptions still materially affect identification.

To allow selection of a seller from the twenty seller categories examined in the demand analysis, demand_uniform_band.py now supports a family-size allocation. For family size twenty, 5,000 Gaussian draws give critical value 5.693366 at band alpha .002 and Monte Carlo allowance .00025, using the maximum order statistic. The learner allowance is separately reduced to .00025. Under the maintained common-demand model, this permits a union bound across twenty seller analyses without requiring independence. It does not validate choosing a favorable economic specification, a new seller outside that family, or a causal model after looking at outcomes. Common slopes, the Gaussian approximation, and the stated counterfactual support remain assumptions.

With those twenty-seller allowances, unrestricted ask-grid continuation costs are [29.134,134.879], margins [5.813,111.559], for D=1. For D=2 the margin floor is 3.076; for D>=5 it returns to the prior floor .088. The smaller five-seller allocation gives costs [36.565,134.162] and margin floor 6.530. These refinements preserve a positive one-learner markup floor but do not deliver a narrow marginal-cost interval. The structured cost-reference profiles and their prior-only bounds still need to be compared under the corrected grid and the selection allowance.

The observed clock remains a substantive limitation: Upgrade has 6,282 continuation rows on 158 dates, only 148 within-listing price changes, and changes on 27 dates. Median concurrent inventory is 46.5 listings. An unchanged quote can be a deliberate pricing choice, but the snapshots do not establish one independent learner update per listing-day. D sensitivity is required; fitting a witness does not establish that clock. The draft's discussion after Theorem CR explicitly permits direct high-probability joint events but also warns that a narrower behavioral law changes the model. The present state-aware, expected-payoff pricing family is such a narrower law, not a consequence of ANR alone.

Next computation: finish the immutable OSC baseline profile; separately freeze the corrected-grid, selection-adjusted version before any new cluster batch. A bounded local search of the predeclared delta=20 class is in askgrid_structured_delta20.log. Do not infer global feasibility from an accepted outer LP box, or global bounds from an incomplete rho profile. No manuscript or referee response has been edited.

The delta=20 full-grid cost search has completed. A witness at rho=.75 passes, with log wealth 25.5410 against 26.3609; the independent full-menu audit reproduces it. Other tried rho values .60, .65, .70, and .80 failed these candidate searches, which does not certify their exclusion. All searches stopped at 250 iterations. Canonical witness: askgrid_structured_delta20_rho0.75_costs.csv; independent audit: ask_grid_witness_common_delta20_rho075.json, including the cost-file hash. No global cost bound follows from this witness alone.


### First complete global baseline cost-reference profile

All sixty rho boxes for delta=20 in the original OSC campaign are accounted for, with no solver failures. Under the earlier displayed-price payoff, single-seller demand band, and persistent |c_l-rho*Ref_l,first|<=20 restriction, the continuation-sale bounds are:

| Learners D | Cost lower | Cost upper | Margin lower | Margin upper |
|---|---:|---:|---:|---:|
| 1 | 82.659 | 128.966 | 12.032 | 58.339 |
| 2 | 73.311 | 131.062 | 9.936 | 67.688 |
| 5 | 54.826 | 134.448 | 6.550 | 86.172 |
| 10 | 33.022 | 137.151 | 3.847 | 107.976 |

The prior alone gives cost [0,140.332] and margin floor .667, at mean displayed price 140.999. Thus the precision is not a direct consequence of a narrow prior mean-cost interval, although the common-reference restriction is substantively important. These are global rho-profile outer bounds, unlike the earlier conditional canary. Source: osc_campaign/swappa_1e112e66_20261016/global_profile_bounds.csv, producing commit 1e112e66a354b0d6bde6124d823dd113ca11c9f2. These baseline values still require the corrected price-grid and twenty-seller uncertainty treatment. The corrected array 15597781 is running; its results must be reported separately. No final estimate is declared from this baseline.
