# Swappa daily pricing: continuous endpoints and robustness

Internal research record, October 17, 2026 UTC. The rescue goal remains active. No manuscript, response letter, or remote publication is changed by this note.

## Status and data

The current data mode is `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. All artifacts below are relative to `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/`. Read `SWAPPA_UNCERTAINTY_ORACLE.md` for the full-source correction and the learning construction. Earlier narrow estimates based on the prematurely filtered data are superseded.

The primary group is Upgrade, iPhone SE 2020, 64GB, Good: 516 listings, 3,831 observed rows, 407 continuation sales, and 161 days with continuation pricing. The empirical sale-weighted mean pre-tax receipt is $130.258305. Initial prices are held fixed in counterfactuals; entry-item profit spillovers remain in joint profit. The target is replacement/opportunity cost in receipt-equivalent units. It is not a verified invoice cost.

## Continuous cost and endpoint inference

`daily_cost_model.py` constructs the best common integer-ask advisor, changing every continuation price jointly within the SKU. A common cost c, or a cost indexed to the public SKU reference quote, enters each candidate profit affinely. The indexed schedule is normalized by the first observed quote, not by a future weighted average. The observed action can outperform the common-price advisor, so regret is allowed to be negative.

The daily range certificate is the sum of maximum continuation-item profits at the highest feasible peer prices and entry-item profit ranges. It decreases coordinatewise in costs. `interval_bounds` combines affine regret bounds and monotone range bounds to enclose wealth over whole cost intervals. `certify` partitions the entire prior support; only tiny boundary cells remain unresolved. The calculation therefore does not exclude unvisited costs by interpolation. Root certificates establish strict derivative signs throughout both boundary brackets and verify that the active advisor remains unchanged. Distinct own-profit optimizer ties are rejected. Roots must reproduce the wealth threshold, so a jump cannot be mistaken for a zero.

At regular boundaries, the endpoint gradient is -W_theta/W_c. `daily_endpoint_inference.py` perturbs the entire shared demand vector in covariance-factor directions, retaining all coefficient correlations. It checks that the covariance's recorded theta matches the loaded fit. The critical value is 3.93460586, allocating .045 first-stage error across 20 sellers, 27 product groups, and two one-sided endpoints. Learning-event failure receives .005. These are **asymptotic conditional outer intervals**, not finite-sample demand confidence intervals. They require correct structural demand, a valid clustered Gaussian approximation, regular boundaries, and the specified learner. A selection allowance does not validate a chosen economic model.

The default history bound is 1,500 daily rounds. This covers the product's existence through the sample only if its learner starts without inherited weights. Longer histories are reported below. The expected-payoff shared-prior learner is stronger than unrestricted ANR. Its information set, normalization, and decision unit are explicit assumptions, not facts established by observing prices.

## Corrected common-cost results

The files with `peerfix` in their names supersede the earlier `certified` versions. A regression test discovered that the shared peer-average helper allocated an integer output when passed integer changes; it truncated fractional peer averages in a range intercept. The helper now always returns floating-point averages. Recomputing changes the main endpoints by less than two cents, but the earlier range certificate must not be used. The independent 200-draw diagnostic was regenerated as well.

| Model/history | Fitted continuous cost set | Asymptotic cost outer interval | Asymptotic gross-margin interval |
|---|---:|---:|---:|
| Common cost, T=1,500 | [65.811,114.710] | [50.022,118.598] | [11.661,80.236] |
| Public-reference index, T=1,500 | [65.266,112.948] | [49.758,116.329] | [13.930,80.500] |
| Common cost, T=5,000 | [64.361,115.109] | [47.912,118.909] | [11.349,82.346] |
| Common cost, T=10,000 | [63.500,115.333] | [46.531,119.086] | [11.173,83.727] |
| Common cost, T=1,000,000 | [57.486,116.738] | [38.440,120.210] | [10.049,91.818] |

The common-cost prior is [0,123.6]. Its margin floor is already $6.658305 before using learning. Do not attribute that portion of a reported lower margin to the method.

Increasing the learner's range normalizer weakens the restriction. At T=1,500, a factor 1.25 gives cost [41.604,119.772] after uncertainty; factor 2 gives [15.109,122.668]. Their fitted sets are [59.939,116.199] and [40.264,119.769]. The main result therefore also uses a substantive normalization choice. Files: `daily_scale1.25_T1500.json` and `daily_scale2_T1500.json`.

## Persistent cost heterogeneity without a cost distribution

`daily_heterogeneous_cost.py` allows each persistent listing cost c_i to lie within delta of the empirical continuation-sale-weighted mean mu. Costs are nonnegative and no larger than any observed receipt for that listing. This is an explicit restriction, not something inferred from the descriptive condition label. The target remains mu; there is no extra +/-delta adjustment to the reported mean bounds.

For a trial mean, let l_i=max(0,mu-delta), h_i=min(listing receipt cap,mu+delta). For a common-price proposal, q'_i-q_i determines the adverse cost endpoint: h_i when the probability change is nonnegative, l_i otherwise. Sum these lower gains, including entry-item spillovers, and maximize over common prices at least as large as every active h_i. Every such proposal is feasible for all costs in the box. Relaxing the mean constraint and choosing costs separately across proposals only lowers regret, so the calculation is conservative.

The range certificate is coordinatewise decreasing. For a nonnegative lower gain use C(l); for a negative gain use C(h), then apply the known normalized lower limit -1. This yields a necessary wealth screen for every compatible true cost vector. Whole-mean-interval bounds use the worst gain endpoints, the smaller guaranteed policy menu, and monotone range bounds. Accepted cells and roots describe an OUTER set; they are not evidence that a feasible latent cost vector exists at every accepted mean.

Heterogeneous root certificates additionally exclude cost-box and menu kinks from the boundary brackets, check nonstructural probability-change sign ties, and verify no clipping kink. Analytic range-derivative enclosures agree with numerical derivatives. The zero-delta implementation reproduces the separate scalar model. Random persistent-cost witnesses verify the direction of the outer wealth inequality.

| Delta | Fitted mean-cost outer set | Asymptotic cost outer interval | Asymptotic gross-margin interval |
|---|---:|---:|---:|
| 2 | [61.529,116.733] | [45.121,120.621] | [9.637,85.137] |
| 5 | [54.847,119.767] | [38.040,123.652] | [6.607,92.218] |
| 10 | [43.316,124.800] | [26.203,128.666] | [1.593,104.056] |
| 20 | [19.726,130.127] | Not computed | Not computed |

Delta=20 reaches the prior upper boundary and fails the two-interior-root condition. No regular two-endpoint confidence interval is reported for it. The deltas are sensitivity assumptions, not estimated cost dispersions or a selected favorable specification.

## Demand pooling and serial dependence matter

The seller-specific first stage adds five focal seller-own-price interactions and retains the same initial-price, quality, calendar, and peer controls. On held-out Upgrade continuation listings, log loss changes from .264639 to .264004 and AUC from .731597 to .735072. This modest predictive improvement is not causal validation. Direct reconstruction of the full counterfactual design verifies the generalized daily model, including the seller interaction.

| Seller-specific demand | Fitted mean-cost outer set | Asymptotic cost outer interval |
|---|---:|---:|
| Common SKU cost | [75.101,116.967] | [49.757,123.491] |
| Delta=5 | [64.778,122.008] | [37.980,128.464] |
| Delta=10 | [53.497,127.023] | [25.422,130.127] |

The corresponding gross-margin floors are $6.767, $1.794, and $.132. The first two are only slightly above their respective prior floors; the last is entirely the prior floor. Therefore the strongest lower-margin claim depends substantially on pooling price slopes. Preserve this unfavorable sensitivity.

A separate covariance sensitivity clusters by listing and calendar week, subtracting listing-week intersection scores rather than individual-row scores. There are 24 weekly clusters, so the Gaussian approximation is particularly tentative. Its PSD projection clips 35 negative eigenvalues (minimum -517.46); this adds a positive semidefinite correction to the estimated meat. Under common demand, weekly-cluster cost intervals are [48.100,118.856] for delta=0, [36.074,123.895] for delta=5, and [24.388,128.900] for delta=10. These do not combine seller-specific demand and weekly clustering. Files: `daily_week_d*_T1500.json`.

## Referee relevance, not completed responses

- R1.1.d and R1.4.c: sale probabilities now depend on the lowest competing price, the price gap on each side of that minimum, competitor counts and means, and owned peer prices. These variables are rebuilt before reference-price filtering. Remaining issue: causal exogeneity and demand specification.
- R1.4.a: the actual integer asking-price grid replaces arbitrary coarse price bins, and receipt/display rounding is separated.
- R1.4.b: cost support is continuous and certified; bounded persistent heterogeneity replaces a Normal cost distribution. The size of the cost restriction remains substantive.
- R1.4.e and R1.3: repeated listings are persistent units, the learner's decision unit is explicit, and first-stage scores allow repeated-listing and common-time dependence. Observation gaps and sale timestamps remain limitations.
- R1.4.f and R1.4.g: the behavioral test concerns the focal seller. Other sellers can supply first-stage demand observations without assuming that every pooled seller follows the focal learning rule. Pooled demand slope restrictions are tested separately.
- R1.2, R1.4.d, R1.5.c, R2.1, R2.2, and R2.3: distinguish expected-payoff information, daily joint advice, payoff normalization, and unknown prehistory from ANR alone. This is a specified extension of the newest method, not an automatic application of the old scalar envelope.

No ledger issue has been marked resolved, and no response claims an unmade manuscript change.

## Validation and next gate

The stable source suite passed 58 tests in `tests_daily_endpoint_final.log`. This includes whole-cost interval enclosures, derivative certificates, integer-input peer means, scalar/heterogeneous equivalence at delta zero, persistent-cost witness containment, and seller-specific counterfactual reconstruction. The earlier full-source and two-phone joint-oracle tests also pass.

The new `daily_multisku_profile.py` is an exploratory next step: one learner processes all daily SKU rounds, with a separate cost level for every SKU. Its continuous-cell LP relaxes the aggregate wealth inequality; fractional cost mixtures are outer bounds only. Every observed SKU in the fixed 27-category universe is included, so favorable SKUs are not chosen after seeing results. This is a stronger shared-learning restriction than separate SKU learners. First-stage inference and the LP relaxation still require validation before this direction can support a final application.

Keep the goal active. The numerical work has produced informative conditional intervals, but the main lower-margin result is sensitive to demand pooling and cost heterogeneity. The next priority is to test the shared learner across products and propagate demand uncertainty coherently, while retaining these adverse results.


## Shared learner across product groups: current promising extension

The full Upgrade target inventory contains sixteen observed SKUs within the fixed 27-category universe, 804 continuation sales, 1,031 daily SKU rounds, and mean receipt $144.000149. `daily_multisku_profile.py` treats these as rounds of one expected-payoff pricing learner, with a separate persistent cost level for each SKU. Prices of all continuation listings within a round still change jointly, entry spillovers remain, and cross-SKU demand effects are excluded. The ordering of SKU rounds must be part of the learning model; simultaneous daily prices do not by themselves establish this sequential shared-learning restriction.

A single predeclared state-aware advisor selects the best common ask for the current SKU. The cost vector need not obey a common reference-price ratio, and no Normal distribution is imposed. The aggregate wealth statistic is the sum of SKU wealth terms. At one million lifetime SKU rounds, its threshold is log(4)+log(1,000,000)+log(20/.005)=23.495855. There is one aggregate advisor event rather than 27 separate SKU events. The first-stage critical value is therefore 3.05508066 for twenty aggregate seller targets and two one-sided endpoints. This comparison assumes the declared twenty-seller target family; it is not an unlimited post-selection guarantee over arbitrary empirical specifications.

For each SKU, the code partitions its continuous mean-cost support and records a lower wealth bound on every cell. A feasible true cost vector picks one cell per SKU and satisfies the sum of these lower bounds <= B. The LP permits fractional cell choices and replaces mean costs by cell endpoints in the objective. This relaxes the true problem, so its extrema are valid outer bounds even when fractional solutions cannot be realized by persistent costs. Finer cells reduce enclosure error but do not automatically remove the convex-mixture relaxation. A grid minimizer with common SKU costs passes the fitted advisor event; this does not prove attainability by a particular learning algorithm.

`daily_multisku_inference.py` differentiates the LP bound itself. It verifies a positive wealth multiplier, positive basic weights, strictly positive nonbasic reduced costs, the wealth equality, and the primal objective. For every active wealth coefficient, `daily_cell_regularity.py` checks advisor gaps, cost-endpoint and gain-sign branches, nonstructural probability-change ties, relevant own-profit maxima, and clipping. At a regular LP optimum, the endpoint derivative is the signed wealth multiplier times the primal-weighted derivative of active lower-wealth coefficients. Whole-vector demand perturbations preserve the clustered covariance. This yields an asymptotic outer interval for an already conservative bound, not a sharp identified set.

| Demand and within-SKU costs | Fitted mean-cost outer interval | Asymptotic cost outer interval | Asymptotic gross-margin interval |
|---|---:|---:|---:|
| Common demand, constant SKU costs | [95.322,119.103] | [85.347,122.559] | [21.442,58.653] |
| Common demand, delta=5 | [83.112,124.948] | [74.136,128.760] | [15.240,69.864] |
| Seller-specific demand, constant SKU costs | [103.810,121.484] | [84.333,126.393] | [17.607,59.667] |
| Seller-specific demand, delta=5 | [90.294,127.900] | [75.147,134.498] | [9.502,68.853] |

The .2-dollar cell profiles are `daily_multisku_common_d0_h02.json`, `daily_multisku_common_d5_h02.json`, `daily_multisku_seller_d0_h02.json`, and `daily_multisku_seller_d5_h02.json`. Their inference files append `_inference`. The common-demand delta=10 fitted outer interval is [72.118,130.284], but its active own-profit tie check failed even after limiting the audit to the denominator branch actually used. No regular Gaussian endpoint interval is reported for that case. Do not bypass the gate.

Two additional tests pass: actual cost-vector witnesses remain inside the joint LP bounds, and all active cell coefficients for the four reported inference models pass their regularity audit. A tenfold gradient-step check for the seller-specific delta=5 result is running in `daily_multisku_seller_d5_h02_h1e4.log`. Further required checks include nonlinear first-stage perturbations, the strength of shared learning/normalization, feasible price-domain sensitivity, and the economic interpretation of bounded replacement-cost dispersion. The result is promising but the goal remains active.


### Independent feasibility and dependence follow-up

The shared-learner common-cost grid vector passes an independent full-joint oracle audit: the certified upper bound on total log wealth is 14.109626, below 23.495855. Its empirical mean cost is 109.584618 and gross margin 34.415532. These are witness values, not point estimates. The oracle allows arbitrary feasible joint price vectors within each SKU, including all owned spillovers, and uses a five-dollar mean-price partition with a valid Lagrange upper bound. Files: `daily_multisku_common_d0_h02_oracle_h5.json` and its hashed cost CSV; script `daily_multisku_oracle_audit.py`.

The tenfold gradient-step replication changes the seller-specific delta=5 lower confidence endpoint by $0.001699 and the upper endpoint by less than 1e-8. The lower gradient's relative norm difference is .000134; the upper difference is 6.46e-9. Numerical resolution is immaterial at cent precision.

Weekly dependence and seller-specific demand can now be combined. The covariance reweighting checks that the alternative covariance lies in the recorded gradient's estimable subspace, with relative projection residuals below 5.3e-8. For common demand and delta=5 the weekly interval is [72.965,129.164]; for seller-specific demand and delta=5 it is [76.251,134.124]. The latter implies gross margins [9.876,67.749]. The day-cluster seller-specific upper cost 134.498 remains slightly more conservative. These are asymptotic sensitivities with only 24 weekly clusters, not exact small-sample coverage results.

`daily_multisku_gaussian_check.py` now prepares a nonlinear diagnostic that re-solves the entire cost LP under coherent Gaussian perturbations. It checks the local envelope approximation, never treats draw quantiles as confidence intervals, and records invalid slope or infeasible draws rather than dropping them. Inspect its progress artifacts before drawing conclusions about finite-sample curvature. The goal remains active pending that diagnostic and the remaining economic scope checks.
