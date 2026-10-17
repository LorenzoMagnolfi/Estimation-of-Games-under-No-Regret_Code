# Swappa daily pricing: continuous endpoints and robustness

Internal research record, October 17, 2026 UTC. The shared-learner analysis and its planned computational validation are complete. No manuscript, response letter, or remote publication is changed by this note.

## Current conclusion and entry point

A conditional version of the Swappa application is now informative under the newest normalized-wealth method. It treats Upgrade as one learner across daily SKU pricing rounds, allows sale probabilities to depend on the lowest competing price and owned inventory, and estimates separate cost levels across product groups. The result is an extension with an explicit learner and feedback restriction, not identification from unrestricted ANR alone.

The preferred seller-specific demand specification gives mean receipt-equivalent replacement cost **[84.333,126.393]** with constant within-SKU costs, and **[75.147,134.498]** under a primitive within-SKU cost span of at most $5. At mean receipt $144.000149, the corresponding gross dollar margins are **[17.607,59.667]** and **[9.502,68.853]**. These are asymptotic conditional outer intervals, not point estimates or exact finite-sample confidence sets. The 20-seller allowance is part of these specifications. Allowing selection among 10,000 sellers, together with seller-specific demand and the $5 span, widens cost to **[61.125,139.274]** and gross margin to **[4.726,82.876]**.

The $5-span prior alone gives costs [0,139.754540] and margins [4.245609,144.000149]. Thus the broad-selection result adds little to the positive margin floor, but substantially excludes low costs and high markups. The constant-cost prior alone gives costs [0,135.541082] and margins [8.459067,144.000149]. Report the prior comparison alongside the inference results.

The maintained economic assumptions are structural conditional demand, the observed-availability and terminal-sale interpretation, no cross-SKU demand substitution, nonnegative costs and no observed below-cost prices, stable within-SKU replacement costs or the stated bounded span, and the shared learner's information, payoff range, and lifetime clock. Data do not establish causal price exogeneity, exact seller feedback, or the cost-span restriction. These are defensible modeling choices for a bulk dealer, not measured facts. Observation gaps remain. The results concern incremental replacement/opportunity cost represented in the profit model, not verified acquisition invoices. Scope, feedback, dependence, and selection sensitivities below make their consequences visible; they have not all been combined in one specification.

Read the shared-learner and final-validation sections below for the current application. Earlier single-SKU sections preserve the less informative specifications and should not be presented as the main result. `daily_multisku_bounds.png` and `.pdf` compare the fitted bounds, uncertainty intervals, and exact cost-span priors. Source validation passed 62 tests; the whole-LP nonlinear diagnostics completed 60 draws per demand specification with no invalid draws. Seller-specific curvature remains material, so the asymptotic qualification cannot be omitted. No manuscript or referee response has been revised, and no result has been published remotely.

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

Increasing the learner's range normalizer widens the fitted intervals reported here. This is a sensitivity, not a general set-inclusion claim when common-advisor regret can be negative. At T=1,500, a factor 1.25 gives cost [41.604,119.772] after uncertainty; factor 2 gives [15.109,122.668]. Their fitted sets are [59.939,116.199] and [40.264,119.769]. The main result therefore also uses a substantive normalization choice. Files: `daily_scale1.25_T1500.json` and `daily_scale2_T1500.json`.

## Persistent cost heterogeneity without a cost distribution

`daily_heterogeneous_cost.py` computes an outer relaxation in which each persistent listing cost c_i lies within delta of the empirical continuation-sale-weighted mean mu. A pre-outcome primitive that justifies this computation is max_i c_i - min_i c_i <= delta within each SKU. Every realized weighted mean lies between the minimum and maximum cost, so the mean-centered box contains every primitive-admissible vector on every outcome path. This is not arbitrary +/-delta variation around an unknown fixed external center, which can have span 2*delta. Costs are nonnegative and no larger than any observed receipt for that listing. This is an explicit restriction, not something inferred from the descriptive condition label. The target remains mu; there is no extra +/-delta adjustment to the reported mean bounds.

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

## Historical validation before the shared-learner extension

The stable source suite passed 58 tests in `tests_daily_endpoint_final.log`. This includes whole-cost interval enclosures, derivative certificates, integer-input peer means, scalar/heterogeneous equivalence at delta zero, persistent-cost witness containment, and seller-specific counterfactual reconstruction. The earlier full-source and two-phone joint-oracle tests also pass.

The single-SKU results motivated the shared-learner extension documented and validated below. Their weaker margin information is retained as an unfavorable comparison.

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

Two additional tests pass: actual cost-vector witnesses remain inside the joint LP bounds, and all active cell coefficients for the four reported inference models pass their regularity audit. The completed gradient-step, nonlinear, normalization, advice-menu, and cost-span checks are recorded below.


### Independent feasibility and dependence follow-up

The shared-learner common-cost grid vector passes an independent full-joint oracle audit: the certified upper bound on total log wealth is 14.109626, below 23.495855. Its empirical mean cost is 109.584618 and gross margin 34.415532. These are witness values, not point estimates. The oracle allows arbitrary feasible joint price vectors within each SKU, including all owned spillovers, and uses a five-dollar mean-price partition with a valid Lagrange upper bound. Files: `daily_multisku_common_d0_h02_oracle_h5.json` and its hashed cost CSV; script `daily_multisku_oracle_audit.py`.

The tenfold gradient-step replication changes the seller-specific delta=5 lower confidence endpoint by $0.001699 and the upper endpoint by less than 1e-8. The lower gradient's relative norm difference is .000134; the upper difference is 6.46e-9. Numerical resolution is immaterial at cent precision.

Weekly dependence and seller-specific demand can now be combined. The covariance reweighting checks that the alternative covariance lies in the recorded gradient's estimable subspace, with relative projection residuals below 5.3e-8. For common demand and delta=5 the weekly interval is [72.965,129.164]; for seller-specific demand and delta=5 it is [76.251,134.124]. The latter implies gross margins [9.876,67.749]. The day-cluster seller-specific upper cost 134.498 remains slightly more conservative. These are asymptotic sensitivities with only 24 weekly clusters, not exact small-sample coverage results.

`daily_multisku_gaussian_check.py` implements a nonlinear diagnostic that re-solves the entire cost LP under coherent Gaussian perturbations. It checks the local envelope approximation, never treats draw quantiles as confidence intervals, and records invalid slope or infeasible draws rather than dropping them. Its completed diagnostics and their limitations are recorded in the final-validation section.


### Scope and approximate-information sensitivities

`daily_scope.py` can restrict advice to displayed prices between .8 and 2 times the current lowest rival price, together with the highest feasible ask as a predeclared fallback. It never uses the observed continuation action to choose the menu. The fallback preserves feasibility and domination of negative-margin common-price proposals. The full action range remains in the payoff certificate. With delta=5 and .5-dollar cells, this advisor gives fitted cost [78.348,125.266] and asymptotic cost [69.487,129.074], gross margins [14.927,74.513]. It is a weaker benchmark, not a claim to have estimated the optimal price of an arbitrary new listing outside the observed market.

A 1.25 range-scale sensitivity gives fitted cost [78.105,126.413]; its uncertainty result is in `daily_multisku_scale125_d5_h05_inference.json`. The broad 10,000-seller allowance gives fitted cost [77.205,126.526] and asymptotic cost [62.845,132.214], gross margins [11.786,81.155]. There are 5,828 distinct sellers in the rebuilt panel. The broader allowance makes the finite-roster assumption explicit; it still does not provide unlimited protection over arbitrary models chosen after inspecting results. Upgrade had 87 listings and ranked fourth in inventory on the first observed date, July 11, 2023. This supports its status as a bulk dealer but is not a substitute for the stated selection correction. The combined seller-specific, 10,000-seller result is reported below.

`daily_proxy_error.py` gives a direct relaxation of exact expected-payoff feedback. Suppose the learner receives normalized payoff forecasts u_tilde in [0,1], with uniform absolute error at most epsilon over feasible joint actions. If it includes the best full-joint proxy-payoff advisor, proxy-oracle regret r_tilde is nonnegative, while any true advisor regret r is at most r_tilde+2epsilon. Therefore, for eta=.99,

    log(1+eta*r) <= log(1+eta*r_tilde) + log(1+2*eta*epsilon).

Over N observed SKU rounds, add N*log(1+2*.99*epsilon) to the wealth threshold. This is an explicit approximate-learning extension. For positive persistent epsilon it does NOT establish exact ANR for true payoffs. The uniform normalized error is an assumption about the seller's payoff forecasts; it is not the analyst's first-stage standard error, and the two allowances are separate. Randomized tests verify the pathwise inequality.

At epsilon=.005, the additional allowance is 10.156707 and the fitted delta=5 cost interval is [73.755,127.219]. The epsilon=.01 fitted result and both uncertainty runs are in files beginning `daily_multisku_proxy`. Do not label these as ordinary ANR results or as requiring only the common-price proxy expert: the proof uses the nonnegative regret of the full proxy oracle.

Completed scope results: the 1.25 normalizer gives asymptotic cost [68.359,130.133]. Combining seller-specific demand with the 10,000-seller allowance gives [61.125,139.274], gross margins [4.726,82.876]. The exact-feedback, twenty-seller result remains narrower; both must be shown rather than silently selecting the favorable correction. The .005 proxy-error extension gives asymptotic cost [63.841,130.861] under common demand, gross margins [13.139,80.159]. These sensitivity dimensions have not all been imposed simultaneously. The full stable suite now passes 62 tests (`tests_scope_checkpoint.log`).


## Final validation, adverse results, and interpretation

### Nonlinear first-stage diagnostics

Both 60-draw coherent Gaussian diagnostics finished under source commit bb9543e. Every draw re-solved the entire continuous-cell LP; none violated the structural conditions or failed feasibility. Common-demand lower/upper endpoint residuals relative to the linear approximation have means [-.161,-.143], RMS values [.343,.204], and observed ranges [-1.520,.126] and [-.882,-.031]. Seller-specific residuals have means [-.713,-.381], RMS values [1.456,.685], and ranges [-8.098,.091] and [-3.756,-.032]. The seller-specific endpoint standard errors are [4.958,2.160]. Curvature is therefore noticeable, especially in the lower tail; this diagnostic supports computation of the functional but does not establish finite-sample accuracy of the Gaussian confidence approximation. Sixty draws cannot validate the extreme tails required by the seller-selection adjustment. Do not use these quantiles as calibrated confidence bounds or silently inflate/shift the reported intervals using their extrema. Artifacts: `daily_multisku_common_d5_h02_gaussian_diagnostic.json` and `daily_multisku_seller_d5_h02_gaussian_diagnostic.json`.

### Primitive cost span and exact prior

The heterogeneity calculations use a necessary mean-centered box, justified pathwise by a primitive total within-SKU cost span. For listing receipt caps b_i and cost span d, every admissible c_i is at most min(b_i,min_j b_j+d). This vector is itself admissible, hence its empirical sale-weighted average is the exact prior upper bound. `daily_cost_span_prior.py` records [0,135.541082], [0,139.754540], [0,142.070473], and [0,143.452052] for spans 0,5,10,20. The estimator's internal box prior is slightly looser; the figure uses the exact primitive prior. The reported main uncertainty endpoints all lie inside it. No cost dispersion is estimated from the data.

### A failed time-varying cost model

`daily_multisku_profile.py --index reference --delta 0 --step .2` tests costs proportional to each SKU's public reference quote, normalized at the first observation. The target weights account for the sales-weighted index multiplier. The continuous-cell lower bound on total wealth is 25.215851, exceeding 23.495855, so the entire fitted-demand outer relaxation is empty. Artifact: `daily_multisku_reference_d0_h02.json`. This is a fixed-demand failure of that exact cost schedule, not a statistical rejection after first-stage uncertainty. Do not change the learning threshold solely to make this specification pass. Public retail quotes need not track wholesale replacement costs. The default constant-cost regression profile is unchanged to 1e-8 after adding indexed support.

### Feedback and learner versions

The common-price-advisor learner and the full-joint-oracle learner can be alternative model versions, each allocating prior 1/4 to its designated normalized expert. Do not assign both 1/4 while also retaining a half-prior legacy mixture without checking the resulting allocation. Under the stronger full-oracle version, the common-price wealth screen remains necessary because its regret is no larger than full-oracle regret. The certified full-oracle witness establishes nonemptiness of this stronger fitted event, not that a named algorithm reproduces the observed prices.

The approximate-feedback extension requires the full proxy oracle. At normalized forecast error .005, common-demand span-5 cost inference is [63.841,130.861]; at .01 it is [54.121,132.229]. This establishes informative conditional bounds even with a modest explicit error allowance, but persistent positive forecast error changes exact true-payoff ANR into an approximate-learning restriction. These errors are relative to the prescribed payoff normalizer, not percentage errors in sale probabilities.

### Monetary units and fee conversion

Reported receipts equal 1.03 times integer ask, before sales tax and display rounding. They are not net seller payouts. Under an illustrative constant schedule with 3% buyer and seller fees, PayPal processing of 3.49% plus $.49, and a constant expected tax rate between 0 and .10, net payout is a*r-.49 with a in [.9033576,.9068476]. Then net cost is a*c-.49 and the operating margin is a*(r-c); normalized learning is unchanged by the common positive scale. Subtracting fees again from that margin would double count them.

`daily_fee_conversion.json` gives conditional net cost [75.693,114.129] and operating margin [15.906,54.109] for seller-specific constant SKU costs; for span 5 it gives [67.394,121.479] and [8.584,62.440]. Combining span 5 with the broad seller allowance gives [54.727,125.811] and [4.269,75.156]. Mean net receipts under this schedule are [129.594,130.096]. These are illustrations, not observed transaction-specific fees, payout records, or acquisition invoices. The cost includes all per-sale incremental costs represented in the model. Sources: Swappa's [fee discussion](https://swappa.com/blog/swappa-fees/) and [PayPal refund discussion](https://swappa.com/blog/who-pays-the-fees-when-issuing-a-paypal-refund/). Publication dates on current pages do not verify their historical contents or the seller's actual schedule.

### Reproduction and remaining research choices

Set `SWAPPA_DATA_VERSION=fullsource_refscreen_known` before running any current script. For the seller-specific span-5 result run `daily_multisku_profile.py --delta 5 --step .2 --fit-name continuation_seller_full_1.joblib --cov-name demand_cluster_seller_covariance.npz --tag daily_multisku_seller_d5_h02`, then `daily_multisku_inference.py --profile daily_multisku_seller_d5_h02.json --cov-name demand_cluster_seller_covariance.npz`. Add `--seller-family 10000` and use step .5 with a distinct tag for the broad-selection version. Constant-cost results use delta 0. The diagnostics, prior calculation, fee conversion, and figure have separate named scripts. Outputs retain hashes of their actual producing sources; later source commits do not retroactively change provenance.

The requested search has found an informative, internally consistent conditional application. Choosing which assumptions to defend in the manuscript, obtaining invoice or payout validation, and developing a stronger finite-sample first-stage procedure are subsequent research decisions. None is represented as already resolved by these estimates. The current code and results preserve the failed specifications, the stronger shared-learning restriction, the nonlinear diagnostic caveat, and the contribution of the prior to positive margins. Referee mappings above identify concrete progress rather than completed responses.
