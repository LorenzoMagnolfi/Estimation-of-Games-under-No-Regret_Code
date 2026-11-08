# SWAPPA revival: completed demand robustness and prioritized repairs

Date: 6 November 2026. Audience: Lorenzo and the agent implementing the SWAPPA revival. Status: completed local robustness calculations and proposed application repairs; no manuscript or response changes. This is the governing implementation memo. It supersedes the more expansive priorities in the earlier version and in `docs/swappa-demand-repair-note-2026-11-06.md`; that companion preserves supporting demand diagnostics.

## Decision

Keep the revival. Treat the demand extensions as robustness, with baseline demand as the main specification, richer rival controls as a close comparison, and flexible relative-price shape as the meaningful sensitivity. Do not turn these checks into a new demand paper. The necessary repairs concern what behavioral restriction is imposed, what inference is justified, and what the counterfactual establishes.

The calculations below hold the sample and reference-price construction fixed across specifications. They are a controlled comparison of demand models, not an exact reproduction of every number on the revival branch. The richer-rival extension has small downstream effects. Flexible shape changes inferred margins more materially; assess its fee results separately rather than inferring their size from the sales elasticities.

## Completed calculations

The common panel has 457,189 listing-days and 22,792 terminal-sale labels over 164 dates. The three full-sample logit fits retain SKU, date, and seller-group effects, age, battery, inventory counts, and entry-price controls. Richer rivals adds rival mean, median, and log-price dispersion. Flexible shape retains those richer rival controls and adds four relative-price hinges. Costs are c(j,d)=theta(i) r(j,d), with reference values fixed under deviations. The reported margin is 1 minus theta times average reference value among sold listings divided by average sale price among those listings; it is not a procurement-accounting estimate.

The fee is 3% of seller revenue. The finite deviation class scales all own portfolio prices by +/-0.5, 1, 2.5, 5, and 10 percent. Rival scenarios hold rival prices fixed or scale them with the candidate own-price level. Even in the latter scenario, rivals stay fixed during a unilateral deviation. All exercises condition on observed inventory and use continuous price scaling.

| Demand | Seller | Margin, provisional (%) | Fitted fee price set, rivals fixed (%) | Fitted fee price set, rivals following (%) |
|---|---|---:|---:|---:|
| Baseline | GadgetPickup | 10.26 to 15.17 | 1.75 to 2.50 | 2.25 to 3.50 |
| Baseline | We Buy Mobile | 9.73 to 14.89 | 2.25 to 3.25 | 2.50 to 3.75 |
| Richer rivals | GadgetPickup | 9.72 to 14.63 | 1.75 to 2.75 | 2.00 to 3.75 |
| Richer rivals | We Buy Mobile | 9.45 to 14.08 | 2.25 to 3.00 | 2.50 to 3.75 |
| Flexible price gap | GadgetPickup | 7.26 to 11.08 | 3.50 to 4.50 | 2.75 to 3.25 |
| Flexible price gap | We Buy Mobile | 7.28 to 11.90 | 4.25 to 5.25 | 2.75 to 3.25 |

The fee columns are direct fitted moment sets: retain cost/price pairs satisfying both status-quo and fee inequalities at their fitted means, then project onto the price change. They are not confidence intervals. Under fixed rivals, flexible curvature approximately doubles the fitted price response relative to baseline; following-rival fitted ranges overlap substantially across specifications. The following-rivals case is a scenario, not an upper equilibrium bound.

The separate optimal-level-shift calculation was also run. For baseline and richer rivals it gives approximately 1.8–3.1% under fixed rivals and 2.8–3.2% when rivals follow, over provisional cost sets. For flexible demand it is unstable under fixed rivals. The initial grid omits four GadgetPickup and five We Buy Mobile cost values because a root is missing. Extending the root search from price levels 0.2 to 3.0 finds up to two positive-to-negative crossings, and the mechanical first-root subtraction produces upper shifts of 132.6% and 129.7%. These are **diagnostics of branch switching and extrapolation, not economically credible fee estimates**. Do not silently drop these cost values or headline a truncated 3–7% range. The flexible following-rivals calculation remains near 2.9–3.2%. Use the direct moment-set exercise as the primary counterfactual; retain the first-order-root calculation only if the objective and root selection are separately justified. This additional finding makes the distinction in M7 consequential.

For comparison, the direct replayed-window fee sets under fixed rivals are:

| Demand | Seller | Fitted fee price set (%) | Separate-test projection (%) |
|---|---|---:|---:|
| Baseline | GadgetPickup | 1.75 to 2.50 | -2.00 to 7.50 |
| Baseline | We Buy Mobile | 2.25 to 3.25 | -2.00 to 8.75 |
| Richer rivals | GadgetPickup | 1.75 to 2.75 | -1.75 to 7.25 |
| Richer rivals | We Buy Mobile | 2.25 to 3.00 | -1.75 to 8.50 |
| Flexible price gap | GadgetPickup | 3.50 to 4.50 | -1.50 to 25.00 (upper search edge) |
| Flexible price gap | We Buy Mobile | 4.25 to 5.25 | -2.50 to 25.00 (upper search edge) |

The fitted column requires both status-quo and fee moment means to be nonpositive. The separate-test column inverts 20 Bonferroni-adjusted inequalities, reproducing the convention in the revival's `cf_fee.py`. It does **not** rerun the sharper Gaussian matched-combination test in `cf_fee_joint.py`. Do not compare these endpoints with that test's endpoints as though only demand had changed. These wider projections include zero; they do not establish a positive fee effect with repaired confidence coverage.

All inferential quantities here remain **provisional**. They retain HAC with seven lags plus a two-way listing/date clustered demand delta variance, with negative covariance eigenvalues clipped as in the revival. They omit the same-data covariance between the demand estimate and payoff moments, and do not correct for seller selection. The conventional per-seller nominal level is 95%. The point of retaining this convention is a controlled robustness comparison, not to certify it.

### What the earlier demand exercises add

On native reference values, a 5% own-price cut increases predicted sales by 38.6%, 40.2%, and 61.6% for GadgetPickup under baseline, richer rivals, and flexible shape; corresponding We Buy Mobile responses are 42.8%, 44.5%, and 71.1%. Richer product-by-week effects give 41.6% and 45.7%. Thus most control refinements are modest, while curvature matters. Flexible shape improves aggregate holdout log loss but does not solve focal-seller calibration: held-out actual sales are 330 and 72, against approximately 297 and 58 predicted under that model.

Matched current/previous-quote checks reduce the 5% cut response by about 14–15%. Treat this as timing sensitivity, not an instrument. The complete-history restriction disproportionately removes We Buy Mobile observations (retaining about 23% of its rows, versus 88% for GadgetPickup); it is not a clean replacement baseline. Eleven specifications and 22 full/holdout fits are preserved in the supporting evidence. No additional demand exercise is required before interpreting the present table.

## Must fix before integrating the application

1. **M1/M2: align the empirical assumption with the method.** Call the maintained restriction finite-Phi regret against proportional transformations. Remove claims that this is generally weaker than external regret. State the population moment implication for this exact class. Separate stationarity/ergodicity and integrability, which support the mean implication, from the dependence and moment conditions needed for a CLT and HAC inference. Stable margins alone do not establish stationary dollar gains. **Check:** every note, theorem reference, and proposed response uses the same comparator, units, and asymptotic versus finite-sample claim. A separate external-regret application is optional, not a prerequisite.

2. **M3/M4: repair coverage if retaining formal confidence claims.** Stack the first-stage score contribution and the daily payoff-moment contribution, including their cross covariance, using dependence assumptions appropriate to both listings and dates. Alternatively specify and justify a joint resampling scheme that refits demand and recomputes moments. The existing fixed-first-stage bootstrap is not that exercise. Account for the original 13-seller screening family if reporting simultaneous statements for two sellers selected after inspection; two-seller Bonferroni alone does not resolve selection. **Check:** validate against a controlled data-generating process with known parameters and report coverage and Monte Carlo uncertainty; then rerun baseline plus the two demand alternatives with one common joint procedure. Until then, retain the provisional label rather than claiming verified 95% coverage.

3. **M5/M6: close the bounded implementation discrepancies.** Reconcile native workbook versus revival extracted-text reference construction with a staged count/hash comparison: unlocked definition, numeric/positive filters, SKU mapping, date alignment, aggregation, and fallback. Native inputs contain 55,783 usable name-unlocked rows versus 46,867 reported in the branch; the precise difference remains unresolved. Preserve both versions until identified. Specify whether actions are posted asks or displayed prices and their feasible tick. Recompute owned-peer averages from full preselection inventory after each own-price transformation. Either use feasible rounded maps or explicitly present continuous prices as an approximation. **Check:** reference counts and a hand-worked inventory example agree; zero transformation is an identity; reported deviations obey the chosen action definition. Do not silently substitute the native reconstruction for the branch series.

4. **M7: narrow fee claims to what was calculated.** State observed-inventory conditioning and the two rival-price scenarios. Keep optimal-level shifts, fitted moment sets, and joint statistical projections in separate columns. A buyer-share interval extending below 50% does not establish that buyers bear most of the fee. **Check:** zero-fee placebo, endpoint and grid checks, unilateral rival-price hold fixed, and identical units across tables. Rerun the preferred joint fee test after M3; the separate-test numbers above do not replace it.

5. **M8/M9: qualify or remove the auxiliary claims rather than rebuild the paper around them.** For the recommender, distinguish external/Phi/swap obedience, label evaluations at three cost points as finite sensitivity, and avoid claiming implementability by a particular seller learner. Either project over the full retained cost region or narrow the claim. For mature-learner rows, experience counts are not certified learning rounds. A bound on a signed mean gain does not justify Markov on its positive part. Remove probability guarantees unless a valid nonnegative bound, clock, feedback model, and later-window bridge are supplied. **Check:** every retained guarantee has the required inequality and information assumptions; otherwise label the exercise heuristic or omit it.

These are necessary corrections to claims or implementation. They do not all require new large computational projects. In particular, removing an unsupported guarantee is an acceptable repair.

## Optional robustness, in priority order

First, retain the richer-rivals and flexible-gap tables already calculated, and carry them through the repaired joint inference. Second, if feasible-price rounding changes economically relevant endpoints, report the rounded-map comparison. Third, add product-by-week effects, lagged quotes, complete-history restrictions, and seller-specific response heterogeneity only where they answer a remaining concern. A new instrument, additional scrape window, dynamic inventory model, complete recommender extension, and new Gazelle comparison are not prerequisites for this conditional application. Predictive robustness does not establish causal price exogeneity; that remains a maintained assumption to state plainly.

## Provenance, checks, and next action

The audited revival is paper-repository branch `claude/busy-dijkstra-qpj2t3`, commit `16be52968a0776f0aec086a2bbeabf5b8f6abfcb`. Its research and review notes are `JPE Revision/swappa-revival-research-note-2026-11-03.tex` and `JPE Revision/swappa-revival-review-note-2026-11-04.md`; source scripts are under `JPE Revision/swappa-revival-2026-11-03/code/`. The canonical manuscript and response are unchanged.

New producing code is `python/swappa_rescue/demand_downstream_robustness_20261106.py`, followed by `python/swappa_rescue/extend_demand_fee_roots_20261106.py`; independent checks are `python/swappa_rescue/check_demand_downstream_20261106.py`. Run those scripts in that order with the documented temporary pyarrow directory on PYTHONPATH. Inputs are the native-reference panel and saved `raw_d0`, `raw_d1`, and `raw_d6` fits under `output/swappa_demand_repair_20261106/`. Detailed results and log remain there. Durable aggregate outputs are `docs/evidence/swappa_demand_downstream_results_20261106.json` and `docs/evidence/swappa_demand_downstream_checks_20261106.json`; they contain input/script hashes, numerical checks, and covariance diagnostics. Earlier demand results are `docs/evidence/swappa_demand_repair_results_20261106.json`. Raw rows are not included in these evidence files.

Verified checks: saved-fit design reconstruction; direct row-profit versus daily-aggregate moment identity; six independent finite-difference gradient checks; interior cost-grid endpoints; zero-fee identity at zero level shift. The cost grid is [0.2,1.2] in steps of 0.0025. The level grid is [-5%,10%] for baseline and richer rivals and was extended to [-15%,25%] for flexible shape, always in steps of 0.25 percentage points. The supplementary flexible root search spans [-80%,200%] price changes with 1 percentage-point bracketing and Brent root refinement; all cost values then have roots, but multiplicity makes the fixed-rival optimal-shift interpretation fail. 8 scenario/fee projections hit a level-grid boundary; see the check artifact for exact cases. Treat those endpoints as censored, not closed interval endpoints. Grid-set endpoints are discrete approximations, and the checks do not validate statistical coverage or rule out disconnected accepted components beyond the searched domain. This run does not repair endogenous pricing or certify a learner.

The next action belongs to the revival agent: first correct the statements in items 1, 4, and 5; then resolve the reference/action mapping and implement the joint inference repair. The methods author should review the dependence and behavioral assumptions. Return one old/new table under the common demand specifications, the updated issue matrix, failed checks, and exact proposed manuscript changes. Stop adding robustness exercises unless that table reveals a material instability. Commit the producing code and record its commit before any number enters the manuscript. These local results have not been published or sent externally.
