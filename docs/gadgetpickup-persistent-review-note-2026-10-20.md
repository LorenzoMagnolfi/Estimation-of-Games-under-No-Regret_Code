# GadgetPickup: persistent prices, review performance, and demand

Date: October 20, 2026. Audience: future research agents. Status: completed computational extension, conditional economic comparisons, and demand diagnostics. No calibrated statistical confidence region or credible marginal-acquisition-cost estimate is claimed. This note governs the persistence extension and follows `gadgetpickup-allowance-demand-audit-note-2026-10-20.md`. Preserve the earlier notes as historical specifications and results.

## Findings and decision

Accounting for persistence changes the appropriate profit comparison. A phone can sell once before the next review; multiplying a one-day profit by a spell length can count repeated sales of the same item. We replace the daily sale probability with a survival-adjusted probability over an anticipated review interval. The main interval distribution uses batch-repricing gaps before November, rather than the realized date on which a particular phone later sold.

This extension does not yet produce both an empirically disciplined, informative regret allowance and credible cost bounds. Requiring the adjustment inequalities at every review remains incompatible with several phone models. Aggregating across reviews, which permits individual mistakes and is closer to the new method's performance premise, leaves only iPhone 6s Plus incompatible under the primary horizon. Aggregating at the actual seller level retains every product and is feasible, but permits zero cost even when adjustment costs and the dollar allowance are inferred jointly.

A stronger assumption about the prices selected when the seller actively reprices gives a positive conditional lower bound. The review-weighted mean lies in [58.70,416.30] under baseline demand and the main horizon distribution. An independent attribution calculation shows that the entire positive lower bound comes from the restriction that, on adjustment dates in aggregate, asks $5 lower would not improve expected contribution. Removing all the adjustment-cost calibration restrictions leaves exactly the same interval. The positive bound is therefore not evidence that the inferred allowance has become informative.

Returning to the original targets matters. Retaining the previously used daily participation assumption, the seller-wide core model gives the original expected-sales mean cost [0,407.58]. The stronger restriction gives [26.87,407.58]. For mean cost among the phones that actually sold, its interval is [0,31480.25]. The original target has not been rescued by changing the weighting to review opportunities.

Demand diagnostics continue to support a negative price association, while leaving its causal magnitude uncertain. Coordinated markdowns are followed by higher sale rates, but the increase is smaller when compared to several preceding days rather than only the day just before the cut. Comparisons that absorb category-by-date demand shocks are imprecise. A frozen daily hazard predicts reasonably close aggregate sales through the first week of unchanged prices, then overpredicts at longer durations. These are empirical inputs and checks; none proves a 90 or 95 percent performance level.

## Data, eligible decisions, and units

Use `SWAPPA_DATA_VERSION=fullsource_refscreen_known` and the corrected sample documented in the preceding audit. November-December has 6,039 listing-days, 832 listings, 50 dates, and 556 terminal-sale labels. The full demand fit has 20,096 listing-days. The copied November 19 snapshot and incomplete exposure intervals remain excluded. Terminal-sale labels and retained listings are observed proxies with the observation limitations already documented.

A batch-repricing date has at least ten changed continuing GadgetPickup quotes across at least three phone models. There are 35 such dates, including 17 in November-December. A model-date enters the review comparisons only if every retained offer has a consecutive fresh previous quote. This gives 268 model-review occasions, 149 with changes and 119 without changes, covering 1,346 listing-review observations and 484 distinct listings.

These are proxies for review opportunities. A price change reveals that some pricing decision occurred; it does not establish that every other phone model was reviewed. Reviews that produce no observed changes are not visible. The interval between observed batch changes is therefore not automatically the interval between all actual reviews. Treating it as the payoff horizon requires an economic assumption.

Let a be integer ask and x=1.03a the receipt-equivalent unit, with displayed price ceil(x). A $5 ask change changes x by $5.15. This convention is retained from the prior analysis. It is not net seller payout. Unknown listing costs c_j are nonnegative and persistent, without an upper cap or equality across listings. The modeled c_j is an opportunity value of selling rather than retaining the phone, inclusive of the payoff convention. Interpreting it as acquisition cost requires a further link to replenishment or continuation value.

## Persistent-price contribution

For a fixed price vector k at review t, let q_jt^k be the conditional daily sale probability of listing j. If this hazard stays constant while the price persists for H days, the probability of a sale before the next review is

    P_jt^k(H) = 1-(1-q_jt^k)^H.

For a horizon distribution F fixed at the review, use

    P_jt^k = E_F[1-(1-q_jt^k)^H],
    V_gt^k(c) = sum_{j in model g at review t} P_jt^k (x_jt^k-c_j).

V is expected contribution relative to retaining the item at the maintained continuation value. A phone contributes at most one sale within the review interval. For example, q=.2 and H=7 give sale probability .7902848, whereas multiplying daily q by seven gives 1.4. The implementation verifies the formula against the sum of first-sale probabilities.

The main F is the empirical distribution of 17 gaps between consecutive batch dates strictly before November 1:

    17, 1, 6, 10, 5, 4, 1, 15, 7, 1, 4, 3, 3, 3, 1, 13, 10 days.

Its mean is 6.12 days and median 4. The specified comparisons use fixed H=1,3,7,14. No horizon was selected for producing nonempty or narrow cost bounds. The later 16 November-December gaps average 2.94 days, with median 2.5. That change is evidence against taking the preperiod distribution as automatically stable; the fixed three-day comparison is reported alongside every other horizon, not promoted after seeing its results.

The calculation assumes that hazard, prices, competitor state, the owned-peer price summary, and opportunity values stay constant until review; there is no replenishment and no discounting within these short horizons. Owned-peer effects are updated for the proposed price vector at the review, but not after subsequent sales deplete inventory. The horizon law is common across actions. It does not model endogenous review timing or seller learning between reviews. Some review horizons extend beyond the final observed date, so those payoffs also extrapolate beyond the panel. These approximations are stronger than merely observing that prices persist. Summed review values equal an actual multi-period profit comparison only under a correctly specified nonoverlapping review process; the current exercises do not establish that link.

The demand specifications are the prior corrected additive fit, the fit without initial-price controls, and the category-by-week fit with C=1. The five listing folds are unchanged. The ten refits needed for previous-quote alternatives reproduce their cached factual probabilities within 1e-10 and converge without warnings. The category-week model's poorer predictive calibration remains a limitation; it is included as a declared comparison rather than selected by its cost implications.

## Individual-review and aggregate comparisons

At an eligible review define

    H_gt(c) = V_gt^actual(c)-V_gt^old(c),
    G_gtk(c) = V_gt^shift-k(c)-V_gt^actual(c).

Here old keeps the previous fresh quotes; shift k changes continuing asks by -10,-5,+5,+10 dollars. K_g is an unknown nonnegative stable adjustment cost for changing the price vector of phone model g. Both c and K are projected out jointly. Unknown costs are the estimation target, not a missing prerequisite.

The initial extension requires H_gt>=K_g on each adjustment date and G_gtk<=K_g on each inaction date. The complete-menu version also requires G_gtk<=0 on each adjustment date. Aggregate actual contribution is nonnegative within each model. These individual-review inequalities are stronger than a window-level performance condition.

Under the primary horizon law, the core version is incompatible with seven models at baseline demand, six without initial-price controls, and eight with category-week demand. The complete-menu versions fail for 13, 12, and 15 models. No prespecified fixed horizon fits every model either. These are contradictions at fixed estimated demand, not statistical rejections of rationality or learning.

### Aggregating within models

Let C_g and I_g count change and inaction reviews. To permit individual mistakes, replace the pointwise inequalities with

    sum_change H_gt(c) >= C_g K_g,
    sum_inaction G_gtk(c) <= I_g K_g,  for every k.

The complete-menu version adds sum_change G_gtk(c)<=0. These restrictions allow gains and mistakes to offset across dates. They are maintained finite-window performance assumptions, not an empirical finding that net mistakes are exactly zero. For all three demand fits, only iPhone 6s Plus remains incompatible under the primary horizon, in both versions. The other 25 models are not selected and reported as a successful seller sample.

### Aggregating at the seller level

GadgetPickup is the decision maker. Requiring performance separately for every phone model is an additional restriction. The seller-wide specification permits offsets across models while retaining a different unknown K_g for every model:

    sum_g sum_change H_gt(c) >= sum_g C_g K_g,
    sum_g sum_inaction G_gtk(c) <= sum_g I_g K_g, for every k,
    sum_g,t V_gt^actual(c) >= 0.

This retains all 26 models. It does not assume that all products share an adjustment cost. The complete-menu version adds sum_g sum_change G_gtk(c)<=0 for each k. Seller-wide systems are feasible under all three demand fits and all five horizon laws.

For the core system, the conditional dollar allowance is B(c,K)=sum_g I_g K_g. The maximum gain among deviations that change prices only on inaction reviews is at most B. Under the complete-menu inequalities the same B bounds the four shifts applied on both kinds of review. The comparison uses one specified shift across the relevant reviews, not an independently optimal price at each review.

This is an exact conditional implication of the inequalities: the total gain separates into change and inaction contributions; the first is nonpositive in the complete-menu system and the second is at most B. It is not a learning-rate theorem or a bound on regret against every possible pricing policy. The deviations depend on actual actions and past quotes, so they belong to a restricted action-dependent comparison class. Seller-wide aggregation does not imply separate guarantees for every model.

## Conditional cost bounds and allowance size

The initial target weights c_j by expected sales across eligible review opportunities, using the survival-adjusted P. Repeated review anchors can include the same phone. These weights summarize conditional review opportunities; they are not predicted distinct sales over a simulated inventory history.

Primary horizon results:

| Demand | Core review-mean cost | Complete-menu review-mean cost |
|---|---:|---:|
| Baseline additive | [0,416.30] | [58.70,416.30] |
| Without initial-price controls | [0,416.50] | [54.06,416.50] |
| Category-week, C=1 | [0,418.74] | [49.79,418.74] |

For the baseline core system, the jointly feasible allowance ranges from 0 to 243792.27 in review-payoff dollars. The complete-menu range is 0 to 121634.31. These are extrema over different feasible cost and adjustment-cost vectors, not a point estimate of actual loss and not a free allowance plugged into a second inversion. The cost intervals already account for joint feasibility. They show that the core comparisons do not exclude zero cost even when the allowance is restricted by the implemented decisions.

The baseline complete-menu review lower bounds at fixed H=1,3,7,14 are 87.48,73.54,58.26,43.70. Longer price persistence does not mechanically strengthen identification: selling probabilities saturate, which changes the benefit of a further markdown. The corresponding core lower bound is zero at every horizon and demand fit.

An independent reconstruction removes all adjustment-cost restrictions and retains only participation plus the complete-menu comparisons on adjustment dates. It reproduces [58.704711,416.299515] exactly. The lower-bound dual certificate places weight only on the -5 ask comparison on adjustment dates. Thus the positive information comes from the stronger aggregate performance assumption when repricing, not from empirically narrowing B. This is a potentially interpretable behavioral hypothesis, but it must be labeled as such.

The still weaker previously studied comparison, that implemented changes improve contribution relative to old prices in aggregate, was also extended to persistent prices. Under the primary horizon it gives [0,416.30] seller-wide and [0,415.85] when imposed separately by model at baseline demand. It supplies no positive lower bound. Some longer-horizon model-specific versions are incompatible; all are retained in the result file.

## Returning to the original cost targets

All 832 November-December listing costs are included in the transfer calculation. Only 484 listings appear in the eligible review comparisons. Listings never reviewed in that sample account for 19.12 percent of baseline expected daily sales and 37.23 percent of recorded sales. Their costs cannot be implicitly omitted from the original targets.

Review inequalities alone give no finite upper bound on either original mean, because positive target weight falls on otherwise unrestricted costs. With the previously used additional daily-window participation condition, sum_jt q_jt(x_jt-c_j)>=0, the baseline primary-horizon results are:

| Target | Core comparisons | Complete-menu comparisons |
|---|---:|---:|
| Review-exposure expected-sales mean | [0,416.30] | [58.70,416.30] |
| Original daily expected-sales mean | [0,407.58] | [26.87,407.58] |
| Original realized-sales mean | [0,33591.93] | [0,31480.25] |

Daily participation is not implied by review participation; both cases are reported. Without it, the original complete-menu expected-sales interval is [26.87,infinity), and the original realized-sales interval is [0,infinity). All three demand fits and five horizons were transferred, not only the primary example. Across the three demand fits and five horizons, the complete-menu lower endpoints for the original expected-sales mean range from 10.89 to 51.75. No cost cap or common cost was added. The large realized-sales upper bounds are a consequence of weak restrictions and differing weights, not a forecast that phones cost those amounts.

## Demand around coordinated markdowns

The event design uses categories defined by phone model, storage, and condition. On a batch date, a treated category has at least three continuing cuts and no increases; a control category has at least three valid continuing quotes and no change. Assignment uses current quotes, not future outcomes or subsequent price changes. Events have a calendar window from seven days before to seven days after, with category and date effects within each stacked event. Weights are category inventory on the preceding day, held fixed within the event. Standard errors cluster by original category and date.

There are 29 event dates and 65 categories in the full sample, with 12 event dates and 37 categories for November-December. Relative to the day before the cut, the average full-sample ask in days 0-3 falls by 11.62 dollars, and the daily sale rate rises by 3.49 percentage points (standard error 1.60). Price differences persist through the event window.

However, the mean sale-rate difference on days -7 through -2 is already 1.89 percentage points above day -1. Comparing days 0-3 to that earlier average gives a 1.60-point increase with standard error 1.16. A descriptive normal interval is [-0.68,3.88] percentage points. The analogous ask change is -9.80 dollars. The alternative normalization was added as an explicitly labeled interpretation check after inspecting the initial event coefficients; no sample or model was changed.

The full-sample pretrend Wald p-value is .225. This does not reject flat pretrends, but it does not prove them either. The November-December lead covariance is indefinite, so its joint p-value is suppressed; one negative individual variance is also suppressed rather than reported as zero uncertainty. Small event samples do not justify a precise causal elasticity.

Other outcomes change around the same events. Mean log listing age rises by .133 between the earlier days and days 0-3, and rival minimum price falls by 5.20 dollars. Composition and competitive conditions are therefore not constant. Subsequent repricing is common: 138 of 226 control-category episodes and 49 of 166 treated episodes have another change within the next week. Controls were not removed based on future choices. Missing category-days are 603 of 5880 possible observations. These features limit a simple causal interpretation of the event contrast.

The diagnostic figure is `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/gadget_demand_20261020/persistent_review_demand_diagnostics.png`, with a matching PDF figure. It has been visually inspected. Its intervals describe clustered regression uncertainty, not causal identification or a demand confidence set for structural inversion.

### Stable prices and common demand shocks

Linear probability checks retain the earlier continuous controls and use the same $10 joint shift contrast. On 5,115 November-December rows with fresh prior quotes, listing and date effects give 67.06 additional predicted sales (standard error 13.05). Restricting to 4,382 unchanged-quote rows gives 47.04 (standard error 10.77). These are different samples; their aggregate counts should not be compared as estimates of the same effect. The negative price association persists when same-day repricing observations are removed.

Absorbing both listing and category-by-date effects makes the relevant contrast imprecise: 628.18 with standard error 387.57 for fresh-quote rows, and 134.00 with standard error 291.31 for unchanged quotes. These linear predictions are not probabilities for cost inversion. Common category-day prices leave little variation to identify a common markdown; transferring individual within-day variation into a coordinated price effect relies on functional restrictions. The large point estimates do not justify choosing a steeper demand response.

### Does a frozen hazard describe unchanged prices?

For each batch anchor, the hazard audit follows a listing while the anchor price remains posted, stopping before a changed quote or gap, after a sale, or at 14 days. It compares observed terminal-sale labels to the frozen anchor probability over the actually observed at-risk exposures. It does not condition on a future sale to select the initial horizon. Repeated anchors can duplicate listing-days, and censoring can be informative; no independent-sample interval is claimed.

November-December results:

| Days after anchor | At-risk anchor exposures | Recorded sales | Frozen-hazard expected sales |
|---|---:|---:|---:|
| 0 | 2041 | 217 | 206.10 |
| 1-2 | 2495 | 185 | 193.04 |
| 3-6 | 2508 | 153 | 161.92 |
| 7-13 | 909 | 39 | 58.46 |

The first-week totals are relatively close; the longer-duration prediction overstates sales by 19.46. Updating probabilities with current observed covariates improves the Brier score in later bands, but is not an ex ante forecast or a causal intervention estimate. These checks support considering short horizons while retaining uncertainty about how hazards evolve and how price changes affect them.

## What is established for the new method

The contribution formula and aggregate-regret implication are proved conditional on the stated review model. The empirical durations, observed choices, demand diagnostics, and optimization results are reproducible. No proprietary algorithm needs to be known. The seller need not know every conditional profit; the restrictions evaluate its realized sequence of choices using the researcher's maintained conditional payoffs.

The extra assumptions are the review proxy, the action-independent horizon law, stable hazard and continuation-value approximation, persistent listing costs, aggregate finite-window performance, and the conditional demand response. The complete-menu case adds a stronger comparison on dates when prices were changed. None of those exact performance inequalities follows from asymptotic no regret on a finite observed prefix.

In the paper's prediction-set formulation, costs, adjustment costs, demand, and the review environment are candidate parameters, and these comparisons are additional obedience restrictions on the relevant empirical witness. The conditional finite bound B is a function of that candidate environment. A full confidence statement would also require a valid consistency set and a justified demand/environment uncertainty set. These computations fix those inputs; they are not completed Q_f confidence projections or 95 percent intervals.

Projecting over uncertainty means retaining a cost when some admissible demand/review environment and witness satisfy the restrictions. Intersecting results from different fitted demand models would impose them all simultaneously. The event regressions do not themselves supply a valid causal uncertainty set, and a change in event-study normalization is not a license to choose a counterfactual slope that produces preferred bounds.

## Provenance and research sequence

The initial persistence, event, stable-price, and hazard design was fixed at ed9005c before its results. Aggregation and target-transfer designs were added after seeing the initial failures, each before its own calculation. Their motivation and timing are explicit in the design files. This is a sequential exploratory investigation, not preregistration of a single final specification. All horizons, demand fits, product failures, and target definitions are retained.

| Producing script in python/swappa_rescue | Producing commit | Result in docs/evidence |
|---|---|---|
| gadget_persistent_review.py | e658fd5 | gadgetpickup_persistent_review_results_20261020.json |
| gadget_batch_demand_audit.py | 0b3663c | gadgetpickup_batch_demand_results_20261020.json |
| gadget_persistent_purposeful.py | f228b43 | gadgetpickup_persistent_purposeful_results_20261020.json |
| gadget_aggregate_review.py | 3645f49 | gadgetpickup_aggregate_review_results_20261020.json |
| gadget_seller_aggregate_review.py | 9b505c2 | gadgetpickup_seller_aggregate_review_results_20261020.json |
| gadget_review_target_transfer.py | 9be4749 | gadgetpickup_review_target_transfer_results_20261020.json |
| gadget_persistence_audit_summary.py | 73596d6 | gadgetpickup_persistence_audit_summary_20261020.json |

The primary shared design is `gadgetpickup_persistent_review_design_20261020.json`; separate purposeful, model-aggregate, seller-aggregate, and target-transfer designs carry matching names. The shared design was amended for the covariance and comparison audit after the first event run. Validate each result against its historical producing-commit design hash. The initial event result remains archived as `batch_demand_initial.json` in the ignored run directory, SHA256 `7181b53006339040195f2a20822b061b750d53a9c1251afaee90091003c54378`.

The new ignored inputs are `persistent_review_predictions.parquet` and `batch_event_stack.parquet`. The main manifest is `docs/evidence/gadgetpickup_persistence_manifest_20261020.json`. Direct coefficient-profit identities, horizon limiting cases, probability bounds, solver residuals, and an independent seller-wide LP reconstruction are checked. The attribution LP agrees on the lower endpoint within 1e-7. Target-transfer scaling was strengthened after a residual check failed; a separate JSON integer conversion was fixed. Neither changed economic restrictions, and failed runs produced no interpreted result.

## Remaining research task

Persistence has been incorporated and the stronger claim has been audited: the data-informed adjustment-cost exercise still does not supply informative bounds on the original mean. The conditional positive lower bound is explicitly attributable to an additional performance comparison. No product was discarded, common cost imposed, or allowance selected to obtain that result.

A further extension should begin with demand identification and the review process, not a chosen target interval. A concrete demand check is to examine whether the all-seller panel supplies price variation after common market-day shocks and persistent listing differences are controlled for, with the overlap and seller-specific omitted information stated explicitly. A concrete review check is to distinguish observed batch changes from unobserved no-change reviews and to let forecast hazards and review timing respond to observed state. Neither extension is completed here. There is no claim that the present failures prove that Swappa can never be informative.
