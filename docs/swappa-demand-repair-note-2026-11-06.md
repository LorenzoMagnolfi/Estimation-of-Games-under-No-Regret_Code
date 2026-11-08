# SWAPPA demand: an implementable repair

Date: 6 November 2026. Audience: Lorenzo and the revival agent. Status: research proposal with a local diagnostic pilot; empirical results are recorded separately below as they become verified. The consolidated `swappa-revival-agent-memo-2026-11-06.md` now governs priorities and includes completed downstream costs and fee robustness. This supporting note preserves earlier diagnostics; its broader proposed work is optional unless the consolidated memo says otherwise. It does not replace the original design log or approve a manuscript change.

## Recommendation

Retain a listing-level sale hazard, repair its observation and conditioning assumptions, and make the focal seller's unilateral portfolio-price response the primary demand object. Use flexible relative-price controls and product-specific calendar effects to assess robustness. Keep a market-wide price response as an additional sensitivity because the available variation identifies it less credibly. The main fee exercise should initially hold rivals fixed and condition on the observed inventory distribution.

The available data can materially improve the demand specification and discipline sensitivity. They do not yet supply a verified instrument or experiment. A causal interpretation needs either conditional price exogeneity or a separately justified design; a good predictive score does not establish it. The final application can still be useful if the assumptions and the consequences of weakening them are explicit.

## Evidence already available

The revival source is paper commit `16be52968a0776f0aec086a2bbeabf5b8f6abfcb`, folder `JPE Revision/swappa-revival-2026-11-03/`. The logit uses SKU, date, and seller-group effects; a log own/reference price; a kink at the cheapest rival; owned-peer prices; age, battery, counts, and entry-price controls. Its reported sample has 457,189 listing-days, 22,792 terminal-sale labels, and 164 dates. Its price-response sensitivities move GadgetPickup's Lerner interval from 9.9--15.7 percent to 23.3--33.7 percent when the fitted response is halved.

Earlier work is not a clean causal benchmark. `docs/gadgetpickup-persistent-review-note-2026-10-20.md` reports 29 coordinated-markdown events. The sale-rate change is 3.49 percentage points relative to day -1 but 1.60 points (SE 1.16) relative to days -7 through -2. Listing age, rival prices, and subsequent control repricing change around events. A new event study must address these problems rather than reuse the day -1 contrast as an elasticity.

The public-data provenance is established in `docs/DATA_AVAILABILITY_STATEMENT.md`. These are author-collected public listings, not institutional restricted or licensed records. No raw rows should be published with this note.

## D0. Clarify the estimand before selecting a model

For listing j of seller i at date d, define q(j,d;p_i,p_-i) as the probability of a terminal sale during the next verified observation interval, conditional on being active at its start and on the modeled history. The primary contrast scales all offers of i by 1+k, updates owned-peer prices, and keeps rival prices fixed. Its aggregate is the change in expected daily sales over a fixed, explicitly named inventory set.

Report this contrast for k in +/-0.5, 1, 2.5, and 5 percent. Report both continuous and feasible rounded price maps. Report the sales-weighted derivative near zero and the finite-change contrast; the kink and rank changes can make them differ. Do not use a global LPM coefficient or a cross-sectional bin slope as a direct estimate of this portfolio contrast.

Separate three claims:

1. The hazard predicts terminal sales on held-out listings/dates.
2. Its counterfactual price response is credible conditional on the covariates and pricing-information assumption.
3. Its response can be transported to a different fee/inventory regime.

A successful prediction exercise supports the first claim. It is useful evidence for specification but does not by itself establish the other two.

## D1. Observation and risk-set repair: required for the main specification

### A single terminal sale is compatible with a hazard model

A duration likelihood is built from non-sale exposures followed by a sale or censoring. A random spell length ending at sale is part of that model, not a reason to discard it. The issue is whether the risk set, exposure, covariates, censoring, and unobserved heterogeneity are modeled correctly. A conditional fixed-effects binary logit applied as if this were an ordinary balanced panel is not automatically a valid solution to endogenous stopping and heterogeneity.

Use the official [Stata discrete-time survival discussion](https://www.stata.com/manuals/stdiscrete.pdf) as a concise reference for logit/cloglog hazards, censored spells, and late entry. It also warns that frailty with late entry needs a likelihood that conditions appropriately on entry.

### Build one observation ledger

Produce one row per listing interval, containing the snapshot start/end, known active status, quote used, first appearance, verified creation date if valid, disappearance/reappearance flags, terminal sold status, and terminal-label confidence. Compute rivals before any outcome/reference filtering.

- Drop the copied November 19 snapshot. Preserve November 18 to 20 as an interval, not a second independent day.
- Count every listing-level gap and return. A missing listing that later returns is not a sale; a gap does not reveal the unobserved price path.
- Report the final-date censoring rule and unknown/late terminal statuses separately.
- Keep the stock present on July 11 separate from observed entry cohorts. First-seen age is not true listing age for the opening stock.
- Use actual interval lengths. Under a constant within-interval hazard rate h=exp(eta), q=1-exp(-Delta*h), so a cloglog model has offset log(Delta). This construction requires a maintained within-interval covariate path; it does not justify treating an unobserved long gap as constant exposure.
- Use daily complete-history and entry-cohort restrictions as sensitivity samples. Selecting complete histories uses future observation information and may itself select listings; do not call that restriction an unbiased correction.

**Checks:** one terminal sale per code; no modeled sale before an observed return; rival summaries verified against raw offers; source-to-risk-set counts; current/previous-quote comparisons on identical rows; alternative conservative terminal-label definitions. Investigate differences before fitting a more elaborate demand model.

## D2. Main candidate: nonlinear hazard with richer competition and product-time controls

Start from the revival logit rather than replacing it with a large new demand system. The recommended specification ladder is:

| Specification | Change | Purpose |
|---|---|---|
| B0 | Revival-like logit on reconciled observations | Reproduction anchor |
| B1 | Add rival mean/median relative to reference and rival log-price dispersion | Test whether the minimum sufficiently summarizes competition |
| B2 | B1 plus SKU-by-week and date effects | Absorb product-specific depreciation and demand shifts |
| B3 | B2 with flexible relative-price splines or prespecified bins | Check the imposed kink and tail curvature |
| B4 | B2 with seller-specific price-response interactions for sufficiently active sellers | Assess pooling of demand slopes |
| B5 | Cloglog version of B2 with correct exposure, and a limited heterogeneity model | Assess link, duration, and survivor-selection assumptions |

Rich controls should reflect the actual substitution set: exclude all own listings from rivals; distinguish own count from rival count; consider rank and the number of cheaper rival offers; update every price-dependent control under the deviation. Quantity/availability/rank may be endogenous and require the same conditioning argument; they are not automatically instruments.

Use the same observations and the same held-out listings when comparing B0--B3. Compare common-sample contrasts separately from changes caused by censoring or entry-cohort restrictions. Report local overlap of proposed prices in SKU-time and seller/product cells. Do not discard a specification for yielding wide or empty cost sets.

For validation, use listing-disjoint folds and blocked forward validation with a model that can predict new dates. Date fixed effects learned from the test dates cannot validate a prospective forecast. Forward validation can use smooth calendar controls or product-age effects, estimated in training, and should report new-SKU support. Include held-out sales counts, revenue calibration, sale rates by relative price/age, and cumulative incidence over 1/3/7 days. Calibrate a multi-day forecast by survival probabilities rather than summing hazards as if one phone could sell repeatedly.

## D3. Separate unilateral substitution from the common market-price response

The own/reference price term in the revival combines relative-price and market-level movements. Algebraically,

    log(p_j/r_gd) = log(p_j/P_-i,gd) + log(P_-i,gd/r_gd).

The first term measures position relative to rivals. The second moves when the market becomes expensive relative to the external reference. Strong product-time effects can absorb much of the second variation; that is informative about the limits of the data, not a reason to omit the controls.

A unilateral fee response uses the first channel plus changes in the seller's own portfolio. A scenario where all sellers match also needs the market-level response and potentially the outside option. Market size/traffic is not directly observed here. State the invariance assumptions rather than interpreting both channels as equally identified.

**Main:** conditional unilateral changes with rivals fixed. **Robustness:** rivals-follow scenarios across an explicit range of market-level responses. Do not label these equilibrium bounds. The centered internal reference is an ex post measurement index, not a predetermined information variable or an instrument. Compare it with a trailing index and external-only quotes; preserve the distinction between a normalization and a causal demand shifter.

## D4. The most useful design for stronger causal evidence

Revisit coordinated repricing only with an explicit timing and comparison design. The October pre-cut dip makes an event-time comparison with day -1 unreliable.

1. Define events from price changes alone, before reading post-event sales: a seller-wide or multi-SKU batch, with a fixed minimum number/fraction of continuing offers changed.
2. Fix the at-risk inventory at the last pre-event snapshot. Follow every member for cumulative sale within 1/3/7 days, including those that disappear; account separately for uncertain exits. Never condition on surviving into the post-event period.
3. Construct controls from untreated offers in the same SKU-time environment and, where possible, untreated SKUs within the same seller. Report how much treatment variation remains after seller-day and product-day controls; do not force a high-dimensional design when it has no effective variation.
4. Show at least seven days of leads, use an earlier multi-day baseline, and report rival-price, inventory, quality, and listing-age changes. Keep subsequent repricing in the primary intent-to-treat definition and report it; do not remove controls because of future decisions.
5. Estimate the price first stage and sales reduced form separately. Convert to a local elasticity only with an explicit exclusion/timing argument. Batch timing may respond to private demand information or slow sales.
6. Search for documented mechanical repricing rules, schedules, or administrative discontinuities. A verified implementation rule can strengthen the design. Do not infer one from visually regular prices alone.

This is initially a robustness/identification diagnostic. Promote it to main causal evidence only if the comparison and timing assumptions have support. Rival prices, rival exits, the iPhone launch, and Decluttr price movements have direct demand channels and are not automatic excluded instruments. Shipping-price shocks require actual data and a verified exclusion argument; no such shock is established in this corpus.

## D5. Demand sensitivity should be an output of the application

Retain a family of demand responses justified by model comparisons, support, and event diagnostics. For every response, re-estimate or explicitly calibrate factual demand and then recompute the structural moments and fee outcomes. Scaling only the change in the index while holding the factual fit fixed is a useful sensitivity intervention, but it is not an estimated alternative demand model.

Keep two axes separate: residual price endogeneity and competition/functional-form specification. An attenuation scalar on every price term need not capture uncertainty in own substitution and market-wide demand. Use a two-dimensional sensitivity if retaining the rivals-follow scenario.

For each demand scenario h, construct the appropriate cost/fee region and report their union when h is maintained to lie in the scenario class. A finite menu of fitted models is not automatically a confidence set for the true demand function. If claiming joint coverage, build a nuisance confidence region with a justified joint procedure and allocate/propagate uncertainty consistently with the repaired inference in M3.

Return a table with fitted and sampling-adjusted markups, fee-price effects, and profit effects for each model and sensitivity. State exactly which economic conclusions survive. Positive margins, positive pass-through, and a numerical claim that buyers bear most of the fee are different conclusions.

## Work performed in this review

The new standalone pilot is `python/swappa_rescue/demand_repair_pilot_20261106.py`. It reads the existing full-source public panel, leaving both the raw data and revival code unchanged. Outputs are under `output/swappa_demand_repair_20261106/`; `manifest.json` records source hashes, script hash, software versions, and the baseline repository commit. Models are exploratory predictive comparisons, not causal estimates or new confidence regions.

The pilot was designed after inspecting the revival estimates and the October diagnostics. It is not a preregistered analysis. Its seven specifications compare the baseline, richer competition, SKU-week effects, complete daily histories, current/previous quotes on a matched sample, and a flexible price-gap function. Each fit has a listing-disjoint holdout and the same portfolio-price contrasts. The complete-history restriction is a selection sensitivity; the previous-price comparison is not an instrument.

### Verified results of the pilot

We completed seven specifications using the existing Stata-derived reference series and four additional specifications using reference prices reconstructed from all 166 native Decluttr workbooks. Each has a full-sample fit and one independently held-out-listing fit, for 22 fitted models. This is one prespecified random split within this exploratory pilot, not five-fold validation or forward validation. All optimizations passed the score check. No cost region or fee confidence interval was recomputed.

**Risk set.** The reconstructed sample matches the revival counts exactly: 457,189 rows, 34,079 listings, 22,792 terminal labels, and 164 dates. There are 11,891 intermediate non-daily listing gaps across 5,665 histories, excluding the known copied-date gap. A complete-daily-history restriction, also excluding the November 18 two-day interval, retains 279,225 rows and 19,979 sales. This is a selection sensitivity, not evidence that the excluded rows are invalid.

For GadgetPickup it retains 18,500 of 21,012 rows (88.0 percent) and 1,551 of 1,638 sales. For We Buy Mobile it retains 3,263 of 14,276 rows (22.9 percent) and 196 of 336 sales. The removed We Buy Mobile rows have higher mean displayed prices (361.80 versus 319.33 dollars) and longer first-seen ages (54.55 versus 15.87 days). Observation gaps are economically patterned. This raises the priority of the observation model for We Buy Mobile; it is not a reason to drop the seller after seeing its estimates.

**Reference reconstruction.** The native-workbook audit found 55,783 usable records whose names end in UNLOCKED, versus 46,867 reported in the revival's text-extraction build. The network-field restriction instead gives 54,288, exactly the old Stata intermediate's count. Name and network restrictions disagree on 1,495 usable rows. The new name-based, named-column reconstruction changes 13.43 percent of the analysis reference values relative to the existing panel (including small changes in the fallback normalization).

The raw-reference baseline yields coefficients (-4.871, -4.814, 0.501, 0.706) on own/reference price, the positive and negative rival-minimum gaps, and the owned-peer term. These are close to the revival's (-4.798, -4.865, 0.593, 0.676), but are not an exact replication. The audit records 269 screened SKU-date quotes and 23,380 retained quoted SKU-dates, versus 99 and 19,533 in the branch build. The extracted-text payload is not present locally, so the precise loss/transformation in that route remains unresolved. The revival agent should compare native workbook rows against its text extraction before integrating either reconstruction. Do not silently promote the new reconstruction to the canonical cost reference.

**Common-sample specification comparisons, native-workbook references.** The table gives the percentage change in total predicted sales over the fixed observed inventory exposures after a uniform 5 percent price cut by the named seller, with rival prices fixed and owned-peer prices adjusted. These are fitted counterfactual contrasts, not causal estimates. All rows use the same observations and holdout assignment.

| Model | GadgetPickup sales change | We Buy Mobile sales change | Holdout log loss |
|---|---:|---:|---:|
| Revival-like baseline (`raw_d0`) | +38.58% | +42.77% | 0.168206 |
| Richer rival summaries (`raw_d1`) | +40.22% | +44.48% | 0.167854 |
| Richer rivals plus SKU-week effects (`raw_d2`) | +41.60% | +45.69% | 0.168240 |
| Richer rivals plus flexible price-gap curve (`raw_d6`) | +61.60% | +71.11% | 0.165867 |

The extra rival summaries modestly improve prediction and leave the response similar. Product-week effects do not reduce the response, although their aggregate holdout log loss is slightly worse; the improvement in in-sample fit alone does not select that model. Flexible curvature improves this holdout score and materially steepens the response. It is a candidate for further validation, not a license to report the largest elasticity.

Focal-seller calibration remains imperfect. Under the raw-reference baseline the held-out sales are 330 and 72 for GadgetPickup and We Buy Mobile, against predictions of 297.53 and 59.33. The flexible model predicts 297.40 and 58.44, so its aggregate predictive improvement does not resolve focal-seller calibration. Product-week effects give 310.16 and 61.73. These figures motivate seller heterogeneity and blocked validation rather than an automatic choice of the flexible model.

**Timing comparison, existing-panel references.** On exactly 395,393 matched rows, replacing current own prices and owned-peer quotes with previous quotes changes GadgetPickup's predicted response to a 5 percent cut from +35.13 to +30.28 percent and We Buy Mobile's from +38.71 to +32.86 percent. The finite-change response falls by 13.8 and 15.1 percent respectively. Rival state remains contemporaneous, and a fresh complete prior owned-peer quote is required. This is a timing sensitivity, not an instrument. Its magnitude does not support treating 0.08 as a directly established local response multiplier.

**Complete-history comparison, existing-panel references.** The rich model on the selected complete-history sample gives +36.83 and +34.43 percent, against +35.34 and +38.74 on the full sample. Both estimation and target composition change, so these numbers cannot isolate a measurement-bias correction. Their holdout losses are not comparable across the different samples.

**Checks and evidence.** `check_demand_repair_20261106.py` imports the original branch's demand design and verifies the independently standardized baseline predictor to 2.2e-14 and the original-design score to 1.5e-10 on the same inputs. It checks all 22 fit scores, the identity deviation, matching rows/outcomes in the timing comparison, and local signs of portfolio-price responses. It does not establish causal identification or confidence coverage. The checked compact evidence is `docs/evidence/swappa_demand_repair_results_20261106.json`; detailed outputs and coefficients remain under `output/swappa_demand_repair_20261106/`. Script hashes are recorded in the evidence, including a later guard that refuses rounded deviations until a full-inventory ask/display-price mapping is supplied. No rounded-deviation results are claimed.

### Immediate next experiment

Reconcile the reference series first, then run a crossed comparison of rich competition, product-time controls, and flexible shape, with prespecified seller-slope heterogeneity and several held-out-listing/blocked-time splits. The present pilot varied these separately, so it has not established that the flexible-shape improvement survives rich product-time effects. Report common-sample local contrasts and focal-seller calibration. Keep the existing timing comparison and complete-history sensitivity. If nonlinear shape remains important, propagate a demand family through the cost and fee calculations; do not replace that family with an arbitrary scalar attenuation alone.

## Main versus robustness: final decision rule

The main specification requires a reconciled observation model, a stable nonlinear fit, an explicit conditional-exogeneity statement, and end-to-end inference. B1/B2 are candidates, not preselected winners. A richer model that improves prediction but leaves economic contrasts unstable should widen the reported demand uncertainty rather than silently replace the original model.

Minimum/mean/dispersion controls, product-time controls, feasible deviations, and common-sample timing comparisons are high priority. Cloglog, entry cohorts, measured quality, and seller slope heterogeneity are targeted robustness. A full consumer-choice model with estimated arrival intensity, a dynamic inventory model, and new external instruments are larger extensions; they are not necessary to complete a conditional unilateral-fee application.

Background on the distinction between demand fit and identification: Berry and Haile, [Foundations of Demand Estimation](https://www.nber.org/papers/w29305), discusses models, data requirements, instruments, and nonparametric identification. The recommendations here apply that distinction to the observed Swappa listing panel; they are not a claim that these data satisfy a theorem from that chapter.
