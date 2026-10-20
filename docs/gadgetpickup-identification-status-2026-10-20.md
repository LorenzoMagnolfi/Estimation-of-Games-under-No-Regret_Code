# GadgetPickup: corrected diagnostics and identification status

Date: October 20, 2026. Audience: Lorenzo and future research agents. Status: available-data investigation at an identification impasse. Informative, empirically defended cost and markup bounds have not been obtained. The research goal is not complete.

## What the correction changes

The source audit established that November 19 repeats November 18. The new diagnostic restricts estimation to dates followed by a fresh next-day snapshot. It does not impute missing prices, move sales across dates, or assume that a two-day interval is a one-day exposure. The retained GadgetPickup sample has 20,096 observations, 1,565 terminal-sale labels, 1,997 listings, and 163 dates.

The design was fixed at `b867ff2`; `python/swappa_rescue/gadget_observation_correction.py` ran at `a583e93`. It uses the previously specified additive demand model and the same five whole-listing folds. All folds converge without a convergence warning. The comparison below evaluates the cached old fit and the refitted model on exactly the same retained November--December rows: 6,039 exposures and 556 sale labels.

| Diagnostic | Old fit on retained rows | Refit on retained rows |
|---|---:|---:|
| Predicted sales | 576.98 | 572.31 |
| Recorded sales | 556 | 556 |
| Predicted mean sale receipt | $409.20 | $407.58 |
| Recorded mean sale receipt | $386.49 | $386.49 |
| Log loss | .255312 | .255383 |
| Additional fitted sales under a joint $10 price cut | 128.88 | 130.51 |

Receipt is the existing receipt-equivalent convention, not verified net payout. The joint-price calculation changes continuation prices and their own-peer terms together. It is a fitted response, not a causal estimate or a simulated path of future inventories. The correction is necessary but does not remove the prediction discrepancy or identify a counterfactual demand response.

Over the full retained period, predicted sales are 1,614.16 versus 1,565 recorded, and predicted mean sale receipt is $404.48 versus $405.09 recorded. Better full-period revenue composition does not resolve the later-period discrepancy. Listing-disjoint validation also permits common calendar shocks across folds; it is not a new independent time holdout.

## Pagination matters, but does not supply an instrument

The audit flags an offer when its raw device-date-exact-price group spans a 50-row page boundary. There are 4,949 flagged rows in the corrected GadgetPickup sample. This flag indicates potentially unstable coverage, not a proven error for that listing. All eligible observations remain in the fit.

Among consecutive appearances of a listing after removing the copied date, temporary gaps occur in 78 of 4,697 transitions following a flagged offer (1.66%), compared with 48 of 14,376 following an unflagged offer (.334%). Thus gaps are about five times as frequent near these boundaries. Ninety-eight of the 126 gap transitions have the same price at both observed ends. The gap calculation conditions on eventual reappearance and does not identify whether a phone was actually withdrawn in between. It also cannot recover offers never captured in the sample.

The corrected November--December predictions show receipt-composition discrepancies in both groups. For flagged rows, predicted mean sale receipt is $386.87 versus $365.97 recorded; for the other rows it is $414.76 versus $394.01. Accordingly, removing all flagged rows would not by itself eliminate the current model's composition discrepancy. Such removal would also change the selected product and price distribution.

The independent source evidence is consistent with pagination-related missingness. It does not establish that the researcher's captured row order is the position seen persistently by buyers. A rank instrument would require that additional exposure mechanism and an exclusion argument. Rival prices also directly affect a buyer's alternatives and are not excluded instruments for the seller's own price without further structure.

## Corrected sales-information calculation

We repeated the previous full-history information calculation on the same one-day calendar restriction. The supplemental design and driver were committed at `e513270`. `gadget_observation_feedback.py` supplies corrected input and a new output directory to the unchanged mixture-bound calculation; no earlier artifact is overwritten.

There are 4,572 evaluated category-days across 158 categories. Under both observation models (daily sales fractions and the stronger individual-exposure model), and with either exact-price bands or the additional monotone-demand restriction, there are no empty confidence-set intersections and no category-days on which the weak no-demonstrated-inferiority comparison excludes zero cost. This replaces the numerical information diagnostic that included the copied date.

The statistical guarantee remains conditional on constant conditional arm means, predictable observation and action selection, and valid timely feedback. It is not an empirical finding that the seller has those beliefs, uses those bounds, or obeys the proposed choice restriction. Nor does nonrejection of zero cost in these comparisons establish its membership in the full new-method set, which also needs consistency restrictions and joint treatment of demand uncertainty.

## Why the public pricing practices do not yet yield a finite guarantee

The remaining obstacle is a quantitative behavioral restriction. A known software name is not logically necessary. For example, a justified requirement that the seller performs at least as well as a feasible benchmark learner could support a bound without requiring literal adoption of that learner. But benchmark dominance, its information set, and the conditions of its performance theorem would still be additional assumptions. The available pricing records and public sources do not calibrate them.

Publicly motivated rule families do not automatically have sublinear regret. Consider a competitor-following rule that repeatedly sets a price p_L while another feasible price p_H exceeds p_L. Let the sale probability be the same positive q at both prices and let cost be below p_L. This demand is weakly decreasing in price. The regret relative to p_H after T dates is T q (p_H-p_L), which is linear in T. A rule that repeatedly marks down to a floor can have the same problem once it reaches that floor. Adding smooth strictly decreasing demand does not remove the counterexample if the higher price still earns more profit. These examples do not assert that GadgetPickup follows either rule. They show why evidence that a rule is used, or available commercially, would not alone establish the guarantee the application needs.

Unrestricted asymptotic no regret also does not place a numerical bound on a particular finite prefix. For any fixed finite observed history and finite candidate cost vector with bounded action payoffs, a learner can reproduce that history and subsequently start a standard no-regret procedure. The finite initial loss disappears after division by a growing horizon. Thus the unknown cost is not the objection: the missing finite performance restriction is substantive even if cost were known to the researcher.

The previously derived uncertainty-based guarantee is a valid sufficient construction under its maintained information and choice restrictions. The corrected calculation shows that its weaker behavioral version is not informative about a positive cost floor in the comparison exercise we can currently support. Replacing it with exact optimization or a tighter unexplained allowance would change the economic model, not extract a result already established by the data.

## Audit against the research objective

| Requirement | Authoritative evidence | Assessment |
|---|---|---|
| Diagnose the demand problem | Demand audit, chronological and listing-disjoint comparisons; corrected results JSON | Substantial diagnosis completed; calibration and causal interpretation remain unresolved |
| Ground additional assumptions in data or reality | Empirical-foundations ledger, dated seller comment, procurement sources, inventory and observation audits | Supports coordinated pricing, repeated procurement, and certain observable information; does not establish a quantitative performance restriction |
| Derive a finite theoretical guarantee | Demand-regret note's simultaneous-band proof and choice implications; published learner guarantees reviewed there | Conditional guarantees derived; their behavioral/information premises are not substantiated for this seller |
| Obtain informative credible cost and markup bounds using the new method | Earlier cost frontiers are fitted-demand allowance sensitivities; corrected information comparison does not exclude zero | Not achieved; no full consistent-and-obedient confidence set with an empirically defended finite envelope |
| Preserve reproducibility and distinguish evidence from assumptions | Designs, code commits, saved predictions, manifests, and observation notes | Completed for the investigations reported here |
| Report unsupported essentials honestly | This audit and the linked source records | Essential gaps are explicit; the goal must not be marked complete |

The same demand-identification and finite-performance gaps remained through the inventory, observation, and corrected-estimation passes. The correction and information checks are now executed, and no job is running. Further tuning of this demand model, deletion of observations, or recalculation of arbitrary-allowance frontiers would not establish the missing behavioral premise. Repeating the same public vendor searches is also unlikely to change it without a new lead.

We are therefore at an impasse for the requested credible cost bounds with the available information. This is not a theorem that every possible Swappa model is uninformative. It is a limit of the evidence and maintained models that can presently be defended. Work can resume if there is new evidence about pricing review and performance, a defensible source of exogenous price variation, richer historical order/inventory records, or an explicitly justified alternative behavioral model. A particular named algorithm is one possible route, not a requirement in itself. The user has confirmed that no additional private GadgetPickup records are currently available.

## Reproduction and files

Use `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. Run `python/swappa_rescue/gadget_observation_correction.py` for the five-fold fit and pagination statistics, and `python/swappa_rescue/gadget_observation_feedback.py` for the corrected information calculation. The core model and confidence-bound routines are unchanged. The design's added feedback check was recorded after the first-stage run and before the feedback run; the original first-stage design hash remains in its result.

Tracked outputs: `docs/evidence/gadgetpickup_observation_correction_results_20261020.json` and `gadgetpickup_observation_feedback_results_20261020.json`. Detailed predictions and gap records are in the existing `gadget_demand_20261020` output directory; corrected feedback paths are under its `observed_intervals/` subdirectory. The final manifest records hashes without changing earlier producing hashes. No canonical paper, old demand object, cost interval, remote repository, or external account was changed.
