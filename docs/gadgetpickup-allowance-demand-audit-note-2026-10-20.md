# GadgetPickup: empirical allowance calibration and demand audit

> Subsequent extension: `gadgetpickup-persistent-review-note-2026-10-20.md` evaluates survival-adjusted persistent prices, aggregate performance, the original cost targets, and coordinated markdown demand. This note retains the preceding daily-payoff audit.

Date: October 20, 2026. Audience: future research agents. Status: completed exploratory computation and source review; no statistically validated cost region. This implementation note governs interpretation of the new allowance and demand checks. It supersedes the recommendation in `gadgetpickup-performance-assumptions-note-2026-10-20.md` to use model-specific 90 or 95 percent performance as the leading application specification. That note remains the record of the conditional cost frontiers and the new-method mapping.

## Decision

Neither 90 nor 95 percent was estimated from GadgetPickup's choices or supported by a quantitative external benchmark. Treat both as sensitivity values. Professional operations, coordinated prices, and available repricing software motivate examining performance restrictions; they do not measure the permitted loss.

A new attempt to discipline the allowance jointly with unknown costs uses observed adjustment and inaction decisions. It does not specify a learning algorithm. Its simplest daily-payoff version is infeasible under the baseline fitted demand: even on observed batch-repricing dates, six of 26 phone models cannot satisfy the restrictions in November-December. This is a refutation of a joint fitted-demand/economic specification, not a statistical rejection of seller competence. Do not add an unrestricted slack or discard the six models to obtain a preferred answer.

Demand evidence is mixed. A negative price association survives richer time controls, removal of initial-price controls, and comparisons within the same listing. The magnitude is sensitive to specification and timing. Most proposed markdown prices lie outside the observed price range for that product category and week. The current evidence motivates a negative response more convincingly than the exact response used in the cost inversion. It does not yet justify a narrow causal demand set or an empirical upper regret allowance.

## Sample, units, and targets

All calculations use the corrected known-terminal-status sample, selected by `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. The fit sample has 20,096 GadgetPickup listing-days, 1,997 listings, 163 dates, and 1,565 terminal-sale labels. November-December has 6,039 listing-days, 832 listings, 50 dates from November 1 through December 22, 2023, and 556 terminal-sale labels. November 18 and 19 exposure starts are excluded because November 19 duplicates the previous snapshot; the final snapshot is not a complete exposure interval. Prior notes retain the underlying observation audit and residual concerns about missing listings and sale timing.

A category is phone model by storage by condition. A phone model is the broader `device_id` group. The receipt-equivalent unit is x=1.03a for integer ask a; displayed price is ceil(x). This is a maintained historical grid convention, not net seller payout. The four alternatives change continuing asks by -10, -5, +5, or +10 dollars. Entrant prices are unchanged; predicted spillovers to retained owned offers are included. Rivals' observed prices and the recorded market histories are held fixed. These are one-period deviations at observed histories, not a simulated alternative inventory path.

The expected-sales mean cost weights persistent nonnegative listing costs by fitted actual sale probabilities. It differs from mean cost among the phones that actually sold. The prior informative intervals concerned the former; the latter remained broad. None of the new checks changes that distinction. Acquisition expense may already be sunk. The modeled cost is an opportunity value of selling rather than retaining the phone, inclusive of the payoff convention. Recovering acquisition cost requires an additional economic link to replenishment or continuation values.

## What outside evidence can say about the allowance

A useful comparison is a randomized pricing study with a professional Amazon used-book seller. More than 5,000 books were assigned to each policy for three months, with the same repricing frequency. The study reports approximately 23 percent higher accumulated profit under one alternative. This is suggestive evidence that professional operation alone does not imply 95 percent efficiency. The comparison involves a broader policy and a different cost definition from our local Swappa deviations; it does not show that GadgetPickup fails the local restriction. See [Schlosser and Boissier, Tables 7-8](https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2018/dynamic_pricing_kdd_2018.pdf).

For any positive-profit comparison, if an alternative raises incumbent profit by fraction g, the incumbent's shortfall relative to that alternative is g/(1+g). Thus a 23 percent improvement corresponds to about a 19 percent shortfall, not 23 percent. Crucially, the experiment identifies performance relative to one feasible alternative. It does not bound losses relative to the best available policy from above. Transporting its magnitude to another seller, menu, market, or horizon would be another assumption. A distribution of gains across selected case studies is not an independently representative distribution of regret.

Other evidence has different outcomes. A Rue La La pricing experiment reports a 9.7 percent revenue improvement, with a 90 percent interval of 2.3 to 17.8 percent. Revenue improvements do not calibrate profit regret; see [Ferreira, Lee, and Simchi-Levi](https://www.hbs.edu/ris/Publication%20Files/kris%20Analytics%20for%20an%20Online%20Retailer_6ef5f3e6-48e7-4923-a2d4-607d3a3d943c.pdf). A randomized Mercari suggested-price study finds that lower suggestions increase sale likelihood and seller revenue. This supports considering a negative price response in used-goods markets, but its abstract does not supply a transferable GadgetPickup demand elasticity or efficiency bound; see [Fong, Manchanda, and Song](https://journals.sagepub.com/doi/abs/10.1177/00222437261432198).

The complete source ledger is `docs/evidence/gadgetpickup_allowance_demand_sources_20261020.json`. Its Table 7 calculation uses rounded published entries and yields a 22.83 percent profit gain; Table 8 reports 23.1 percent. Use approximately 23 percent rather than treating the rounding-level difference as a separate finding. No new evidence establishes seller algorithm adoption, and adoption is not required for the proposed economic approach.

## Attempt to infer a dollar allowance from choices

Let c_j be the unknown persistent opportunity cost of listing j. For phone model g on review date t, define the fitted conditional expected contribution under price vector k as

    Pi_gt^k(c) = sum_{j in g,t} q_jt^k (x_jt^k-c_j).

Let 0 denote actual prices, old denote retaining the previous fresh daily quotes, and k one of the four simple alternatives. Define

    H_gt(c) = Pi_gt^0(c)-Pi_gt^old(c),
    G_gtk(c) = Pi_gt^k(c)-Pi_gt^0(c).

H is the gain from an adjustment actually made. G is the gain left unexploited by a simple alternative. Suppose reviewing and changing the prices of model g costs K_g dollars, with K_g nonnegative and stable over the window. Both c and K are unknown parameters. The primary restrictions are

    change date:       K_g <= H_gt(c),
    inaction date:     G_gtk(c) <= K_g for every k,
    participation:     sum_t Pi_gt^0(c) >= 0.

Equivalently, conditional on c,

    max{0, max_inaction_dates,k G_gtk(c)} <= K_g <= min_change_dates H_gt(c).

This brackets an allowance using two different observed decisions. A seller that declines a larger gain but pays to obtain a smaller gain cannot satisfy a stable adjustment-cost explanation under the maintained payoffs. Costs need not be known in advance: feasibility and cost bounds project jointly over c and K. This is different from choosing the smallest tolerance that rationalizes the sample and then treating it as an empirically established upper bound.

The substantive assumptions are substantial and explicit:

1. On a qualifying review date the seller considers the specified simple alternatives; unchanged prices then reveal a decision, rather than absence of attention.
2. The adjustment cost is stable within a phone model and applies once per model's price-vector adjustment. It is not proportional to the number of changed listings or allowed to vary freely by date.
3. Conditional expected one-day gains describe the relevant benefit. This omits the benefit of keeping an improved price for several days. K therefore cannot be interpreted automatically as an estimated staff expense.
4. The researcher has the relevant conditional demand and payoff state. Private signals, negotiations, inventory objectives, changing continuation values, and omitted markets can violate this restriction.
5. Costs are nonnegative and stable within listings, and aggregate contribution is nonnegative within each model over the window. No common cost or arbitrary upper cap is imposed.

Two review schedules were specified: every retained date, and dates with at least ten changed GadgetPickup quotes across at least three phone models. There are 35 batch dates over the full sample, 17 in November-December. The latter schedule uses observed behavior as a review proxy, but it still assumes that each modeled product was considered on those dates. A model-date qualifies only if every retained offer in that model has a consecutive fresh prior quote. Entry and gap dates are excluded from the behavioral inequalities; contribution and target weights still use the full window.

The primary restriction bounds ignored gains only on inaction dates. It does not control regret on adjustment dates. A separately specified stronger version requires actual adjusted prices to perform at least as well as the four other adjusted vectors on change dates. The common adjustment cost then cancels. Under that stronger version, restricted cumulative regret against the four shifts over eligible review dates is at most sum_g n_g^I K_g, where n_g^I is the number of eligible inaction dates. There is no such bound for omitted or unreviewed dates without an additional argument. This is a conditional finite-window inequality, not an ANR learning-rate result.

### Results at fitted demand

| Window | Review proxy | Infeasible models, primary | Infeasible models, stronger menu |
|---|---|---:|---:|
| Full sample | Every retained date | 25 / 27 | 25 / 27 |
| Full sample | Observed batch dates | 12 / 27 | 20 / 27 |
| November-December | Every retained date | 13 / 26 | 15 / 26 |
| November-December | Observed batch dates | 6 / 26 | 12 / 26 |

All models have both eligible adjustment and inaction dates. The six November-December batch failures are iPhone 12 mini, 12 Pro, 6s Plus, 8, X, and XS. All six remain infeasible after removing the aggregate participation restriction. For iPhone 12 mini, just inaction on December 7 against the $10 markdown and adjustment on December 8 already create a contradiction at fitted demand and nonnegative costs. This example illustrates the source of the restriction; it is not a claim that either day alone proves irrationality.

The feasible models give nontrivial conditional ranges for K and costs, recorded in the full result JSON. They are not a selected subsample estimate of the seller's competence. No seller-wide calibrated percentage or new headline cost interval is reported. Numerical infeasibility is verified independently, including nonnegative weighted combinations of the inequalities that yield a contradiction. Some normalized contradiction margins are small. Demand uncertainty could restore feasibility; no statistical significance is attached to these failures.

The next economically motivated revision would value an adjustment over the time it is expected to persist, with the sale/retention decision and review timing modeled explicitly. This is motivated by how price changes work, not by the desire to remove particular failed inequalities. It must be specified before evaluating its cost results. Freely varying K by date would eliminate much of the identifying content and is not an empirical calibration.

## Demand: what survives and what does not

The baseline and new nonlinear models use the same corrected sample and five whole-listing folds. Category-week models add interactions of model/storage/condition with week. C=10 reduces shrinkage relative to C=1. The no-initial model drops initial-price residual and reference controls. All fifteen new fold fits converge without convergence warnings. This is retrospective validation across listings, not out-of-time prediction or proof that prices are conditionally exogenous.

| Specification | Predicted Nov-Dec sales | Predicted mean receipt | Extra sales from $10 ask reduction | Log loss |
|---|---:|---:|---:|---:|
| Baseline additive | 572.31 | 407.58 | 135.19 | 0.25538 |
| Category-week, C=1 | 690.75 | 406.93 | 156.50 | 0.26524 |
| Category-week, C=10 | 778.07 | 413.82 | 212.48 | 0.31931 |
| Drop initial-price controls | 572.26 | 408.70 | 124.19 | 0.25575 |
| Recorded outcomes | 556 | 386.49 | Not observed | Not applicable |

The sign does not depend on the initial-price controls. Adding category-specific weekly effects does not eliminate it either, but those nonlinear fits predict too many sales and perform worse on the predictive criterion. Selecting the largest response because it produces tighter cost bounds would be unjustified. The baseline's aggregate count is close to the observed count, while it assigns sales to higher-price offers than the data do. Aggregate count calibration alone is inadequate validation of price sensitivity.

Exact fixed-effect linear probability regressions offer a separate check. They use the baseline's continuous regressors and absorb the listed effects without the nonlinear shrinkage. The contrast applies the same joint $10 ask reduction to November-December offers.

| Absorbed effects | Extra sales, linear contrast | Two-way clustered standard error |
|---|---:|---:|
| Category and date | 113.73 | 13.24 |
| Category-by-week and date | 126.79 | 26.77 |
| Listing and date | 76.67 | 14.07 |

Clustering is by category and date. These are descriptive regression contrasts; linear probabilities need not lie between zero and one. They are not alternative structural probability models for direct cost inversion. Listing effects remove persistent listing differences, but not changing buyer interest, selective survival, negotiations, or private seller information. The smaller within-listing response is evidence that the magnitude deserves further examination, not a causal estimate of the amount of bias in the baseline. Ordinary regression intervals cannot be promoted to a confidence set for the causal response.

### Observed price support

There are 5,305 continuing listing-days in November-December. For the $10 markdown, 82.62 percent of proposed prices are below the observed minimum for that category in the same week. These outside-envelope observations account for 86.40 percent of the fitted positive sales gain. For the $5 markdown, 74.48 percent of proposed prices are outside the envelope. About 31.10 percent of continuing observations belong to category-weeks with only one observed price.

This is an envelope diagnostic, not a formal conditional overlap test. Pooling across weeks and related products can identify a parametric response under stable demand restrictions; the absence of a same-week comparison does not logically invalidate all such models. It does show that much of the 135-sale calculation relies on transferring a response beyond locally observed price variation. Positive aggregate predictive performance does not independently validate that transfer.

### Price timing on a common sample

The timing audit keeps observations with a consecutive fresh previous quote. Its November-December sample has 5,115 listing-days and 435 sales. Current and lagged price models are compared on this same sample. Lagged variants reconstruct owned-peer prices using previous quotes where available, hold current market controls fixed, and retain current prices for owned offers without a usable lag.

| Regressors | Category-week and date contrast | Listing and date contrast |
|---|---:|---:|
| Current price vector | 105.83 | 67.06 |
| Previous price vector | 69.01 | 59.03 |
| Current vector plus previous own-price residual | 111.89 | 80.47 |

The lagged-price contrast is a change in a lagged predictor, not the causal effect of a current markdown. Prior prices are not presumed exogenous. The attenuation from 105.83 to 69.01 is a timing diagnostic; it must not be described as a comparison of 135.19 and 69.01 on the same observations. Including the lag as a control does not eliminate the current association.

Unadjusted sale rates are 7.76 percent after unchanged prices, 10.82 percent after coordinated category cuts, 13.68 percent after other cuts, and 26.92 percent after increases. The increase group contains only 26 observations and seven sales. Both directions of repricing are associated with more selling activity, which is suggestive of selection into repricing; those rates do not isolate a causal effect.

Swappa documents a mechanism that can create simultaneity: a buyer makes an offer, the seller edits the posted price, then the buyer purchases. Its current FAQ explicitly excludes Enterprise Seller listings from this process. This is a possible mechanism, not an estimate of its prevalence for GadgetPickup in 2023. The historical GadgetPickup comment previously recovered shows an offer refusal, not an accepted negotiated sale. See [the official FAQ](https://swappa.com/faq/answer/how-to-accept-an-offer-lower-price).

A fixed-prediction calendar-week bootstrap also records count and receipt calibration discrepancies. It uses only eight blocks, does not refit demand, and shares fitted week controls across listings. Its intervals are descriptive conditional-on-fit diagnostics, not independent validation or structural uncertainty bounds. They are preserved in the JSON but do not support a formal rejection claim.

## Implications for cost and the new method

For a candidate cost vector c and demand specification q, the previous calculation determines the performance loss necessary to rationalize that candidate. It does not establish an upper bound on the seller's true loss. Choosing that upper bound to obtain a preferred cost interval would reverse the intended inference.

The allowance and demand response matter separately. In the previous results, imposing 95 percent performance separately for each phone model gives expected-sales mean cost [183.96,407.51] at baseline demand. Halving the counterfactual log-odds response gives a lower endpoint of 61.71 under the same performance restriction. The half-response case is a sensitivity calculation, not a newly validated demand estimate. The current within-listing and timing checks explain why response uncertainty deserves economic attention; they do not uniquely select a scaling factor.

The new method can accommodate an explicitly stated finite-performance restriction without knowing the seller's algorithm. It still needs a credible payoff/demand environment, a consistency set for the observed distribution, and a finite-window performance restriction whose magnitude is economically defended. Restricted shifts are action-dependent comparisons. External ANR alone does not supply their finite-window guarantee. A conditional coverage proof requires the true environment and empirical-distribution witness to satisfy those restrictions on their stated probability events. A 5 percent profit tolerance is not a 5 percent error probability.

If demand is uncertain, candidate costs should be retained when some admissible demand model and witness satisfy the restrictions. Projection over that uncertainty enlarges the set; intersecting estimates from alternative demand fits would impose all fitted models simultaneously. The current collection of fits is not a confidence set and must not be labeled one. The adjustment-cost failures apply at the fixed baseline fit and do not rule out a joint demand-and-behavior explanation.

## Reproduction, verification, and next work

The producing scripts, designs, and results are in the code repository. Detailed parquets remain ignored under `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/gadget_demand_20261020/`.

| Script in python/swappa_rescue | Producing commit | Result stem in docs/evidence |
|---|---|---|
| gadget_demand_credibility.py | c0c090f | gadgetpickup_demand_credibility_results_20261020.json |
| gadget_demand_price_timing.py | a5219e0 | gadgetpickup_demand_timing_results_20261020.json |
| gadget_adjustment_cost_calibration.py | fd27d4a | gadgetpickup_adjustment_cost_results_20261020.json |
| gadget_allowance_audit_checks.py | 5d3c96c | gadgetpickup_allowance_audit_checks_20261020.json |

Use the research Python with `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. The shared design `gadgetpickup_empirical_allowance_demand_design_20261020.json` was subsequently amended to document the stronger adjustment-menu variant. The demand run used its c0c090f version; validate that historical design hash rather than comparing to the amended file. Timing has its separate `gadgetpickup_demand_timing_design_20261020.json`. Designs were committed before their respective new computations, after the earlier research results were known; this is not preregistration of the overall project.

Numerical checks independently reconstruct all 26 November-December batch systems, reproduce all feasibility classifications, verify the six contradiction certificates, and compare coefficient inequalities to direct profits. Maximum profit-identity discrepancy is 1.71e-13. Removing participation leaves all six failures intact. The adjustment solver's first attempted run encountered an indeterminate numerical status; scaling and fallback solvers fixed computation without altering economic restrictions. No inference was drawn from the failed run. Checks and source/input hashes are preserved in `docs/evidence/gadgetpickup_allowance_demand_manifest_20261020.json`.

Next executable work should proceed in this order:

1. Define review episodes and a persistent-price payoff horizon from observed pricing spells, inventory, and terminal outcomes. Specify the treatment of sale-induced stopping and omitted dates before constructing an allowance. Analyze both observed batch reviews and a weaker uncertain-review interpretation without choosing dates by resulting bounds.
2. Audit demand around routine batch changes using contemporaneous comparisons and explicit leads, separating calendar shocks, product aging, and buyer-initiated adjustments as far as observed data permit. The existing price lags are diagnostics, not instruments. Report overlap and sale-label uncertainty before using any event comparison as causal evidence.
3. Form an economically justified range of demand responses, including weaker responses when supported, and project jointly over costs, adjustment/attention parameters, and that range. Preserve failures rather than adding an uncalibrated residual tolerance.
4. Only after a quantitative performance bound survives those checks, implement the consistency component and the full witness projection. Report expected-sales and realized-sales targets separately, with receipt-equivalent and acquisition-cost interpretations distinguished.

These are research proposals, not completed identification results. The remaining task is economic calibration and demand validation. A named proprietary learner is neither necessary nor the requested next input.
