# GadgetPickup: performance assumptions and the information on cost

Date: October 20, 2026. Audience: future research agents. Status: completed conditional calculations and proposed economic specification; statistical inference remains to be implemented. This note governs the new performance exercise. It supersedes the identification-impasse conclusion in `gadgetpickup-identification-status-2026-10-20.md`, while retaining that note's observation and demand diagnostics. The user explicitly permits suggestive evidence to motivate assumptions and does not require a verified pricing algorithm.

## Decision and interpretation

An economically motivated restriction on performance can be used without knowing GadgetPickup's algorithm. The most useful candidate examined here requires that its pricing earn a substantial fraction of the profit available from four routine adjustments: lowering or raising continuation asking prices by $5 or $10. Apply this restriction over the whole evaluation window, either to the seller's portfolio or separately to each phone model. These alternatives use the same demand fit and allow every listing to have a different unknown cost.

The distinction between seller-wide and model-specific performance matters. Under the corrected fitted demand and a 5 percent permitted shortfall, the seller-wide expected-sales mean cost is in [67.29,407.58]. Requiring the same performance for each of 26 phone models gives [183.96,407.51]. A 10 percent shortfall gives [35.38,407.58] and [93.41,407.52], respectively. The extra information in the second specification comes from restricting offsets across phone models. It does not come from identifying the algorithm or imposing equal costs.

A weaker qualitative assumption was also tested: actual repricing does not systematically reduce expected profit relative to retaining the previous quote. Pooled over changes, it gives no improvement on the expected-mean interval [0,407.58]. Applying it separately to increases and decreases raises the lower endpoint only to 4.16. Thus purposeful repricing alone is insufficient for a useful overall mean in this specification.

We propose model-specific limited performance as a candidate economic restriction, with the seller-wide restriction as a weaker benchmark. We do not claim a uniquely weakest assumption or that the evidence estimates a 5 or 10 percent allowance. Report the complete frontier. The evidence motivates deliberate, coordinated product pricing and the comparison menu; the numerical degree of competence remains a maintained economic judgment. All reported sets condition on a fitted counterfactual demand model.

## Observations, units, and the target

The evaluation window contains 50 retained daily observations from November 1 through December 22, 2023, 6,039 listing-days, 832 listings, and 556 terminal-sale labels. A listing-day contributes only if it belongs to the previously corrected one-day observation sample. November 18 and 19 exposure starts are excluded because November 19 copies the preceding snapshot. The fit uses all 20,096 corrected observations, with five whole-listing folds. This is retrospective demand fitting, not a chronological holdout or a claim about information available to the seller.

The corrected fit predicts 572.31 sales and a mean receipt of 407.58, compared with a recorded mean of 386.49. The exact counterfactual calculation preserves the maintained historical asking-price grid. If a is the integer ask, displayed price is ceil(1.03 a), and the receipt-equivalent unit is x=1.03 a. A $10 ask change changes x by $10.30. The four comparisons change all retained continuing offers together and update owned-peer prices. There are 5,305 continuing listing-days. Entrant prices remain fixed, but their predicted sales and profit spillovers remain in the calculation.

Let j index listings, t retained dates, and k a price comparison. Conditional on the observed market history, q_jt^0 is the probability of a sale at the actual price vector and q_jt^k is the probability at the comparison vector. Let x_jt^k denote its receipt. Define

    Q_j^k = sum_t q_jt^k,
    R^k = sum_jt x_jt^k q_jt^k,
    Pi^k(c) = R^k - sum_j Q_j^k c_j.

Costs c_j are nonnegative, persistent within a listing, and otherwise unrestricted. They are estimated, not supplied to the calculation. The primary target is

    mu(c) = sum_j Q_j^0 c_j / Q^0,    Q^0 = sum_j Q_j^0.

This is mean opportunity cost weighted by conditional expected sales along the observed histories. It is different from the mean cost of the listings that actually sold. Expected margin per sale is R^0/Q^0 - mu(c). With the 5 percent model-specific restriction its interval is [0.07,223.61] in receipt-equivalent dollars. These are contribution margins, not net accounting profit.

The original realized-sales-weighted target remains weakly bounded. Under the same 5 percent model-specific restriction its interval is [2.79,4810.83]. Seller-wide performance gives [0,25489.67]. Reporting the expected-sales result must explicitly acknowledge the different estimand; it does not resolve inference on the original target.

The receipt convention also requires a cost interpretation. If net proceeds equal a common fraction a of x minus a listing-specific transaction expense b_j, and the economic opportunity cost is k_j, then true contribution is a[x-(k_j+b_j)/a]. Consequently the estimated c_j equals (k_j+b_j)/a, and multiplying all contributions by positive a leaves relative performance unchanged. Recovering net opportunity costs requires the applicable fees and expenses. Unknown price-dependent fees that cannot be represented this way require a different payoff model.

For stocked used phones, acquisition expense is already sunk when the daily price is chosen. A payoff proportional to (x-c_j)q can describe the value of selling instead of retaining a phone when c_j includes the value of retention and transaction expenses. It describes replacement cost under the additional economic approximation that a sale is followed by replacement and the future inventory state is restored. Repeated procurement supports considering that approximation; it does not establish immediate replacement. Persistent opportunity values over a listing's life, additive continuation losses, and the adequacy of the recorded inventory are maintained restrictions.

## A limited performance assumption

Let K contain the four changes {-10,-5,+5,+10}, measured in integer asking-price dollars, and include withholding the relevant inventory as the zero-contribution benchmark. For a specified rho in [0,1), assume

    Pi^0(c) >= (1-rho) max{0, Pi^k(c): k in K}.                 (P)

For example, rho=.05 says that over this window actual pricing earns at least 95 percent of the expected contribution available from the best of these four adjustments or withholding. This permits imperfect pricing every day. It does not require optimality over all prices, knowledge of expected demand, a particular updating formula, common costs, or independent learning across phones. The benchmark is a sum of one-period payoffs at observed histories, not the profit from a simulated alternative inventory path.

The normalization is economic. Withholding has zero incremental contribution relative to retaining the inventory. Relative performance would not be invariant to an arbitrary additive constant in utility; the payoff origin must therefore be fixed as above.

For a fixed rho and demand model, (P) is linear in costs. The exact restrictions are

    sum_j Q_j^0 c_j <= R^0,
    sum_j [(1-rho) Q_j^k - Q_j^0] c_j >= (1-rho) R^k - R^0,  k in K.

The first inequality is aggregate nonnegative contribution. It alone bounds mu by [0,R^0/Q^0], which here is [0,407.58]. The price comparisons supply additional information. For example, a discount can be profitable at low cost because it increases sales, while a larger cost makes the additional sales less valuable. That is why markdown comparisons can exclude low costs.

For any single row Bc>=A with A>0, c>=0, and positive target weights w_j=Q_j^0/Q^0,

    mu(c) >= A / max_j(B_j/w_j),

provided the denominator is positive. This follows from Bc <= max_j(B_j/w_j) sum_j w_j c_j. The programs optimize the full system, not just this certificate. No cost cap or equality across listings is imposed. The weak pooled endpoints can concentrate costs on a small number of listings, which explains why their mean bounds are broad.

## What the fitted demand contributes

At zero cost, contribution equals revenue. The corrected fit gives the following November-December counterfactual totals.

| Price change | Expected sales | Expected revenue | Revenue change |
|---|---:|---:|---:|
| Actual prices | 572.31 | 233261.02 | 0 |
| Ask minus 10 | 707.50 | 271809.32 | 38548.30 |
| Ask minus 5 | 635.78 | 251819.19 | 18558.17 |
| Ask plus 5 | 515.75 | 216007.07 | -17253.95 |
| Ask plus 10 | 466.14 | 200204.09 | -33056.93 |

The positive demand response to a markdown is supplied by the estimated model, not observed as an experiment. It is the source of the revenue gain used to exclude low costs. Purely predictive cross-validation does not establish that causal response.

For pooled performance, zero cost is excluded exactly when

    rho < max_k {1 - R^0/R^k}_+ = 0.1418211181.

Thus the fit says that zero cost entails a 14.18 percent shortfall relative to the $10 markdown. It does not say that the seller's actual shortfall is less than 14.18 percent. The comparison needs the latter economic restriction. The discount's revenue gain is 16.53 percent of current revenue; this is a different denominator from the 14.18 percent shortfall relative to the alternative.

| Permitted shortfall rho | Pooled lower mean cost | Pooled upper mean cost |
|---|---:|---:|
| 0 | 103.40 | 407.58 |
| .025 | 80.24 | 407.58 |
| .05 | 67.29 | 407.58 |
| .075 | 52.48 | 407.58 |
| .10 | 35.38 | 407.58 |
| .125 | 15.42 | 407.58 |
| .15 | 0 | 407.58 |

Using only the $10 markdown gives a lower endpoint of 91.66 at rho=0 and the same lower endpoints as all four changes at rho=.025,.05,.075,.10,.125. The $5 markdown supplies the sharper exact-performance endpoint. The upward comparison alone does not yield a positive lower mean. These results do not support presenting a rich menu of comparisons as the source of the lower bound.

## Separating performance across product groups from common costs

A product-specific version of (P) applies it separately to each phone model g:

    Pi_g^0(c) >= (1-rho) max{0, Pi_g^k(c): k in K}, for each g.

There are 26 models. This remains a pooled restriction across dates and storage/condition variants of the same model. It allows heterogeneous listing costs and periods of poor performance. It prevents expected losses on one model from being offset without limit by earnings on other models. Grouping by model is coarser than the 153 model-storage-condition categories with observed coordinated prices.

The counterfactual demand specification permits substitution within the modeled category but omits substitution across these groups. Under that maintained separation, the simultaneous counterfactual predictions also evaluate changing only a particular group's prices. If cross-model or cross-channel spillovers matter, group contributions and the relevant deviations must include them.

| Assumption added to nonnegative listing costs | Cost interval at rho=.05 | Cost interval at rho=.10 |
|---|---:|---:|
| Seller-wide performance | [67.29,407.58] | [35.38,407.58] |
| Seller-wide performance plus nonnegative contribution for every model | [103.00,407.58] | [45.03,407.58] |
| Separate performance for each model | [183.96,407.51] | [93.41,407.52] |
| Separate performance for every model-storage-condition category | [227.76,404.07] | [136.22,404.84] |
| Common category opportunity costs, seller-wide performance | [73.02,407.58] | [38.71,407.58] |

The comparison isolates two sources of information. Model-specific participation helps, but model-specific performance supplies further restrictions. Common category costs with only pooled performance contribute much less. The common-cost row is a separate approximation about the opportunity value of similar phones, not a claim of equal acquisition invoices. It is not required for the recommended model-specific specification.

We also inverted the calculation in the other direction. The table reports the critical rho below which every cost vector with a mean at or below the specified threshold is excluded. A zero means that such a mean is already feasible at exact performance.

| Mean cost hypothesis | Pooled performance | Model-specific performance | Category-specific performance |
|---|---:|---:|---:|
| mu <= 0 | 14.1821% | 27.2153% | 34.2996% |
| mu <= 50 | 7.8862% | 12.8497% | 15.0380% |
| mu <= 100 | .3451% | 9.6632% | 11.7681% |
| mu <= 150 | 0 | 7.3893% | 9.4528% |
| mu <= 200 | 0 | 3.5379% | 6.9596% |

For example, the model-specific fit excludes mean cost at or below 100 if actual pricing earns more than 90.3368 percent of the comparison profit for every model. This is a precise statement of the performance needed, not an estimate of achieved efficiency. The high zero-cost threshold in the grouped rows can be driven by a group with little sales weight; it should not be interpreted as a comparably strong restriction on economically substantial mean costs.

## Sensitivity to counterfactual demand

The additional stress test keeps all actual fitted sale probabilities fixed and multiplies each counterfactual change in log odds by a factor s. The formula is

    q_jt^k(s) = logistic[logit(q_jt^0) + s{logit(q_jt^k)-logit(q_jt^0)}].

This isolates the influence of the price response. It is not an estimated confidence set and does not address all possible omitted variables or outcome errors.

| Response scale s | Model-specific interval, rho=.05 | Model-specific interval, rho=.10 |
|---|---:|---:|
| .50 | [61.71,407.37] | [1.59,407.41] |
| .75 | [151.04,407.46] | [37.69,407.48] |
| 1.00 | [183.96,407.51] | [93.41,407.52] |

For a mean-cost hypothesis of 100, the critical model-specific shortfall changes from 9.6632 percent at s=1 to 6.9133 percent at s=.75 and 4.0387 percent at s=.5. The behavior assumption and the demand assumption cannot be treated as independent sources of already established information. The latter controls how much profit the former says the seller could be missing.

## A qualitative alternative: purposeful repricing

A separate design compares actual prices with the immediately preceding fresh daily quote for the same listing. The comparison is available only for consecutive calendar-day observations; November 19 cannot serve as a fresh previous observation. Entrants and other ineligible prices stay fixed. The profit calculation includes spillovers on retained owned listings.

There are 733 eligible changes in November-December: 707 reductions and 26 increases, on 30 distinct dates. The increases involve 23 listings on 15 dates; their median receipt increase is 9.785. This calculation uses a different eligibility rule from earlier counts of price changes, and those counts should not be substituted.

The weakest version examined is

    Pi^actual(c) >= Pi^retain_previous_quote(c).

It requires that repricing be weakly profitable in aggregate under the maintained conditional demand. It does not require that unchanged prices be good choices. This is a substantive no-harm performance restriction, although it has no numerical allowance. It can fail when repricing experiments sacrifice current profit, when the seller uses private information omitted from demand, or when the payoff model misses inventory objectives.

At fitted demand, undoing all changes loses 6018.69 of expected revenue and reduces expected sales by 17.31. Actual markdowns can therefore be purposeful at zero cost. Together with pooled nonnegative contribution, the comparison gives [0,407.58]. Separately requiring increases and reductions to be purposeful gives [4.16,407.58]. Requiring those two comparisons for every phone model gives [18.67,407.46]. The last two add conditioning by the direction of the seller's action and by product, respectively; they are stronger restrictions. They are not selected or dropped on a formal significance test.

The qualitative restriction is defensible as a starting point, but it does not recover a useful overall mean here. This result is why the proposed near-performance calculation compares actual prices with routine changes that the seller did not necessarily implement.

## Empirical motivation and its limits

Three types of evidence motivate the proposed restrictions.

First, the data show coordinated pricing. In the earlier audited panel, 92.44 percent of multi-listing category-days have a common price. Among 658 entrants facing a unique incumbent category price, 75.84 percent adopt it. The seller has repeated entries, sales, and batched markdowns. These are descriptive findings from the preserved pricing and inventory audits; the percentages were not recomputed on the new one-day sample. They support product-level pricing decisions and simple coordinated comparisons. They do not establish category-specific profit maximization or a numerical efficiency level.

Second, GadgetPickup's [November 2023 BankMyCell announcement](https://gadgetpickup.com/blog/gadgetpickup-bankmycell-team-up/) describes a professional buyback operation and device valuation. This is suggestive of ongoing attention to product values. It concerns procurement and does not disclose a retail pricing algorithm. Its current page displays a November 12, 2023 publication date; we do not treat that as independently archived proof of the exact publication content.

Third, a [2021 Swappa/WholeCell demonstration](https://www.wholecell.io/blog/56-demo-swappa-incoming-inventory-wholecell-integrations) documents tools for bulk listings, serialized inventory, and sales-order imports before this sample. These tools make organized inventory and sales feedback plausible for this environment. GadgetPickup's adoption of WholeCell is not established. A separately preserved 2023 [GadgetPickup listing](https://swappa.com/listing/view/LXKC36753) contains an explicit refusal of a lower offer after a recorded markdown. This is suggestive of deliberate posted-price decisions; it does not identify a cost floor or numerical regret allowance.

Taken together, these observations support considering a seller that evaluates routine product-level pricing changes over weeks. They do not justify treating 95 percent as measured, assuming that all demand shocks are observed, or imposing the same performance separately on every individual listing-day. A 5 percent and a 10 percent shortfall are transparent benchmarks for economic judgment. Neither is a theorem about all professional sellers.

## Relation to ANR and the new method

Asymptotic no regret does not bound regret on these 50 observed dates. It also does not automatically cover a rule that adds or subtracts a fixed amount from the actual chosen price. Such a rule depends on the chosen action and is a restricted swap deviation. It is not one of the fixed alternative prices in contextual external regret. The paper's extension to action-dependent deviations is relevant; the added comparison class must be stated explicitly.

Let the restricted cumulative regret be

    R_N^K(c) = max{0, Pi^k(c)-Pi^0(c): k in K}.

Assumption (P) is equivalent to nonnegative actual contribution and

    R_N^K(c) <= rho/(1-rho) Pi^0(c).

Thus it supplies a candidate-cost-dependent finite-window performance bound. It makes no claim about a learning rate, the speed of convergence at other horizons, or adoption of a particular learner. A process may satisfy this limited comparison while having large regret against other prices. ANR and this one-window restriction are not logically interchangeable.

For the paper's notation, lambda contains costs, demand, and the payoff/observation environment. A complete application requires a joint empirical-distribution witness compatible with the observed distribution and the consistency set C. The paper's Q_f(C,epsilon;lambda) separates consistency from obedience. Here the proposed performance inequalities replace the relevant obedience rows with

    E_nu[u_i(a,t;lambda)] >= 0,
    E_nu[u_i(a,t;lambda) - (1-rho)u_i(d_k(a,h),a_-i,t;lambda)] >= 0,

within each maintained product group. Conditional expectations in this display stand for the corresponding unnormalized group sums when a group has zero empirical mass. Since rho is fixed in an inversion, these are linear inequalities in the witness nu at a fixed candidate lambda. They define an explicit extension of the paper's relaxed prediction set. They do not result from substituting a universal numerical epsilon into the unmodified external-regret theorem: the bound depends on the actual contribution in the witness and the comparisons depend on the chosen action.

A coverage argument is nonetheless direct. Suppose the true demand/payoff model belongs to its maintained uncertainty set with probability at least 1-alpha_D, the true empirical state distribution belongs to C with probability at least 1-alpha_C, and the stated finite-window performance event holds with probability at least 1-alpha_P. On their intersection, the true empirical distribution is a feasible witness for the true lambda. Projection therefore covers the true parameter with probability at least 1-alpha_D-alpha_C-alpha_P. No independence is needed for this union bound.

This is a conditional coverage theorem. It does not establish the three probability statements from the descriptive evidence. If performance is imposed pathwise as a maintained economic model, alpha_P is zero within that model. If it is intended to hold only with high probability, that probability is another substantive assumption. A 5 percent permitted profit shortfall is not a 5 percent rejection probability. The current calculations fix demand and the observed histories and implement only the profit inequalities; they are not 95 percent confidence intervals or a completed joint Q_f projection.

The finite-window assumption is imposed on the retained one-day observations. It is not deduced by deleting dates from a full-history regret guarantee. Private signals, omitted inventory and channels, delayed feedback, and demand uncertainty must be handled in the full observation and consistency model rather than implicitly absorbed into rho.

## Separation of evidence and assumptions

| Component | Role in these calculations | Evidentiary status |
|---|---|---|
| Observed prices, stock histories, outcomes, copied snapshot | Defines the evaluation sample and feasible historical price comparisons | Observed proxies, audited for specified defects |
| Counterfactual sale probabilities | Determines revenue and quantity gains from a price change | Maintained fitted-demand response; sensitivity reported |
| Nonnegative persistent listing opportunity costs | Defines the unknown payoff parameters | Economic approximation, not observed invoices |
| Nonnegative contribution | Supplies the main upper bound on expected mean cost | Participation restriction relative to retention/withdrawal |
| Small-menu performance | Excludes low costs inconsistent with sufficiently competent pricing | Economic assumption; numerical rho is not estimated |
| Performance for every model | Prevents unrestricted offsets across product models | Stronger assumption motivated by coordinated product pricing |
| Common category costs | Separate pooling restriction on opportunity values | Optional comparison; not needed in the recommended specification |
| Full statistical coverage | Requires demand, performance, and consistency events | Derived conditionally above; not implemented in these intervals |

## Reproduction and audit

Run with `SWAPPA_DATA_VERSION=fullsource_refscreen_known` and the machine's research Python. Each producing script was committed before its calculation. Result JSON files include the producing commit, script/design hashes, and hashes of prediction inputs or outputs.

| Producing script in python/swappa_rescue | Producing commit | Tracked result in docs/evidence |
|---|---|---|
| gadget_performance_frontier.py | 29f6b7f | gadgetpickup_performance_results_20261020.json |
| gadget_performance_groups.py | e8939e7 | gadgetpickup_performance_group_results_20261020.json |
| gadget_repricing_performance.py | 7753e3f | gadgetpickup_repricing_performance_results_20261020.json |
| gadget_performance_thresholds.py | 9817edb | gadgetpickup_performance_thresholds_20261020.json |

The corresponding designs are `gadgetpickup_performance_design_20261020.json`, `gadgetpickup_performance_group_design_20261020.json`, and `gadgetpickup_repricing_performance_design_20261020.json`. These are prospective descriptions of each computation, not preregistration of the overall research project. The product and purposeful-repricing extensions were specified after seeing the pooled results, with their timing recorded explicitly.

Detailed predictions are in `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/gadget_demand_20261020/`: `performance_predictions.parquet` and `repricing_performance_predictions.parquet`. The new fits reproduce the corrected factual predictions to within 1e-10. All five folds converge. Checks cover exact ask-grid compatibility, LP feasibility, the analytic participation bound, nesting as rho increases, an independent two-product analytic solution, signs against direct profit calculations, and agreement between analytical and numerical zero-cost thresholds.

## Returns and next executable work

The completed exercise returns a defensible candidate class of behavior restrictions, their exact linear cost implications, a numerical decomposition by product pooling, an unsuccessful weaker qualitative restriction, and a demand-performance sensitivity calculation. It removes verified algorithm adoption as a prerequisite for further work.

The recommended next step is a joint demand-and-performance analysis with the model-specific menu as a declared candidate, retaining the pooled and purposeful-repricing alternatives. Expand the counterfactual-demand sensitivity set using already estimated alternative specifications and observation uncertainty; project over that set rather than intersecting selected favorable models. Keep the expected-sales and realized-sales targets separate. Implement the consistency component and the direct witness argument before describing any result as a confidence region. These are executable extensions; lack of a named repricer is not a blocking condition.
