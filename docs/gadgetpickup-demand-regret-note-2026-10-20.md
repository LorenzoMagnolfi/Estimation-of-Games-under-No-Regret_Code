# GadgetPickup: demand, sales feedback, and cost identification

Date: October 20, 2026. Audience: Lorenzo and future research agents. Status: empirical audits and a proposed sufficient regret restriction; no credible marginal-cost confidence interval has yet been established. This note governs the new diagnostics. `SWAPPA_RULE_TESTS_2026-10-20.md` remains the record of the earlier conditional cost calculations; its demand validation and next-step sections are superseded here. No existing demand object or cost result was replaced.

## Result and decision

We improve the prediction of which GadgetPickup phones sell, but have not established the causal effect of its price changes. A seller-specific demand fit, evaluated on entire listing histories excluded from training, predicts a November–December sales-weighted receipt of $412.65, compared with $392.61 observed. The previous pooled fit predicted $466.16. More detailed category and time controls do not improve all validation measures, and future-period predictions remain unstable.

We also construct simultaneous bounds on demand from past sales. These supply a statistical measure of uncertainty rather than a chosen profit allowance. In five categories selected by exposures before October, even the sharper construction cannot rule out zero cost under the weak behavioral requirement that the observed price is not demonstrably inferior to another previously observed price. This result concerns the stated information model and comparison set. It is not an impossibility theorem for every learner.

Public sources support individual-device accounting, professional valuation, public transaction comparisons, and batch listing tools. We found no verified disclosure of GadgetPickup's repricer, updating equation, exploration probabilities, learning rate, or finite-sample performance guarantee. A dated 2023 seller comment refusing a lower offer is useful evidence on posted-price selling; it does not identify a cost floor or a learning rule.

**Decision.** Do not carry the earlier conditional cost intervals into the application as credible estimates. Keep costs unknown in the structural parameter. The unresolved inputs are a defensible demand response, the economic meaning of cost, and a finite-window restriction connecting sales information to price choices. The absence of observed cost is not an objection to estimation.

## Data, observation model, and provenance

Input root: `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/`. Set `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. The function `demand_timing_sensitivity.terminal_status_data()` starts from the full prepared histories, retains observed reference prices and rival availability, and excludes the final calendar date and unknown terminal outcomes. GadgetPickup has no unknown terminal-status rows in the raw prepared panel. The retained sample contains 20,486 listing-day observations, 1,594 terminal sale indicators, and 165 dates.

A terminal sale indicator assigns a later verified sold status to the final observed active exposure. It is an observation model; the `Updated` field is not an order timestamp. Current/future seller comments, actual review times, exploration probabilities, and the repricer's identity are absent from the input columns. A successful demand regression does not validate this timing convention.

Receipt equivalents are computed as `1.03 * floor(displayed_price / 1.03)`. They remain the earlier convention, not verified net payouts after seller fees, payment processing, shipping, and returns. All reported dollar diagnostics use this convention. A category is `device_id|Storage|Condition`. The observed listing count is not a count of sequential pricing updates.

The run directory is `gadget_demand_20261020/` below the input root. Scripts and tracked results are paired as follows:

| Script under `python/swappa_rescue/` | Design/result under `docs/evidence/` | Producing code |
|---|---|---|
| `gadget_demand_audit.py` | `gadgetpickup_demand_design_20261020.json`; `gadgetpickup_demand_audit_20261020.json` | 7dd7df2 |
| `gadget_demand_extension.py` | `gadgetpickup_demand_extension_20261020.json`; `gadgetpickup_demand_extension_results_20261020.json` | a51d344 |
| `gadget_demand_crossfit.py` | `gadgetpickup_crossfit_design_20261020.json`; `gadgetpickup_crossfit_results_20261020.json` | 1d7b960 |
| `gadget_demand_time_controls.py` | `gadgetpickup_time_controls_design_20261020.json`; `gadgetpickup_time_controls_results_20261020.json` | 2ee49ee |
| `gadget_feedback_audit.py` | `gadgetpickup_feedback_design_20261020.json`; `gadgetpickup_feedback_results_20261020.json` | 0851103 |

The first audit's script received a subsequent performance-only change replacing repeated column insertion with concatenation. Its saved result records the producing script hash. Each later design was committed before its calculation; later designs were informed by earlier results. The sample had already been examined. These are retrospective exploratory checks, not fresh confirmatory holdouts. `gadgetpickup_demand_regret_manifest_20261020.json` records current input, script, and output hashes without rewriting historical producing hashes.

## Demand diagnosis

The previous model's aggregate sales fit concealed price-composition errors. In November–December it predicted 93.05 sales at prices up to $200, where 148 terminal sales were recorded. Above $600 it predicted 177.62 sales, compared with 113 recorded sales. Expensive iPhone 14 Pro/Pro Max variants account for several of the largest revenue residuals. A seller intercept alone cannot correct this allocation.

We estimate all declared alternatives without inspecting new cost bounds. Logistic models include the established reference-price, competitor, owned-inventory, age, battery, and initial-price controls. All new logistic fits use regularization parameter C=1. Gadget-only models replace pooling across sellers. Category models use joint model/storage/condition intercepts. Chronological category models use weekday and linear calendar time rather than future week indicators; the extension removes that time trend or uses a fixed shallow tree as a predictive benchmark. The tree is not inverted for costs.

### Chronological prediction

Training ends August 31, September 30, or October 31; evaluation is respectively September, October, or November 1–December 22. The November–December sample contains 6,429 rows and 585 sales. Its observed mean receipt is $392.61.

| Model trained through October | Predicted sales | Predicted mean receipt | Log loss |
|---|---:|---:|---:|
| Gadget-only additive product controls | 600.71 | 435.09 | 0.2601 |
| Joint category intercepts, linear time | 414.48 | 445.01 | 0.2773 |
| Joint category intercepts, no time trend | 562.38 | 439.77 | 0.2711 |
| Category-specific age responses | 547.95 | 467.84 | 0.3515 |
| Category model with price splines | 416.41 | 447.60 | 0.2772 |
| Shallow tree | 470.74 | 423.25 | 0.2670 |

Lower log loss is better. The no-trend model avoids part of the extrapolation error; the tree improves price composition but underpredicts the number of sales. Neither observation establishes a reliable counterfactual demand curve. Earlier-month results are in the tracked JSON, including the poor performance of the category-specific age model. No model was selected because it generated tighter cost bounds.

### Listing-disjoint prediction within the observed period

We assign whole listing histories to five folds using a stable hash of `Code`. Each held-out prediction comes from a model trained on the other listings. Calendar weeks can appear in both training and evaluation. This estimates predictive performance for another listing in the observed environment; it does not evaluate a seller forecasting future weeks. Common shocks across training and evaluation listings can remain.

| Model | November–December predicted sales | Predicted mean receipt | Log loss |
|---|---:|---:|---:|
| Gadget-only additive controls and week effects | 601.45 | 412.65 | 0.2535 |
| Joint category and week effects | 700.08 | 416.68 | 0.2621 |
| Joint category/week effects and price splines | 700.23 | 417.19 | 0.2620 |
| Additional category-by-week effects | 727.42 | 411.87 | 0.2622 |
| Category-by-week effects, no initial-price controls | 727.59 | 412.82 | 0.2624 |

The full-sample additive crossfit predicts 1,643.85 sales against 1,594 observed and a mean receipt of $406.41 against $407.00 observed. Thus restricting the model to GadgetPickup largely removes the full-sample composition discrepancy. November–December still has a residual discrepancy, and finer intercepts overpredict sales.

The estimated response to a joint $10 price reduction remains negative-sloped in all reported fits. Among continuation rows in November–December, the additive model predicts 135.17 additional sales; the category model predicts 150.76; category-by-week effects predict 156.75; removing initial-price controls gives 145.75. These are changes in fitted probabilities summed over realized exposure rows, with histories and inventory held fixed. They are not predictions of a recursively evolving inventory path or evidence of a causal effect. They also differ from the earlier previous-ask deviation used for cost inversion. New code changes all owned prices before selecting held-out rows, so peers assigned to other folds are included in the joint change.

### Identifying variation and timing

The full-sample price standard deviation is $254.61. Removing category means leaves residual RMS variation of $41.12; removing category-by-week means leaves $7.07; removing category-by-date means leaves $2.33. On 90.99% of full-sample rows there is no price residual within category/date. Among category-days with multiple owned listings, 92.44% have a common price. These are descriptive variation measures, not a formal weak-identification statistic.

Within the same category, most useful price variation is over time. A causal interpretation therefore requires a restriction on demand shocks, buyer contacts, changing inventory composition, and prices. Regularized category-week effects do not automatically absorb arbitrary category-week shocks. Initial prices can proxy quality, but are also seller choices. Holding those controls fixed under a continuation-price deviation is a maintained specification.

There are 1,924 consecutive-day cuts, with 184 terminal sales on those rows; unchanged consecutive rows have 1,139 sales among 16,314 rows. The corresponding rates are 9.56% and 6.98%. Moreover, 345 cuts occur on the terminal or preceding day of a subsequently sold listing. This association can reflect a price effect, negotiation, seller information about buyers, or inventory selection. It cannot distinguish these explanations alone.

## What public evidence now supports

The extended search covered exact seller-name variants, the public managing-member/business identity, dated listings and comments, employment advertisements, vendor integrations and changelogs, public code, trade-in partners, and archival searches. It supplemented the earlier five-seller/vendor search in `SWAPPA_EMPIRICAL_FOUNDATIONS_2026-10-20.md`. No seller was contacted and no account was created. The user confirms that no additional private records or contacts are available.

The following sources were read or retrieved in indexed text on October 20, 2026. A displayed publication date is not an independently verified archival snapshot.

| Source | Evidence admitted | Restriction it does not establish |
|---|---|---|
| [GadgetPickup/BankMyCell announcement](https://gadgetpickup.com/blog/gadgetpickup-bankmycell-team-up/), displayed November 12, 2023 | GadgetPickup describes expert valuation and a trade-in comparison partnership. Indexed text was accessible; direct opening failed. | Retail price optimization, a demand model, or learning rate. The announcement concerns procurement. |
| [GadgetPickup listing LXKC36753](https://swappa.com/listing/view/LXKC36753), September–October 2023 | Seller's dated October 2 reply: “Unfortunately, the price is firm.” | A universal no-negotiation policy, an acquisition-cost floor, or acceptance of later offers. |
| [July 2023 buyer reviews](https://swappa.com/en/bp/gadgetpickup/ratings?page=468) | Dated reviews of completed dealings and responses to inquiries. | Historical order-level price histories or experimental exploration. Review samples are selected. |
| [GadgetPickup terms](https://gadgetpickup.com/terms-and-conditions/), displayed May 2025 | Buyback valuation uses market conditions and inspected device condition; quotes have limited validity. | 2023 retail pricing, constant acquisition costs, or observed seller net receipts. |
| [GadgetPickup current Swappa trade-in profile](https://swappa.com/en/trade-in/partners/gadgetpickup) | Current procurement offers and service conditions. | Historical cost data. Do not attach today's quotes to 2023 inventory. |
| [Shipping/inventory supervisor advertisement](https://hireourheroes.com/jobs/?categories%5B%5D=Distribution-Shipping&categories%5B%5D=Human+Resources&categories%5B%5D=Nonprofit-Social+Services), displayed October 18, 2026 | Advertised duties include stock reconciliation, quality standards, and using sales trends for demand forecasts. | Duties in 2023, the software used, or near-optimal pricing. The page is a changing job index. |
| [WholeCell/Swappa demonstration](https://www.wholecell.io/blog/56-demo-swappa-incoming-inventory-wholecell-integrations), May 9, 2021 | Historical bulk listing, serialized devices, sales-order records, and distinction between power and enterprise workflows. | GadgetPickup adoption or automated price optimization. Representative photos do not establish multi-quantity listings. |
| [WholeCell changelog](https://www.wholecell.io/updates) | Dated inventory/accounting features; Back Market BackPricer support in July 2022; Swappa single-item automatic delisting in March 2024. | A Swappa repricer in 2023. Platform-specific features must remain separate. |
| [GadgetPickup listing LZJF49808](https://swappa.com/listing/view/LZJF49808), August–November 2025 | Seller again refuses a lower offer; a buyer reports a subsequent price increase. | 2023 behavior or a verified change log. The price-increase statement is a buyer report. |

The historical listing LXKC36753 is in our panel. Its displayed price was $468 from September 20–29 and $424 from September 30 through October 6. It has `Real_Sold=0` and an October 7 update. The seller's October 2 refusal occurs after the observed markdown. This reconciles the public comment with the scrape: a firm response to an individual buyer can coexist with later or earlier general repricing. The buyer's $390 and $410 messages do not establish executed trades.

We also searched GadgetPickup together with WholeCell, Sellercloud, Reusely, Rewhole, RepairDesk, Sellbrite, Linnworks, ChannelAdvisor, ShipStation, repricer, dynamic pricing, automation, pricing specialist, and e-commerce. No verified adoption link was found. Current listing badges identify Phonecheck, which tests devices; they do not identify pricing software. Boilerplate about proprietary software likewise supplies no updating rule.

Excluded or unresolved leads: a freelance job uses GadgetPickup as an example for another company's proposed buyback site; it is not a GadgetPickup software contract. TaskMarket, TradeDeck, Quixess, and Commercium advertise current capabilities without verified sample-period adoption. Forum claims about buyback offers or common ownership were not used as identifying restrictions. Search results for our own paper are circular evidence. A Wayback CDX request failed, so no archived 2023 snapshot was verified. The negative search finding does not prove that private or unindexed software does not exist.

## An allowance derived from sales uncertainty

A useful additional assumption can be stated without naming a particular algorithm. The distinction between a statistical bound and a behavioral restriction must remain explicit.

Let u_n(a;c) be the conditional expected portfolio profit at date n from feasible price vector a, given the information available before that choice. Costs c remain unknown to the researcher. Suppose simultaneous bounds L_n(a;c) and U_n(a;c), constructed using past feedback, cover u_n(a;c) for every relevant date and action on an event of probability at least 1-delta. Write W_n(a;c)=U_n(a;c)-L_n(a;c).

**Sufficient behavioral restriction, proposed but unverified.** Suppose the seller chooses a_n to maximize U_n(a;c). It chooses the price with the best profit that is still consistent with its uncertainty. This is an optimistic choice rule. It need not know expected profits, use exponential weights, or have a learning rate. It nevertheless adds a substantive restriction on behavior and on the bounds the seller uses.

On the simultaneous coverage event, for every feasible alternative a,

    u_n(a;c) <= U_n(a;c) <= U_n(a_n;c)
               = L_n(a_n;c)+W_n(a_n;c)
               <= u_n(a_n;c)+W_n(a_n;c).

Summing gives cumulative expected-profit regret against any included fixed rule at most Gamma(c)=sum_n W_n(a_n;c). A rule's recommendation must be feasible at each realized history. This is a one-step comparison, not a policy that changes future inventory and rivals' histories.

**Weaker behavioral restriction.** Suppose only that U_n(a_n;c) >= max_a L_n(a;c): the chosen price has not been shown to be inferior by the available bounds. Then the same reasoning gives

    u_n(a;c)-u_n(a_n;c) <= W_n(a_n;c)+W_n(a;c),

and Gamma(c)=sum_n [W_n(a_n;c)+max_a W_n(a;c)] suffices. This allows a broad class of choices, but its bound can be much larger. Neither restriction follows from ANR or from accurate prediction of observed prices.

For separable expected payoffs sum_j (x_j-c_j) q_j, a rectangular probability band [l_j,u_j] gives W=sum_j |x_j-c_j| (u_j-l_j). Taking absolute values is essential when candidate margins are negative. A valid joint confidence set can be used instead of a rectangle. Costs need not be revealed to the researcher and need not have a common fixed upper bound for this candidate-specific calculation.

**Connection to the new method.** The paper's Q_f construction needs an obedience bound and a consistency set for the signal-state distribution. The inequalities above give an obedience event for a maintained class of behaviors. Their allowance is random and depends on candidate costs. Its coverage must be proved by the simultaneous event and the same witness argument; it is not an unmodified invocation of Theorem CR, which uses an expected-regret envelope with its specified concavity and conditioning properties. A deterministic envelope would require a separate bound on the expectation of Gamma. Demand and signal-state uncertainty must be included before projection onto costs. Divide cumulative portfolio dollars by eligible listing-days when comparing to earlier epsilon units, rather than by the number of calendar days.

The proposed route avoids choosing epsilon from the width of a cost interval. Its remaining economic burden is that GadgetPickup's choices obey the stated restriction with the specified information bounds. Expert valuation and commercial software availability do not establish that claim.

## Sequential feedback diagnostic

We implement a simple version of the information calculation before imposing the proposed choice restriction. The five categories are chosen by exposure counts through September 30. Feedback accumulation begins October 1, and evaluation ends December 22. We use category-days with a common displayed price. Prices are separate actions, with no assumed transfer across prices or categories. This restriction is deliberately transparent; it omits the seller's earlier experience and is not a claim that it restarted in October.

For a category and price, let Z_m be the sale fraction on its m-th observed day, a number in [0,1]. The maintained diagnostic assumption is E[Z_m | prior information and the current chosen price/inventory]=q for a constant q. Conditional mean constancy includes the relevant selection and feedback timing. It is stronger than merely observing a repeated price, and is not established here. The daily-fraction version allows arbitrary dependence among devices within a day. An alternative treats each exposure as a Bernoulli observation and requires the constant conditional mean to hold even after other outcomes within the batch; it is an additional assumption, not additional pricing rounds.

We use two simultaneous constructions. Hoeffding intervals spend error probability over five categories, introduced prices, and sample counts. A sharper mixture construction spends error only over categories and introduced prices. If price j is the j-th new price encountered after October in that category, set alpha_j=.05/[5 j(j+1)]. For S_m=sum_{r<=m} Z_r, define

    M_m(q) = B(S_m+1/2, m-S_m+1/2)
             / [B(1/2,1/2) q^S_m (1-q)^(m-S_m)].

The confidence interval consists of q for which M_m(q)<=1/alpha_j. It is [0,1] before any feedback. At each date the calculation uses completed prior dates only.

**Guarantee.** For each theta in (0,1), the factor (theta/q)^Z ((1-theta)/(1-q))^(1-Z) is convex in Z and lies below its endpoint chord. Its conditional expectation is therefore at most one when E[Z|past]=q. Products are nonnegative supermartingales, as is their mixture over theta with a Beta(1/2,1/2) density. The probability that M ever reaches 1/alpha_j is at most alpha_j by Ville's inequality. Summing alpha_j over the five categories and all introduced prices gives .05. This argument permits fractional Z and does not require independent devices within a daily fraction. Predictable entry into a price action and valid outcome timing remain necessary. Related general methods appear in [Howard et al. (2021)](https://arxiv.org/abs/1810.08240).

The mixture calculation gives the following widths of the probability interval for the currently chosen price, computed before that day's sales. Means below weight category-days equally.

| Category | Common-price days | Exposures | Sales | Mean width, daily fractions | Mean width, exposure model |
|---|---:|---:|---:|---:|---:|
| iPhone X, 256 GB, Good | 81 | 172 | 11 | .625 | .513 |
| iPhone XS Max, 64 GB, Good | 74 | 391 | 15 | .804 | .491 |
| iPhone 8, 128 GB, Good | 55 | 81 | 6 | .814 | .752 |
| iPhone 11, 128 GB, Fair | 30 | 43 | 9 | .916 | .850 |
| iPhone 11, 128 GB, Good | 44 | 97 | 23 | .902 | .838 |

At zero cost, U_n(a_n;0) is at least the largest lower expected receipt among previously encountered prices on every evaluated category-day, under both constructions and both observation models. Thus the weak no-demonstrated-inferiority restriction does not exclude zero cost in this diagnostic. The result does not compare every feasible portfolio, unobserved price, or demand model. It does not imply that zero cost fits the earlier parametric profit inequalities.

A tighter construction could use prior experience, neighboring prices, or related products. Each route needs a justified transfer restriction and a valid conditional mean model. Adding more calendar rows as if they were independent pricing experiments, or narrowing a probability band until it excludes zero cost, is not an admissible correction.

## Why ANR alone cannot certify the observed allowance

A finite-prefix argument makes the limitation precise. Fix a finite observed window of N decisions, any finite candidate cost vector, and bounded payoffs for that candidate. A decision process can follow the observed actions during those N dates and then start an ordinary no-regret learner. If D(c) bounds one-period payoff differences, its later cumulative regret is at most N D(c) plus the restarted learner's regret. Dividing by the growing horizon gives zero. Any finite prefix is therefore compatible with asymptotic no regret in an environment admitting the continuation learner, even when the prefix has large regret.

This argument concerns what ANR alone implies. It does not invalidate the new method when a justified finite envelope is part of the maintained model. It shows why smooth prices, a long business history, or small predictive errors cannot supply that envelope by themselves. Unknown cost is not the logical obstacle.

## Other learning guarantees examined

- **EXP3-IX.** The earlier calculation remains: [Neu (2015), Theorem 1](https://proceedings.neurips.cc/paper_files/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf) exceeds the maximum normalized regret on the 52-day design, even for two actions. Adoption, inherited state, feedback, and realized-payoff normalization also remain unverified.
- **Stochastic upper-confidence algorithms.** [Auer, Cesa-Bianchi, and Fischer (2002)](https://doi.org/10.1023/A%3A1013689704352) provides a route for stationary arm rewards and the specified exploration/choice policy. Repeated prices alone do not establish either requirement. The feedback diagnostic above shows the information available under a transparent stationary-arm approximation.
- **Contextual regression.** [Foster and Rakhlin (2020), Theorem 1](https://proceedings.mlr.press/v119/foster20a/foster20a.pdf) requires realizability, an online regression guarantee, and SquareCB's action-sampling rule. Its displayed bound is 4 sqrt(K T RegSq(T)) + 8 sqrt(K T log(2/delta)). A retrospective prediction score is not RegSq(T), and a good demand fit does not identify the sampling rule.
- **First-order bounds.** [Pogodin and Lattimore (2020)](https://proceedings.mlr.press/v115/pogodin20a/pogodin20a.pdf) analyzes specified algorithms and payoff assumptions. Sparse sales do not imply small normalized loss: under loss=1-profit/C, a nonsale has loss one. Gains-based and small-loss results require their own admissible normalization and assumptions.
- **Subinterval guarantees.** [Daniely, Gonen, and Shalev-Shwartz (2015)](https://proceedings.mlr.press/v37/daniely15.html) studies strongly adaptive learning. A prefix theorem cannot be applied as a November restart without justification. Extra-feedback results such as [Lu et al. (2024)](https://arxiv.org/abs/2401.09278) use an information model that is not established by observing several different phones on a day.

These sources identify possible maintained models. None is evidence that GadgetPickup satisfies the associated guarantee.

## Verification and next executable work

All new logistic folds converged without recorded convergence warnings. Exact fold membership keeps entire listing histories together. Joint price changes account for peers across folds. The feedback implementation passes three focused tests: one-observation closed forms, exact enumeration of first-crossing probabilities through 50 observations at four Bernoulli means, and the convexity inequality supporting fractional observations. These checks validate implementation and the stated mathematical construction; they do not validate its economic assumptions.

The next empirical distinction is between replenished inventory and fixed-stock selling. Individual acquisition payments are sunk once a phone is in stock; a one-period payoff (price-cost) times sale probability then identifies a continuation or replacement opportunity cost only under an explicit economic reduction. GadgetPickup's procurement business and common category prices motivate investigating that reduction, but do not prove it. Audit listing arrivals, sales, cancellations, inventory levels, and repeated model-price cohorts before imposing common or stationary costs. Keep acquisition cost, replacement cost, and inventory continuation value separate.

For demand, a future extension should use a declared causal restriction or instrument with an institutional argument, and carry its uncertainty into cost inference. For regret, investigate a behavior class with a proved envelope and an empirical implication beyond price persistence. Do not rerun the same vendor searches or append another arbitrary-epsilon frontier. The current result remains an improvement in diagnosis and a proved candidate route, rather than a completed rescue of the application.


## Inventory and use of earlier feedback: subsequent pass

**Status.** This pass adds evidence about the economic meaning of cost and strengthens the earlier information diagnostic. It does not supply a verified finite regret allowance. The design was committed at `a25ab49`; the producing implementation is `85e01d7`, `python/swappa_rescue/gadget_inventory_feedback_extension.py`. Results are in `docs/evidence/gadgetpickup_inventory_feedback_extension_results_20261020.json`. The old cost estimates and demand fits are unchanged.

### What replenishment looks like in the listing panel

The inventory calculation uses the prepared full-source reference-screened panel before the demand routine removes missing references, absent rivals, unknown outcomes, or the final date. It contains 21,322 GadgetPickup listing-days, 2,052 listing identifiers, and 191 model-storage-condition categories over 166 calendar dates. There are 1,640 terminal sales before the last date, 301 terminal nonsale exits, and 111 listings present on the last date. These counts differ from the demand sample because they precede its exclusions. Listing identity is observed; physical-device identity and warehouse inventory are not.

Accounting for 126 returns after observation gaps reconciles daily stock changes exactly. The accounting identity is current stock minus prior stock equals new listings plus returns after gaps minus prior-day final exits and prior-day starts of gaps. Ignoring gaps would confuse observation interruptions with acquisitions and withdrawals.

After removing fourteen dates at each edge, average listed stock is 133.73 phones. There are 1,640 newly observed listings and 1,396 sales over 138 dates. Sold listings are observed for a median of five dates and a ninetieth percentile of twenty-six dates. Thus the seller repeatedly brings phones to the market; the sample is not a single initial stock being liquidated without further entry.

Aggregate entry is not one-for-one replenishment within a product category. The following entries occur strictly after the event date. Sale events are weighted by the number of sales on the category-day; nonsale exits use the analogous count.

| Window after event | Sale followed by any same-category entry | Nonsale exit followed by any same-category entry |
|---|---:|---:|
| One day | 5.4% | 5.3% |
| Three days | 15.7% | 17.7% |
| Seven days | 39.3% | 38.3% |
| Fourteen days | 64.5% | 62.6% |

The calculation includes 1,396 sales and 243 nonsale exits in the interior. Entry and sale events may overlap across windows; these are descriptive proportions, not independent observations or estimates of a causal replenishment response. Earlier entries partly represent the same cohorts that subsequently sell, so the before/after asymmetry is not a treatment effect. Active-category-day comparisons also change the category weights. All before/after quantities and baseline counts are retained in `inventory_entry_windows.csv`.

There are 563 observed spells with no listed stock between two dates with stock in the same category, with median duration nine days. These are absences from the panel, not verified warehouse stockouts. Of 1,930 new listings after the first sample date, 670 face continuing stock in their category. Continuing stock has one common price in 658 cases; 75.84% of those entrants receive that same price. The continuing phones in those matched cases have a mean observed age of 16.96 days. This is consistent with some category-level price coordination across differently aged listings. It does not establish a common acquisition cost, a price-independent marginal continuation value, or immediate replenishment.

The additional public-source check was directed at procurement and inventory, rather than repeating repricer-name searches. GadgetPickup's announcement displaying November 12, 2023 describes quotes held for fourteen days and customer shipment of the device before payment. That describes a procurement pipeline with a shipping stage. It does not establish the timing or cost of replacement for a particular Swappa sale. The displayed publication date has not been independently archive-verified. Source: [GadgetPickup and BankMyCell partnership announcement](https://gadgetpickup.com/blog/gadgetpickup-bankmycell-team-up/). Searches for seller-specific wholesale arrangements and inventory software produced no additional verified 2023 contract or retail-pricing procedure.

### The cost interpretation requires an economic restriction

Unknown cost remains the object of inference. The issue here is which cost enters a sale decision, rather than whether the researcher observes its value.

Consider first a single phone and a fixed current state. Let x be receipt from a sale net of any explicitly modeled sale-contingent expenses, q(p) the sale probability, and V_keep and V_sale the expected continuation values conditional on retaining or selling the phone. Suppose changing today's price affects the next state only through that sale outcome, and these two conditional continuation values do not otherwise depend on the price. Apart from a term common to prices, the expected current and discounted future payoff is

    q(p) [x - c_eff],    c_eff = beta (V_keep - V_sale).

Any additional expense incurred only on sale enters c_eff as well. With no replenishment, c_eff is the value of retaining the phone for future use or sale. The original purchase payment has already been incurred. Estimating c_eff therefore need not recover that historical payment.

A sufficient replacement model instead assumes a sold unit can immediately be replaced for an unknown expense k, after which the distribution of next-period states is the same as if the unit had not sold. Conditional on the current state, k must not depend on today's proposed retail price. Expected profit then differs across prices by q(p)(x-k). In this special case the cost in the static expression is marginal replacement expense. The researcher need not know k. The new-method inversion would recover bounds on it if the other restrictions held.

For a seller offering many phones, an additional issue arises. If D is the vector of sales and I is stock, continuation payoff generally includes beta V(I-D). Writing this as a sum of listing sale probabilities times fixed listing costs requires the relevant expected continuation loss to be additive in D, or another explicit restriction that yields that representation. Otherwise the marginal value of selling one phone can depend on which other phones sell, and therefore on the joint price vector. Common category prices and frequent aggregate arrivals do not establish additivity.

The current evidence motivates a model of repeated procurement and coordinated prices. It does not justify immediate replacement, an unchanged next-period stock distribution, additive continuation values, or constant per-listing replacement expense. Inventory held outside the observed marketplace could make rapid replacement possible, so the observed entry gaps do not logically reject all replenishment models. They do prevent us from treating that particular reduction as an observed fact. Expanding the estimand to state-dependent opportunity costs is possible, but it also expands latent heterogeneity and requires a new identification calculation.

### Earlier experience and monotonicity do not resolve the information problem

The extension accumulates all available qualifying common-price category-days from the beginning of the retained sample, before each evaluated day from October 1 through December 22. It includes 158 categories and 4,724 evaluated category-days. It does not assume that the firm had no experience before the observation window. As before, categories with more than one current price do not supply observations to this simple arm calculation.

To avoid using a retrospectively chosen set of categories for error allocation, category i is indexed at its first observed qualifying date, with a fixed lexical tie rule; price j is indexed when it first appears in that category. Set

    alpha_ij = .05 / [i(i+1) j(j+1)].

Use the same beta-mixture interval as above at alpha_ij. The sum of these budgets over all positive i and j is .05. Under predictable category-price introduction, the stated constant conditional mean assumption, and valid timely outcomes, the simultaneous event therefore covers every introduced arm at every sample size with probability at least .95. The adaptive allocation is fixed before the corresponding feedback. The larger category family changes error allocation relative to the original five-category calculation, so differences in interval widths are not the isolated effect of adding prior history.

For the second version additionally maintain that q(p) is nonincreasing in price within a category throughout the observed period. If [l(a),u(a)] are the price-specific bands, the sharp marginal projections of their intersection with monotonicity are

    l_mono(p) = max_{a >= p} l(a),
    u_mono(p) = min_{a <= p} u(a),

where p itself is among the introduced prices. If any lower endpoint exceeds its upper endpoint, the shape-restricted set is empty. No additional probability spending is needed: monotonicity is an extra maintained restriction, not an extra sample. The current-price probability widths for the previously reported five categories are:

| Category | Daily-fraction model, exact price | Daily-fraction model, monotone | Exposure model, exact price | Exposure model, monotone |
|---|---:|---:|---:|---:|
| iPhone X, 256 GB, Good | .639 | .619 | .424 | .363 |
| iPhone XS Max, 64 GB, Good | .919 | .916 | .602 | .594 |
| iPhone 8, 128 GB, Good | .878 | .547 | .824 | .405 |
| iPhone 11, 128 GB, Fair | .882 | .881 | .744 | .736 |
| iPhone 11, 128 GB, Good | .944 | .942 | .871 | .843 |

Across all evaluated categories, there are no empty shape intersections and no dates on which the weak no-demonstrated-inferiority comparison excludes cost zero, under either observation model and either price restriction. The comparator set consists of previously introduced prices plus withdrawal. This is an exact result of this diagnostic, not a universal lower bound on the information available to the firm or a finding that zero cost belongs to the full new-method set. In particular, the check is not a joint cost inversion with the paper's consistency restrictions, other deviations, and first-stage uncertainty.

The exposure model imposes the same arm mean conditional on earlier outcomes within each daily batch. It is stronger than the daily bounded-fraction model and does not turn individual phones into separate price-setting rounds. Both versions require stationarity conditional on the exact category and price despite changes in rivals, inventory, device composition, and calendar time. Monotonicity further pools across different prices observed in different states. The existing demand diagnostics give reasons to question that pooling. Failure to reject a restriction with wide confidence bands is not evidence that the restriction holds.

Three focused tests pass: equality with the earlier mixture routine at a common error budget, agreement of monotone marginal projections with independently solved linear programs on twelve generated feasible systems, and detection of an infeasible shape system. The stock identity is checked exactly in the producing script. A pytest cache-write warning occurred; all test calculations completed successfully.

### Implication for the next step

The strongest presently supported additions are coordinated prices for comparable stock, repeated listing entry, a procurement pipeline, and firm refusal of at least one dated buyer offer. None bounds finite-window profit regret by itself. The earlier-price information extension does not overcome that gap, even under its strong stationarity approximation.

A credible implementation still needs two independent pieces: an institutionally supported causal restriction for counterfactual sales, and an economically supported finite performance restriction with a valid information and timing model. A precise replacement or continuation-value interpretation is also needed for the cost target. We should not replace these requirements with an arbitrary dollar allowance, an inferred software name, or a common-cost restriction chosen because it improves the endpoints. The next useful investigation is a direct audit of potentially identifying price variation and observation timing; further variants of stationary confidence intervals are not the priority.
