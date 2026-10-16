# Swappa rescue checkpoint, 2026-10-16

Audience: Lorenzo and future research agents. Status: active exploratory work, no validated confidence region. Owner: Codex chat 01a14662-bc4d-7902-b104-04742ceb3ca5. The user explicitly reopened the application and authorized methodological modifications and several days of computation. This supersedes the instruction to leave the application parked, but preserves the previous results as an audit trail.

## Objective and governing method

Obtain informative marginal-cost and markup estimates using the current method, with empirically appropriate assumptions. The paper at commit 235704d contains the rate-agnostic Theorem CR, separate consistency and regret uncertainty, and permission to use joint or high-probability regret envelopes. Its observation-window initialization qualification applies. Do not restore the old exact-population-consistency calculation or select a tolerance to make an attractive interval appear.

Paper sources: `../Estimation-of-Games-under-No-Regret/CURRENT DRAFT.tex`, `JPE Revision/Application_State_Of_Play.md`, and `JPE Revision/Response.tex`, R1.4.a-g and R2.4. The July budgeted-regret proposal is in `../../DynamicFoundations/JPE Revision/inferential_bounds_ideas.md`. The correct directory on this machine is `JPE Revision`, not the obsolete `Revision` path in the handover.

## Verified data findings

Input: `data/intermediate/raw_data_cleaned.dta`, SHA256 `69f34f96ecc886b335afb5e011f81e7073d4fac89545dcf31e702c4c97a22894`. It contains 260,745 observations and 25,504 listings. The dates cover July 11 through December 23, 2023.

1. The old code constructs competitors separately after retaining the top 15 sellers. Consequently, demand estimation using all sellers and structural estimation using top sellers do not use the same competitive environment or quantile boundaries.
2. Competitor averages include other listings owned by the focal seller. In the complete cleaned panel, 140,850 observations have additional own-seller inventory in the product market. The rebuild excludes every listing of the focal seller when computing rival mean and minimum prices. It passes 500 independent brute-force comparisons.
3. The old minimum-price code assigns a strictly larger competing price to the first of several tied minimum-price listings when a higher price exists. Its comment claims to handle this tie, but its condition does not count tied minima. The new code first collapses to seller minima, then takes the minimum across other sellers.
4. The legacy sale indicator contains 9,071 positive observations, while 17,614 listings have `Real_Sold=1`. Among those sold listings, 8,960 have Updated equal to the last observed date and 6,241 have Updated exactly one day later. The raw, pre-cleaning file exhibits the same pattern: 13,528 and 8,875, respectively, among 24,441 sold listings. This is not solely an artifact of matching to Decluttr.
5. The scraper collects the listing page's Updated field, not a separately labeled transaction timestamp. Its interpretation as a sale date remains an assumption. Three outcome definitions are preserved: legacy same-date sales; sales during `(date,date+1]`; and a terminal-exposure label for sold listings whose Updated date is zero or one day after the last observed date. The latter produces 15,201 positives before analysis restrictions. Do not call the latter an established correction until the timestamp and disappearance mechanism are audited.
6. Persistence of dollar costs differs from persistence of cost residuals. A cost fixed at listing entry remains fixed in dollars as Decluttr prices change. The new persistent-cost programs use the reference price at the first analyzed observation as an anchor and retain the same latent cost throughout the listing's observed life.

Generated data and results are under `output/swappa_rescue_20261016/`, ignored by Git. No raw data were uploaded.

## Implemented experiments

- `python/swappa_rescue/audit_panel.py`: full-market reconstruction and timing audit.
- `demand_pilot.py`: logistic sale models with rival mean, rival minimum, both, and a richer specification with a kink at the minimum rival price and a price-by-reference-value interaction. Controls include device, storage, condition, seller group, calendar week, age, own inventory, and rival inventory. Validation holds out whole listings. Its original 500-iteration fits produced some convergence warnings; these outputs are preliminary. The cached richer terminal-exposure model in the continuous-cost runs uses 2,000 maximum iterations.
- `persistent_cost_lp.py`: discrete-support exploratory bounds with one latent cost assignment shared by all observations of a listing. Both focal sellers can be rationalized at zero budgeted contextual regret, whereas the single-cost pilots have positive minimum regret. This is a feasibility finding, not a validated confidence statement.
- `continuous_cost_lp.py`: continuous costs within intervals, with interval-specific obedience and a total regret budget. The endpoint representation is an exact representation of interval probabilities and first moments for affine cost payoffs, not a restriction that physical costs are endpoint atoms. The LP remains a convex outer relaxation of deterministic listing-level cost assignments. Targets include the mean across listings and the mean across observed sold listings. Cost positivity is imposed; broad upper support and support sensitivity are explicit. Optional no-offer and no-below-cost-posting assumptions are separately flagged.
- `affine_cost_pilot.py`: persistent cost equal to beta times the entry reference plus an intercept, tested against additive and proportional reference-price deviations and the option not to offer. At the richer terminal-exposure first stage, the best listing-day fits have minimum regret 0.561 dollars (AMS) and 0.699 dollars (GadgetPickup). The corresponding best-fit gross margins are 42.85 and 38.40 dollars. These are minimum-regret diagnostics, NOT point estimates or confidence intervals.
- `test_rescue.py`: analytical cost-bound checks, monotonicity/support checks, equivalence of solver formulations, and independent competitor reconstruction checks. Six tests passed before the no-offer/individual-rationality extension; rerun after further edits.

## Mathematical interpretation

For listing l, let chi_l be the initial reference price and c_l=chi_l+z_l the persistent dollar cost. For a fixed counterfactual reference-price residual a, define the sum of expected payoff gains over the listing's observed risk set as D_la(z)=A_la-B_la*z. Prices at or below zero are replaced by one dollar. The conditional demand model supplies all sale probabilities.

Partition residual costs into intervals k. Let x_lk be latent mass and y_lk its residual-cost first moment. The program imposes sum_k x_lk=1, lower_lk*x_lk <= y_lk <= upper_lk*x_lk, and sum_l (A_la*x_lk-B_la*y_lk)/N <= e_k for every interval and deviation. It imposes e_k>=0 and sum_k e_k<=epsilon. The same x and y enter every observation of the listing. All reported cost means are empirical projections; no iid population cost composition is imposed.

The total budget uses dollars of expected profit per observed listing-day. It is NOT a confidence level. The current programs omit first-stage uncertainty and do not certify the regret budget. A uniform or conditional coverage claim requires a defensible regret envelope, feedback model, observed-window initialization treatment, and first-stage uncertainty set. Repeated listings invalidate iid daily cost draws; they do not by themselves establish the correct learning horizon for a pathwise regret bound. Timing and feedback must determine that horizon.

For a phone already acquired, pricing can identify an inventory opportunity value rather than its original acquisition cost. Calling the inferred object procurement cost requires an additional replenishment, salvage, or procurement model. Seller fees, payment processing, shipping, returns, and negotiation also separate posted prices from net proceeds. Preserve this distinction throughout. Current dollar outputs use posted-price-equivalent costs and gross margins.

## Current assessment

Replacing rival average price by rival minimum alone does not improve held-out predictive performance. Adding both measures and richer controls does improve it in the first split. Persistent heterogeneity resolves the single-cost rejection but unrestricted cost bounds widen materially with the regret budget and upper support. Allowing continuous interval costs avoids the five-point Normal assumption, but the unrestricted pilots do not yet yield credible positive lower bounds on transaction markups. No successful specification has been declared.

## Next executable work

1. Finish and inspect `continuous_ir.log`, which adds the no-offer action and an explicit no-below-cost-posting assumption. Run the no-offer-only comparator. Compare all results with prior-only bounds; do not attribute precision from participation restrictions to the regret restrictions.
2. Refit validation models to convergence, add held-out calendar blocks, and fit initial-listing-price/quality controls to assess the causal-demand assumption. Use the timing alternatives throughout. Cluster demand uncertainty by listing and calendar shocks rather than treating listing-days as independent.
3. Extend action grids and cost-interval partitions, expand upper supports, and verify that reported bounds are not numerical endpoints. Record every specification, including failed ones.
4. Calculate candidate-dependent regret envelopes under explicit feedback and initialization assumptions. Investigate exclusion of loss-making actions only under an explicit participation restriction. A smaller expected-payoff range is legitimate only if all permitted actions and states satisfy the bound. Do not truncate rare cells because doing so narrows the interval.
5. Evaluate economic interpretations: persistent acquisition/replenishment costs versus inventory shadow values; one listing decision with lifetime payoff versus daily repricing. A listing-level sale-within-H-days regression is not a valid constant-price counterfactual when observed subsequent prices change. Do not select only eventual non-repricers without addressing selection.
6. Once a model survives, add first-stage uncertainty, empirical coverage simulations under the exact maintained model, and transparent markup aggregation and fee treatment. Then map evidence to R1.4.a-g and R2.4. The revision ledger and manuscript remain unchanged until their gates are satisfied.

## Operations

The goal is active in this chat. A two-hour heartbeat, `continue-swappa-research`, is scheduled through an explicit prompt cutoff of October 19, 2026, 4:30 p.m. America/New_York. It must not create duplicate writers. A temporary Windows sleep-prevention helper was launched for 72 hours; its status and stop file are in the user's TEMP directory as `swappa_research_keepawake_20261016.*`. Windows process inspection requires the elevated execution context here; a process invisible to the sandbox is not necessarily dead. Do not change permanent power settings.

Read the most recent logs and repository status before resuming. Work only in the new rescue directory and its outputs unless another change is necessary. Existing application scripts, raw inputs, manuscript text, and referee replies have been preserved. Local code commits are authorized by the research task; external sharing was not requested.

## Latest checkpoint addition

The continuous-cost calculation with no offer and no below-cost posting completed. At an assumed total regret budget of one dollar per listing-day, transaction-weighted gross markup outer bounds are [0, 148.44] dollars for AMS and [0.22, 111.28] for GadgetPickup. These are not confidence intervals. The tiny positive GadgetPickup lower bound can arise from the difference between the minimum posted price used in the participation restriction and the terminal price; do not attribute it to identification of positive margins. At zero budget the corresponding intervals are [9.71, 98.40] and [7.98, 74.68]. Nonparametric heterogeneity alone therefore does not supply a tight headline.

`python/swappa_rescue/run_robustness.py` runs six predeclared follow-ups: wider support; finer action and cost-interval grids; forward and legacy outcome timing; and simpler mean/minimum demand specifications. It writes source hashes, commit, dirty state, command arguments, statuses, and logs to `output/swappa_rescue_20261016/robustness_manifest.json`. Check that manifest before launching anything else.

The next theoretical calculation should investigate candidate-dependent payoff ranges and realized context counts under a defensible feedback model. A full-information contextual Hedge benchmark can supply bounds without iid daily cost draws, but requires an actual decision/feedback opportunity per round and a justified initial state. An anytime bound must also handle endogenous context counts (listing survival depends on past actions); do not silently condition a fixed-count tail inequality on those counts. Removing negative-profit actions can lower payoff ranges only when the learner is assumed to know its cost and to exclude such dominated actions. All observed actions must be compatible with that assumption. Compare bounds under several explicit warm-start entropy allowances. Compute the implied numerical envelopes before investing in expensive demand bootstraps.
