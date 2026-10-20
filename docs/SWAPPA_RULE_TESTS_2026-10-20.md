# Swappa rule tests and cost inversion: implementation record

**Subsequent demand and regret work:** `gadgetpickup-demand-regret-note-2026-10-20.md` records the completed demand audits and sequential sales-information calculation. It supersedes this note's prospective demand-validation status; the original conditional cost results below remain unchanged.

Date: 2026-10-20. Status: chronological rule validation and exact fitted-demand cost sensitivity calculations complete. No empirically validated confidence interval is claimed. The human note in the paper sibling, `JPE Revision/swappa-rule-tests-note-2026-10-20.tex`, governs economic interpretation and mathematics. This record governs implementation and provenance. Earlier source research remains in `SWAPPA_EMPIRICAL_FOUNDATIONS_2026-10-20.md`; the old learner remains closed under `UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md`.

## Decision and user correction

Recovering unknown costs is the purpose of the method. Neither the researcher lacking costs nor the seller using its own cost records is an identification objection. In the new computations every listing cost is unknown. The substantive outstanding restrictions concern counterfactual demand, economic cost interpretation, feasible alternatives, and a valid finite-window performance bound. Do not repeat the earlier argument that cost must be observed before the method can work.

The tested repricing families do not validate GadgetPickup's adoption of a specific rule. They remain usable profit comparisons under an explicitly broader performance restriction. No new exact learner is inferred from them. Small ask reductions and withdrawal can bound a separate expected-sales cost average without a cost cap, common costs, or a cost-dispersion bound. The fitted bounds are informative at sufficiently small assumed allowances, but no certified informative allowance has been established here. An ordinary EXP3-IX guarantee on the 52-calendar-day window is vacuous even with two actions.

## Frozen design and provenance

- Original design: `docs/evidence/gadgetpickup_rule_design_20261020.json`, commit `3047402`, before any fits. Training through September 30, validation October 1--31, test November 1--December 22, 2023.
- Prediction code: `python/swappa_rescue/empirical_pricing_rules.py`, introduced `3853379`. Corrected ask-grid projection at `2413ce4` before examining fit metrics. The correction replaced rounding with floor inversion to preserve an admissible unchanged displayed price. Initial outputs were overwritten before selection was inspected. Statistical design was unchanged.
- Price-only cost code: `python/swappa_rescue/policy_cost_frontier.py`, introduced `ccb4148`; shortened output filenames for Windows at `3b4b77f`. Original price predictions were frozen before loading demand or costs.
- Withdrawal design: `docs/evidence/gadgetpickup_withdrawal_design_20261020.json`, `74d2283`, after the original cost results but before withdrawal calculations. Code `withdrawal_cost_frontier.py` introduced `1c9ee82`; trusted local object-array loading corrected `e465482` before successful completion.
- Expected-sales target: `docs/evidence/gadgetpickup_expected_cost_design_20261020.json`, `c5775a1`, after the observed-sales target proved very wide, before calculating the new target. Code `expected_cost_frontier.py`, `83c9668`. This is a disclosed estimand change, not an additional cost restriction or a pre-analysis primary result.
- Attribution and focused tests: `cost_attribution.py` and `test_empirical_rules.py`, `e188ebe`.

Machine: ASCECN-NC330225. Environment variable: `SWAPPA_DATA_VERSION=fullsource_refscreen_known`. Python: `C:/Users/magnolfi.1/.venvs/research/Scripts/python.exe`. No HPC job, push, or seller contact was used.

Run from the code root, in this order:

```text
python python/swappa_rescue/empirical_pricing_rules.py
python python/swappa_rescue/policy_cost_frontier.py
python python/swappa_rescue/withdrawal_cost_frontier.py
python python/swappa_rescue/expected_cost_frontier.py
python python/swappa_rescue/cost_attribution.py
python python/swappa_rescue/test_empirical_rules.py
```

Use the absolute interpreter above and set the environment variable first. The scripts save to the established ignored output tree and to tracked evidence files. They do not refit demand.

## Inputs and artifacts

Input directory: `output/swappa_rescue_20261016/fullsource/reference_screened/known_status/`.
Input panel: `panel.parquet`, read through `demand_timing_sensitivity.terminal_status_data()`.
Frozen demand objects: `continuation_seller_full_1.joblib` and `multiproduct_initial_terminal.joblib`.
Run directory: `empirical_rules_20261020/` below the input directory.

The run directory contains seller forecast Parquets, `validation_candidates.csv`, `rule_validation.json`, `policy_cost_frontier.json`, `withdrawal_frontier.json`, `expected_frontier.json`, and four compact coefficient archives: `gp_seller_coef.npz`, `gp_common_coef.npz`, `up_seller_coef.npz`, `up_common_coef.npz`. Coefficients preserve observed and proposed probabilities, proposed prices, listing IDs, both target ingredients, and every comparison label.

Tracked results in `docs/evidence/`:

- `gadgetpickup_rule_validation_20261020.json`
- `gadgetpickup_policy_cost_frontier_20261020.json`
- `gadgetpickup_withdrawal_frontier_20261020.json`
- `gadgetpickup_expected_frontier_20261020.json`
- `gadgetpickup_cost_attribution_20261020.json`

The prediction result records design, script, and panel SHA256 plus code commit. The price-only cost result additionally records forecast and demand-object hashes. Withdrawal records hashes of its coefficient inputs. Expected-target and attribution results use those same coefficient archives and record their script/design/commit provenance. Follow this chain rather than treating a local ignored artifact as an unspecified input.

## Prediction construction

Five sellers: GadgetPickup; UPGRADE SOLUTION INC; AMS Traders Inc.; Certified Cell, Inc; SaveGadget. There are 1,562 candidates per seller. Families and all grids are in the frozen design. Categories are `device_id|Storage|Condition`.

All scored rows have consecutive retained dates for a listing. A rival minimum is lagged within the listing. The category median is calendar-lagged one day. Sales are terminal sold-status indicators aggregated by category, with seven prior calendar days and missing dates filled with zero. The waiting clock is elapsed time since the last observed price change, initialized at first observation. It is not complete inventory age. Initial and gap rows are excluded from scoring. Training price floors are category minimum or 10th percentile; unseen categories use the declared fallback of 10. GadgetPickup has 301 unseen-category evaluation rows, so this limit remains visible.

Selection uses October category-day-weighted MAE and separately an equal-weight average of change and no-change MAE. Exactly tied candidates retain the first declared grid value. The selected forecasts for each criterion are saved; duplicates are deliberately retained. October selection precedes November evaluation. The frozen demand fits themselves use the full sample: this is price-forecast holdout validation, not demand holdout validation.

GadgetPickup: 5,638 eligible test pairs, 870 changes. Primary MAE: persistence 1.883975; own-category median 1.936643; competitor 10.051237; waiting rule 1.942266; sales rule 1.911299. Selected competitor: offset -20, training minimum, Thursday, deadband 20. Selected waiting rule: 28 days, step 10, daily, no floor. Selected sales rule: low threshold 0, high 2, step 10, cooldown 28, daily. Balanced selection does not reverse the substantive failure.

Upgrade: 1,563 pairs, 17 changes. MAE: persistence .029333, category 1.340175, competitor 1.744479, waiting .097831, sales .088770. Full metrics for all five sellers and both selectors are in the JSON. No formal universal test rejects all repricers; this evaluates the specified families and information proxies.

## Cost program and units

There are 52 calendar dates. GadgetPickup has 6,429 retained rows, 832 listing-cost variables, and 471 eligible terminal sales; Upgrade has 1,820 rows, 178 cost variables, and 131 eligible terminal sales. Costs persist over the evaluation exposures and differ freely across listings.

Let L be eligible rows. For comparison k, A[k] is the change in total expected receipts divided by L; B[k,j] is the change in listing j's cumulative expected sales divided by L. The exact gain is A[k] - B[k,:] @ c. LP constraints are -B @ c <= epsilon - A. Costs are nonnegative. The supplementary cap is each listing's minimum receipt equivalent in the test window; this cap is never silently imposed in the primary nonnegative-only case.

All numerical prices obey display=ceil(1.03*integer ask); x=1.03*ask is a receipt equivalent, not verified seller net payout. Fixed ask shifts are relative to the previous ask, not current contemporaneous price. They and rule forecasts affect only eligible rows. Initial/gap rows remain at their actual prices, but their predicted sales spillovers enter total seller profit. Own total price is updated within market ID and raw own count preserved. Rival prices, quality controls, initial-price controls, calendar controls, and reference quotes remain fixed. This is a one-step comparison on realized histories, not a recursively simulated inventory path.

Withholding all modeled inventory has A0=-sum(x*q)/L and B0j=-sum_n(q_jn)/L. It requires zero incremental payoff relative to retention and a behavioral allowance covering this enlarged participation choice. Current platform cancellation feasibility is documented at https://swappa.com/faq/answer/cancel-listing (accessed October 20, 2026); no historical snapshot was verified. This is stronger historical evidence only if 2023 feasibility is independently established. The FAQ does not certify any performance guarantee.

Both priors and all allowance profiles [0,.01,.05,.1,.25,.5,1,2,5,10] are saved. No allowance was chosen from interval width.

## Results and attribution

Observed-sales mean: under nonnegative costs alone, all original price-only cases are [0,infinity], even at zero allowance. Withdrawal reduces the GadgetPickup upper bound to 12,145.59 (seller fit, all comparisons, zero allowance), still unusable. Under the extra no-below-receipt cap, GadgetPickup's interval is [165.59,399.54] at zero and [70.19,399.54] at one dollar. The rule-only original comparisons yield the entire cap-defined prior even at zero. The cap, not learning theory, supplies the upper restriction.

Separate expected-sales target: v_j=sum_n(q_jn)/sum_jn(q_jn) includes all retained inventory, including entry/gap rows. It enters expected seller profit exactly. Nonnegative unrestricted costs plus all comparisons and withdrawal give GadgetPickup [120.58,466.16] at zero, [92.58,475.87] at one, and [64.58,485.59] at two dollars. Upgrade: [76.34,147.63], [47.54,158.52], [25.40,169.41]. Costs can be highly heterogeneous; no hidden cost span is used.

GadgetPickup's entire lower endpoint is certified by the -10 ask comparison: max_j B_kj/v_j=.03570988948 and zero-cost gain A_k=4.305968231. Thus lower=max(0,(4.305968231-epsilon)/.03570988948), attained by the LP at reported budgets. The fitted rules alone plus withdrawal yield lower zero. The upper is expected mean receipt + L*epsilon/expected_sales, also attained. At zero allowance the implied expected gross margin is [0,345.58]; positive allowances allow negative aggregate margins. No strictly positive markup is established without additional restrictions.

The expected target's lower endpoint disappears at epsilon=4.305968231 for GadgetPickup and 3.164545975 for Upgrade, under the seller fit. These are breakdown points, not calibrated errors. Common-demand GadgetPickup zero-allowance bounds are [129.50,470.56]. Do not select the tighter fit; a maintained ambiguity between models calls for a union, and two fits do not form a statistical confidence set.

## Diagnostics and guarantees

Within retained market-days, raw own count exactly equals retained seller count in both focal samples. The maximum omitted count is zero. This resolves the possible within-market omitted-owned-profit issue for these particular runs. It does not establish economic coverage of entirely omitted markets or outside channels.

GadgetPickup all-row sales: 585 realized terminal indicators versus 580.447 predicted (seller fit). Mean receipt across those actual sales is 392.61135 versus expected-sales mean 466.16149. For the common fit predicted sales are 598.46997 and expected mean receipt 470.55909. Aggregate sales calibration conceals substantial price-composition discrepancy. Upgrade: 157 realized versus 143.54152 predicted; receipt means 138.71541 versus 147.63262. These are descriptive, not formal rejection tests, and fitted full-sample demand is not a causal validation.

Neu (2015), Theorem 1: https://proceedings.neurips.cc/paper_files/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf . For normalized losses and an initialized prefix, eta=2*gamma=sqrt(2*log(K)/(K*N)). At N=52 and 95 percent, the bound is 214.31060 for K=14, 222.21982 for K=15, and 91.60527 for K=2, all above the logical maximum of 52. This is an optimistic diagnostic, not an adoption fit or dollar allowance. Realized-payoff normalization, feedback delays, expected-profit conversion, selection, and inherited initial state remain unverified. With unbounded candidate costs, a common finite payoff range cannot be silently assumed.

The exact new-method mapping is in the human note, including the paper's Q_f consistency requirement, Theorem CR, unit conversion from cumulative regret to epsilon, expanded history-rule comparison class, and demand uncertainty. These LPs implement an obedience-component sensitivity exercise. They are not the full finite-sample prediction set or confidence region.

## Verification and unresolved work

Completed checks:

- All admissible displayed prices remain unchanged under the price-grid projection.
- Perturbing current/future prices, rival minima, and sale indicators does not change forecasts based on information through the prior date; tested over the entire declared candidate family.
- A toy cost problem recovers both analytic endpoints and becomes unbounded above when withdrawal is removed, verifying constraint signs and normalization.
- Direct expected-profit differences agree with A-Bc at heterogeneous test costs within 1e-10.
- Leaving all prices unchanged reproduces observed fitted demand within 1e-12.
- Withdrawal endpoints agree with analytic formulas; supplemental intervals expand monotonically with the allowance.
- GadgetPickup's lower endpoint agrees with its single-comparison analytic certificate.

The three focused unit tests pass. Runtime identity and analytic endpoint assertions also pass. The trusted local npz files contain object listing IDs; allow_pickle is used only to read these files generated in this run. No external serialized object is accepted by that path.

Next executable research step: audit demand calibration by model/storage/condition and price range using the retained raw-status and transaction-timing information. Predetermine the demand validation target before further cost inversion. Independently seek data on actual review timing, feedback, and the observed-window performance restriction. Do not choose a new demand fit, cost cap, learning parameter, or cost span by the tightness of bounds. Extra invoice data would help validate interpretation, but recovering unknown opportunity costs remains the method's task.

Substantive audit: checked inequality signs, denominators, target changes, timing, full-sample demand use, current-versus-2023 feasibility, and claims against saved results. Reader audit: human note leads with the result, explains the economic gain before equations, distinguishes costs from net markups, and separates assumptions from data. Remaining material limitations are explicitly reported rather than silently resolved. Source-only human note: static checks and local verification are recorded in the paper handover; no PDF or canonical manuscript edit is claimed.
