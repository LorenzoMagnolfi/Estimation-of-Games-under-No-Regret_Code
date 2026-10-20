# Upgrade learning-rule exercise: closure and interpretation

Date: 2026-10-20. Status: closed as a conditional methodological exercise; not an empirically validated Swappa application. Audience: future researchers and agents. This note governs the interpretation of the earlier Swappa rescue notes. Numerical artifacts at code commit e58b319 remain the record of the calculations; no estimate is recomputed here.

## Read this first

The exercise found informative cost restrictions by constructing a seller who already knows the current expected profits relevant for its price recommendation. The seller combines that recommendation with other rules through specified randomized updates. A probability argument gives a necessary restriction on the seller's observed expected-profit shortfalls. Imposing that restriction, a structural demand model, and tight cost restrictions produces informative conditional intervals.

We have not established that Upgrade uses this algorithm, knows these counterfactual expected profits, or satisfies the resulting finite-sample guarantee. If the seller already knows current expected profits and can freely choose prices, direct maximization is available. The construction supplies no economic reason for weighting other recommendations. Calling the application empirically salvaged was premature. Retain the computation as an example of what stronger assumptions deliver; do not use its precision to justify those assumptions.

The next research question is which additional restrictions can be motivated independently by observed seller behavior, dated platform documentation, or verifiable software practices. Software availability is not adoption, inventory synchronization is not repricing, and repricing is not no-regret learning.

## Economic objects and sample

Upgrade means the Swappa seller UPGRADE SOLUTION INC. The rebuilt source has 465,900 distinct listing-day rows and 34,413 listings. The demand sample has 422,620 rows and 22,151 terminal sale indicators. Upgrade has 10,301 rows, 1,017 listings, and 950 terminal sale indicators. The continuation-pricing exercise uses 804 continuation sales and 1,031 category-day rounds over 165 dates, July 11 through December 22, 2023. Categories distinguish model, storage, and reported condition; 16 occur in the declared universe of 27.

There are 666 gaps in Upgrade histories, or 1,684 missing interior listing-days. The descriptive count of 159 observed within-listing price changes on 32 dates includes changes across gaps. A constant observed price does not establish that the seller did not review it, and a daily scrape does not establish daily updating.

The target is the continuation-sale-weighted mean persistent replacement/opportunity cost represented in a static profit model. The numerical units are receipt-equivalent: integer ask a becomes x=1.03a and displayed p=ceil(x). This is not verified net revenue or invoice acquisition cost. Fees, shipping, returns, repair expense, and the value of waiting require separate economic accounting. The mean x is 144.000149. See daily_fee_conversion.py for the earlier conversion sensitivity.

## Assumptions, separately from observations

1. Estimated conditional demand is structural: changing price as specified changes sale probabilities according to the fitted function. Conditioning on rivals and observed quality does not by itself establish price exogeneity.
2. Prices jointly affect all owned listings within a category; cross-category substitution is omitted. Entry prices are held fixed, including their profit spillovers in comparisons.
3. Persistent costs are nonnegative and do not exceed observed receipt-equivalent prices. Costs are constant within category, or their total within-category span is at most the stated delta. A delta of 5 is an imposed five-dollar total span, not an estimated dispersion.
4. The observed opportunities represent a contiguous window of the modeled updating process; skipped nonzero updates cannot be silently removed.
5. A single learner shares weights across categories, has the specified initial mass and updating coefficients, and has no more than one million lifetime updates.
6. Current expected profits and the profit-range bound are known to the learner before it forms its current best-common-price recommendation. Feedback from actual sales alone does not satisfy this condition.
7. Inference on estimated endpoints uses a clustered Gaussian approximation and regularity of the optimization problem. Numerical tests do not establish those statistical assumptions.

## Exact inequality and its proof

For category-day n, let u_n(a;c,beta) be expected total profit. The designated rule chooses the best common continuation ask given the current state, true costs c, and true demand beta. Let Delta_n be its expected profit minus expected profit at the observed vector of prices. Delta_n can be negative because the observed vector need not be common across listings. Let C_n bound the entire feasible expected-profit range. The restriction is

    W(c,beta) = sum_n log(1 + .99 Delta_n/C_n)
              <= log(4) + log(Tmax) + log(M/.005).

For Tmax=1,000,000 and M=20, the allowance is 23.49585456. For M=10,000, it is 29.71046266. These are log-statistic allowances, not dollars, not standard external regret, and not a bound on average percentage markup.

The sufficient algorithm starts rule e with prior pi_e. It draws action a with probability proportional to the sum of eta_e,t*w_e,t over rules recommending a. After drawing a_t, it forms

    v_e,t = w_e,t [1 + eta_e,t (u_t(a_e,t)-u_t(a_t))],
    w_e,t+1 = t/(t+1) v_e,t + pi_e/(t+1).

The designated copy has pi=1/4 and eta=.99/C_t. Smaller-rate copies and ordinary bounded-payoff comparison rules have positive prior weight. The latter supply the connection to unnormalized ANR on an appropriate bounded game; ANR does not imply the displayed finite-sample restriction.

The action probabilities give conditional expected aggregate weighted excess profit zero. Set Y_t=t*sum_e v_e,t. Then E[Y_t | pre-choice information]=Y_t-1+1 and E[Y_T]=T. Thus Y is a nonnegative submartingale. For every contiguous window [s,t], injection of prior weights implies

    pi_e product_{tau=s}^t [1 + eta_e,tau Delta_e,tau] <= Y_t.

Doob's maximal inequality gives P(max_{t<=Tmax}Y_t >= Tmax/alpha)<=alpha. Taking logs, with alpha=.005/M, gives the restriction simultaneously over contiguous windows. The total weight itself is NOT generally a martingale after additive injection. Earlier notes using that phrase are corrected or explicitly superseded.

This is a proof for the constructed process, not a theorem about commercial repricers. The paper's general feasibility logic can combine this event with an event for uncertain demand; it does not supply the missing empirical justification for this new behavioral event. Passing the necessary restriction does not reproduce the full probability law of the algorithm. A different algorithm could support it only with a corresponding proof, or the performance restriction could be imposed directly as a stronger behavioral assumption.

## Numerical results and what causes the information

| Specification | Conditional asymptotic mean-cost outer interval | Gross dollar margin interval |
|---|---:|---:|
| Seller-specific demand, constant category costs, 20 sellers | [84.333,126.393] | [17.607,59.667] |
| Seller-specific demand, total span 5, 20 sellers | [75.147,134.498] | [9.502,68.853] |
| Seller-specific demand, total span 5, 10,000 sellers | [61.125,139.274] | [4.726,82.876] |

The span-5 prior alone gives cost [0,139.754540] and margin [4.245609,144.000149]. Much of the broad-selection positive margin floor therefore comes from the prior. The performance restriction mainly excludes very low costs and very high margins. For the 20-seller span-5 case the fitted-demand outer interval is [90.293987,127.900108], before widening for estimated demand.

The implementation minimizes and maximizes sale-weighted mean cost subject to conservative lower bounds on W over continuous cost cells. A linear-program relaxation permits mixtures across cells. Consequently accepted endpoints are outer bounds, not proof that a persistent cost vector or a full algorithmic history exists there. A separate fitted-demand witness for the stronger joint-price comparison has mean cost 109.584618 and log statistic at most 14.109626; this is not a point estimate.

The final intervals allocate .005 to the learning event and .045 to asymptotic demand inference. They are not exact finite-sample 95% intervals. Seller-specific nonlinear diagnostics show material curvature. Sixty nonlinear demand perturbations and 62 passing implementation tests are useful checks but do not establish confidence coverage. Larger within-category cost spans and separate category learners weaken the information; failures remain in SWAPPA_DAILY_ENDPOINTS.md.

## Actual-sales learners: diagnostic and limits

Neu (2015), Theorem 1, gives EXP3-IX, for losses in [0,1], K fixed actions, and T rounds,

    R_T <= 2 sqrt(2 K T log K)
           + [sqrt(2 K T/log K)+1] log(2/delta),
    eta=2 gamma=sqrt(2 log K/(K T)).

Source: https://proceedings.neurips.cc/paper_files/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf

For K=339, T=1,031, delta=.05, eta=.00577392396 and the allowance is 5,317.53775, exceeding the logical maximum regret T. This is an idealized feasibility diagnostic, not a fitted bandit model for Upgrade. It grants a known experiment horizon and a fixed action representation. Actual joint prices, inherited learning, and feedback timing require their own treatment. A tighter confidence-tuned bound from the proof is still above T at these values. No informative actual-sales cost interval has been established.

This negative diagnostic does not prove that every bandit guarantee is useless. Structured demand learning, a restricted policy family, small-loss bounds, and informative historical experience are different possibilities. Each must have a justified information set, comparator, payoff normalization, time clock, and finite-sample theorem. Realized Bernoulli sales can have a much larger payoff range than expected profits: the old expected-profit C_n cannot simply normalize realized sales. Standard fixed-action regret does not justify comparison with the best current-state price every day. Translating realized-payoff guarantees to structural expected-profit restrictions also requires an explicit stochastic argument.

A read-only diagnostic on October 19 additionally bounded expected dynamic-oracle rewards at cost zero by 164.099913 after dividing daily total profit by 350 times daily owned inventory. Even a confidence-tuned two-action EXP3-IX allowance at .05 is 208.286127. This is supplementary fixed-demand feasibility evidence under that normalization, not a new estimator, theorem for the observed action space, or an impossibility result. It is not used in a reported cost bound.

## Preserved work and exact entry points

- Governing interpretation: this closure note.
- Human exposition and proof: paper sibling, JPE Revision/swappa-shared-pricing-note-2026-10-19.tex, revised October 20.
- Numerical record: docs/SWAPPA_DAILY_ENDPOINTS.md; history in docs/SWAPPA_RESCUE_CHECKPOINT.md and docs/SWAPPA_UNCERTAINTY_ORACLE.md.
- Data setting: SWAPPA_DATA_VERSION=fullsource_refscreen_known.
- Data/output root: output/swappa_rescue_20261016/fullsource/reference_screened/known_status/.
- Demand fits: continuation_seller_full_1.joblib and multiproduct_initial_terminal.joblib.
- Main code: python/swappa_rescue/daily_cost_model.py, daily_heterogeneous_cost.py, daily_multisku_profile.py, daily_multisku_inference.py; covariance files demand_cluster_seller_covariance.npz and demand_cluster_initial_covariance.npz.
- Earlier output filenames and validation logs are indexed in SWAPPA_DAILY_ENDPOINTS.md. Numerical code of record: e58b319.

## Decision and next action

Close the custom expected-profit learner as a conditional construction. Do not add its cost estimates to the canonical paper or claim a referee concern is resolved on this basis. Retain corrected demand construction, persistent-unit accounting, competing-price variables, continuous-cost computation, and uncertainty diagnostics for future applications.

The next task is an evidence-led specification: document available seller software and historical practices, separate facts from proposals, test candidate price rules and decision timing on the existing panel, and only then assess their identification content. Search failure must be reported as lack of verified evidence, never as proof that a seller has no repricer. No external contact or publication is authorized by this research request.
