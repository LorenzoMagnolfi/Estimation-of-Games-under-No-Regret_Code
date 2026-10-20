# Swappa profit-sensitive learning: conditional research candidate

> **October 20, 2026 interpretation update:** The custom expected-profit learner is closed as a conditional construction, not a verified rescue of Swappa. Read `UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md` first. Earlier success language describes numerical informativeness under added assumptions; it does not establish their empirical credibility.


Internal research note, 2026-10-16. This is not manuscript text or a claim that the application has been rescued. No first-stage confidence region has yet been incorporated. Read this together with SWAPPA_ANY_WINDOW_REGRET.md and SWAPPA_RESCUE_CHECKPOINT.md.

## What changed

The separable economic-envelope calculation was too optimistic: a seller changes the sale probabilities of its other listings when it changes one price. Total-inventory profit bounds and an allowance for unobserved learning history weakened those results. A common global payoff cap is also costly because a few high-profit states determine the learning rate for all observations.

The current candidate combines persistent listing costs, total-inventory profits, and confidence-rated experts whose weights depend on a predictable certificate of the profits at stake. It does not impose a Normal cost distribution or independent daily cost draws. The useful numerical case does impose a strong cross-listing restriction: costs lie within a stated dollar distance of a common fraction of the reference price at first observation. That assumption, the learning rule, and the demand model must be defended separately.

The algorithmic antecedent is multiple-rate Prod with confidence-rated advice: Gaillard, Stoltz and van Erven (2014), especially Section 4, [original paper](https://proceedings.mlr.press/v35/gaillard14.pdf). The note below derives the randomized, shared-prior, candidate-cost version used here; it does not claim this construction is the behavior of the observed seller.

## Payoffs and predictable range certificates

For a focal listing i, hold other current prices and inventory fixed. Its counterfactual price x lies in [c_i,P]. Profit includes its own expected unit profit and all the seller's other same-product listings. There is no delisting counterfactual. The fitted demand model has negative own-price log-odds slopes and a positive coefficient b_peer on the mean price of the seller's other listings. Moving the focal price from p_i to x changes peer j's log odds by b_peer*(x-p_i)/(n-1).

Write k_i for the smallest absolute own-price log-odds slope in the current state. Let z_i(c_i) be the focal log odds at cost, holding peers' current prices fixed. A bound on its nonnegative expected profit is

    U_i(c_i) = min(P-c_i, W(exp(z_i(c_i)-1))/k_i),

where W is the principal Lambert W function. This uses the current state and cost, not a uniform cap on sale probability at cost. Peer profits are increasing in the focal price. Hence

    C_i(c) = U_i(c_i)
             + sum_{j != i} (p_j-c_j) [q_j(P)-q_j(c_i)]

bounds the range of total-inventory profit over all focal prices in [c_i,P]. Costs are nonnegative and no greater than every observed price of the listing. Missing risk-set peers, if any, receive a conservative analytic bound; they are never silently assigned zero profit. The terminal-status observation model has complete own-market groups for the focal seller.

The certificate is predictable only if the seller knows the relevant costs and has the maintained range information before selecting its price. Current rivals' prices, quality controls, and current other owned listings are part of that information. Algebraically, the certificate does not depend on the focal action: the focal price contained in peers' current log odds cancels its subtraction in the counterfactual shift. This is a stronger information assumption than feedback arriving only after the action. Estimated demand is treated as known in the pilots, not as established causal information possessed by sellers.

## Learning class and exact finite-sample restriction

For each fixed price policy a, cap index h>=0, and rate index j>=0, set

    K_h = 2^h,
    eta_hj = s * 2^(-j) / K_h,  0<s<1,
    pi_ahj = (1/A) / [(h+1)(h+2)(j+1)(j+2)].

All three index sets define one learner; priors on h and j sum to one. The finite action menu includes every integer price from zero to P, clipped from below at cost, and eleven cost-plus markup policies clipped at P. At P=350 it has A=362 entries. Duplicate actions in some states are harmless.

Before each action, define f_ht=min(1,K_h/C_t(c)). A sleeping variant uses the indicator that C_t(c)<=K_h. Choose among policy copies with probabilities proportional to eta_hj*f_ht*w_ahj,t-1. For realized total-profit excess r_at=u_t(a)-u_t(actual), update

    w'_ahj,t = w_ahj,t-1 [1+eta_hj*f_ht*r_at].

Since |f_ht*r_at|<=K_h, weights remain positive. The conditional expected total wealth after updating equals the preceding total wealth. The action lottery, rather than an assertion that observed actions are mixed actions, supplies this martingale identity. Mix wealth back toward its prior after lifetime round t using share 1/(t+1), as in SWAPPA_ANY_WINDOW_REGRET.md. Sharing preserves total wealth and gives each copy a floor pi_ahj/(t+1).

For an observation window starting after s0 rounds and ending after s0+N, the floor and retention fractions imply

    log(w_ahj,s0+N/W_s0)
      >= log(pi_ahj)-log(s0+N+1)
         + sum_window log(1+eta_hj*f_ht*r_at).

Ville's inequality therefore gives, jointly over all policies, caps, rates, and window endpoints, probability at least 1-alpha that

    sum log(1+eta_hj*f_ht*r_at)
      <= log(1/pi_ahj) + log(T_max) + log(1/alpha),

provided T_max bounds s0+N+1. There is no sample-start reset. For D interleaved copies, a union bound with alpha/D and summation gives D times the logarithmic right side, using the sum over their observations on the left. The pilots use alpha=.005, T_max=1,000,000, and D=1 or 10.

This exact wealth inequality avoids the extra quadratic approximation in the earlier second-order calculation. It can also be written as a regret envelope by moving the exact log compensation to the right. Its coverage argument is the same true-witness projection argument as the new method. The weighted restrictions are additional restrictions on a stronger learning class; they are not consequences of contextual ANR alone and are not a verbatim application of a theorem requiring a deterministic, data-independent scalar envelope. A formal paper extension would need to state this admissible joint event explicitly.

The class is ANR against its finite policy menu when payoff ranges are uniformly bounded: some finite h then has f=1 always, and the countable rate prior contains arbitrarily small rates. Shared-prior penalties grow logarithmically. A finite grid of fixed rates alone would not establish this asymptotic statement. The numerical program tests necessary subsets of the infinite inequalities. For a concrete candidate, the full-policy audit explicitly checks 20 cap scales and 10 rates and bounds the remaining tails analytically.

## Cost model and numerical outer bounds

The structured candidate is

    c_l = rho * reference_at_first_observation_l + e_l,
    rho >= 0,  |e_l| <= delta.

Costs remain fixed over each listing's observations. The errors can be arbitrarily related across listings and to observed characteristics; no iid error, mean-zero error, or density assumption is imposed. Delta is a maintained sensitivity parameter, not an estimated confidence radius. A five-dollar band may be too restrictive without further quality or procurement evidence.

For each interval of rho, the program encloses every admissible cost for every listing, intersecting with nonnegative cost and the minimum posted price. It then bounds the own-profit extrema exactly over each interval, using endpoints, the rival-minimum kink, the price-cap transition, and all interior stationary points. Peer profits and confidence weights receive interval bounds. If the lower bound on an exact log-wealth restriction exceeds its threshold, every cost vector in the box is rejected. Surviving boxes give outer bounds, not an existence result. This is a continuous-parameter enclosure, not acceptance of a finite grid as the parameter space.

A separate script constructs actual listing-cost assignments and tests them against the entire policy menu. It uses static joint-profit first-order conditions only to initialize a cost vector. Those first-order conditions are not imposed in inference and their point estimates are not reported as the no-regret result.

The objective reported as expected_sales is sum_t q_t*c_l / sum_t q_t; its gross margin counterpart subtracts that cost from sum_t q_t*p_t / sum_t q_t. The sold target uses the observed sale labels and prices. Mapping an empirical random sold target into a population transaction-cost estimand needs separate outcome/composition control. None of these gross posted-price-equivalent quantities is automatically procurement cost or a net-of-fees margin.

## Results to preserve, and limitations

The first exact-log profile used cap scales h=2,...,10 and five rate indices. At delta=5, D=1, P=350, terminal-status expected-sales costs were [99.449,129.253] dollars and gross margins [14.654,44.458]. The prior-only cost range was [0,129.253]: the markup lower bound in that profile came from the cost model and no-below-cost restriction, whereas the learning restrictions sharply reduced the upper markup bound. Exact proportional costs (delta=0) were rejected after refinement. At delta=10 the outer cost range widened to [79.287,136.776]; delta=25 was largely uninformative.

For the same initial tested subset, alternative timing gave delta=5 costs [104.256,126.685] under sale_last_seen and [110.044,125.793] under sale_forward. Raising the price ceiling to 650 gave terminal-status costs [98.606,128.501]. These are separate conditional scenarios, not a model-selected confidence interval. Differences in their observation assumptions cannot be resolved by selecting the narrowest result.

The first proposed cost vector at rho=.70 passed the initial subset and all 362 policies at its tested scales, but failed at the previously omitted h=0 by about 7.48 total payoff dollars. This was caught by the expanded audit. A second vector at rho=.69, delta=5 passes all 362 policies, 20 cap scales and 10 rates: its maximum violation is -11.62 dollars. All remaining cap/rate tails pass by analytic bounds. This establishes a concrete witness for the tested exact-wealth restrictions, not that the full structural and learning model is true. The expanded h>=0/eight-rate profile is complete: at delta=5, D=1, costs are [99.449,126.959] and gross margins [16.948,44.458]. At D=10, costs are [85.972,129.253]. Delta=0 remains rejected at D=1. These results supersede the initial subset.

Remaining gates are substantial: causal demand and its uncertainty; defensible delta and cost covariates; a believable feedback/decision clock; sensitivity to the observed sale process; initialization support for the algorithm class; economic interpretation of inventory opportunity cost versus replacement/acquisition cost; and historical fee/net-receipt treatment. The first-stage two-way cluster covariance is only an asymptotic diagnostic and is not a simultaneous confidence set. No paper or response-letter claim is authorized by these pilot calculations alone.

## Files

- profit_specialists.py: range certificates, sleeping/confidence interval bounds, exact own-profit extrema, and unrestricted-cost outer pilot.
- specialist_cost_profile.py: continuous rho-box screening, second-order and exact-log versions, dispersion/timing/support sensitivity.
- specialist_witness.py: candidate assignments and full-menu/tail audit.
- test_profit_specialists.py: profit-range, interval, and exact action-tree checks.
- coordinate_profit*.py: preceding total-inventory and history-aware outer programs.

All outputs are in output/swappa_rescue_20261016 and are ignored by Git. Preserve a source-hash manifest alongside numerical outputs before further code changes.
