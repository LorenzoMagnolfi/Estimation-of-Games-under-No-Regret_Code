# Second-order Swappa diagnostic: argument and limitations

> **October 20, 2026 interpretation update:** The custom expected-profit learner is closed as a conditional construction, not a verified rescue of Swappa. Read `UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md` first. Earlier success language describes numerical informativeness under added assumptions; it does not establish their empirical credibility.


Internal research record, 2026-10-16. Status: mathematical proposal and conditional computational diagnostic, not an application confidence statement. No manuscript edits are authorized by this note. Related implementation: `python/swappa_rescue/second_order_diagnostic.py`.

## Why this extension

Worst-range envelopes make every pricing deviation pay for extreme losses on unrelated actions. An alternative controls the sum of squared excess payoffs separately for each comparator. The construction below extends the multiple-rate Prod argument to realized randomized actions. It supplies a joint high-probability regret event suitable for the current method's witness argument. It does not follow from ANR alone.

Reference: Gaillard, Stoltz and van Erven (2014), *A Second-order Bound with Excess Losses*, COLT, https://proceedings.mlr.press/v35/gaillard14.html. Their Theorem 1 is for mixture loss. The randomized-action martingale step below is our derivation, not a claim that their theorem directly covers this application.

## Explicit learner and proof

There are A experts, here residual-price policies, and J learning-rate copies per expert. Copy (a,j) has a fixed positive dollar learning rate eta_aj and positive wealth w_aj. At the beginning of the observation window normalize total wealth to one. Assume w_aj >= exp(-H)/(A J). This is an explicit initialization restriction.

At round t, draw expert (a,j) with probability proportional to eta_aj w_aj. Its policy chooses the price. Let U_t denote its realized expected structural payoff, and u_at the counterfactual expected payoff of expert a at the same state. Feedback reveals the entire payoff vector. The state and payoff vector may depend on past actions but not on the current random draw. Private cost and observable covariates can be inputs to a policy; iid cost draws are unnecessary.

Update each wealth by w_aj <- w_aj [1 + eta_aj (u_at - U_t)]. Require a known, pre-sample one-sided support bound u_at - U_t >= -K_a and eta_aj <= 1/(2 K_a). Write r_at=u_at-U_t, R_a=sum_t r_at and V_a=sum_t r_at^2.

The total wealth W_t is a nonnegative martingale: its conditional increment is sum_aj eta_aj w_aj [u_at - E(U_t | past,state)], which is zero by the choice probabilities. Ville's inequality implies P(sup_t W_t >= 1/alpha) <= alpha. The elementary inequality log(1+x) >= x-x^2 holds for x >= -1/2. Therefore log w_aj,t >= log w_aj,0 + eta_aj R_a - eta_aj^2 V_a. On W_t < 1/alpha, simultaneously at every horizon and for every copy,

    R_a <= [log(A J) + H + log(1/alpha)] / eta_aj + eta_aj V_a.

The minimum over the J predetermined rates is valid because all rate copies are actually part of the learner. Optimizing a continuum of rates after observing the sample without this construction would be invalid. The implementation uses J=10 dyadic rates from 1/(2 K_a).

For D separate learners on interleaved subsequences, allocate alpha/D to each. Summing their bounds at the same comparator and rate gives

    R_a <= D [log(A J) + H + log(D/alpha)] / eta_aj + eta_aj V_a.

This is an explicit alternative feedback architecture, not an adjustment for iid sample size. It illustrates the cost of infrequent feedback. No restriction is required on the sequence of contexts beyond nonanticipation and the support bound. The Ville event also handles endogenous stopping and random context counts. Multiple contexts would require separate learners and an explicit probability allocation or joint wealth construction.

## Link to the new inference method

For each candidate cost parameter and demand primitive, compute the exact structural excess payoffs. Retain a candidate if its observable/latent-cost witness satisfies every comparator-specific inequality. At the true cost and demand primitive, the displayed event supplies a feasible witness with probability at least 1-alpha. For empirical-composition targets and a model imposing no population composition restriction, no iid composition approximation is needed. Any population restriction added later needs its own valid consistency treatment. Any estimated-demand confidence set must contain the true payoff primitive with its own allocated probability; profiling over that set then gives the usual joint coverage argument. We have not implemented that first-stage uncertainty component yet.

The code tests only 51 residual deviations plus no offer. The learner's action menu is all 501 integer residuals from -250 to 250 plus no offer (A=502). The smaller tested set gives necessary, potentially weak, restrictions. Actual prices of the five focal sellers fit this integer menu relative to rounded contemporaneous reference prices. Payoffs are evaluated on the continuous price interval as an upper bound on the menu maximum. Costs are fixed in dollars for each listing, anchored to its initial analyzed reference price.

The one-sided bound may be tighter than a two-sided range: for a comparator a, max_p u_t(p)-u_t(a) bounds its possible negative excess payoff. Positive excess payoffs need no upper bound for the log inequality. No-below-cost posting is NOT imposed in this diagnostic.

## What the current code does not establish

1. K_a is calculated from the observed state envelope, with scale factors one and two. This is a diagnostic of the numerical potential, not a justified ex-ante uniform support bound. A data-dependent learning-rate grid is not licensed by the proof. To claim coverage, specify an appropriate known state/payoff support or a valid scale-adaptive learner and its exact additional terms.
2. The initialization sensitivity is H in {0, log A, 5 log A}. These bounds are not automatic for sellers active before the sample. A strongly adaptive learner might avoid a restart assumption, but its own proof and constants are needed.
3. Listing-days are not necessarily sequential full-feedback decisions. D in {1,10,100} is a sensitivity exercise. Actual timing and feedback opportunities must justify one architecture.
4. The first stage is fitted demand, not a known structural primitive. Predictive improvements do not establish causal price effects. Sale timing remains ambiguous. The quality model uses age since first observed for every listing and battery controls; the initial-price model is excluded here to avoid changing the counterfactual meaning at entry.
5. Proportional or additive entry-reference costs are restrictive deterministic models. Their finite-grid accepted points are not certified continuous parameter projections. No population cost-distribution estimate follows from these diagnostics.
6. The payoff cost is an inventory opportunity value unless a procurement/replenishment model connects it to acquisition marginal cost. All current proceeds and margins are gross of fees and other selling costs.

The useful empirical test is whether second-order restrictions retain nonempty, economically informative ranges across demand specifications, support allowances, initialization, and feedback frequencies. A tight range obtained only with uniform initialization and one full-feedback update per listing-day is not a successful rescue.
