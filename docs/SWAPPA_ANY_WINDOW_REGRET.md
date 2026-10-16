# Any-window second-order regret with a shared prior

Internal derivation, 2026-10-16. This extends the randomized Prod argument in `SWAPPA_SECOND_ORDER.md`. It is not yet manuscript text. The purpose is to remove the sample-start reset assumption and make the benchmark genuinely asymptotically no-regret, rather than stopping at a finite fixed grid of learning rates.

## Learner

For each of A pricing policies and each integer j>=0, create a copy with eta_j=2^(-j)/(2K) and prior mass pi_aj=1/[A(j+1)(j+2)]. The prior sums to one by telescoping. Use the realized-action randomized Prod choice rule: select copies with probability proportional to eta_j times their current wealth. After observing the full payoff vector, multiply each wealth by 1+eta_j r_at, where r_at is that policy's excess payoff over the chosen action. Its negative part must be bounded by K, so every factor is at least one half.

After global lifetime round t, mix wealth back toward the prior:

    w_aj,t = (1-gamma_t) w'_aj,t + gamma_t pi_aj W'_t,
    gamma_t = 1/(t+1).

This sharing preserves total wealth W_t=W'_t and therefore preserves the nonnegative martingale property. It also ensures w_aj,t/W_t >= pi_aj/(t+1). The algorithm is a benchmark class with a provable bound; we do not claim these merchants literally implement it.

## An arbitrary observation window

Let the window start after s lifetime rounds and contain N further rounds. Normalize by W_s. The share floor at the start is pi_aj/(s+1). The product of the retention fractions during the window is

    product_{t=s+1}^{s+N} t/(t+1) = (s+1)/(s+N+1).

Using log(1+x)>=x-x^2 for x>=-1/2 gives

    log(w_aj,s+N/W_s) >= log pi_aj - log(s+N+1)
                           + eta_j R_a - eta_j^2 V_a.

Conditional on the entire pre-window history, Ville's inequality bounds the probability that W_t/W_s ever exceeds 1/alpha by alpha. Thus simultaneously over policies, rates and window endpoints,

    R_a <= [log(1/pi_aj) + log(s+N+1) + log(1/alpha)]/eta_j
           + eta_j V_a.

A known upper limit T_max on s+N+1 can replace the unknown lifetime clock. There is no assertion of uniform initial weights at the data-collection date. For D separate interleaved learners, a union bound with alpha/D and summation yields D times the logarithmic numerator plus eta_j times the total squared excess payoffs. Independent learners need not have statistically independent state processes.

The inference program tests j=0,...,7, a subset of the restrictions implied by the infinite learner. The prior weights remain the infinite-prior weights, not 1/8. The implementation profiles lifetime clock bounds 10^5, 10^6 and 10^7, and D=1 or 10. Choosing a lifetime bound because it produces a desired estimate would be invalid; these are explicit maintained histories and sensitivity cases.

## Why this benchmark is ANR

For bounded excess payoffs, V_a<=N K^2. The countably infinite rates contain a rate within a factor of two of the order sqrt(log N/N)/K. The policy/rate prior penalty at such a rate is O(log A+log log N), while the sharing term is O(log(s+N)). The displayed average regret therefore tends to zero for fixed s. The high-probability event can be taken at arbitrarily small alpha, yielding the corresponding almost-sure asymptotic result; boundedness also gives expected convergence. A finite set of fixed positive rates would leave a nonvanishing worst-case variance term as N grows. Earlier finite-grid pilots are finite-horizon diagnostics and should not alone be described as an asymptotic guarantee.

## Application restrictions remain separate

This construction addresses history and learning-rate support. It does not establish that each listing-day is a sequential feedback opportunity, that sellers receive full expected-payoff feedback, that estimated demand is causal or known, or that inventory values equal procurement costs. The total-profit range must include own-inventory substitution. The current application uses the explicit demand and price-support restrictions in `coordinate_profit_shared.py`, and continues to treat its fitted-demand outputs as conditional pilots.
