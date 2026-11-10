# The equilibrium benchmark against no regret on Swappa: a horse race

*Lorenzo Magnolfi, 9 November 2026. Audience: Lorenzo; Niccolò Lomys and his agents if shared. Status: new results, not yet discussed with Niccolò; manuscript, response letter and the 8 and 9 November notes untouched. Code: `python/swappa_rescue/swappa_horserace_{tables,core,eval,lab,figs,evidence}_20261109.py` (uncommitted at the time of writing). Unity jobs 11392145 (payoff tables) and 11392198 (evaluation and laboratory), run root `~/NoRegret_runs/swappa_horserace_20261109/`. Evidence `docs/evidence/swappa_horserace_20261109.json`; figures `docs/evidence/swappa_horserace_*.{pdf,png}`.*

## Bottom line

The natural benchmark for the Swappa application is the equilibrium supply side: the seller's prices satisfy the first-order condition of expected profit, with the same logit-hazard demand and the same value-linked cost, c = θr. In the stationary world that the test of record maintains, the benchmark does not get it wrong. Across 500 simulated windows on GadgetPickup's own environment, with the regularized leader and the measured experience, its 95 percent interval covers the true cost ratio in every window, its mean error is 0.01 Lerner points, and its interval is 2.1 points wide against 18.0 for the test of record.

It gets it wrong when the window is not the seller's steady state. When the seller's experience predates a 3 percent change in its cost relative to the reference, the seller forgoes 1.0 percent of expected profit in the window. The equilibrium estimate is then off by 2.2 to 2.4 Lerner points (GadgetPickup and AMS Traders), and its interval covers the truth in none of 500 windows; the steady-state set covers in 16 to 44 percent, the test of record in all 500. In the data, the benchmark's own restriction, one cost ratio for the window, fails. Its interval on the first half of the window is disjoint from its interval on the second half for 9 of 13 sellers under the Decluttr reference and for 3 of 13 under the internal index; the test-of-record sets of the two halves overlap for all 13 sellers under both references. The regret frontier of Section 3 maps forgone profit into error: a GadgetPickup that forgoes 1 percent of its expected profit is consistent with any Lerner index in [10.5, 15.2] percent.

## 1. The benchmark

The researcher who assumes equilibrium recovers marginal cost from the first-order condition of each seller's pricing problem (Rosse, 1970; Bresnahan, 1989; Berry, Levinsohn and Pakes, 1995; Nevo, 2001). We implement it with the objects of the application, so that the comparison isolates the behavioral assumption. Demand is `raw_d0`, cost is c = θr with r the Decluttr quote (the internal index as robustness), and the price direction is the seller's level. The condition is that the derivative of the window's expected profit along a proportional shift of the seller's prices is zero:

mean_t D_t(θ) = 0, with D_t(θ) = [R_t(+0.5%) − R_t(−0.5%)] − θ [CR_t(+0.5%) − CR_t(−0.5%)],

where R_t and CR_t are the day's expected revenue and expected cost-reference mass, Σ p q and Σ r q, at the shifted prices (owned-peer means rescaled, rivals fixed). The estimate is θ_N = Σ_t ΔR_t / Σ_t ΔCR_t. The 95 percent interval inverts |mean D(θ)| ≤ 1.96 sd(θ) on the θ grid, with the variance conventions of the test of record: lag-0 sample variance of the daily series, demand delta method, and lag-0 demand cross term.

Three remarks place the benchmark. First, it is the limit of the steady-state inequalities (A2, the moment inequalities of Pakes, Porter, Ho and Ishii, 2015) as the shift goes to zero, since the inequalities for +k and −k bound the derivative from both sides. The horse race therefore isolates one assumption, zero regret against bounded regret, and the steady state enters as the set-valued version of equilibrium. Second, with a logit in log price and small daily hazards, the equilibrium Lerner index is close to the inverse of the seller-level price elasticity: prices move the inferred cost, not the inferred margin. Third, the benchmark conditions on realized rival prices, as our moments do; no difference between Nash and coarse correlated play enters, and the only wedge is whether the seller's level is optimal in the window.

## 2. What the three procedures say on the data

Lerner index of sold phones (revenue weighted), percent, 95 percent, demand `raw_d0`, eight lattice level shifts (±1, ±2.5, ±5, ±10 percent in logs), learner action set of nine levels (d = 9). Halves split the window at 1 October 2023. The last column is the forgone profit at the equilibrium point when the seller is compared, month by month, with the best level shift of that month (share of expected daily profit).

| Seller | H (cells) | Sales | Equilibrium | Steady state | Test of record | Equilibrium, halves | Disjoint halves (eq / st / rec) | Monthly forgone profit |
|---|---:|---:|---|---|---|---|---|---:|
| The Cellular Resource (Decluttr) | 186,460 | 360 | 17.9 [17.0, 18.9] | [16.0, 19.7] | [8.2, 23.4] | [14.9, 18.1], [17.6, 19.1] | no / no / no | 2.02% |
| (internal) | | | 16.7 [15.8, 17.6] | [15.2, 18.7] | [9.1, 23.7] | [17.6, 22.7], [14.7, 16.3] | yes / no / no | 2.22% |
| We Buy Mobile (Decluttr) | 170,697 | 336 | 12.2 [11.1, 13.5] | [10.0, 14.9] | [4.6, 19.8] | [8.9, 12.2], [13.0, 16.0] | yes / no / no | 1.22% |
| (internal) | | | 14.3 [13.2, 15.7] | [12.4, 16.8] | [7.2, 21.7] | [12.9, 15.9], [12.9, 15.4] | no / no / no | 2.37% |
| GadgetPickup (Decluttr) | 49,196 | 1,638 | 12.6 [11.6, 13.8] | [10.5, 14.9] | [2.9, 20.9] | [9.2, 11.1], [13.8, 16.3] | yes / yes / no | 1.25% |
| (internal) | | | 12.9 [11.8, 14.1] | [10.7, 15.2] | [3.6, 21.2] | [11.3, 13.5], [12.1, 14.6] | no / no / no | 0.16% |
| AMS Traders (Decluttr) | 38,159 | 1,316 | 12.2 [11.3, 13.4] | [10.4, 14.5] | [2.9, 19.6] | [8.8, 11.0], [12.9, 14.8] | yes / no / no | 0.94% |
| (internal) | | | 11.6 [10.7, 12.6] | [9.8, 13.7] | [3.5, 18.9] | [11.2, 14.0], [10.1, 11.8] | no / no / no | 0.45% |
| PDX Cellular (Decluttr) | 31,463 | 300 | 11.1 [10.0, 12.5] | [9.3, 13.4] | [−0.2, 19.6] | [7.3, 11.0], [12.2, 14.2] | yes / no / no | 1.99% |
| (internal) | | | 11.5 [10.3, 12.8] | [9.3, 13.8] | [0.3, 19.6] | [9.3, 13.1], [10.8, 12.8] | no / no / no | 1.02% |
| Sellworld (Decluttr) | 29,696 | 357 | 12.9 [11.4, 14.5] | [10.3, 15.9] | [0.4, 26.4] | [4.0, 11.4], [13.4, 15.3] | yes / no / no | 0.90% |
| (internal) | | | 13.5 [12.2, 15.1] | [11.1, 16.8] | [2.6, 27.9] | [11.4, 19.4], [12.2, 14.2] | no / no / no | 0.08% |
| Wireless Source (Decluttr) | 23,944 | 389 | 21.3 [19.2, 23.7] | [18.3, 24.9] | [−14.4, 39.5] | [19.7, 30.4], [18.5, 19.9] | no / no / no | 43.93% |
| (internal) | | | 22.4 [20.2, 25.0] | [19.3, 26.2] | [−14.6, 40.3] | [21.4, 31.9], [19.0, 20.7] | yes / no / no | 61.36% |
| NYCPhonebuyer (Decluttr) | 15,359 | 747 | 12.8 [11.5, 14.1] | [10.3, 15.9] | [1.8, 27.6] | [10.3, 12.9], [12.6, 15.6] | no / no / no | 0.21% |
| (internal) | | | 12.6 [11.4, 14.1] | [10.2, 15.8] | [1.9, 27.7] | [12.9, 15.8], [9.6, 12.3] | yes / no / no | 0.58% |
| SaveGadget (Decluttr) | 13,247 | 581 | 15.3 [14.4, 16.3] | [13.4, 17.4] | [0.9, 25.1] | [12.3, 15.0], [15.5, 16.8] | yes / no / no | 0.60% |
| (internal) | | | 15.0 [14.2, 16.1] | [13.1, 17.2] | [2.9, 26.3] | [14.7, 18.2], [13.7, 15.0] | no / no / no | 0.55% |
| TD'S Cell Shop (Decluttr) | 10,872 | 418 | 16.5 [15.3, 17.8] | [14.2, 19.2] | [6.0, 26.6] | [12.5, 16.7], [16.4, 18.7] | no / no / no | 0.68% |
| (internal) | | | 16.0 [14.6, 17.4] | [13.8, 18.8] | [5.7, 26.3] | [14.9, 19.6], [14.4, 16.6] | no / no / no | 0.86% |
| Gilly's (Decluttr) | 6,468 | 425 | 13.2 [11.8, 14.7] | [10.3, 16.5] | [−0.7, 31.0] | [8.0, 10.7], [14.7, 18.5] | yes / yes / no | 2.41% |
| (internal) | | | 13.8 [12.6, 15.2] | [11.4, 16.9] | [1.6, 29.9] | [11.7, 14.3], [13.4, 16.3] | no / no / no | 0.20% |
| Upgrade Solution (Decluttr) | 1,265 | 952 | 13.0 [11.8, 14.4] | [10.9, 15.3] | [−34.5, 29.0] | [9.5, 11.8], [15.0, 19.1] | yes / yes / no | 3.11% |
| (internal) | | | 12.7 [11.7, 13.8] | [10.6, 14.9] | [−19.6, 27.2] | [12.2, 13.8], [10.6, 13.8] | no / no / no | 0.20% |
| Certified Cell (Decluttr) | 784 | 554 | 11.1 [9.7, 12.6] | [8.6, 13.7] | [−35.9, 77.4] | [3.8, 10.0], [11.1, 13.4] | yes / no / no | 4.16% |
| (internal) | | | 10.2 [8.9, 11.7] | [8.0, 12.8] | [−35.0, 77.5] | [6.1, 12.2], [9.7, 12.0] | no / no / no | 3.61% |

The three procedures agree on the center: the equilibrium point lies inside the steady-state set and the test-of-record set for every seller under both references, and between 10 and 18 percent for every seller except Wireless Source. The steady-state set for GadgetPickup reproduces the 9 November value, [10.54, 14.90]; the test-of-record set, [2.9, 20.9], differs from the 9 November [2.63, 20.90] only through the lattice version of the level shifts. Wireless Source's monthly figure reflects a July in which its expected profit at the equilibrium point is close to zero; we do not read it.

## 3. The regret frontier

For each candidate θ, the frontier reports the largest window-average gain of the eight level shifts, as a share of expected daily profit at θ: the profit the seller must be forgoing for θ to be its true cost ratio. It is computed from the data alone and is the object behind every set in Table 1 (figure `docs/evidence/swappa_horserace_frontier.png`). For GadgetPickup under the Decluttr reference:

| Forgone profit, share of expected profit | 0.1% | 0.25% | 0.5% | 1% | 2% | 5% | 6.9% (record allowance) |
|---|---|---|---|---|---|---|---|
| Lerner index consistent with it | [11.9, 13.5] | [11.6, 13.8] | [11.1, 14.4] | [10.5, 15.2] | [10.0, 16.3] | [8.6, 19.0] | [8.1, 20.4] |

The internal reference gives the same shape, [10.7, 15.4] at 1 percent. Expected profit is flat at the optimum, so a small loss of profit admits a large error in the price level and hence in the inferred cost: the frontier widens with the square root of forgone profit, and it is asymmetric because pricing below the optimum, toward zero margins, is costlier than pricing above it. The equilibrium interval, [11.6, 13.8], coincides with the 0.25 percent band. Equilibrium inference on GadgetPickup thus amounts to the claim that the seller forgoes less than one quarter of one percent of its expected profit, 1.48 dollars a day out of 591.

## 4. The laboratory

We build a laboratory with known truth on each seller's actual window. The environment is the seller's 164 days of SKU-day cells with their states; its observed prices are the relative-price profile; demand `raw_d0` is the truth; cost is θ* r with θ* the equilibrium estimate on the data (0.800 for GadgetPickup), so that the simulated prices look like the real ones. The seller chooses a level each day from nine levels (0, ±1, ±2.5, ±5, ±10 percent) with the learner of `08_CLOSING_PROOFS.md`: the regularized leader with entropy, c = 1, constant scale equal to the root-mean-square cell span at θ*, step √(ln 9 / s) per cell, full feedback once a day, one level drawn per day for all of the seller's cells. Its H presample cells are whole days drawn from the window's days. We run three families of data-generating processes. First, Assumption S holds: presample and window days i.i.d., with the measured H, with H/16, H = 784 and H = 0, and with θ* moved so that the payoff-maximizing level falls between grid points (off-grid). Second, the window days in calendar order. Third, the seller's experience predates a change: the presample is generated under θ*(1 + b), the window under θ*, b from −5 to +5 percent, with the measured H and with H/16. In every window we run the three procedures exactly as on the data; the demand delta-method term of the real data is kept in every interval, although demand is known in the laboratory, because a practitioner would report it. Every cell has 500 windows.

GadgetPickup (AMS Traders gives the same picture; full tables in the evidence file):

| Process | H | Forgone profit in the window | Equilibrium error, Lerner points (s.d.) | Coverage: equilibrium (without demand term) | Steady state | Test of record | Width: eq / st / rec |
|---|---:|---:|---|---|---|---|---|
| Assumption S, measured H | 49,196 | 0.00% | +0.01 (0.30) | 1.00 (0.96) | 1.00 | 1.00 | 2.1 / 4.2 / 18.0 |
| Assumption S, H = 0 | 0 | 0.00% | +0.03 (0.26) | 1.00 (0.99) | 1.00 | 1.00 | 2.2 / 4.3 / 108.4 |
| Calendar order | 49,196 | 0.00% | +0.10 (0.13) | 1.00 (1.00) | 1.00 | 1.00 | 2.1 / 4.3 / 18.0 |
| Off-grid, optimum at −3.5% | 49,196 | 0.13% | −1.08 (0.33) | 0.89 (0.11) | 1.00 | 1.00 | 2.6 / 5.0 / 19.9 |
| Change b = −1% | 49,196 | 0.05% | +0.66 (0.32) | 0.95 (0.44) | 1.00 | 1.00 | 2.2 / 4.4 / 18.3 |
| Change b = −2% | 49,196 | 0.35% | +1.34 (0.30) | 0.32 (0.01) | 1.00 | 1.00 | 2.3 / 4.5 / 18.6 |
| Change b = −3% | 49,196 | 0.95% | +2.34 (0.33) | 0.00 (0.00) | 0.44 | 1.00 | 2.4 / 4.7 / 18.9 |
| Change b = −5% | 49,196 | 2.47% | +3.57 (0.31) | 0.00 (0.00) | 0.00 | 1.00 | 2.5 / 4.9 / 19.6 |
| Change b = +2% | 49,196 | 0.39% | −1.34 (0.29) | 0.28 (0.00) | 1.00 | 1.00 | 2.0 / 3.8 / 17.6 |
| Change b = +3% | 49,196 | 1.00% | −2.17 (0.29) | 0.00 (0.00) | 0.32 | 1.00 | 1.9 / 3.6 / 17.5 |
| Change b = +5% | 49,196 | 2.95% | −3.51 (0.32) | 0.00 (0.00) | 0.00 | 1.00 | 1.9 / 3.5 / 18.4 |
| Change b = −3%, H/16 | 3,075 | 0.14% | +0.92 (0.30) | 0.82 (0.18) | 1.00 | 1.00 | 2.2 / 4.4 / 91.4 |

Four results follow. First, under Assumption S the regularized leader is at the optimum on average within the window for every experience level, including a fresh learner: the window holds 10,120 cells, the learner concentrates within days, and the softmax of a flat-topped profit is centered at the optimum. The benchmark is then unbiased and precise, and the forgone profit is below 0.001 percent of expected profit. Second, a learner whose experience predates a change carries the old level into the window, since 49,196 cells of history outweigh 10,120 cells of new evidence; the error of the benchmark is the lag, and it grows with the square root of forgone profit: 1.3 Lerner points at 0.35 percent, 2.2 to 2.3 at 1 percent, 3.5 to 3.6 at 2.5 to 3 percent. Third, these errors lie on the frontier computed from the data (right panel of `docs/evidence/swappa_horserace_lab_GadgetPickup.png`), so the frontier predicts how wrong the benchmark is for a given forgone profit. Fourth, the steady state, which keeps the slack of the ±1 percent shifts, survives changes up to 2 percent and fails beyond. Certified Cell, with 784 cells of experience and wider equilibrium intervals, is harder to fool: its equilibrium interval covers in 91 percent of windows at b = −3 percent and 52 percent at b = −5 percent.

## 5. The benchmark's own restriction in the data

The benchmark asserts one cost ratio for the whole window. Its intervals on the two halves of the window are disjoint for 9 of 13 sellers under the Decluttr reference (GadgetPickup [9.2, 11.1] against [13.8, 16.3]) and for 3 of 13 under the internal index; at least one pair of its six monthly intervals is disjoint for 13 of 13 sellers under the Decluttr reference and 10 of 13 under the internal index. The steady-state halves are disjoint for 3 of 13 and 0 of 13; the test-of-record halves for 0 of 13 under both. Every interval includes the demand term, which is common to both halves, so disjointness is a conservative signal.

Two readings fit these facts, and prices alone do not separate them. Under the first, the seller's cost relative to the reference changes within the window; this is the drift of the Decluttr reference of finding F21, and the internal index removes most of it (GadgetPickup's monthly equilibrium intervals have a common point under the internal index). The cost model A1 is then misspecified for every procedure, and the test of record survives only because its allowance is wide. Under the second, the seller adjusts its level with a lag to a market that moves, the third family of the laboratory; the benchmark reads the lag as cost, and the test of record covers. The month-by-month forgone profit in Table 1, between 0.45 and 2.4 percent for 7 of 13 sellers under the internal index, has the magnitude at which the benchmark fails in the laboratory. It is an interval regret, not the window-level regret that biases the benchmark, so it shows that departures of the right size occur in the data, not that they persist over the window.

## 6. What we can claim, and what we cannot

We can claim that the equilibrium benchmark is valid only within a narrow band of regret. For GadgetPickup its interval is the set of cost ratios with forgone profit below 0.25 percent of expected profit; a learner that forgoes 1 percent moves its estimate by 2.2 Lerner points and its interval never covers the truth in the laboratory; and its intervals on the two halves of the window are disjoint for 9 of 13 sellers under the declared reference. The test of record covers in every laboratory cell and gives overlapping half-sample sets for every seller.

We cannot claim that the benchmark is wrong on Swappa's margins. The three procedures agree on the center, between 10 and 18 percent outside Wireless Source, and every equilibrium point lies inside the test-of-record set. Under Assumption S as calibrated, the benchmark is right and its interval is 8.6 times narrower than the test-of-record set; the test-of-record allowance, 6.9 percent of expected profit, is about 8,000 times the mean forgone profit of the maintained learner in the laboratory, 0.0008 percent. In the change processes, the test of record covers because its allowance exceeds the realized forgone profit (at most 3.2 percent), not by Lemma 2, which needs Assumption S across the presample and the window. The half-sample overlap of the test of record is partly mechanical, since wide sets rarely separate.

The horse race therefore yields a statement about robustness, not a different answer. The equilibrium benchmark obtains a 2-point interval by assuming that the seller forgoes less than a quarter of one percent of its expected profit. The data cannot verify this assumption, and a 3 percent change in costs relative to the reference, smaller than the drift of the Decluttr reference within the window, is enough to break it.

## 7. Proposed exhibits and next steps

1. The frontier figure as the application's main exhibit on what no regret adds over equilibrium, with the GadgetPickup row of the frontier table and the halves of Table 1 in one paragraph.
2. The laboratory figure and table in the appendix, next to the simulation sections' coverage exhibits, as a third laboratory on Swappa's own environment.
3. A sharper allowance under Assumption S. The laboratory shows that the maintained learner forgoes almost nothing, so a gap-dependent bound (Proposition 2's route, offered on 8 November and not run) could tighten the test of record substantially in the stationary case without giving up the guarantee.
4. Laboratory extensions, each one Unity job of under ten minutes on the saved tables: a fixed-share learner in the change processes (where the any-window layer is a theorem), a change at the iPhone 15 launch inside the window, and the off-grid and change processes for all 13 sellers.

## 8. Provenance

| Item | Location |
|---|---|
| Payoff tables, 13 sellers, lattice j = −40..40 (step 0.005), R, CR, CRI, S per cell; demand gradients, V, date influence | `output/swappa_horserace_20261109/tables_raw_d0_*.npz`, `demand_raw_d0.npz` (gitignored; Unity run root `results/`); script `swappa_horserace_tables_20261109.py`, sha256 `ad61d1b2…`; job 11392145, 1 min 5 s; checks: saved standardization reproduced, level-0 revenue and the +1 percent shift equal `Seller.ev` to 1e−8 |
| Estimators shared by data and laboratory | `swappa_horserace_core_20261109.py`, sha256 `f7c17339…` |
| Real-data evaluation, 13 sellers, two references | `swappa_horserace_eval_20261109.py`, sha256 `25166e26…`; job 11392198; `output/swappa_horserace_20261109/horserace_eval_raw_d0.json` |
| Laboratory, GadgetPickup, AMS Traders, Certified Cell, 17 processes, 500 windows each, seed 20261109 | `swappa_horserace_lab_20261109.py`, sha256 `1933a270…`; job 11392198 (9 min 46 s in all); `horserace_lab_*.json` |
| Compact evidence and tables | `swappa_horserace_evidence_20261109.py`; `docs/evidence/swappa_horserace_20261109.json` |
| Figures | `swappa_horserace_figs_20261109.py`; `docs/evidence/swappa_horserace_frontier.{pdf,png}`, `swappa_horserace_lab_*.{pdf,png}` |
| Base | code `main` at `38dd2e7`; demand fit and panel hashes as in the 8 November records (`raw_d0_coef.npz` `f43f031e…`, `raw_reference_panel.parquet` `c1fd3e64…`) |

Not validated: coverage of the steady-state and test-of-record sets on the real data beyond the laboratory (the block bootstrap of 8 November covers the steady state under A2 only); the laboratory uses one draw of the level per day and the A9 lattice, and the equilibrium interval keeps the real-data demand term although demand is known in the laboratory, which favors the benchmark.
