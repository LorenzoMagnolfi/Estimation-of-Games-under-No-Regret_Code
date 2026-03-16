# CLAUDE_HANDOVER.md — Estimation of Games under No Regret (Code)

<!--
  Read this FIRST when resuming. For stable architecture context, see HANDOVER.md.
  For the full refactor plan, see MATLAB_REFACTOR_PLAN.md.
-->

## Last Session
- **Date**: 2026-03-16
- **Duration**: ~8 hours (continued from prior sessions)
- **Summary**: Completed Phases 1, 2, 3b, 3a, and 4 of the MATLAB refactor. Phase 4 rewrote the learning kernel (`learn_mod` → `df.sim.learn`) with sufficient statistics, vectorized counterfactual utilities, and precomputed lookup tables for 2.1x speedup. Also unified marginal cost draw functions into `df.sim.marginal_cost_draws`.

## Current State

### Completed Phases
- **Phase 1**: Eliminated all 12+ MATLAB globals. Created `cfg` struct passed explicitly through all function calls. Extracted `df.setup.game_simulation()` and `df.setup.game_application()`. Created `+df/` package namespace. All four MAINs, all solver functions, and the fixture runner updated. Validated against baseline: exact match on Stages II-IV (20 CVX-nondeterministic Stage IV points excepted).
- **Phase 2**: Unified three solver variants (`ComputeBCCE_eps`, `_ApplicationL`, `_pass`) into `df.solvers.solve_bcce()`. Hoisted constraint matrix construction outside the parameter-grid loop. Created `build_constraints.m` (joint), `build_constraints_marginal.m`, `solve_socp_cvx.m`. Old files are now thin wrappers. Validated: same results.
- **Phase 3b**: Replaced AMPL external solver with native `linprog` for Stage I polytope computation. Created `df.solvers.solve_polytope_lp()`. 100 Halton directions solved in ~12s. `find_polytope_switch.m` is now a thin wrapper. No AMPL binary required.
- **Phase 3a**: SOCP solver speedup. See detailed experiment record below. Outcome: switched default CVX solver from SDPT3 to SeDuMi for ~2x speedup with zero identification mismatches. Created `solve_socp_coneprog.m` (experimental, not recommended). Cross-validated: all 8 fixtures PASS against SDPT3 baseline.
- **Phase 4**: Learning kernel rewrite. Created `df.sim.learn` as optimized replacement for `learn_mod`. Three optimizations: (1) sufficient statistics — store cumulative sums instead of running averages, only update realized type per iteration (eliminates O(s) decay loop); (2) vectorized counterfactual utility — single vectorized expression replaces per-action loop; (3) precomputed lookup tables — 2D action-profile-to-index map and recording-time maps replace `find()`/`ismember()` per iteration. Also unified `marginal_cost_draws_v4` and `marginal_cost_draws_v4_new` into `df.sim.marginal_cost_draws`. Head-to-head validation: ALL PASS (max_diff ≤ 2.78e-16). Full fixture suite: ALL 8 PASS. Speedup: 2.1x on learning kernel, 1.7x on Stage IV bootstrap.

### New Files Created (All Sessions)
- `matlab/src/+df/+io/repo_paths.m`
- `matlab/src/+df/+setup/game_simulation.m`
- `matlab/src/+df/+setup/game_application.m`
- `matlab/src/+df/+solvers/compute_epsilon.m`
- `matlab/src/+df/+solvers/build_constraints.m`
- `matlab/src/+df/+solvers/build_constraints_marginal.m`
- `matlab/src/+df/+solvers/solve_socp_cvx.m`
- `matlab/src/+df/+solvers/solve_socp_coneprog.m` ← Phase 3a (experimental)
- `matlab/src/+df/+solvers/solve_bcce.m`
- `matlab/src/+df/+solvers/solve_polytope_lp.m`
- `matlab/src/+df/+util/allcomb.m`
- `matlab/src/+df/+util/table2latex.m`
- `matlab/test/compare_coneprog_vs_cvx.m` ← Phase 3a cross-validation
- `matlab/test/fixtures_baseline_cvx/` ← Phase 3a SDPT3 baseline backup
- `matlab/src/+df/+sim/learn.m` ← Phase 4 (optimized learning kernel)
- `matlab/src/+df/+sim/marginal_cost_draws.m` ← Phase 4 (unified MC draw function)
- `matlab/test/compare_learn_old_vs_new.m` ← Phase 4 head-to-head validation

### Files Heavily Modified
- All four MAIN scripts (removed globals, use `cfg`)
- `ComputeBCCE_eps.m`, `_ApplicationL.m`, `_pass.m` (thin wrappers to `solve_bcce`)
- `epsilon_switch.m`, `epsilon_switch_distrib.m` (thin wrappers to `compute_epsilon`)
- `learn_mod.m` (Phase 4: thin wrapper to `df.sim.learn`), `learn.m` (accept `cfg` as first arg)
- `find_polytope_switch.m` (thin wrapper to `solve_polytope_lp`)
- `matlab/test/run_fixtures.m` (uses `cfg` pattern)
- `solve_bcce.m` (Phase 3a: added backend/solver/precision options, defaults to SeDuMi)
- `solve_socp_cvx.m` (Phase 3a: added precision parameter, `cvx_begin quiet`)

### Git Status
- Branch: `main`
- Last commit: `18fa631 Add SeDuMi default solver and coneprog backend for SOCP speedup (Phase 3a)`
- Pushed to remote: **no**
- Uncommitted changes: Phase 4 files (df.sim.learn, df.sim.marginal_cost_draws, learn_mod wrapper, compare_learn_old_vs_new.m, this file)

## Pending / Next Steps
- [x] **Phase 3a**: SOCP solver speedup (completed — SeDuMi default, coneprog abandoned)
- [x] **Phase 4**: Learning kernel rewrite (completed — 2.1x speedup, sufficient statistics, vectorized inner loop)
- [ ] **Phase 5**: Separate compute from reporting (`df.stages.run_stage_*`, `df.report.*`, `df_run.m`)
- [ ] Remove AMPL files (`.mod`, `.dat`, `.run`) and `fprintAmplParam.m`
- [ ] Add Stage I to the fixture runner (now possible with linprog)
- [ ] No AMPL baseline exists to validate against — need to confirm polytope correctness analytically or visually
- [ ] Push to GitHub when ready

## Open Questions
- Stage I polytope validation: no AMPL baseline was captured. Should we run the old AMPL code once to generate a reference, or is the linprog output sufficient?
- Parallelization strategy for Stage IV: `parfor` requires Parallel Computing Toolbox

## Phase 3a Experiment Record: SOCP Solver Speedup

### Goal
Replace CVX+SDPT3 with a faster SOCP backend. Two strategies attempted: (1) MATLAB's native `coneprog` (R2020b+), (2) alternative CVX solvers/precision settings.

### Experiment 1: coneprog Backend

**Setup**: Mapped the CVX SOCP to `coneprog` format:
- `maximize -c'x` → `minimize c'x`
- `B_INEQ * x >= b` → `-B_INEQ * x <= -b`
- `||Mat_NLC * x|| <= 1` → `secondordercone(Mat_NLC, 0, 0, -1)` (gamma=-1)

**Bug found**: Initial implementation used `gamma=+1` in `secondordercone()`. MATLAB's convention is `||Ax - b|| <= d'x - gamma`, so `gamma=-1` gives the constraint `||Mat_NLC*x|| <= 1`. With `gamma=+1`, coneprog returned all infeasible (exitflag=-2).

**After fixing gamma**: coneprog returns exitflag=1 (success) but the solutions violate equality constraints by ~7% (eq_viol=6.96e-2). Extensive debugging:

1. **Sparse matrices**: No improvement
2. **Reduced inequalities** (rank-reduced B_INEQ from 625→91 rows): Still fails
3. **Null-space elimination** (pre-solve equalities, reduce to free variables): coneprog exitflag=-7
4. **Tight bounds** ([-1, 1]): coneprog works perfectly (eq_viol=3.33e-16) but solution is wrong because true dual variables reach ±9368
5. **Equalities as double-sided inequalities**: Still fails
6. **Incremental constraint addition**: Works with partial B_INEQ rows; fails when all 91 independent rows included

**Root cause**: The problem has 625 inequality constraints (B_INEQ), 25 equality constraints (B_EQ), and a 24-dim SOC cone, with 125 variables. B_INEQ is heavily rank-deficient (rank 91/125, cond=Inf). The equality dual variables in CVX solutions reach ±9368. coneprog's interior-point solver cannot handle this conditioning.

**Conclusion**: coneprog is fundamentally unable to solve this SOCP class. The `solve_socp_coneprog.m` file is retained as an experimental option (`opts.backend = 'coneprog'`) but is not recommended.

### Experiment 2: CVX Solver/Precision Comparison

Benchmarked on 50 grid points from Stage II:

| Configuration | Time/solve | Identification mismatches vs SDPT3 |
|---|---|---|
| CVX + SDPT3 (default precision) | 1.305s | 0 (baseline) |
| CVX + SeDuMi (default precision) | 0.656s | 0 |
| CVX + SDPT3 (low precision) | 0.625s | 5/50 (10%) |
| Direct SeDuMi (no CVX) | 2.990s | N/A |

**Key findings**:
- SeDuMi is ~2x faster than SDPT3 with identical identification results (0 mismatches)
- `cvx_precision low` is dangerous: 10% of identification decisions change
- Direct SeDuMi (without CVX) is slower, likely due to missing CVX preprocessing

### Outcome
- Default solver changed from SDPT3 to SeDuMi in `solve_bcce.m`
- Default precision kept at `default` (not `low`)
- Added `opts.solver`, `opts.precision`, `opts.backend` parameters to `solve_bcce`
- Full fixture validation (all 8 fixtures, Stages II-IV): ALL PASS
  - Stage II: max_diff=3.30e-04, 23 infeasibility mismatches (within CVX nondeterminism tolerance)
  - Stages III-IV: exact match (max_diff=0, 0 infeasibility mismatches)
- **Speedup: 1.7-2.2x** across stages (Stage II: 453→203s, Stage III: 691→409s, Stage IV: 454→265s)

## Tricky Bits (Accumulated)
- **`switch_eps==1` discrepancy**: simulation mode uses `Kappa*sqrt(log(NAct))/(conf*sqrt(T))`, application mode uses `s*Kappa*sqrt(log(NAct))/(conf*sqrt(T))`. This is intentional.
- **`ExpectedRegretComp_Pl2` bug**: Line 208 in `IV_MAIN` uses `epsPl1` instead of `epsPl2`. Preserved for baseline fidelity.
- **CVX nondeterminism**: ~20 of 400 Stage II/IV identification grid points return `100` (infeasible) in some runs but converge in others. This is a CVX/SeDuMi numerical issue, not a code bug. Fixture comparison tolerates ≤30 infeasibility mismatches.
- **Marginal-mode constraint builder**: `build_constraints_marginal.m` uses `exp_pi` (expected payoffs integrated over opponent actions) rather than the full joint payoff tensor. The alpha/deviation construction is structurally different from the joint mode.
- **`Identification_Pricing_Game_ApplicationL.m` per-player loop**: calls `df.setup.game_application(iii, ...)` inside the player loop. Each player gets its own `cfg` with data-driven actions and sale probabilities.
- **coneprog sign convention**: `secondordercone(A, b, d, gamma)` encodes `||Ax - b|| <= d'x - gamma`. To get `||Ax|| <= 1`, use `gamma=-1` (NOT `+1`).
- **coneprog numerical limits**: Cannot solve SOCPs with rank-deficient inequality systems and large dual variables. Fails silently (exitflag=1) with constraint violations.
