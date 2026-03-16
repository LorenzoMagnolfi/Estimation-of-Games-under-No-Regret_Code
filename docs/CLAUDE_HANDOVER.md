# CLAUDE_HANDOVER.md — Estimation of Games under No Regret (Code)

<!--
  Read this FIRST when resuming. For stable architecture context, see HANDOVER.md.
  For the full refactor plan, see MATLAB_REFACTOR_PLAN.md.
-->

## Last Session
- **Date**: 2026-03-16
- **Duration**: ~10 hours (continued from prior sessions)
- **Summary**: Completed Phase 5 of the MATLAB refactor. Extracted computation into `df.stages.run_stage_{i,ii,iii,iv}` orchestrators and reporting into `df.report.*` shared modules. Each MAIN script is now a thin wrapper (~20-80 lines) calling setup → compute → report. Net -1297/+236 lines across 5 modified files; 10 new modules created. All 8 fixtures PASS cross-validation against CVX baseline.

## Current State

### Completed Phases
- **Phase 1**: Eliminated all 12+ MATLAB globals. Created `cfg` struct passed explicitly through all function calls. Extracted `df.setup.game_simulation()` and `df.setup.game_application()`. Created `+df/` package namespace. All four MAINs, all solver functions, and the fixture runner updated. Validated against baseline: exact match on Stages II-IV (20 CVX-nondeterministic Stage IV points excepted).
- **Phase 2**: Unified three solver variants (`ComputeBCCE_eps`, `_ApplicationL`, `_pass`) into `df.solvers.solve_bcce()`. Hoisted constraint matrix construction outside the parameter-grid loop. Created `build_constraints.m` (joint), `build_constraints_marginal.m`, `solve_socp_cvx.m`. Old files are now thin wrappers. Validated: same results.
- **Phase 3b**: Replaced AMPL external solver with native `linprog` for Stage I polytope computation. Created `df.solvers.solve_polytope_lp()`. 100 Halton directions solved in ~12s. `find_polytope_switch.m` is now a thin wrapper. No AMPL binary required.
- **Phase 3a**: SOCP solver speedup. Switched default CVX solver from SDPT3 to SeDuMi for ~2x speedup with zero identification mismatches. Created `solve_socp_coneprog.m` (experimental, not recommended). Cross-validated: all 8 fixtures PASS against SDPT3 baseline.
- **Phase 4**: Learning kernel rewrite. Created `df.sim.learn` as optimized replacement for `learn_mod`. Three optimizations: sufficient statistics, vectorized counterfactual utility, precomputed lookup tables. Also unified `marginal_cost_draws_v4` and `marginal_cost_draws_v4_new` into `df.sim.marginal_cost_draws`. Speedup: 2.1x on learning kernel, 1.7x on Stage IV bootstrap.
- **Phase 5**: Separated compute from reporting. Created 5 `df.report.*` modules (build_param_grid, classify_identified_set, plot_identified_set, write_tables, plot_regret_histogram) and 5 `df.stages.*` orchestrators (run_stage_i through iv, plus compute_cost_statistics). Each MAIN is now a thin wrapper. Validated: all 8 fixtures PASS cross-validation against CVX baseline with identical results.

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
- `matlab/src/+df/+report/build_param_grid.m` ← Phase 5 (shared grid construction)
- `matlab/src/+df/+report/classify_identified_set.m` ← Phase 5 (shared SVM classification)
- `matlab/src/+df/+report/plot_identified_set.m` ← Phase 5 (shared visualization)
- `matlab/src/+df/+report/write_tables.m` ← Phase 5 (Stage III table export)
- `matlab/src/+df/+report/plot_regret_histogram.m` ← Phase 5 (Stage IV broken-axis histogram)
- `matlab/src/+df/+stages/run_stage_i.m` ← Phase 5 (Stage I orchestrator)
- `matlab/src/+df/+stages/run_stage_ii.m` ← Phase 5 (Stage II orchestrator)
- `matlab/src/+df/+stages/run_stage_iii.m` ← Phase 5 (Stage III orchestrator)
- `matlab/src/+df/+stages/run_stage_iv.m` ← Phase 5 (Stage IV orchestrator)
- `matlab/src/+df/+stages/compute_cost_statistics.m` ← Phase 5 (Stage III MC helper)

### Files Heavily Modified
- All four MAIN scripts (removed globals → cfg → thin wrappers)
- `ComputeBCCE_eps.m`, `_ApplicationL.m`, `_pass.m` (thin wrappers to `solve_bcce`)
- `epsilon_switch.m`, `epsilon_switch_distrib.m` (thin wrappers to `compute_epsilon`)
- `learn_mod.m` (Phase 4: thin wrapper to `df.sim.learn`), `learn.m` (accept `cfg` as first arg)
- `find_polytope_switch.m` (thin wrapper to `solve_polytope_lp`)
- `Identification_Pricing_Game_ApplicationL.m` (Phase 5: uses `df.report.build_param_grid`)
- `matlab/test/run_fixtures.m` (uses `cfg` pattern)
- `solve_bcce.m` (Phase 3a: added backend/solver/precision options, defaults to SeDuMi)
- `solve_socp_cvx.m` (Phase 3a: added precision parameter, `cvx_begin quiet`)

### Git Status
- Branch: `main`
- Last commit: `6db10f9 Separate compute from reporting with stage orchestrators and shared modules (Phase 5)`
- Pushed to remote: **no**
- Uncommitted changes: **none** (clean working tree)

## Pending / Next Steps
- [x] **Phase 1**: Eliminate globals (completed)
- [x] **Phase 2**: Unify solvers (completed)
- [x] **Phase 3b**: Replace AMPL with linprog (completed)
- [x] **Phase 3a**: SOCP solver speedup (completed — SeDuMi default)
- [x] **Phase 4**: Learning kernel rewrite (completed — 2.1x speedup)
- [x] **Phase 5**: Separate compute from reporting (completed — stage orchestrators + shared modules)
- [ ] Remove AMPL files (`.mod`, `.dat`, `.run`) and `fprintAmplParam.m`
- [ ] Add Stage I to the fixture runner (now possible with linprog)
- [ ] No AMPL baseline exists to validate against — need to confirm polytope correctness analytically or visually
- [ ] Push to GitHub when ready

## Open Questions
- Stage I polytope validation: no AMPL baseline was captured. Should we run the old AMPL code once to generate a reference, or is the linprog output sufficient?
- Parallelization strategy for Stage IV: `parfor` requires Parallel Computing Toolbox

## Phase 5 Design Notes

### build_param_grid auto-detection
`df.report.build_param_grid` auto-detects application vs simulation mode via `isscalar(mu)`. In simulation mode, `gridparamM`/`gridparamV` are multipliers applied to `mu`/`sigma2`; in application mode, they are absolute grid endpoint values. `Identification_Pricing_Game_ApplicationL` passes `mu(1,1)` (scalar) to trigger application mode.

### Stage III reference price path
`run_stage_iii` always uses `df.io.repo_paths()` to find the data directory for reference prices, rather than inferring paths from the distribution file location.

## Phase 3a Experiment Record: SOCP Solver Speedup

### Goal
Replace CVX+SDPT3 with a faster SOCP backend. Two strategies attempted: (1) MATLAB's native `coneprog` (R2020b+), (2) alternative CVX solvers/precision settings.

### Experiment 1: coneprog Backend

**Setup**: Mapped the CVX SOCP to `coneprog` format:
- `maximize -c'x` → `minimize c'x`
- `B_INEQ * x >= b` → `-B_INEQ * x <= -b`
- `||Mat_NLC * x|| <= 1` → `secondordercone(Mat_NLC, 0, 0, -1)` (gamma=-1)

**Bug found**: Initial implementation used `gamma=+1` in `secondordercone()`. MATLAB's convention is `||Ax - b|| <= d'x - gamma`, so `gamma=-1` gives the constraint `||Mat_NLC*x|| <= 1`. With `gamma=+1`, coneprog returned all infeasible (exitflag=-2).

**After fixing gamma**: coneprog returns exitflag=1 (success) but the solutions violate equality constraints by ~7% (eq_viol=6.96e-2). Extensive debugging confirmed coneprog cannot handle the heavily rank-deficient inequality system (rank 91/125, cond=Inf) with large dual variables (±9368).

**Conclusion**: coneprog is fundamentally unable to solve this SOCP class. Retained as experimental option only.

### Experiment 2: CVX Solver/Precision Comparison

| Configuration | Time/solve | Identification mismatches vs SDPT3 |
|---|---|---|
| CVX + SDPT3 (default precision) | 1.305s | 0 (baseline) |
| CVX + SeDuMi (default precision) | 0.656s | 0 |
| CVX + SDPT3 (low precision) | 0.625s | 5/50 (10%) |
| Direct SeDuMi (no CVX) | 2.990s | N/A |

### Outcome
- Default solver: SeDuMi. Default precision: `default`.
- **Speedup: 1.7-2.2x** across stages.

## Tricky Bits (Accumulated)
- **`switch_eps==1` discrepancy**: simulation mode uses `Kappa*sqrt(log(NAct))/(conf*sqrt(T))`, application mode uses `s*Kappa*sqrt(log(NAct))/(conf*sqrt(T))`. This is intentional.
- **`ExpectedRegretComp_Pl2` bug**: Line 208 in original `IV_MAIN` uses `epsPl1` instead of `epsPl2`. Preserved for baseline fidelity in `run_stage_iv`.
- **CVX nondeterminism**: ~20 of 400 Stage II/IV identification grid points return `100` (infeasible) in some runs but converge in others. Fixture comparison tolerates ≤30 infeasibility mismatches.
- **Marginal-mode constraint builder**: `build_constraints_marginal.m` uses `exp_pi` (expected payoffs integrated over opponent actions) rather than the full joint payoff tensor.
- **`Identification_Pricing_Game_ApplicationL.m` per-player loop**: calls `df.setup.game_application(iii, ...)` inside the player loop. Each player gets its own `cfg`.
- **coneprog sign convention**: `secondordercone(A, b, d, gamma)` encodes `||Ax - b|| <= d'x - gamma`. To get `||Ax|| <= 1`, use `gamma=-1` (NOT `+1`).
- **coneprog numerical limits**: Cannot solve SOCPs with rank-deficient inequality systems and large dual variables.
- **build_param_grid application mode**: Pass `mu(1,1)` as scalar to trigger application mode. If `mu` is a vector, simulation mode is triggered (grid uses multipliers).
