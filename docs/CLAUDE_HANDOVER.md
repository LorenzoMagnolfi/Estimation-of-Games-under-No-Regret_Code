# CLAUDE_HANDOVER.md — Estimation of Games under No Regret (Code)

<!--
  Read this FIRST when resuming. For stable architecture context, see HANDOVER.md.
  For the full refactor plan, see MATLAB_REFACTOR_PLAN.md.
-->

## Last Session
- **Date**: 2026-03-16
- **Duration**: ~16 hours (continued from prior sessions)
- **Summary**: Completed Phases 1-5 refactor. Then moved to result reproduction and computational exploration. Generated Stage I figures (validated against paper). Benchmarked SOCP solver alternatives (coneprog, adaptive grid, parfor). Discovered coneprog is numerically unreliable at production scale. Implemented stabilization items. Full-grid CVX+SeDuMi run for 500k identified set in progress.

## Current State

### Completed Phases
- **Phase 1**: Eliminated all 12+ MATLAB globals. Created `cfg` struct passed explicitly.
- **Phase 2**: Unified three solver variants into `df.solvers.solve_bcce()`.
- **Phase 3b**: Replaced AMPL with native `linprog` for Stage I polytope.
- **Phase 3a**: SOCP solver speedup — SeDuMi default (~2x over SDPT3).
- **Phase 4**: Learning kernel rewrite — 2.1x speedup via sufficient statistics.
- **Phase 5**: Separated compute from reporting — stage orchestrators + shared modules.
- **Phase 6 (in progress)**: Result reproduction + stabilization for revision.

### Phase 6: Result Reproduction & Stabilization

#### Figures Generated
- **Stage I**: Polytope plot + convergence simplex scatter — validated against paper ✓
- **Stage II (500k, adaptive)**: Identified set with SVM boundary — shape matches paper, some nonconvexities from adaptive approximation
- **Stage II (500k, full-grid)**: CVX+SeDuMi run in progress (~2hrs, 10,201 solves at ~0.65s/solve)

#### Computational Exploration Results
- **Benchmark (21×21 = 441 grid)**: CVX 0.91s/solve, coneprog 0.21s (4.3x faster), adaptive 7.2x, parfor 3.8x
- **Production scale (101×101 = 10,201 grid)**: coneprog UNRELIABLE — returns all-negative values, 92% disagreement with CVX. Closed as a viable path.
- **Architecture**: Fast backend now uses CVX+SeDuMi with precomputed objectives (constraints built once, objective vectors vectorized). Adaptive grid is exploration-only, not production.

#### Stabilization Items (from code audit)
- [x] **Close coneprog path**: Deprecated `solve_socp_coneprog.m` and `solve_grid_coneprog.m` with warnings. Updated `solve_bcce.m` and `run_stage_ii.m` docstrings. Fixed misleading "coneprog" log message.
- [x] **Halton quality parameter**: Added `opts.quality` ('draft' 50k / 'final' 500k) to `classify_identified_set.m`
- [x] **Stage IV bootstrap parfor**: Added `opts.use_parfor` to `run_stage_iv.m`. Bootstrap loop is textbook parfor — each draw is independent.
- [x] **Batch objective push to Stage IV**: Converted identification exercise from `ComputeBCCE_eps_pass` to `df.solvers.solve_bcce` (builds constraints once). Added progress logging.
- [x] **Fix build_param_grid indexing bug**: `idx = (ind1-1)*NGridM + ind2` → `(ind1-1)*NGridV + ind2`. Benign when NGridM==NGridV but wrong for asymmetric grids.
- [x] **Fix Stage IV NGrid computation**: `NGrid = opts.NGridV * opts.NGridM` → `(NGridV+1) * (NGridM+1)` to account for leading 1 in gridparamV/M.
- [ ] **Replace Excel I/O in Stage III**: Deferred — requires MATLAB to convert multi-sheet XLSX to .mat
- [ ] **Push batch objectives to Stage III**: Deferred — requires refactoring ApplicationL marginal solver path

### New Files Created (This Session)
- `matlab/src/+df/+solvers/solve_grid_cvx.m` — Batch CVX+SeDuMi solver (serial, production reliable)
- `matlab/src/+df/+solvers/solve_grid_coneprog.m` — Batch coneprog solver (DEPRECATED, unreliable)
- `matlab/src/+df/+solvers/solve_grid_adaptive.m` — Adaptive coarse-to-fine grid solver (exploration only)
- `matlab/src/+df/+report/plot_polytope.m` — Native convhull polytope plotting (no MPT3)
- `matlab/src/+df/+report/plot_convergence.m` — Native simplex scatter plotting
- `matlab/test/benchmark_solvers.m` — Comprehensive solver benchmark
- `matlab/test/compare_backends_production.m` — Production-scale coneprog vs CVX comparison
- `matlab/test/run_stage_ii_fast.m` — Stage II fast-backend test runner
- `matlab/test/run_stage_i_fast.m` — Stage I test runner

### Files Modified (This Session)
- `run_stage_ii.m` — Added fast backend (precomputed objectives, CVX+SeDuMi batch, adaptive grid option)
- `run_stage_iv.m` — Added parfor bootstrap, unified solver for identification, fixed NGrid bug
- `solve_bcce.m` — Updated coneprog deprecation warning
- `solve_socp_coneprog.m` — Added DEPRECATED header
- `solve_grid_coneprog.m` — Added DEPRECATED header
- `classify_identified_set.m` — Added quality presets ('draft'/'final')
- `build_param_grid.m` — Fixed indexing bug (NGridM→NGridV)
- `I_MAIN_Simul_2acts.m` — Updated to use native plotting functions
- `solve_grid_adaptive.m` — Added backend option (CVX or coneprog)

### Git Status
- Branch: `main`
- Last commit: `a3e0600 Add native polytope/convergence plots, fix run_stage_ii grid size`
- Pushed to remote: **yes**
- Uncommitted changes: **many** (stabilization items above)

## Pending / Next Steps

### Immediate
1. Wait for full-grid CVX run to finish → validate 500k identified set figure against paper
2. Commit stabilization changes
3. Excel I/O replacement (Stage III)

### Then: NEW RESULTS for REVISION
User has requested: "After stabilization, we move to produce the NEW RESULTS for the REVISION."
- Run all four iterations (500k, 1M, 2M, 4M) for Stage II identified sets
- Stage III application identification
- Stage IV bootstrap regrets + identification
- Generate all paper figures

## Tricky Bits (Accumulated)
- **`switch_eps==1` discrepancy**: simulation mode uses `Kappa*sqrt(log(NAct))/(conf*sqrt(T))`, application mode uses `s*Kappa*sqrt(log(NAct))/(conf*sqrt(T))`. Intentional.
- **`ExpectedRegretComp_Pl2` bug**: Line 90 in `run_stage_iv` uses `epsPl1` instead of `epsPl2`. Preserved for baseline fidelity.
- **CVX nondeterminism**: ~20 of 400 Stage II/IV points return `100` (infeasible) in some runs but converge in others. Fixture comparison tolerates ≤30 mismatches.
- **coneprog sign convention**: `secondordercone(A, b, d, gamma)` encodes `||Ax - b|| <= d'x - gamma`. gamma=-1 gives `||Ax|| <= 1`.
- **coneprog production unreliability**: Returns large negative spurious values at 101×101 scale. 92% disagreement with CVX. Interior-point method cannot handle the ill-conditioned inequality system (rank 91/125, cond=Inf, dual ±9368). DO NOT USE for production.
- **Grid leading 1**: `gridparamV = [1; linspace(..., NGridV)']` gives NGridV+1 elements. Actual grid is (NGridV+1)×(NGridM+1) = 10,201 for NGridV=NGridM=100.
- **Adaptive grid**: Exploration only. Solves ~31% of grid points but boundary approximation introduces nonconvexities. Not suitable for inference/publication figures.
- **build_param_grid application mode**: Pass `mu(1,1)` as scalar to trigger application mode (absolute grid values, not multipliers).
