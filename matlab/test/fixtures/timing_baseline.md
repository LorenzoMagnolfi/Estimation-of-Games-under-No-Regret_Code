# Timing Baseline

## Purpose
Track wall-clock times for each computational stage before and after refactoring.
The fixture runner (`run_fixtures.m`) saves timing data to `fixture_timing.mat`.
This file documents the baseline measurements and post-refactor comparisons.

## Reduced-Scale Config
| Parameter | Stage II | Stage III | Stage IV |
|-----------|----------|-----------|----------|
| maxiters  | 5,000    | data-driven | 5,000  |
| NGridV    | 20       | 20        | 20       |
| NGridM    | 20       | 20        | 20       |
| s (types) | 5        | 5         | 5        |
| Bootstrap B | —      | —         | 10       |
| Solver calls | 400  | 800 (2 pl) | 400    |

## Baseline Run (Pre-Refactor)
- Date: (pending — run `run_fixtures.m` on MATLAB machine)
- MATLAB version:
- Machine:

| Stage | Time (s) | Notes |
|-------|----------|-------|
| II    |          |       |
| III   |          |       |
| IV    |          |       |
| TOTAL |          |       |

## Post-Refactor Runs
Record each refactor phase here after running `run_fixtures.m` again.

### Phase 1: Eliminate Globals + Extract Game Setup
- Date:
- MATLAB version:

| Stage | Time (s) | Speedup | Notes |
|-------|----------|---------|-------|
| II    |          |         |       |
| III   |          |         |       |
| IV    |          |         |       |
| TOTAL |          |         |       |

### Phase 2: Unified Solver (coneprog)
- Date:
- MATLAB version:

| Stage | Time (s) | Speedup | Notes |
|-------|----------|---------|-------|
| II    |          |         |       |
| III   |          |         |       |
| IV    |          |         |       |
| TOTAL |          |         |       |

### Phase 3a: Parallelize Solver Grid
- Date:

| Stage | Time (s) | Speedup | Notes |
|-------|----------|---------|-------|
| II    |          |         |       |
| III   |          |         |       |
| IV    |          |         |       |
| TOTAL |          |         |       |

### Phase 3b: Replace AMPL with linprog
- Date:

| Stage | Time (s) | Speedup | Notes |
|-------|----------|---------|-------|
| I     |          |         |       |

## Notes
- All timings from the reduced-scale fixture runner, NOT full production runs.
- The fixture runner exercises the same code paths at ~1/100 scale.
- Expected baseline total: 15-25 min.
- Full production run: ~7.6 h (estimated from original MAIN scripts).
