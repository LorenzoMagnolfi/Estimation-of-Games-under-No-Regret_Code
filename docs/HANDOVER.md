# Handover — Estimation of Games under No Regret (Code)

## Purpose
Replication codebase for "Estimation of Games under No Regret." Two tracks:
1. Reproduce current paper outputs from the existing replication package.
2. Deep MATLAB refactor to support revision-track work (new experiments, new replication package).

## Architecture

### Pipeline
```
Stata (data cleaning)  →  matlab/data/ (inputs)  →  MATLAB (compute)  →  output/ (figures, tables)
python/ (scrapers)     →  data/ (raw)            →  Stata             →  ...
```

### MATLAB Compute Stages
| Stage | Script | Solver | What it does |
|-------|--------|--------|-------------|
| I | `I_MAIN_Simul_2acts.m` | AMPL (LP) | Polytope vertex enumeration |
| II | `II_MAIN_simul.m` | `ComputeBCCE_eps` (CVX SOCP) | Simulation: learning + identification |
| III | `III_MAIN_Estim_Application_PrefSpec.m` | `ComputeBCCE_eps_ApplicationL` (CVX SOCP) | Application: data-driven identification |
| IV | `IV_MAIN_Emp_Distrib_Regrets.m` | `ComputeBCCE_eps_pass` (CVX SOCP) | Bootstrap regret distribution + identification |

Three solver variants, not two: `ComputeBCCE_eps` (joint), `_ApplicationL` (marginal), `_pass` (pre-computed epsilon).

### Key Technical Details
- **CVX parsing overhead** dominates runtime (~1.5-2.5s per solve); actual SOCP is fast
- **Constraint matrices** (B_EQ, B_INEQ, Mat_NLC, bounds) are invariant across inner loops; only objective `c` changes
- **12+ global variables** passed via `global` declarations across the entire codebase
- **Epsilon formulas**: `epsilon_switch.m` (cases 0-5) and `epsilon_switch_distrib.m` (cases 0-9) differ for same switch values (intentional per-stage)
- **Joint vs marginal**: genuinely different constraint dimensions (`dv = s^2*NA` vs `dv = s*Nactions`)
- **AMPL**: used only by Stage I for an LP; replaceable with `linprog`

## Project Structure
```
docs/
  HANDOVER.md              ← this file (stable context)
  CLAUDE_HANDOVER.md       ← volatile session state (read first when resuming)
  MATLAB_REFACTOR_PLAN.md  ← full refactor plan (single source of truth)
  FIRST_PORT_NOTES.md      ← initial port documentation
  REPLICATION_README_AUDIT.md
  REVISION_TRACK_NOTES.md
  PROVENANCE_RECOVERY_NOTES.md
matlab/
  src/                     ← all MATLAB source (MAINs, solvers, utilities)
  data/                    ← replication input data (tracked in git)
  test/
    run_fixtures.m         ← reduced-scale fixture runner
    verify_refactor.m      ← regression test: compare new vs baseline
    fixtures/              ← fixture .mat outputs (gitignored)
    fixtures/baseline/     ← frozen pre-refactor fixtures (gitignored)
    fixtures/timing_baseline.md  ← timing records
stata/                     ← .do files for data pipeline
python/                    ← scrapers (Swappa, Decluttr, Gazelle)
```

## MATLAB Refactor Plan
Full plan: `docs/MATLAB_REFACTOR_PLAN.md` (5 phases, 50+ work items, 13-17 sessions).

Summary of phases:
1. **Eliminate globals + extract game setup** — mechanical, low risk
2. **Unified solver** — replace CVX with `coneprog` (R2020b+); biggest speedup
3. **Parallelize + kill AMPL** — `parfor` solver grid; `linprog` replaces AMPL
4. **Stage wrappers** — thin MAIN scripts calling modular `+df/` functions
5. **Polish + validation** — end-to-end check against paper outputs

Key decisions:
- Kill AMPL dependency entirely
- Paper outputs serve as final artifact oracle (no full 7.6h rerun needed for baseline)
- Lightweight fixture capture at 1/100 scale for module-level regression testing
- Package namespace: `+df/` with `+io`, `+setup`, `+solvers`, `+sim`, `+stages`, `+report`, `+util`

## Baseline Fixture Infrastructure
- `run_fixtures.m` exercises Stages II-IV at reduced scale (20x20 grid, 5k iters, B=10)
- Baseline captured 2026-03-16 on R2024a: **43 min total** (II: 732s, III: 991s, IV: 864s)
- Baseline .mat files frozen in `matlab/test/fixtures/baseline/`
- `verify_refactor.m` compares post-refactor outputs against baseline (dual tolerance: strict 1e-10, solver 1e-6)

## Environment
- MATLAB R2024a on Windows (lorem@LoresLG)
- CVX installed at `matlab/cvx/` (gitignored)
- Git identity: `Lorenzo Magnolfi <lorenzo.magnolfi@wisc.edu>` (repo-local config)

## Known Limitations
- MATLAB not yet run end-to-end at full scale from the new repo
- Figure 12 (Stata) depends on `heatplot`, not installed
- Former RAs have not replied with scraping/provenance details
- `SCRAPING_PROTOCOL.md` not yet written
