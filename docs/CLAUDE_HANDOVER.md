# CLAUDE_HANDOVER.md — Estimation of Games under No Regret (Code)

<!--
  Read this FIRST when resuming. For stable architecture context, see HANDOVER.md.
  For the full refactor plan, see MATLAB_REFACTOR_PLAN.md.
-->

## Last Session
- **Date**: 2026-03-16
- **Duration**: ~2 hours (continued from a prior session that ran out of context)
- **Summary**: Completed Prep phase of MATLAB refactor — built and ran fixture infrastructure, established git repo with initial commit and baseline tag.

## Current State

### What Was Done
- Audited and critiqued the MATLAB refactor plan (across 2 sessions); consolidated into `MATLAB_REFACTOR_PLAN.md`
- Instrumented all four MAIN scripts with `save` calls for key intermediates
- Created `matlab/test/run_fixtures.m` — reduced-scale fixture runner (Stages II-IV)
- Created `matlab/test/verify_refactor.m` — regression comparison script
- Created `matlab/test/fixtures/timing_baseline.md` — timing record template
- Fixed `.gitignore` to allow `matlab/data/*.xlsx` through while excluding `data/`
- Set repo-local git config (Lorenzo Magnolfi)
- Initial commit to `main`, tagged `v0-baseline` (78 files)
- Ran fixture runner on MATLAB R2024a: 43 min total, all 9 fixture files captured
- Copied fixtures to `matlab/test/fixtures/baseline/`
- Updated timing template with actual results

### Files Touched
- `matlab/src/I_MAIN_Simul_2acts.m` — added fixture save calls
- `matlab/src/II_MAIN_simul.m` — added fixture save calls
- `matlab/src/III_MAIN_Estim_Application_PrefSpec.m` — added fixture save calls
- `matlab/src/IV_MAIN_Emp_Distrib_Regrets.m` — added fixture save calls
- `matlab/test/run_fixtures.m` — created, then fixed path setup for batch mode
- `matlab/test/verify_refactor.m` — created
- `matlab/test/fixtures/timing_baseline.md` — created, filled with baseline data
- `docs/MATLAB_REFACTOR_PLAN.md` — created (previous session)
- `docs/HANDOVER.md` — rewritten for stable/volatile separation
- `.gitignore` — fixed `data/` vs `matlab/data/` scoping, added `.claude/`, `**/*.mat`

### Decisions Made
- **Kill AMPL**: user explicitly prefers removing AMPL dependency entirely; replace with `linprog`
- **Paper-as-oracle**: paper outputs suffice as final artifact check; no need for full 7.6h baseline rerun
- **Fixture scale**: 20x20 grids (vs 100x100), 5k iters (vs 500k-4M), B=10 (vs 500) — exercises same code paths
- **Git on `main`**: initial commit captures everything; refactor work will branch off as `refactor/matlab`
- **No remote yet**: repo is local + Dropbox only; GitHub push deferred

### Git Status
- Branch: `main`
- Last commit: `7568fbe Fix fixture runner path setup and record baseline timing`
- Tag: `v0-baseline` on `836a074`
- Pushed to remote: **no** (no remote configured)
- Uncommitted changes: none (clean)

## Pending / Next Steps
- [ ] Push to GitHub (when ready)
- [ ] **Phase 1**: eliminate globals + extract game setup into struct-passing functions
- [ ] **Phase 3b** (independent): replace `find_polytope_switch.m` AMPL calls with `linprog`
- [ ] After Phase 1: re-run `run_fixtures.m`, compare with `verify_refactor.m`
- [ ] Phase 2: unified solver (CVX → `coneprog`)

## Open Questions
- GitHub repo: private or public? Organization?
- Phase 1 vs 3b: which to tackle first? (Both are independent; Phase 1 is prerequisite for Phase 2)
- Should fixture runner also cover Stage I once linprog replacement exists?

## Tricky Bits
- **`mfilename('fullpath')` in batch mode**: `run('../test/run_fixtures.m')` from `src/` changes context; the fixture runner must `addpath(src_dir)` explicitly before calling any source functions
- **`.gitignore` scoping**: `data/` (no leading `/`) catches `matlab/data/`; must use `/data/` + negation rules (`!matlab/data/`) to allow MATLAB input data through
- **Epsilon formula discrepancy**: `switch_eps==1` has `s*` multiplier in `epsilon_switch_distrib.m` but not in `epsilon_switch.m` — this is intentional, do not "fix" it
- **`ExpectedRegretComp_Pl2` bug?**: Line 208 in `IV_MAIN` uses `epsPl1` instead of `epsPl2` — appears to be a bug in the original code, but we reproduce it exactly for baseline fidelity
- **CVX baseline timing**: 43 min at 1/100 scale → the CVX parsing overhead is ~1.5-2.5s per solve, confirming the refactor plan's prediction that `coneprog` will be the single biggest speedup
