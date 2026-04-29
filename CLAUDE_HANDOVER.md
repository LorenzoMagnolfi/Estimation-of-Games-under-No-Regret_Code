# CLAUDE_HANDOVER.md — Estimation-of-Games-under-No-Regret_Code

<!--
  This file is for Claude session continuity. Update at the END of each session.
  Read this FIRST when resuming work on a project.
-->

## Last Session
- **Date**: 2026-04-29
- **Summary**: Three pilot threads supporting the JPE simulation-section rebuild. (1) Pre-flight rate-landscape pilot at four canonical sample sizes with the corrected R1 radius, and a follow-up Pass B with consistency slack. (2) Realization-conditional identification pilot at `s = 5` — central 3×3 block at `N = 4M`, truth in set. (3) Realization-conditional pilot at `s = 9` (diagnostic for grid-vs-LP-flexibility) — 81/81 feasible at every `N`. Decisive: LP-flexibility-at-unrealized-types is the binding identification floor. Addendum for Niccolò drafted in paper repo's design memo.

## Current State

### What Was Done

**`+df/+solvers/`:**
- `compute_epsilon.m`: added `switch_eps == 10` (R1, Hedge corollary) and `switch_eps == 11` (R2, Exp3 corollary) — both simulation and application branches.
- `compute_consistency_slack.m`: NEW. Coordinatewise Hoeffding (`box`) and joint Bretagnolle-Huber-Carol (`L1`) variants for the consistency-block sampling-error slack.
- `build_constraints.m`: extended to support optional `consistency_slack` argument; when positive, consistency-equality rows move to inequality block as a coordinatewise box.
- `build_constraints_realization.m`: NEW. Slice-equality LP for App. SIM-RC. Replaces the marginal-on-A constraint with a per-action equality at the realized type-pair slice. Mirrors `build_constraints.m` otherwise.

**`+df/+sim/`:**
- `learn_fixed.m`: NEW. Hedge / regret-matching with FIXED `(t_1*, t_2*)` realizations throughout the trajectory (vs `df.sim.learn` which redraws iid every period).

**`+df/+stages/`:**
- `run_stage_ii.m`: extended to support optional `consistency_slack_kind` (`box`, `L1`), `alpha_R / alpha_C` budget split, and `precomputed_distY` cache (for re-solving under different rate/slack without re-running learn).

**Top-level entry runners (`matlab/src/`):**
- `II_RUN_rate_landscape_pilot.m`: NEW (v1, R1 only, 4-N overlay).
- `II_RUN_rate_landscape_pilot_v2.m`: NEW (R1 with vs without consistency slack, two-panel figure, reuses v1 trajectories).
- `replot_overlay_R1.m`: NEW (re-renders v1 overlay with hardcoded axis crop).
- `II_RUN_realization_pilot.m`: NEW (s=5 baseline; 5×5 feasibility heatmap).
- `II_RUN_realization_pilot_s9.m`: NEW (s=9 finer-grid diagnostic).

**Outputs (gitignored):**
- `matlab/output/rate_landscape_pilot/` — `results_R1.mat`, `results_R1_no_slack.mat`, `results_R1_box_slack.mat`, run logs.
- `matlab/output/figures/rate_landscape_pilot/` — `overlay_R1{,_cropped}.{png,pdf}`, `overlay_2panel.{png,pdf}`.
- `matlab/output/realization_pilot/` — `results_realization_R1{,_s9}.mat`, run logs.
- `matlab/output/figures/realization_pilot/` — `heatmap_realization_R1{,_s9}.{png,pdf}`.

### Key Design Decisions
- **R1 per-cell radius**: `Kappa * sqrt(2 ln|A| / N) / alpha`. The lower-order `K * ln|A| / (alpha * N)` correction is omitted (negligible at our N and cannot be threaded through the per-cell `sqrt(phi)` factor).
- **Consistency slack**: not needed in realization-conditional setup (prior held fixed; rate enters only via obedience). Used in the parametric pipeline's v2 pilot as an optional Pass B.
- **Realization-conditional LP**: prior `pi` uniform `1/s^2`, slice equality replaces marginal-on-A. Constraint matrix structure depends on the realized index, so build per candidate (cheap: `s^2 = 25` rebuilds at `s = 5`).
- **Pre-flight pilots use adaptive solver** (`solve_grid_adaptive`) — exploration only, NOT for production/inference.
- **Realization pilot uses `solve_socp_cvx` per candidate** — full grid solve, no adaptivity needed at this small candidate count.

### Files Touched (this session)
**Modified:**
- `matlab/src/+df/+solvers/build_constraints.m`
- `matlab/src/+df/+solvers/compute_epsilon.m`
- `matlab/src/+df/+stages/run_stage_ii.m`

**Created:**
- `matlab/src/+df/+solvers/compute_consistency_slack.m`
- `matlab/src/+df/+solvers/build_constraints_realization.m`
- `matlab/src/+df/+sim/learn_fixed.m`
- `matlab/src/II_RUN_rate_landscape_pilot.m`
- `matlab/src/II_RUN_rate_landscape_pilot_v2.m`
- `matlab/src/replot_overlay_R1.m`
- `matlab/src/II_RUN_realization_pilot.m`
- `matlab/src/II_RUN_realization_pilot_s9.m`

### Git Status
- Branch: `main`, several commits ahead of `origin/main` from prior sessions (DAS audit) plus this session's uncommitted work.
- 3 modified files (R1/R2 + consistency slack), 8 untracked new files (pilots + helpers).
- 1 stale `.bak` file (`matlab/test/fixtures_baseline_cvx/fixture_stage_ii_iter_5k.mat.bak`, dated 2026-03-16) — likely cruft from fixture refactor; flag for cleanup on next commit pass.
- No commits yet this session — awaiting user approval. Two natural commits proposed: (1) rate-landscape pilot bundle, (2) realization-conditional pilot bundle.

## Pending / Next Steps
- [ ] **Niccolò review** of realization-conditional addendum (paper repo's design memo). Three resolution paths laid out: (a) BCCE-with-fixed-slice (current, too flexible), (b) data-only LP, (c) BCCE with smoothness on `nu(a | t)`.
- [ ] **Implement chosen resolution** — likely (b) data-only LP, ~30 lines of code in a new `build_constraints_realization_dataonly.m` plus a re-run.
- [ ] **Commit pending work** when user is ready: paper repo (1 commit) + code repo (2 commits).
- [ ] **Asymmetric robustness pass** at `(t_1*, t_2*) = (2, 4)` once the LP variant is settled.
- [ ] **Production runs (rebuild plan Exercises 1–6)**: open decisions in plan need resolution before kicking off SSCC compute.
- [ ] **PRM comparison** (`II_RUN_prm_comparison.m`) — still blocked on Theorem 3 feedback resolution.
- [ ] **Figure 12** Stata `heatplot` package install.
- [ ] **JPE-style README** + `requirements.txt` for replication package.

## Open Questions
- **Cost grid in `marginal_cost_draws_v5`**: returns `linspace(0, 6, s)`, hardcoded — NOT TruncNormal-quantile-based as one might expect from the function name. Cost 0 is in the support. Doesn't affect pilot validity but worth examining for production runs and possibly replacing with a more economically natural grid.
- **Single-market identification target** (theoretical, for Niccolò). What's the right BCCE-style object when one realization is observed? Captured in the paper repo's design memo addendum.
- **Theorem 3 feedback assumption** (full-info or bandit rate?): still pending Niccolò.
- **Data redistribution rights** (DAS, low-priority): confirm with NL whether any additional statement is needed for JPE.

## Tricky Bits
- `data/` is gitignored — raw data lives locally only. The copied files are present on disk but not tracked by git. `matlab/data/` IS tracked (whitelisted in `.gitignore`).
- `df_repo_paths` is a **function** returning a struct. Call as `paths = df_repo_paths()`.
- CVX+SeDuMi hangs with >~2,600 candidates per batch. Keep fast-pass grids at ~256.
- The rate-landscape pilots use `adaptive = true` — exploratory only. Production runs need full grid (no adaptive).
- Realization-conditional pilots have 25 (s=5) or 81 (s=9) LPs per N — no adaptive needed; full grid is fast.
- `precomputed_distY` cache in `run_stage_ii`: skips the learn step if a trajectory is provided. Used in v2 pilot to reuse v1's R1 trajectories for the slack comparison (saves ~hours).
- `gridparamV/M` in `run_stage_ii.m` prepend an out-of-order `1` entry that breaks `contour()` plotting; the pilot helper drops `(1, :)` and `(:, 1)` to recover a clean monotonic grid.
- Action-profile index 13 in the canonical lab corresponds to `(a_1, a_2) = (6, 6)` — the median price for both players. No-regret play converges here under the median-cost realization.

## Session Lessons
- Save trajectories from learn step, reuse via `precomputed_distY` for follow-up LP variants. Cuts re-run time from hours to minutes when comparing rate/slack/LP-form choices on the same data.
- Negative-result pilots are sometimes the most informative. The s=9 finer-grid run produced 81/81-feasible everywhere — useless as a result, decisive as a diagnostic. Cheap orthogonal pilots are worth running when identification looks weaker than expected.
- LP feasibility tests with free off-data variables are weakly identifying by construction. The realization-conditional s=5 result (3×3 block) was actually constrained largely by the consistency-with-obedience interaction across the 4×24 unrealized cells, not by the slice equality itself. Worth keeping in mind for future identification-set work.
