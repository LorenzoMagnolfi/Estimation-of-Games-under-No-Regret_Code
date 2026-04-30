# CLAUDE_HANDOVER.md — Estimation-of-Games-under-No-Regret_Code

<!--
  This file is for Claude session continuity. Update at the END of each session.
  Read this FIRST when resuming work on a project.
-->

## Last Session
- **Date**: 2026-04-30
- **Summary**: Completed serious-run characterization of realization-conditional ID set: 3 truths × 5 N's × 2 LP variants (BCCE LP vs data-only). Discovered that the BCCE LP is mis-specified for persistent types and rejects the truth at asymmetric realizations. The data-only spec with per-candidate `kappa` eps is the right calibration — hits asymptotic floor at the smallest N tested. s=20 robustness pass shows the action-grid (not cost-grid) is the binding floor.

## Current State

### What Was Done

**`+df/+solvers/`:**
- `compute_dataonly_regret.m` — NEW. Closed-form per-player max regret of an empirical action distribution under candidate costs (simulation, logit demand structure).

**Top-level entry runners (`matlab/src/`):**
- `II_RUN_realization_serious.m` — NEW. Realization-conditional, 3 truths, 5 N's, BCCE LP + data-only side by side. ~25 min runtime.
- `II_REPROCESS_realization_perK.m` — NEW. Re-process serious-run results swapping max-K eps for per-candidate K eps. <1 min runtime.
- `II_RUN_realization_s20.m` — NEW. s=20 (vs s=5 baseline) for action-grid floor diagnostic. Data-only only. ~17 min runtime.

**Outputs (gitignored):**
- `matlab/output/realization_pilot/results_realization_serious.mat` (with feas_data_perK and eps_perK after reprocess)
- `matlab/output/realization_pilot/results_realization_s20.mat`
- `matlab/output/realization_pilot/run_log_serious.txt`, `run_log_s20.txt`, `reprocess_perK.log`
- `matlab/output/figures/realization_pilot/serious_truth_*.{png,pdf}`, `serious_cardinality_vs_N.{png,pdf}`, `serious_*perK_compare.{png,pdf}`, `s20_*.{png,pdf}`

### Key Findings
1. **BCCE LP truth-rejection at asymmetric realizations** — per-cell `eps_R1/sqrt(s)` decomposition is mis-specified for persistent types. Diagnosis in `JPE Revision/Realization_Identification_Design.md` Addendum 2.
2. **Per-candidate `kappa` eps** — replaces max-K version; hits asymptotic floor at smallest N (250k for s=5).
3. **Action-grid is the binding floor** — s=5→s=20 buys 30–50% cost-space resolution; `|A|` is the real lever.

### Files Touched (this session)
**Created (committed):**
- `matlab/src/+df/+solvers/compute_dataonly_regret.m`
- `matlab/src/II_RUN_realization_serious.m`
- `matlab/src/II_REPROCESS_realization_perK.m`
- `matlab/src/II_RUN_realization_s20.m`
- `CLAUDE_HANDOVER.md` (this file)

**On disk untracked (deferred — Lorenzo paused application work):**
- `matlab/src/+df/+solvers/compute_dataonly_regret_app.m`
- `matlab/src/III_RUN_application_fixedcost.m`
- These are application-side analogs of the simulation infrastructure. Working code, demonstrably runs (sanity test produced `c ∈ [-78.84, 0]` identified intervals for sellers 1 and 2), but on hold while we firm up the simulation design with Niccolò. Do NOT delete; do NOT re-introduce until Lorenzo signals.

## Pending / Next Steps
- [ ] **Commit pending work**: code repo (1 commit, simulation new files + handover refresh).
- [ ] **Niccolò review** of paper-repo addendum with the three simulation findings.
- [ ] **Asymmetric simulation / MC variance**: the s=5 serious run was 1 MC rep per (truth, N). Multi-rep would give variance bands on the identified set.
- [ ] **PRM comparison** (`II_RUN_prm_comparison.m`) — still blocked on Theorem 3 feedback resolution.
- [ ] **Auction appendix** calibrated illustration — still pending.
- [ ] **JPE-style README** + `requirements.txt` for replication package.

**Deferred (Lorenzo-paused):**
- Application-side fixed-cost specification (see "On disk untracked" above).

## Open Questions
- **Cost grid in `marginal_cost_draws_v5`**: hardcoded `linspace(0, 6, s)`. Cost = 0 in the support is unusual. Worth examining for production runs.
- **`-WindowStyle Hidden` in PowerShell launches**: caused MATLAB to hang at startup in this session. Use `-NoNewWindow` instead.
- **bash sandbox missing matlab on PATH**: bash launches via `matlab.exe` fail with exit 127. Use PowerShell `Start-Process` with full path.

## Tricky Bits
- `data/` is gitignored — raw data lives locally only.
- `df_repo_paths` is a **function** returning a struct: call as `paths = df_repo_paths()`.
- The realization-conditional pilots use `df.sim.learn_fixed.m` (fixed type per trajectory), NOT `df.sim.learn` (which redraws iid every period). Don't conflate.
- Per-K eps formula: `eps_i(k_i) = kappa_i(c_i^{k_i}) * sqrt(2 ln |A_i| / N) / alpha_set`. The `kappa` is computed at the candidate cost (NOT the worst-case cost over the grid).
- BCCE LP for persistent types is **broken**. Don't try to use `build_constraints_realization.m` going forward — it's mis-specified. Use `compute_dataonly_regret.m` directly.

## Session Lessons
- **Asymmetric/corner truths are the diagnostic for BCCE-LP mis-specification.** Symmetric truths can hide it.
- **Validate launches before claiming "in flight":** check log files growing, CPU > 0 sustained. Earlier in session I claimed runs were in flight when bash exit 127 had killed them silently.
- **PowerShell `-WindowStyle Hidden` blocks MATLAB startup**; use `-NoNewWindow` instead.
- **Save trajectories from `learn_fixed`** in the .mat output, then re-process with different `eps` schemes via small reprocessor scripts. Saves 17 min per re-run.
