# CLAUDE_HANDOVER.md — Estimation-of-Games-under-No-Regret_Code

## GadgetPickup demand and regret investigation, October 20, 2026

Current writer: none. The algorithm-free performance exercise is complete. Governing note: `docs/gadgetpickup-performance-assumptions-note-2026-10-20.md`. The user permits suggestive economic evidence; do not require verified learner adoption or treat its absence as an impasse. No background process remains.

- New finite-performance results: corrected exact-grid four-shift menu, heterogeneous nonnegative listing opportunity costs. At 5% shortfall, expected-sales mean cost is [67.29,407.58] with pooled performance and [183.96,407.51] with performance separately for 26 phone models. At 10% the latter is [93.41,407.52]. These are conditional fitted-demand sets, not confidence intervals or acquisition-cost estimates. The original realized-sales target remains broad.
- A separate purposeful-repricing comparison is economically weaker but insufficient: pooled [0,407.58], sign-split [4.16,407.58]. Half the fitted counterfactual log-odds response changes the 5% model-specific lower bound to 61.71. Numerical performance strength and demand response are separate assumptions.
- Scripts committed before runs: 29f6b7f (relative performance), e8939e7 (product grouping), 7753e3f (purposeful repricing), 9817edb (critical performance thresholds and independent arithmetic checks). Designs/results are in docs/evidence; detailed prediction parquets remain ignored. No paper edits, PDF, push, or external message in this pass. Prior bullets below are historical and superseded where they characterize the task as blocked.

- Governing conclusion: `docs/gadgetpickup-identification-status-2026-10-20.md`. Corrected five-fold additive fit (a583e93) predicts Nov-Dec sales 572.31 vs 556 and mean receipt 407.58 vs 386.49. Removing the copied interval is necessary but does not solve demand identification. Pagination-related gaps are about five times as frequent after flagged price groups.
- Corrected feedback driver e513270: 4,572 category-days, 158 categories, zero cost never excluded under the weak comparison, for exact/monotone bands and both observation models. This supersedes the old information diagnostic containing the copied day. No verified behavioral guarantee or full new-method cost set.
- Latest source-data gate: `docs/gadgetpickup-observation-audit-2026-10-20.md`. November 19 repeats ALL non-date cells of November 18; 197 raw GadgetPickup rows and 30 terminal labels fall on the copy (195/29 in first-stage). November 18 to 20 is a two-day interval. Do not simply relabel all sales to November 18 or count November 19 as feedback.
- Recovered historical table order from 166 raw workbooks. All 3,121 duplicated listing-days straddle 50-row page boundaries, including 195 GadgetPickup cases. Snapshot rank is not verified persistent buyer exposure; no rank-IV claim. Fourteen public Date Closed checks documented; browser works where web fetch failed, no authentication used.
- Producing scripts: gadget_display_order_audit.py (extraction 1714d4f, cached summary 758951f) and gadget_snapshot_calendar.py (2ae290f). New results, sources, and hashes in docs/evidence; detailed outputs remain ignored.
- Governing record: `docs/gadgetpickup-demand-regret-note-2026-10-20.md`. It supersedes earlier demand-validation status and contains the expanded seller-source search, all models, a proposed uncertainty-based regret guarantee with proof, and the finite-prefix ANR argument.
- Gadget-only listing-disjoint additive demand predicts November–December mean receipt 412.65 vs 392.61 observed (old pooled fit 466.16); 601.45 predicted sales vs 585. Category/time interactions worsen count calibration. Future-period forecasts remain unstable. No causal demand identification or new cost inversion.
- Within category/week residual price RMS is 7.07 dollars; category/day RMS is 2.33. Common prices, sparse changes, and batched inventory do not establish learner adoption or independent learning rounds.
- Sequential sales-information audit on five categories uses prior dates only and simultaneous Hoeffding and mixture bounds. Under the explicit constant conditional mean approximation, zero cost is not demonstrably inferior on any evaluated date, even with the sharper mixture bounds and individual-exposure assumption. The model omits pre-October experience; this is an information diagnostic, not a claim that the seller restarted.
- Expanded public search finds a dated November 2023 procurement/valuation announcement and a September–October 2023 seller comment linked to raw listing LXKC36753. No repricer or finite-sample guarantee verified. User confirms no extra private records and asks for extensive public research. Do not repeat searches without a new lead.
- Scripts: `gadget_demand_audit.py`, `gadget_demand_extension.py`, `gadget_demand_crossfit.py`, `gadget_demand_time_controls.py`, `gadget_feedback_audit.py`. Designs/results and hash manifest are tracked in `docs/evidence`; detailed predictions remain in the ignored `gadget_demand_20261020` output folder. Three focused confidence-bound tests pass.
- Inventory extension (design a25ab49, producing code 85e01d7): 1,640 sales and 301 nonsale exits before the final raw-panel date; stock identity closes after 126 gap returns. Same-category entry follows 5.4% of sales within one day and 39.3% within seven days. 75.84% of 658 entrants facing a unique incumbent price match it. This supports coordinated prices and repeated entry, not immediate replacement or common acquisition costs.
- Full-history feedback extension: valid global category/price budget and optional monotonicity; 4,724 category-days, 158 categories, zero cost never excluded by the weak information comparison. Three focused tests pass. New script gadget_inventory_feedback_extension.py, results and design in docs/evidence; detailed CSVs ignored. No new fits or cost intervals. Economic continuation/replacement reductions documented in the governing note.
- Next step requires a new identification input: justified finite performance restriction and causal demand variation. Do not restart the same vendor search, tune a regret allowance, or report old cost frontiers as a rescue. All planned observation corrections have run; no process is live. A named learner is sufficient but not necessary if a weaker supported performance restriction can be justified.

## Swappa rule tests and cost inversion, October 20, 2026

- Governing implementation: `docs/SWAPPA_RULE_TESTS_2026-10-20.md`. Governing exposition: paper sibling `JPE Revision/swappa-rule-tests-note-2026-10-20.tex`. This section supersedes the earlier "next step: freeze and validate" status below.
- Rule design fixed at 3047402; chronological prediction run at 2413ce4; price-only unknown-cost run at 3b4b77f. Supplementary withdrawal design 74d2283 and run e465482. Separate expected-sales target design c5775a1 and run 83c9668. Attribution and timing/LP checks e188ebe. All original outputs preserved. No demand refit.
- GadgetPickup: 5,638 eligible test pairs, 870 changes. Persistence MAE 1.884; sales 1.911; waiting 1.942; competitor 10.051. No selected primary family improves on persistence. Other four sellers also evaluated; full metrics saved.
- Unknown costs remain the estimation target. Nonnegative unrestricted listing costs, frozen price comparisons, and withdrawal bound GadgetPickup's separate expected-sales mean at [120.58,466.16] for epsilon=0 and [92.58,475.87] for epsilon=1 dollar per eligible listing-day. No common costs, cost span, or upper cap. Lower endpoint comes entirely from the -10 previous-ask comparison; fitted rules supply no positive lower bound for GadgetPickup.
- The original observed-sales target remains unbounded without withdrawal and extremely wide with it. The separate no-below-receipt cap substantially changes the intervals and is labeled an unsupported additional restriction. Do not silently impose it or reinterpret receipt equivalents as net payout.
- New-method requirements and exact Theorem CR mapping are documented. These are fitted-demand obedience-component sensitivity sets, not full Q_f confidence intervals. A history-rule/withdrawal comparison needs its own valid performance envelope, observed-window initialization treatment, consistency set, and demand uncertainty. Standard EXP3-IX bound is vacuous at 52 calendar days even with two actions.
- Demand diagnostic: GadgetPickup predicted total sales 580.45 vs recorded 585, but predicted-sales mean receipt 466.16 vs actual 392.61. Within retained market-days own-inventory counts match retained counts exactly; excluded markets and channels remain outside the calculation.
- Three focused tests pass; direct profit identities, analytic withdrawal bounds, interval monotonicity, and GadgetPickup's lower certificate pass. Human source static checks pass; no new PDF, manuscript edit, push, external contact, or HPC job.
- Current writer: none; this pass is complete and no background process remains. Next research step: predetermine and audit demand calibration/causal response, and independently establish the feedback/review/performance inputs. Do not choose assumptions by tightness. Prior lower sections are historical records.


## Swappa exercise closure, October 20, 2026

- Public-source research and the five-seller pricing audit are complete: `docs/SWAPPA_EMPIRICAL_FOUNDATIONS_2026-10-20.md`; reproducible script `python/swappa_rescue/pricing_practices_audit.py`, tracked result `docs/evidence/pricing_practices_audit_20261020.json`. No focal-seller repricer adoption verified. Actual-sales rule availability is documented by a May 2023 Repricer.com announcement; it is not Swappa adoption or a regret guarantee. No new cost estimates.

Read `docs/UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md` before using the earlier rescue results. The constructed expected-profit learner has a conditional guarantee, but its adoption and information are unverified; no credible actual-sales learner has yet produced informative bounds. Cost-estimation artifacts remain unchanged at e58b319. Public-source research is now complete; the next step is to freeze and validate a small rule family before cost inversion. No new cluster job was submitted.

<!--
  This file is for Claude session continuity. Update at the END of each session.
  Read this FIRST when resuming work on a project.
-->

## Last Session
- **Date**: 2026-09-10
- **Summary**: Unity transition started. The campaign had stalled when SSCC linstat closed (July 2026; last linstat-era commit `389086c`, 2026-07-21). Staged the full port surface on OSU Unity and launched the port gate.

### Unity locations (recorded per unity-skill first-use rule)
- **Code**: `/home/magnolfi.1/NoRegret_Code` — `git archive` of clean local HEAD = `origin/main` = `389086c` (recorded in `STAGED_COMMIT.txt` there). Not a git clone; re-stage the same way after new commits.
- **Solvers**: `/home/magnolfi.1/NoRegret_tools/cvx` — CVX 2.2 + bundled SeDuMi (downloaded from web.cvxr.com 2026-09-10). Export `CVX_DIR=$HOME/NoRegret_tools/cvx`; the prereq check picks it up via `genpath`.
- **Cached artifacts**: `/home/magnolfi.1/NoRegret_Code/matlab/output/artifacts` — all 9 files byte-verified against local Dropbox (incl. the s5/s10/s20 nonparam checkpoints and `prm_comparison_s5.mat`, 99 MB).
- **Run roots**: `/home/magnolfi.1/NoRegret_runs/<name>_<ts>/` with `MANIFEST.md` + `logs/`. First: `smoke_20260910_114613` — **port gate PASSED**, job `11096393` COMPLETED 0:0 (4 min, econ): `SMOKE_SEDUMI: PASS`, all 8 checks (exact/box/L1/CLT dVV<3e-4, zero set flips; learn_only; candidate_mask; eps-monotone; parfor==serial), cvx ~11–12s vs sedumi_direct ~9–10s per 25-point stage call. **The solve lane is live on Unity.**
- **Unity environment lore (differs from linstat; every future job script needs this)**:
  1. `~/Documents/MATLAB/startup.m` runs `cvx_startup.m` — created, persists.
  2. `cvx_setup` must run **in-session** before any CVX-lane solve (`run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m')` at the top of the `matlab -batch` command; saved CVX prefs do not reconstitute across batch sessions here — `cvx_global` dies on `cvx___.solvers.list` otherwise).
  3. `export SEDUMI_DIR=$HOME/NoRegret_tools/sedumi` — a **standalone copy** outside the CVX root; the in-root `{cvx}/sedumi` gets pruned off the path by `cvx_clear` (the f6c73c7 failure mode). Do **not** export `CVX_DIR` (whole-tree genpath pollutes CVX's curated path).
  4. Working job template: `~/NoRegret_runs/smoke_20260910_114613/job_smoke.sh` (single MATLAB session: cvx_setup → prereq check → payload).
- **Launcher port**: the linstat launchers (`run_corrected_bundle_linstat.sh` etc., nohup-based) must become sbatch scripts; the smoke job script in the run root is the template. MATLAB module: `ml matlab/r2024b` (HANDOVER says R2024a+, so compatible). Linstat reference smokes for equivalence: `3778539`, `3779271` (both PASS).

## Previous Session
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

## AuctionNet / Design A on Unity (recorded 2026-09-23, per the unity-skill first-use rule)
- **Rule (Lorenzo, 2026-09-23):** no large Python jobs on the office machine; downloads, conversions, simulations run on Unity.
- **Code on Unity:** `/home/magnolfi.1/NoRegret_Code/python/auctionnet/` (scp copy of this repo's `python/auctionnet/`, uncommitted state of 2026-09-23; re-copy after edits).
- **Data on Unity:** `/home/magnolfi.1/NoRegret_data/auctionnet/{raw,parquet,env,results}`; raw zips downloaded on the login node by `unity/download_unity.sh` (log `download.log`, ends with ALLDONE).
- **Python env:** `~/.conda/envs/auctionnet` (mamba module 25.11.0 with `CONDA_PKGS_DIRS`/`CONDA_ENVS_DIRS` in home; the module prefix is read-only). Jobs call `~/.conda/envs/auctionnet/bin/python` directly.
- **Jobs:** `unity/job_convert.sh` (convert_all.py; econ,batch; 6 h; 32 g) then `unity/job_designA.sh` (run_designA.py; 24 h; 48 g; 8 cores), submitted by the login-node watcher `unity/chain.sh` (log `~/NoRegret_runs/auctionnet/chain.log`, ends with CHAIN_SUBMITTED). Job logs in `~/NoRegret_runs/auctionnet/logs/`.
- **Outputs to pull back:** `results/designA/` (MANIFEST.json, CSVs, npz, figs), `results/rules/`, `results/footprints/`. Local copies of period 7-17 zips/parquet/env remain under `C:/Users/magnolfi.1/AuctionNet/` (outside Dropbox); the local period-13 pilot results are in `C:/Users/magnolfi.1/AuctionNet/results/`.
- **Definitive run complete (2026-09-23):** download ALLDONE 11:24 EDT (11 zips, 18 GB); convert job 11149732 COMPLETED (47:55); designA job 11149733 COMPLETED (1:41:53). Results pulled on 2026-09-24 to `C:/Users/magnolfi.1/AuctionNet/results_unity/{designA,footprints,rules}` plus `designA_log.txt` (the six 140 MB pooled environments `env_role*.npz` stay on Unity). Recap: paper repo `JPE Revision/Simulation_AuctionNet_DesignA_Results.md`; register updated.
- **Robustness array (2026-09-24):** `run_robust.py` + `unity/job_robust.sh` (Slurm array 0-5 over the six focal roles; 12 h; 4 cores; 40 g), job 11157909. Variants lam075/lam150/lamcat, nm8/nm16, nv5/nv9, wide; seeds 2-3 at 1e8; STZ comparison (`sim_stz.py`, coverage runs as subsamples). Outputs `results/designA/robust/role<r>/` with `MANIFEST.json`; logs `~/NoRegret_runs/auctionnet/logs/auctionnet-robust_11157909_<i>.log`, ending ROBUST_DONE. Pull back with the same tar recipe (exclude `env.npz`).
- **Still to do (as of 2026-09-24; see the 2026-10-07 update below):** `verify_auction.py` on periods other than 13; WIN-EXP; commit `python/auctionnet/`.

### Update 2026-10-07 (mini PC session)
- **Committed** (not pushed): `e19d4ba` = `python/auctionnet/` exactly as run in September (27 files, sha256 equal to the Unity copy up to CRLF line endings; code of record for jobs 11149732/11149733/11157909); `528052f` checks + `provenance.py`; `8348836` Regime O diagnostics (`explain_regimeO.py`, `plot_explain_regimeO.py`, `check_alpha.py`); `6d2ebc8`, `8117590`, `3989eaa` single-learner counterfactual (`cf_single.py`, `plot_cf_single.py`). Also `.gitignore` for Python bytecode.
- **Staging rule now in force:** `git -c core.autocrlf=false archive --format=tar HEAD python/auctionnet | ssh unity 'cd ~/NoRegret_Code && tar -xf -'`, then write the full hash to `python/auctionnet/STAGED_COMMIT.txt` and check sha256 against `git show HEAD:<file>`. Without `core.autocrlf=false` the archive arrives with CRLF and breaks the job scripts. `provenance.code_commit()` reads `STAGED_COMMIT.txt`, so manifests now carry the commit. The September copy is kept on Unity as `python/auctionnet_asrun_e19d4ba/`. Unity currently holds `8117590` (plus the later plot script, which runs locally).
- **New Unity outputs:** `results/verify/` (21 `verify_period-*.json`, `verify_summary.csv`, `alpha_max.csv`; jobs 11252734, 11252741, 11252763), `results/fringe/` (job 11252735), `results/cf/` (array 11252779; supersedes 11252772), `results/designA/tables/` (payoff tables without the rival draws, for local diagnostics). Run manifest: `~/NoRegret_runs/auctionnet/checks_20261007/MANIFEST.md`. Local copies on the mini PC: `C:/Users/lorem/AuctionNet/results_unity/`.
- **Findings:** all 21 periods pass every mechanism check; rival bids independent of the focal context within a time step but not across the day; the Regime O tolerance in `sim_infer.py` uses realized context shares (latent in that regime), to be replaced by candidate shares; write-ups in the paper repo `JPE Revision/Simulation_AuctionNet_*` (Dropbox-only).
- **Regime O tolerance fixed and rerun (same day):** `6f2075b` changes rule `markov` in Regime O to use ψ_j(μ, σ) + r_cons/2 instead of the realized context shares (old rule kept as `markov_realized`); `rerun_regimeO.py` + `unity/job_regimeO_v2.sh` reran Regime O on every run (array 11252877, all tasks rc 0, manifests record `6f2075b`, clean) into `results/designA/regimeO_v2/role<r>/` (September files untouched; run root `~/NoRegret_runs/auctionnet/regimeO_v2_20261007/`). `summarize_regimeO_v2.py` tables pulled to the mini PC. Truth feasible under the corrected rule in all 1,085 checkpoints. `7b1baec` adds the corrected profile to `explain_regimeO.py` (local outputs in `explain_v2/`). Unity holds `6f2075b` (the later diagnostics commit runs locally). Both repos pushed 2026-10-07.
- **Still to do:** paired-environment robustness row; third learner (proposal note) and two-learner counterfactual (design note) pending Lorenzo's decisions.

## Swappa application reopened (2026-10-16, OSU desktop)

Lorenzo authorized a multi-day goal to salvage Swappa using the newest method and appropriate empirical assumptions. Current owner: Codex chat `01a14662-bc4d-7902-b104-04742ceb3ca5`, restricted to `python/swappa_rescue/`, its outputs, and `docs/SWAPPA_RESCUE_CHECKPOINT.md`. Read that checkpoint before any new Swappa work. It records timing and competitor-construction issues, exploratory persistent-cost programs, unresolved identification and inference obligations, and current computations. The application is not yet validated. The older parked-state memo remains a record of the prior specifications, not an instruction to stop this authorized campaign.
