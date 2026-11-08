# Estimation of Games under No Regret Code

This repository is the working code home for two related goals:

1. Reproduce the current paper as it exists now.
2. Support revision-track code changes without losing the baseline.

## Swappa research notes, November 6 to 8, 2026

The notes for coauthors live in the paper repository (`JPE Revision/swappa-revival-learning-layers-note-2026-11-08.tex` governs the shared claims; the `-for-agents` Markdown beside it carries paths and commits). This repository holds the long-form records and the producing code.

- [Defensible results assessment](docs/swappa-defensible-results-assessment-2026-11-08.md): the core Lerner-index result under the steady state, the repaired inference (max-t, cross-covariance, block bootstrap with demand refit), the free-cost bounds, the regret-notion comparison, the demand weak link, and what to demote.
- [Steady-state alternatives](docs/swappa-steady-state-alternatives-2026-11-08.md): the learning-consistent layers that replace Assumption A2 (experience-indexed, any-window, bandit), all 13 sellers ordered by experience, and (Section 5) how the H unobserved rounds enter Theorems 1 and CR and Proposition 2.
- [Revival audit memo](docs/swappa-revival-agent-memo-2026-11-06.md) and [demand repair note](docs/swappa-demand-repair-note-2026-11-06.md): the fix matrix M1 to M9 and the demand pilot on the native-reference panel.
- Evidence: `docs/evidence/swappa_*_20261106.*` and `docs/evidence/swappa_*_20261108.*`; scripts: `python/swappa_rescue/*_20261106.py` and `*_20261108.py` (the common module is `swappa_common_20261108.py`).

## Swappa research notes, October 20, 2026

- [Rule tests and cost inversion](docs/SWAPPA_RULE_TESTS_2026-10-20.md): completed GadgetPickup-focused evaluation, separate expected-sales cost bounds, exact assumptions, and unresolved confidence guarantees. Governs the new calculations.

- [Upgrade exercise closeout](docs/UPGRADE_LEARNING_RULE_CLOSEOUT_2026-10-20.md): assumptions, exact guarantees, conditional estimates, and limitations. The exercise did not establish an empirically validated rescue.
- [Empirical foundations](docs/SWAPPA_EMPIRICAL_FOUNDATIONS_2026-10-20.md): dated software/platform evidence, sources, descriptive pricing audit, and candidate assumptions for further work.
- [Reproducible descriptive results](docs/evidence/pricing_practices_audit_20261020.json).

## Repository status

- First-round port completed from `C:\Users\lorem\Dropbox\DynFoundApplicationData\Replication_2025`.
- First cleanup pass completed for the main Stata and MATLAB entry points.
- Verified locally on March 15, 2026: the Stata intermediate-data rebuild, Appendix tables, and Appendix Figures 8-11.
- This is still a work in progress, not yet a submission-ready replication package.

## Layout

- `stata/`: Stata orchestration, cleaning, figure, and appendix-table scripts.
- `python/`: scraping scripts and the Swappa notebook.
- `matlab/src/`: MATLAB entry scripts, helpers, and optimization templates.
- `matlab/data/`: MATLAB input data currently shipped with the 2025 replication package.
- `docs/`: port notes and audit notes.

## What is intentionally excluded for now

- Very large raw or intermediate files that are poor fits for standard GitHub, especially `Swappa_Data.dta` (>100 MB).
- Vendored MATLAB dependencies such as `cvx/` and `tbxmanager/`.
- Generated figures and tables.

## Baseline vs revision

- Baseline track: preserve and document the code that reproduces the current paper.
- Revision track: pull in experimental or refactored code only after the baseline is stable.
- External GitHub work, including `Bo-Feng-1024/estimation_of_games_under_no_regret`, should be treated as revision-track input unless we verify that a specific change improves baseline reproducibility.

## Important limitations right now

- The original 2025 package references several intermediate Stata `.dta` files under different names than the files it actually ships.
- This repo now rebuilds the intermediate layer and can also fall back to the shipped `Swappa_Data.dta` and `Decluttr_Data.dta` for the baseline path.
- The MATLAB code still requires external software for some steps, especially AMPL and MATLAB toolboxes.
- The root JPE-style replication README and full run manifest still need to be completed.
- Figure 12 still requires the Stata package `heatplot`, which is not yet installed in this environment.

## Current run guidance

- Stata data build: from the repo root, run `do stata/build_intermediates.do`.
- Stata full output pass: from the repo root, run `do stata/master.do`.
- MATLAB: run the entry scripts from `matlab/src/`. They now resolve paths relative to this repository and write outputs under `matlab/output/`.
- If a step requires missing intermediate data or external software, the script should now fail more transparently.

## Next cleanup priorities

- Recover or rebuild the missing Stata intermediate datasets referenced by the current drivers.
- Audit the Bo-Feng branch history for changes worth porting into the revision track.
- Add a top-level manifest that maps scripts to figures, tables, and manuscript references.
- Decide which large data assets stay outside git and which belong in Git LFS.
