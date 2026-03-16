# Estimation of Games under No Regret Code

This repository is the working code home for two related goals:

1. Reproduce the current paper as it exists now.
2. Support revision-track code changes without losing the baseline.

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
