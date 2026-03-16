# First Port Notes

- Source package: `C:\Users\lorem\Dropbox\DynFoundApplicationData\Replication_2025`.
- New code repo: this repository.
- This first pass copied code and current MATLAB inputs, but not the full raw-data archive.
- First cleanup pass fixed the highest-impact path issues in `stata/master.do`, `stata/macros.do`, `stata/generate_matlab_data.do`, and the main MATLAB entry scripts.
- MATLAB entry scripts now use repository-relative paths via `matlab/src/df_repo_paths.m` and write generated outputs under `matlab/output/`.
- Stata exports now target `matlab/data/`, and Stata figure/table outputs now target `output/stata/`.
- The intermediate-data pipeline has now been verified locally through `stata/build_intermediates.do`.
- To make the baseline reproducible without depending on the fragile Swappa daily-Excel import loop, the build driver falls back to packaged `Swappa_Data.dta` and `Decluttr_Data.dta` when those are available under `data/Raw_Data_Clean/`.
- The missing `gazelle_data.do` step has been recreated and verified through the intermediate-data build.
- Downstream Stata verification status:
  - `Figure 8` through `Figure 11` were generated successfully in `output/stata/figures/`.
  - Appendix tables were generated successfully in `output/stata/tables/`.
  - `Figure 12` is still blocked by a missing Stata package: `heatplot`.
- Additional cleanup is still needed for full MATLAB verification and for a true from-scratch rebuild of the raw Swappa import layer.
