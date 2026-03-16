# Bo Feng Revision-Track Import

This folder stores the Bo Feng materials that are relevant for the JPE revision track and should live in the code repository rather than the Overleaf-synced paper repository.

Imported here:
- `matlab/src/revision_track/bo_feng/II_MAIN_nonparam_simul.m`
- `matlab/src/revision_track/bo_feng/ComputeBCCE_eps.m`
- `docs/bo_feng_revision_track/runtime_log.txt`
- `docs/bo_feng_revision_track/SPIKY_DISTRIBUTION_MODIFICATION_SUMMARY.md`
- `docs/bo_feng_revision_track/SVM_Hyperparameter_Analysis_Report.md`

Why these files were kept:
- `II_MAIN_nonparam_simul.m` is the nonparametric simulation branch.
- `ComputeBCCE_eps.m` is the Bo Feng snapshot that adds direct probability-mass-vector support.
- The markdown reports and runtime log document the low-sigma coverage experiments, hyperparameter tuning, and computational cost.

What was not copied here:
- The imported Stata and Python scripts were not duplicated because the code repo already contains those scripts in cleaner baseline-path form.
- `regret_matching_new.m` and `regret_matching_mod.m` were not duplicated because the same files already exist in `matlab/src/`.

Status:
- These files are preserved as revision-track inputs, not yet integrated into the cleaned baseline path.
