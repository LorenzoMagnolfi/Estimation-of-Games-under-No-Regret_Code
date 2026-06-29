%% II_LINT_l1redos  Syntax/undefined-var screen on the D1/G1/T1 L1-redo runners.
%  checkcode does not execute, so it catches typos a long solve would only hit at
%  the end. Run: matlab -batch "II_LINT_l1redos".
files = {'II_RUN_D1_consistency_L1.m', 'II_RUN_G1_consistency_L1.m', ...
         'II_RUN_T1_consistency_L1.m', 'II_RUN_prm_bandit_largeN.m', ...
         'II_RUN_nonparam_L1.m', 'II_RUN_demand_L1.m', 'II_RUN_action_grid_sweep.m', ...
         'II_RUN_nonparam_L1_s10.m', 'II_RUN_costsupport_s20.m', ...
         fullfile('+df','+stages','run_stage_ii_nonparam.m')};
for i = 1:numel(files)
    m = checkcode(files{i}, '-string');
    if isempty(strtrim(m)), fprintf('LINT_CLEAN %s\n', files{i});
    else, fprintf('LINT %s:\n%s\n', files{i}, m); end
end
fprintf('LINT_DONE\n');
