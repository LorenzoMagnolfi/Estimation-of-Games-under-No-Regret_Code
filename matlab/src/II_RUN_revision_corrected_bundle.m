%% II_RUN_revision_corrected_bundle
%
% Server-side bundle for the JPE revision simulations after Niccolo's
% correction to the finite-sample construction.
%
% Default behavior:
%   - Do not rerun the rate-landscape pilot, because existing R1 outputs
%     were already generated with switch_eps = 10.
%   - Rerun nonparametric and demand-cost exercises with switch_eps = 10.
%   - Run PRM comparison with switch_eps = 10 for full feedback and
%     switch_eps = 11 for bandit feedback.
%
% Run from matlab/src on linstat after confirming:
%   which cvx_clear
%   which sedumi

clear; clc;

II_CHECK_corrected_bundle_prereqs;
paths = df_repo_paths();

run_rate_landscape = false;
run_nonparam = true;
run_demand = true;
run_prm = true;

scripts = {};
if run_rate_landscape
    scripts{end+1} = 'II_RUN_rate_landscape_pilot_v2'; %#ok<SAGROW>
end
if run_nonparam
    scripts{end+1} = 'II_RUN_nonparam_revision'; %#ok<SAGROW>
end
if run_demand
    scripts{end+1} = 'II_RUN_demand_identification'; %#ok<SAGROW>
end
if run_prm
    scripts{end+1} = 'II_RUN_prm_comparison'; %#ok<SAGROW>
end

status = struct();
status.started_at = char(datetime('now'));
status.scripts = scripts;
status.failed = {};
status.completed = {};

log_path = fullfile(paths.output, 'corrected_revision_bundle.log');
diary(log_path);
fprintf('=== Corrected revision simulation bundle ===\n');
fprintf('Started: %s\n', status.started_at);
fprintf('MATLAB: %s\n', version);
fprintf('cvx_clear: %s\n', which('cvx_clear'));
fprintf('sedumi: %s\n', which('sedumi'));
fprintf('Output root: %s\n\n', paths.output);

for si = 1:numel(scripts)
    script_name = scripts{si};
    fprintf('\n--- Running %s (%d/%d) ---\n', script_name, si, numel(scripts));
    t_script = tic;
    try
        run([script_name '.m']);
        status.completed{end+1} = script_name; %#ok<SAGROW>
        fprintf('--- Completed %s in %.1f minutes ---\n', script_name, toc(t_script)/60);
    catch ME
        status.failed{end+1} = script_name; %#ok<SAGROW>
        fprintf(2, '--- FAILED %s after %.1f minutes ---\n', script_name, toc(t_script)/60);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        save(fullfile(paths.output, 'corrected_revision_bundle_status.mat'), 'status');
        diary off;
        rethrow(ME);
    end
    save(fullfile(paths.output, 'corrected_revision_bundle_status.mat'), 'status');
end

status.finished_at = char(datetime('now'));
save(fullfile(paths.output, 'corrected_revision_bundle_status.mat'), 'status');
fprintf('\n=== Bundle complete ===\n');
fprintf('Finished: %s\n', status.finished_at);
fprintf('Status artifact: %s\n', fullfile(paths.output, 'corrected_revision_bundle_status.mat'));
diary off;
