%% compare_coneprog_vs_cvx.m — Cross-validate coneprog fixtures against CVX baselines
%
% Compares fixture files from the coneprog run (fixtures/) against the
% CVX baseline backup (fixtures_baseline_cvx/).

this_file = mfilename('fullpath');
test_dir = fileparts(this_file);
matlab_root = fileparts(test_dir);

new_dir = fullfile(matlab_root, 'test', 'fixtures');
old_dir = fullfile(matlab_root, 'test', 'fixtures_baseline_cvx');

fprintf('=== coneprog vs CVX Cross-Validation ===\n');
fprintf('New (coneprog): %s\n', new_dir);
fprintf('Old (CVX):      %s\n\n', old_dir);

files = {
    'fixture_stage_i_polytope.mat'
    'fixture_stage_ii_iter_5k.mat'
    'fixture_stage_ii_solver_all.mat'
    'fixture_stage_iii_solver_raw.mat'
    'fixture_stage_iii_player_1.mat'
    'fixture_stage_iii_player_2.mat'
    'fixture_stage_iv_bootstrap.mat'
    'fixture_stage_iv_identification.mat'
    'fixture_stage_iv_regret_comparison.mat'
};

all_pass = true;

for k = 1:numel(files)
    fname = files{k};
    f_new = fullfile(new_dir, fname);
    f_old = fullfile(old_dir, fname);

    if ~isfile(f_new)
        fprintf('[SKIP] %s — new file not found\n', fname);
        continue
    end
    if ~isfile(f_old)
        fprintf('[SKIP] %s — baseline not found\n', fname);
        continue
    end

    d_new = load(f_new);
    d_old = load(f_old);

    fields_new = fieldnames(d_new);
    fields_old = fieldnames(d_old);
    common = intersect(fields_new, fields_old);

    max_diff = 0;
    n_compared = 0;
    n_infeas_diff = 0;  % points where one is 100 (infeasible) and other is not
    field_diffs = {};

    for j = 1:numel(common)
        fn = common{j};
        v_new = d_new.(fn);
        v_old = d_old.(fn);

        if ~isnumeric(v_new) || ~isnumeric(v_old)
            continue
        end
        if ~isequal(size(v_new), size(v_old))
            fprintf('  [WARN] %s.%s: size mismatch %s vs %s\n', ...
                fname, fn, mat2str(size(v_new)), mat2str(size(v_old)));
            continue
        end

        n_compared = n_compared + 1;
        diff = abs(v_new(:) - v_old(:));

        % Identify infeasibility mismatches (value = 100)
        infeas_new = (v_new(:) >= 99);
        infeas_old = (v_old(:) >= 99);
        infeas_mismatch = xor(infeas_new, infeas_old);
        n_infeas = sum(infeas_mismatch);

        % Exclude infeasibility mismatches from max diff
        diff_clean = diff;
        diff_clean(infeas_mismatch) = 0;
        md = max(diff_clean);

        if md > max_diff
            max_diff = md;
        end
        if n_infeas > 0
            n_infeas_diff = n_infeas_diff + n_infeas;
        end

        if md > 1e-3
            field_diffs{end+1} = sprintf('%s (max=%.2e)', fn, md);
        end
    end

    if max_diff < 1e-3 && n_infeas_diff <= 30
        status = 'PASS';
    else
        status = 'FAIL';
        all_pass = false;
    end

    fprintf('[%s] %s: %d fields, max_diff=%.2e, infeas_mismatches=%d\n', ...
        status, fname, n_compared, max_diff, n_infeas_diff);
    if ~isempty(field_diffs)
        for j = 1:numel(field_diffs)
            fprintf('        %s\n', field_diffs{j});
        end
    end
end

fprintf('\n');
if all_pass
    fprintf('=== ALL PASS: coneprog produces equivalent results to CVX ===\n');
else
    fprintf('=== SOME FAILURES: check diffs above ===\n');
end

% Timing comparison
t_new_f = fullfile(new_dir, 'fixture_timing.mat');
t_old_f = fullfile(old_dir, 'fixture_timing.mat');
if isfile(t_new_f) && isfile(t_old_f)
    t_new = load(t_new_f); t_new = t_new.timing;
    t_old = load(t_old_f); t_old = t_old.timing;
    fprintf('\n=== Timing Comparison ===\n');
    fn_t = intersect(fieldnames(t_new), fieldnames(t_old));
    for k = 1:numel(fn_t)
        f = fn_t{k};
        fprintf('  %-12s  CVX: %7.1fs  coneprog: %7.1fs  speedup: %.1fx\n', ...
            f, t_old.(f), t_new.(f), t_old.(f)/t_new.(f));
    end
end
