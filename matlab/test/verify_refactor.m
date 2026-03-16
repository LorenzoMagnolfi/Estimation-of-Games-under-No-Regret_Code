%% verify_refactor.m — Compare refactored outputs against baseline fixtures
%
% PURPOSE:  After refactoring a module, re-run the fixture runner (or the
%           refactored equivalent) and compare new outputs against the
%           baseline .mat files captured before the refactor.
%
% USAGE:    >> cd matlab/src; run('../test/verify_refactor.m')
%
% PROTOCOL:
%   1. Run run_fixtures.m ONCE before any refactoring (baseline capture).
%   2. Copy the fixture_*.mat files to a 'baseline' subdirectory:
%        matlab/test/fixtures/baseline/
%   3. Refactor the code.
%   4. Run run_fixtures.m AGAIN (produces new fixture_*.mat in fixtures/).
%   5. Run this script to compare new vs baseline.
%
% TOLERANCE:
%   Numerical: atol = 1e-10, rtol = 1e-8
%   Solver outputs (VV, maxvals): atol = 1e-6 (CVX solver tolerance)

clear all; clc; close all;

paths = df_repo_paths();
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
baseline_dir = fullfile(fixture_dir, 'baseline');

if ~exist(baseline_dir, 'dir')
    error(['Baseline directory not found: %s\n' ...
           'Run run_fixtures.m, then copy fixture_*.mat to baseline/ before refactoring.'], ...
           baseline_dir);
end

% Tolerances
atol_strict = 1e-10;   % For quantities that should be bitwise identical
atol_solver = 1e-6;    % For solver outputs (CVX numerical tolerance)
rtol = 1e-8;           % Relative tolerance

%% Discover fixture files
baseline_files = dir(fullfile(baseline_dir, 'fixture_*.mat'));
if isempty(baseline_files)
    error('No fixture_*.mat files found in baseline directory.');
end

fprintf('=== Refactor Verification ===\n');
fprintf('Baseline: %s\n', baseline_dir);
fprintf('Current:  %s\n\n', fixture_dir);

n_pass = 0;
n_fail = 0;
n_skip = 0;
failures = {};

for k = 1:numel(baseline_files)
    fname = baseline_files(k).name;
    current_file = fullfile(fixture_dir, fname);
    baseline_file = fullfile(baseline_dir, fname);

    if ~isfile(current_file)
        fprintf('  SKIP  %s (no current file)\n', fname);
        n_skip = n_skip + 1;
        continue;
    end

    fprintf('  Checking %s ...\n', fname);

    base = load(baseline_file);
    curr = load(current_file);

    % Compare all shared fields
    base_fields = fieldnames(base);
    curr_fields = fieldnames(curr);
    shared = intersect(base_fields, curr_fields);
    missing_in_curr = setdiff(base_fields, curr_fields);
    extra_in_curr = setdiff(curr_fields, base_fields);

    if ~isempty(missing_in_curr)
        fprintf('    WARN: Missing in current: %s\n', strjoin(missing_in_curr, ', '));
    end
    if ~isempty(extra_in_curr)
        fprintf('    INFO: Extra in current: %s\n', strjoin(extra_in_curr, ', '));
    end

    file_ok = true;

    for j = 1:numel(shared)
        field = shared{j};
        b_val = base.(field);
        c_val = curr.(field);

        % Choose tolerance based on field name
        if contains(field, {'VV', 'maxvals', 'id_set', 'ratio'})
            atol = atol_solver;
        else
            atol = atol_strict;
        end

        [match, msg] = compare_values(b_val, c_val, field, atol, rtol);

        if ~match
            fprintf('    FAIL  %s: %s\n', field, msg);
            file_ok = false;
            failures{end+1} = sprintf('%s :: %s :: %s', fname, field, msg);
        end
    end

    if file_ok
        fprintf('    PASS  (%d fields checked)\n', numel(shared));
        n_pass = n_pass + 1;
    else
        n_fail = n_fail + 1;
    end
end

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('  PASS: %d    FAIL: %d    SKIP: %d\n', n_pass, n_fail, n_skip);

if n_fail > 0
    fprintf('\nFailures:\n');
    for k = 1:numel(failures)
        fprintf('  %s\n', failures{k});
    end
    fprintf('\nRefactor verification FAILED.\n');
else
    fprintf('\nRefactor verification PASSED.\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper: compare two values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [match, msg] = compare_values(b, c, name, atol, rtol)
    match = true;
    msg = '';

    % Type check
    if ~strcmp(class(b), class(c))
        match = false;
        msg = sprintf('type mismatch: %s vs %s', class(b), class(c));
        return;
    end

    % Cell arrays
    if iscell(b)
        if ~isequal(size(b), size(c))
            match = false;
            msg = sprintf('cell size mismatch: %s vs %s', mat2str(size(b)), mat2str(size(c)));
            return;
        end
        % Check each cell element
        for idx = 1:numel(b)
            if isnumeric(b{idx}) && isnumeric(c{idx})
                [m, mg] = compare_numeric(b{idx}, c{idx}, atol, rtol);
                if ~m
                    match = false;
                    msg = sprintf('cell{%d}: %s', idx, mg);
                    return;
                end
            elseif ischar(b{idx}) && ischar(c{idx})
                if ~strcmp(b{idx}, c{idx})
                    match = false;
                    msg = sprintf('cell{%d}: string mismatch', idx);
                    return;
                end
            end
        end
        return;
    end

    % Numeric arrays
    if isnumeric(b)
        [match, msg] = compare_numeric(b, c, atol, rtol);
        return;
    end

    % Logical arrays
    if islogical(b)
        if ~isequal(b, c)
            match = false;
            n_diff = sum(b(:) ~= c(:));
            msg = sprintf('%d/%d elements differ', n_diff, numel(b));
        end
        return;
    end

    % Strings
    if ischar(b) || isstring(b)
        if ~strcmp(b, c)
            match = false;
            msg = 'string mismatch';
        end
        return;
    end

    % Struct
    if isstruct(b)
        if ~isequal(b, c)
            match = false;
            msg = 'struct mismatch';
        end
        return;
    end

    % Fallback: isequal
    if ~isequal(b, c)
        match = false;
        msg = 'values differ (fallback comparison)';
    end
end

function [match, msg] = compare_numeric(b, c, atol, rtol)
    match = true;
    msg = '';

    if ~isequal(size(b), size(c))
        match = false;
        msg = sprintf('size mismatch: %s vs %s', mat2str(size(b)), mat2str(size(c)));
        return;
    end

    % Handle NaN
    nan_b = isnan(b);
    nan_c = isnan(c);
    if ~isequal(nan_b, nan_c)
        match = false;
        msg = 'NaN pattern mismatch';
        return;
    end

    % Compare non-NaN values
    valid = ~nan_b;
    if any(valid(:))
        abs_diff = abs(b(valid) - c(valid));
        max_abs = max(abs_diff);

        denom = max(abs(b(valid)), abs(c(valid)));
        denom(denom == 0) = 1;
        max_rel = max(abs_diff ./ denom);

        if max_abs > atol && max_rel > rtol
            match = false;
            msg = sprintf('max_abs=%.2e, max_rel=%.2e (atol=%.0e, rtol=%.0e)', ...
                max_abs, max_rel, atol, rtol);
        end
    end
end
