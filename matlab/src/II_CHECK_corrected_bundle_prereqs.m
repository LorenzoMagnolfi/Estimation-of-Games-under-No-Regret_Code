%% II_CHECK_corrected_bundle_prereqs
%
% Server prerequisite check for the corrected JPE revision simulation bundle.
% Run from matlab/src before II_RUN_revision_corrected_bundle.
%
% Optional environment variables:
%   CVX_DIR           directory containing a CVX installation
%   SEDUMI_DIR        directory containing a SeDuMi installation
%   MATLAB_EXTRA_PATH pathsep-separated extra MATLAB paths

clear; clc;

paths = df_repo_paths();

add_env_path('CVX_DIR');
add_env_path('SEDUMI_DIR');
add_extra_paths('MATLAB_EXTRA_PATH');

fprintf('=== Corrected bundle prerequisite check ===\n');
fprintf('MATLAB: %s\n', version);
fprintf('Working directory: %s\n', pwd);
fprintf('Output root: %s\n', paths.output);

cvx_path = which('cvx_clear');
sedumi_path = which('sedumi');

fprintf('cvx_clear: %s\n', cvx_path);
fprintf('sedumi: %s\n', sedumi_path);

assert(~isempty(cvx_path), ...
    ['CVX is not on the MATLAB path. Set CVX_DIR or MATLAB_EXTRA_PATH, ' ...
     'or run cvx_setup before launching the bundle.']);
assert(~isempty(sedumi_path), ...
    ['SeDuMi is not on the MATLAB path. Set SEDUMI_DIR or MATLAB_EXTRA_PATH, ' ...
     'or add the solver path before launching the bundle.']);

fprintf('Prerequisite check passed.\n');

function add_env_path(var_name)
    path_value = getenv(var_name);
    if isempty(path_value)
        return
    end
    if exist(path_value, 'dir')
        fprintf('Adding %s: %s\n', var_name, path_value);
        addpath(genpath(path_value));
    else
        warning('II_CHECK:MissingPath', ...
            'Environment variable %s points to a missing directory: %s', ...
            var_name, path_value);
    end
end

function add_extra_paths(var_name)
    path_value = getenv(var_name);
    if isempty(path_value)
        return
    end
    pieces = strsplit(path_value, pathsep);
    for ii = 1:numel(pieces)
        this_path = pieces{ii};
        if isempty(this_path)
            continue
        end
        if exist(this_path, 'dir')
            fprintf('Adding %s component: %s\n', var_name, this_path);
            addpath(genpath(this_path));
        else
            warning('II_CHECK:MissingExtraPath', ...
                'MATLAB_EXTRA_PATH component is missing: %s', this_path);
        end
    end
end
