function paths = repo_paths()
% df.io.repo_paths  Resolve repository-relative MATLAB paths.
%
%   paths = df.io.repo_paths() returns a struct of absolute paths for the
%   project's data, output, figure, table, and artifact directories. Creates
%   any missing directories on first call.

src_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matlab_root = fileparts(src_dir);

paths = struct();
paths.src = src_dir;
paths.matlab_root = matlab_root;
paths.repo_root = fileparts(matlab_root);
paths.data = fullfile(matlab_root, 'data');
paths.output = fullfile(matlab_root, 'output');
paths.figures = fullfile(paths.output, 'figures');
paths.tables = fullfile(paths.output, 'tables');
paths.artifacts = fullfile(paths.output, 'artifacts');
paths.figures_i = fullfile(paths.figures, 'part_i');
paths.figures_ii = fullfile(paths.figures, 'part_ii');
paths.figures_iii = fullfile(paths.figures, 'part_iii');
paths.figures_iv = fullfile(paths.figures, 'part_iv');
paths.tables_ii = fullfile(paths.tables, 'part_ii');
paths.tables_iii = fullfile(paths.tables, 'part_iii');
paths.artifacts_iv = fullfile(paths.artifacts, 'part_iv');
paths.legacy_ampl_output = fullfile(src_dir, 'Output_Pricing');

required_dirs = {
    paths.output
    paths.figures
    paths.tables
    paths.artifacts
    paths.figures_i
    paths.figures_ii
    paths.figures_iii
    paths.figures_iv
    paths.tables_ii
    paths.tables_iii
    paths.artifacts_iv
    paths.legacy_ampl_output
};

for ii = 1:numel(required_dirs)
    if ~exist(required_dirs{ii}, 'dir')
        mkdir(required_dirs{ii});
    end
end

addpath(src_dir);

end
