function [label, PGx, SVMMod] = classify_identified_set(distpars, id_set_index, halton_ranges, opts)
% DF.REPORT.CLASSIFY_IDENTIFIED_SET  SVM classification of identified set boundary.
%
%   [label, PGx, SVMMod] = df.report.classify_identified_set(distpars, id_set_index, halton_ranges)
%   [label, PGx, SVMMod] = df.report.classify_identified_set(distpars, id_set_index, halton_ranges, opts)
%
%   Fits a Gaussian-kernel SVM to the identified/non-identified parameter
%   grid points and predicts labels on a dense Halton quasi-random grid.
%   Unifies SVM logic from Stages II, III, and IV.
%
%   Inputs:
%     distpars       — (NGrid x 2) parameter values from build_param_grid
%     id_set_index   — (NGrid x 1) logical: true = identified
%     halton_ranges  — [xmax, ymax] scaling for Halton grid
%                      OR [xmin_offset, xmax, ymin_offset, ymax] for absolute ranges
%     opts           — optional struct:
%       .NGx         — number of Halton points (default: 500000)
%                      OR use .quality preset: 'draft' (50k) | 'final' (500k)
%       .quality     — 'draft' | 'final' (default: 'final')
%                      'draft': 50k points, fast for exploration/iteration
%                      'final': 500k points, publication quality
%       .filter      — function handle for post-filter (e.g., Stage IV clips to rectangle)
%
%   Outputs:
%     label   — (NGx x 1) predicted labels on Halton grid
%     PGx     — (NGx x 2) Halton grid points (after any filtering)
%     SVMMod  — trained SVM model

if nargin < 4, opts = struct(); end

% Quality presets: 'draft' (50k, fast) vs 'final' (500k, publication)
if isfield(opts, 'quality') && ~isfield(opts, 'NGx')
    switch opts.quality
        case 'draft',  opts.NGx = 50000;
        case 'final',  opts.NGx = 500000;
        otherwise, error('opts.quality must be ''draft'' or ''final''.');
    end
end
if ~isfield(opts, 'NGx'), opts.NGx = 500000; end

% Build training data
Xtrain = distpars;
YTrain = id_set_index(:);

% Generate Halton grid
ppx = haltonset(2, 'Skip', 1e3, 'Leap', 1e2);
PG0x = net(ppx, opts.NGx - 1);

if numel(halton_ranges) == 2
    % Simple [xmax, ymax] scaling
    PGx = [PG0x(:,1) * halton_ranges(1), PG0x(:,2) * halton_ranges(2)];
elseif numel(halton_ranges) == 4
    % [xmin_offset, xmax, ymin_offset, ymax] — data-adaptive (Stage III)
    PGx = [PG0x(:,1) * abs(halton_ranges(2) - halton_ranges(1)) + halton_ranges(1), ...
           PG0x(:,2) * abs(halton_ranges(4) - halton_ranges(3)) + halton_ranges(3)];
else
    error('halton_ranges must be [xmax, ymax] or [xmin, xmax, ymin, ymax].');
end

% Optional filtering (e.g., Stage IV clips to a rectangle)
if isfield(opts, 'filter') && ~isempty(opts.filter)
    keep = opts.filter(PGx);
    PGx = PGx(keep, :);
end

% Train Gaussian SVM
SVMMod = fitcsvm(Xtrain, YTrain, ...
    'KernelFunction', 'gaussian', 'Standardize', true, 'KernelScale', 'auto');

% Predict on Halton grid
label = predict(SVMMod, PGx);

end
