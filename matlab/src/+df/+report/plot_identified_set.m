function fig = plot_identified_set(PGx, label, distpars, id_set_index, opts)
% DF.REPORT.PLOT_IDENTIFIED_SET  Visualize identified set with SVM boundary and projections.
%
%   fig = df.report.plot_identified_set(PGx, label, distpars, id_set_index, opts)
%
%   Plots the SVM-classified Halton grid with identified set boundary,
%   horizontal (mu) and vertical (sigma) projections, and optional true
%   parameter marker. Unifies plotting from Stages II, III, and IV.
%
%   Inputs:
%     PGx          — (N x 2) Halton grid points
%     label        — (N x 1) predicted labels (1 = identified)
%     distpars     — (NGrid x 2) or (2 x NGrid) original parameter grid
%     id_set_index — (NGrid x 1) logical: true = identified
%     opts         — struct with fields:
%       .true_param     — [mu, sigma] to plot as marker (default: [])
%       .xlabel_str     — x-axis label (default: '$\mu$')
%       .ylabel_str     — y-axis label (default: '$\sigma$')
%       .hover_offset   — projection offset (default: 0.9)
%       .yticks         — custom y-tick values (default: auto)
%       .legend_loc     — legend location (default: 'northeast')
%       .legend_ncol    — legend columns (default: 1)
%       .font_size      — font size (default: 17)
%       .marker_size    — scatter marker size (default: 20)
%       .scatter_colors — 2-char color spec for gscatter (default: 'wc')
%       .save_path      — if nonempty, saves .epsc and .png (default: '')

if nargin < 5, opts = struct(); end

% Defaults
if ~isfield(opts, 'true_param'),    opts.true_param = []; end
if ~isfield(opts, 'xlabel_str'),    opts.xlabel_str = '$\mu$'; end
if ~isfield(opts, 'ylabel_str'),    opts.ylabel_str = '$\sigma$'; end
if ~isfield(opts, 'hover_offset'),  opts.hover_offset = 0.9; end
if ~isfield(opts, 'yticks'),        opts.yticks = []; end
if ~isfield(opts, 'legend_loc'),    opts.legend_loc = 'northeast'; end
if ~isfield(opts, 'legend_ncol'),   opts.legend_ncol = 1; end
if ~isfield(opts, 'font_size'),     opts.font_size = 17; end
if ~isfield(opts, 'marker_size'),   opts.marker_size = 20; end
if ~isfield(opts, 'scatter_colors'),opts.scatter_colors = 'wc'; end
if ~isfield(opts, 'save_path'),     opts.save_path = ''; end

% Ensure distpars is (NGrid x 2) — Stage II passes (2 x NGrid)
if size(distpars, 1) == 2 && size(distpars, 2) > 2
    distpars = distpars';
    % Also flip id_set_index to column if row
    id_set_index = id_set_index(:);
end

% Extract identified set points from original grid
id_set_points = distpars(id_set_index, :);
min_mu = min(id_set_points(:, 1));
max_mu = max(id_set_points(:, 1));
min_sigma = min(id_set_points(:, 2));
max_sigma = max(id_set_points(:, 2));

% Create figure
fig = figure;
hold on;

% Main scatter
s1 = gscatter(PGx(:,1), PGx(:,2), label(:), opts.scatter_colors, '.', opts.marker_size);

% True parameter marker
if ~isempty(opts.true_param)
    s2 = plot(opts.true_param(1), opts.true_param(2), '.', 'MarkerSize', 25, 'Color', 'k');
end

% Axis labels
xlabel(opts.xlabel_str, 'Interpreter', 'latex');
ylabel(opts.ylabel_str, 'Interpreter', 'latex');

% Horizontal projection (mu range)
ho = opts.hover_offset;
plot([min_mu, max_mu], [min_sigma, min_sigma] - ho, 'r-', 'LineWidth', 1.5);
s3 = plot(min_mu, min_sigma - ho, 'ko', 'MarkerFaceColor', 'r');
plot(max_mu, min_sigma - ho, 'ko', 'MarkerFaceColor', 'r');

% Vertical projection (sigma range)
plot([min_mu, min_mu] - ho, [min_sigma, max_sigma], 'g-', 'LineWidth', 1.5);
s4 = plot(min_mu - ho, max_sigma, 'ko', 'MarkerFaceColor', 'g');
plot(min_mu - ho, min_sigma, 'ko', 'MarkerFaceColor', 'g');

% Build legend
mu_label = strrep(opts.xlabel_str, '$', '');
sig_label = strrep(opts.ylabel_str, '$', '');
mu_str = strcat(['$' mu_label ' \in [', num2str(min_mu, '%.2f'), ', ', num2str(max_mu, '%.2f'), ']$']);
sig_str = strcat(['$' sig_label ' \in [', num2str(min_sigma, '%.2f'), ', ', num2str(max_sigma, '%.2f'), ']$']);

if ~isempty(opts.true_param)
    lgnd = legend([s1(2), s2, s3, s4], 'Confidence region', 'True parameter', ...
        mu_str, sig_str, 'Location', opts.legend_loc, 'NumColumns', opts.legend_ncol);
else
    lgnd = legend([s3, s4], mu_str, sig_str, ...
        'Location', opts.legend_loc, 'NumColumns', opts.legend_ncol);
end
set(lgnd, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', opts.font_size);

% Y-ticks
if ~isempty(opts.yticks)
    yticks(opts.yticks);
end

% Axes formatting
set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'Units', 'normalized', 'FontUnits', 'points', 'FontWeight', 'normal', ...
    'FontName', 'cmr10', 'FontSize', opts.font_size, 'Box', 'off');

% Save
if ~isempty(opts.save_path)
    saveas(gcf, opts.save_path, 'epsc');
    saveas(gcf, opts.save_path, 'png');
end

end
