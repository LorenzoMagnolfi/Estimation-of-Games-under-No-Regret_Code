function fig = plot_identified_cdfs(distribution_parameters, id_set_index, support, opts)
% DF.REPORT.PLOT_IDENTIFIED_CDFS  CDF envelope of identified distributions.
%
%   fig = df.report.plot_identified_cdfs(distribution_parameters, id_set_index, support, opts)
%
%   Plots the empirical CDF envelope (min/max at each support point) of all
%   identified candidate distributions, with the true CDF overlaid. Responds
%   to R1.1.b: "To visually the results you could focus on ... the cdf."
%
%   Inputs:
%     distribution_parameters — {1 x NGrid} cell of (s x 1) probability vectors
%     id_set_index            — (1 x NGrid) or (NGrid x 1) logical
%     support                 — (s x 1) type support values
%     opts                    — struct with optional fields:
%       .true_distrib  — (s x 1) true marginal distribution
%       .save_path     — if nonempty, saves .epsc and .png
%       .font_size     — default: 17
%       .line_width    — default: 2
%       .fill_alpha    — default: 0.3
%       .fill_color    — default: [0.2 0.6 0.9]
%       .title_str     — optional title
%
%   Outputs:
%     fig — figure handle

if nargin < 4, opts = struct(); end
if ~isfield(opts, 'true_distrib'), opts.true_distrib = []; end
if ~isfield(opts, 'save_path'),    opts.save_path = ''; end
if ~isfield(opts, 'font_size'),    opts.font_size = 17; end
if ~isfield(opts, 'line_width'),   opts.line_width = 2; end
if ~isfield(opts, 'fill_alpha'),   opts.fill_alpha = 0.3; end
if ~isfield(opts, 'fill_color'),   opts.fill_color = [0.2 0.6 0.9]; end

support = support(:);
s = numel(support);
id_set_index = logical(id_set_index(:)');

% Collect CDFs of identified distributions
id_idx = find(id_set_index);
n_id = numel(id_idx);

if n_id == 0
    warning('No identified distributions — cannot plot CDF envelope.');
    fig = figure;
    return;
end

% Build CDF matrix: (n_id x s)
cdf_mat = zeros(n_id, s);
for k = 1:n_id
    p = distribution_parameters{id_idx(k)};
    p = p(:) ./ sum(p(:));
    cdf_mat(k, :) = cumsum(p)';
end

% Envelope: min and max CDF at each support point
cdf_lo = min(cdf_mat, [], 1);
cdf_hi = max(cdf_mat, [], 1);

% True CDF
if ~isempty(opts.true_distrib)
    p_true = opts.true_distrib(:) ./ sum(opts.true_distrib(:));
    cdf_true = cumsum(p_true)';
end

% Plot
fig = figure; hold on;

% Shaded envelope (step function style)
% Build step-function coordinates for fill
[x_fill, lo_fill, hi_fill] = cdf_step_coords(support, cdf_lo, cdf_hi);
fill([x_fill; flipud(x_fill)], [lo_fill; flipud(hi_fill)], ...
    opts.fill_color, 'FaceAlpha', opts.fill_alpha, 'EdgeColor', 'none');

% Envelope boundaries
stairs_plot(support, cdf_lo, opts.fill_color * 0.7, opts.line_width, '--');
stairs_plot(support, cdf_hi, opts.fill_color * 0.7, opts.line_width, '--');

% True CDF
if ~isempty(opts.true_distrib)
    h_true = stairs_plot(support, cdf_true, 'k', opts.line_width + 0.5, '-');
end

% Labels
xlabel('Type support', 'Interpreter', 'latex', 'FontSize', opts.font_size);
ylabel('CDF', 'Interpreter', 'latex', 'FontSize', opts.font_size);

if ~isempty(opts.true_distrib)
    legend(h_true, 'True CDF', 'Location', 'southeast', ...
        'Interpreter', 'latex', 'FontSize', opts.font_size - 2);
end

if isfield(opts, 'title_str') && ~isempty(opts.title_str)
    title(opts.title_str, 'Interpreter', 'none');
end

% Formatting
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', ...
    'FontSize', opts.font_size, 'Box', 'off');
ylim([0 1.05]);

% Annotation: number of identified distributions
text(support(1), 0.95, sprintf('%d identified distributions', n_id), ...
    'Interpreter', 'none', 'FontSize', opts.font_size - 3, ...
    'VerticalAlignment', 'top');

% Save
if ~isempty(opts.save_path)
    saveas(gcf, [opts.save_path '.eps'], 'epsc');
    saveas(gcf, [opts.save_path '.png'], 'png');
end

end


function h = stairs_plot(support, cdf_vals, color, lw, ls)
% Step-function plot for discrete CDF
    s = numel(support);
    x = zeros(2*s, 1);
    y = zeros(2*s, 1);
    for k = 1:s
        if k == 1
            x(1) = support(1);
        else
            x(2*k - 2) = support(k);
        end
        x(2*k - 1) = support(k);
        if k < s
            x(2*k) = support(k+1);
        else
            x(2*k) = support(k) + (support(k) - support(k-1));
        end
        y(2*k - 1) = cdf_vals(k);
        y(2*k) = cdf_vals(k);
    end
    h = plot(x, y, 'Color', color, 'LineWidth', lw, 'LineStyle', ls);
end


function [x_fill, lo_fill, hi_fill] = cdf_step_coords(support, cdf_lo, cdf_hi)
% Build step-function coordinates for filled envelope
    s = numel(support);
    x_fill = zeros(2*s, 1);
    lo_fill = zeros(2*s, 1);
    hi_fill = zeros(2*s, 1);
    for k = 1:s
        x_fill(2*k - 1) = support(k);
        if k < s
            x_fill(2*k) = support(k+1);
        else
            x_fill(2*k) = support(k) + (support(k) - support(k-1));
        end
        lo_fill(2*k - 1) = cdf_lo(k);
        lo_fill(2*k)     = cdf_lo(k);
        hi_fill(2*k - 1) = cdf_hi(k);
        hi_fill(2*k)     = cdf_hi(k);
    end
end
