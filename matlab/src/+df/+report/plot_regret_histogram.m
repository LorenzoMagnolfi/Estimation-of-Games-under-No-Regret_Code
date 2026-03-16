function fig = plot_regret_histogram(results, cfg, opts)
% DF.REPORT.PLOT_REGRET_HISTOGRAM  Broken-axis histogram of empirical vs theoretical regrets.
%
%   fig = df.report.plot_regret_histogram(results, cfg, opts)
%
%   Creates a histogram of bootstrap empirical regrets with vertical lines
%   for average empirical regret, 95th percentile, average worst-case
%   expected regret, and average epsilon bound. Uses a broken x-axis to
%   accommodate the large gap between empirical regrets and the theoretical
%   bound. Extracted from IV_MAIN lines 175-249.
%
%   Inputs:
%     results — struct from df.stages.run_stage_iv
%     cfg     — config struct
%     opts    — struct with:
%       .save_path — if nonempty, saves .epsc and .png (default: '')
%       .font_size — (default: 21)

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'save_path'), opts.save_path = ''; end
if ~isfield(opts, 'font_size'), opts.font_size = 21; end

all_regrets = results.all_regrets;
ExpectedRegretComp = results.ExpectedRegretComp;
average_epsilon = results.average_epsilon;
maxiters = results.maxiters;
marg_distrib = cfg.marg_distrib;

% Statistics
percentile_95 = prctile(all_regrets, 95);
mean_expected_regret = mean(ExpectedRegretComp);
average_empirical_regret = mean(all_regrets);

% Create figure
fig = figure('Position', [100, 100, 800, 600]);

% Histogram data
[counts, edges] = histcounts(all_regrets, 20, 'Normalization', 'probability');
centers = (edges(1:end-1) + edges(2:end)) / 2;

% Break point
break_start = max(percentile_95, mean_expected_regret) * 1.3;
break_end = average_epsilon * 0.98;
break_width = (break_end - break_start) / 50;

% Broken axis
h = BreakXAxis([centers average_epsilon], [counts 0.00001], break_start, break_end, break_width);
hold on;

% Bar plot
bar_width = centers(2) - centers(1);
h_bar = bar(centers, counts, bar_width * 1000000, 'FaceColor', 'b', 'EdgeColor', 'b');

% Coordinate mapping after break
map_x = @(x) x - (x > break_start) * (break_end - break_start - break_width);

% Vertical lines
l1 = line(map_x([average_empirical_regret average_empirical_regret]), ylim, ...
    'Color', 'r', 'LineStyle', '-', 'LineWidth', 2);
l2 = line(map_x([percentile_95 percentile_95]), ylim, ...
    'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
l3 = line(map_x([mean_expected_regret mean_expected_regret]), ylim, ...
    'Color', 'g', 'LineStyle', '-', 'LineWidth', 2);
l4 = line(map_x([average_epsilon average_epsilon]), ylim, ...
    'Color', 'g', 'LineStyle', '--', 'LineWidth', 2);

% Labels
xlabel('Regret', 'Interpreter', 'latex');
ylabel('Probability', 'Interpreter', 'latex');

% X-axis ticks
left_ticks = linspace(min(all_regrets), break_start, 3);
right_ticks = [break_end, average_epsilon];
all_ticks = [left_ticks, right_ticks];
mapped_ticks = map_x(all_ticks);
set(gca, 'XTick', mapped_ticks);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1e', x), all_ticks, 'UniformOutput', false));

% Legend
lgnd = legend([h_bar, l1, l2, l3, l4], ...
    {'Empirical regrets', 'Average empirical regret', ...
     '$95^{th}$ percentile empirical regret', ...
     'Average worst-case expected regret', ...
     'Average $\varepsilon(i,t_i;\lambda)$'}, ...
    'Location', 'northeast');
set(lgnd, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', opts.font_size);

% Appearance
yticks([0 0.2 0.4 0.6]);
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', ...
    'FontSize', opts.font_size, 'Box', 'off');
set(findall(gcf, 'type', 'text'), 'FontSize', opts.font_size);

% Save
if ~isempty(opts.save_path)
    saveas(gcf, opts.save_path, 'epsc');
    saveas(gcf, opts.save_path, 'png');
end

end
