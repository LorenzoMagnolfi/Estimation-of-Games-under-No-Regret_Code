%% II_RUN_prm_comparison
%
% R1.1.c: confidence sets under proxy-regret matching.
%
% The comparison separates two objects that should not be conflated:
%   1. the learning rule used to generate the empirical play distribution;
%   2. the finite-sample confidence radius implied by the feedback structure.
%
% Regret matching is paired with the corrected adversarial full-feedback
% radius (switch_eps = 10). Proxy-regret matching is paired with the
% corrected adversarial bandit-feedback radius (switch_eps = 11).

clear all; clc; close all;

paths = df_repo_paths();
rng(12345);

%% Game parameters
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Shared Stage II options
stage_opts_base = struct();
stage_opts_base.maxiters_values = [500000, 1000000, 2000000, 4000000];
stage_opts_base.NGridV = 100;
stage_opts_base.NGridM = 100;
stage_opts_base.alpha_set = 0.05;
stage_opts_base.backend = 'fast';
stage_opts_base.adaptive = true;
stage_opts_base.consistency_slack_kind = 'none';

%% Run 1: regret matching with full-feedback radius
fprintf('\n========== Regret matching, full-feedback radius ==========\n');
rng(12345);
stage_opts_rm = stage_opts_base;
stage_opts_rm.learning_style = 'rm';
stage_opts_rm.switch_eps = 10;
results_rm = df.stages.run_stage_ii(cfg, stage_opts_rm);

%% Run 2: proxy-regret matching with bandit-feedback radius
fprintf('\n========== Proxy-regret matching, bandit-feedback radius ==========\n');
rng(12345);
stage_opts_prm = stage_opts_base;
stage_opts_prm.learning_style = 'prm';
stage_opts_prm.switch_eps = 11;
results_prm = df.stages.run_stage_ii(cfg, stage_opts_prm);

%% Summaries
summary_table = summarize_prm_results(results_rm, results_prm);
disp(summary_table);

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary_table, fullfile(paths.tables_ii, 'prm_comparison_s5.csv'));

%% Figures
if ~exist(paths.figures_ii, 'dir'), mkdir(paths.figures_ii); end

fig_all = figure('Color', 'w', 'Position', [50 100 1600 720]);
tiledlayout(2, numel(results_rm.maxiters_values), ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for mi = 1:numel(results_rm.maxiters_values)
    N = results_rm.maxiters_values(mi);

    ax = nexttile(mi);
    plot_identified_region(ax, results_rm, mi, [0.12 0.32 0.66], ...
        sprintf('RM, R1 radius, N=%s', commas(N)));

    ax = nexttile(mi + numel(results_rm.maxiters_values));
    plot_identified_region(ax, results_prm, mi, [0.55 0.16 0.16], ...
        sprintf('PRM, bandit radius, N=%s', commas(N)));
end

sgtitle('Regret matching versus proxy-regret matching: 95 percent confidence regions', ...
    'Interpreter', 'latex', 'FontSize', 13);
saveas(fig_all, fullfile(paths.figures_ii, 'prm_comparison_s5_allN.png'));
saveas(fig_all, fullfile(paths.figures_ii, 'prm_comparison_s5_allN.pdf'));

last_idx = numel(results_rm.maxiters_values);
fig_last = figure('Color', 'w', 'Position', [100 100 1200 520]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(1);
plot_identified_region(ax, results_rm, last_idx, [0.12 0.32 0.66], ...
    sprintf('Regret matching, R1 radius, N=%s', commas(results_rm.maxiters_values(last_idx))));

ax = nexttile(2);
plot_identified_region(ax, results_prm, last_idx, [0.55 0.16 0.16], ...
    sprintf('Proxy-regret matching, bandit radius, N=%s', commas(results_prm.maxiters_values(last_idx))));

sgtitle('R1.1.c: full feedback versus bandit feedback in the running example', ...
    'Interpreter', 'latex', 'FontSize', 13);
saveas(fig_last, fullfile(paths.figures_ii, 'prm_comparison_s5_4M.png'));
saveas(fig_last, fullfile(paths.figures_ii, 'prm_comparison_s5_4M.pdf'));

%% Save artifact
save(fullfile(paths.artifacts, 'prm_comparison_s5.mat'), ...
    'results_rm', 'results_prm', 'summary_table', ...
    'stage_opts_rm', 'stage_opts_prm', '-v7.3');

fprintf('\n========== PRM comparison complete ==========\n');
fprintf('Artifact: %s\n', fullfile(paths.artifacts, 'prm_comparison_s5.mat'));
fprintf('Figures:  %s\n', fullfile(paths.figures_ii, 'prm_comparison_s5_4M.pdf'));

%% Local helpers
function summary_table = summarize_prm_results(results_rm, results_prm)
    n_iters = numel(results_rm.maxiters_values);
    N = zeros(n_iters, 1);
    rm_count = zeros(n_iters, 1);
    rm_share = zeros(n_iters, 1);
    rm_mu_min = nan(n_iters, 1);
    rm_mu_max = nan(n_iters, 1);
    rm_sig2_min = nan(n_iters, 1);
    rm_sig2_max = nan(n_iters, 1);
    prm_count = zeros(n_iters, 1);
    prm_share = zeros(n_iters, 1);
    prm_mu_min = nan(n_iters, 1);
    prm_mu_max = nan(n_iters, 1);
    prm_sig2_min = nan(n_iters, 1);
    prm_sig2_max = nan(n_iters, 1);

    for ii = 1:n_iters
        N(ii) = results_rm.maxiters_values(ii);
        [rm_count(ii), rm_share(ii), rm_mu_min(ii), rm_mu_max(ii), ...
            rm_sig2_min(ii), rm_sig2_max(ii)] = summarize_one(results_rm, ii);
        [prm_count(ii), prm_share(ii), prm_mu_min(ii), prm_mu_max(ii), ...
            prm_sig2_min(ii), prm_sig2_max(ii)] = summarize_one(results_prm, ii);
    end

    summary_table = table(N, rm_count, rm_share, rm_mu_min, rm_mu_max, ...
        rm_sig2_min, rm_sig2_max, prm_count, prm_share, prm_mu_min, ...
        prm_mu_max, prm_sig2_min, prm_sig2_max);
end

function [count, share, mu_min, mu_max, sig2_min, sig2_max] = summarize_one(results, iter_idx)
    distpars = squeeze(results.distpars_all(iter_idx, :, :));
    VV = results.VV_all(iter_idx, :);
    identified = VV(:) <= 1e-12;
    count = sum(identified);
    share = count / numel(identified);
    if any(identified)
        mu_vals = distpars(identified, 1);
        sig2_vals = distpars(identified, 2);
        mu_min = min(mu_vals);
        mu_max = max(mu_vals);
        sig2_min = min(sig2_vals);
        sig2_max = max(sig2_vals);
    else
        mu_min = NaN;
        mu_max = NaN;
        sig2_min = NaN;
        sig2_max = NaN;
    end
end

function plot_identified_region(ax, results, iter_idx, color, title_text)
    axes(ax);
    distpars = squeeze(results.distpars_all(iter_idx, :, :));
    VV = results.VV_all(iter_idx, :);
    nV = results.NGridV + 1;
    nM = results.NGridM + 1;

    mu_grid = reshape(distpars(:, 1), nV, nM);
    sig2_grid = reshape(distpars(:, 2), nV, nM);
    VV_grid = reshape(VV, nV, nM);

    % The first row and column contain the prepended true grid value and
    % are non-monotone in the plotting grid.
    mu_grid = mu_grid(2:end, 2:end);
    sig2_grid = sig2_grid(2:end, 2:end);
    VV_grid = VV_grid(2:end, 2:end);

    feasible = VV_grid <= 1e-12;
    hold on;
    if any(feasible(:)) && any(~feasible(:))
        contour(mu_grid, sig2_grid, double(feasible), [0.5 0.5], ...
            'LineColor', color, 'LineWidth', 2.2);
    elseif any(feasible(:))
        text(0.5, 0.5, 'All plotted candidates included', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'Interpreter', 'latex');
    else
        text(0.5, 0.5, 'No plotted candidates included', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'Interpreter', 'latex');
    end

    plot(3, 1, 'p', 'MarkerSize', 14, ...
        'MarkerEdgeColor', [0.10 0.10 0.10], ...
        'MarkerFaceColor', [0.85 0.15 0.15], 'LineWidth', 1.2);
    xlim([2.0, 4.0]);
    ylim([0.0, 4.5]);
    xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 11);
    title(title_text, 'Interpreter', 'latex', 'FontSize', 10);
    grid on;
    box on;
    set(gca, 'FontSize', 10);
    hold off;
end

function s = commas(n)
    str = sprintf('%d', n);
    L = length(str);
    s = '';
    for i = 1:L
        s = [s, str(i)];
        rem = L - i;
        if rem > 0 && mod(rem, 3) == 0
            s = [s, ','];
        end
    end
end
