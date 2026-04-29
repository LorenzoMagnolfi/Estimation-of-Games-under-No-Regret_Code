%% II_RUN_rate_landscape_pilot
%
%  Pilot exploratory exercise: re-solve the existing 2-seller pricing-game
%  simulation at the four sample sizes N in {500k, 1M, 2M, 4M} using
%  Niccolo's corrected adversarial + full-feedback radius (R1, switch_eps=10),
%  and produce a single overlay figure with all four confidence-region
%  boundaries on one (mu, sigma^2) plot.
%
%  Setup matches II_MAIN_simul.m exactly.  Only the rate switches.

clear all; clc; close all;

%% Setup (matches II_MAIN_simul)
paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Stage II run with NEW rate (switch_eps = 10)
stage_opts = struct();
stage_opts.maxiters_values = [500000, 1000000, 2000000, 4000000];
stage_opts.NGridV = 100;
stage_opts.NGridM = 100;
stage_opts.alpha_set = 0.05;
stage_opts.switch_eps = 10;          % Niccolo R1: adversarial + full feedback
stage_opts.backend    = 'fast';
stage_opts.adaptive   = true;        % exploratory pilot only

fprintf('\n=== Rate-landscape pilot: switch_eps=10 (Niccolo R1, full-feedback) ===\n\n');
t_total = tic;
results = df.stages.run_stage_ii(cfg, stage_opts);
fprintf('\n=== Total wall time: %.1f s ===\n', toc(t_total));

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'rate_landscape_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_R1.mat'), '-struct', 'results');
fprintf('Saved results to %s\n', fullfile(out_dir, 'results_R1.mat'));

%% Overlay plot: all 4 confidence regions on one (mu, sigma^2) plot
fig = figure('Color', 'w', 'Position', [100 100 750 550]);
hold on;

% Light-to-dark blue palette (small N -> light, large N -> dark)
cmap = [
    0.70 0.78 0.92;
    0.40 0.55 0.85;
    0.18 0.32 0.65;
    0.04 0.10 0.38
];

n_iters = numel(results.maxiters_values);
contour_handles = gobjects(n_iters, 1);
N_labels        = strings(n_iters, 1);

for k = 1:n_iters
    distpars = squeeze(results.distpars_all(k, :, :));
    VV       = results.VV_all(k, :);

    nV = results.NGridV + 1;
    nM = results.NGridM + 1;

    mu_grid = reshape(distpars(:, 1), nV, nM);
    sg_grid = reshape(distpars(:, 2), nV, nM);
    VV_grid = reshape(VV,            nV, nM);

    feasible = VV_grid <= 1e-12;

    % Outline of feasible region only.
    [~, h] = contour(mu_grid, sg_grid, double(feasible), [0.5 0.5], ...
        'LineColor', cmap(k, :), 'LineWidth', 2.2);
    contour_handles(k) = h;
    N_labels(k) = sprintf('N = %s', commas(results.maxiters_values(k)));
end

% True parameter
plot(mu(1), sigma2(1, 1), 'p', 'MarkerSize', 16, ...
    'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.85 0.15 0.15], ...
    'LineWidth', 1.2);
text(mu(1) + 0.04, sigma2(1, 1) + 0.18, '$(\mu^*,\sigma^{2*})$', ...
    'Interpreter', 'latex', 'FontSize', 13);

xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 14);
title({'95\% confidence regions, corrected adversarial + full-feedback radius (Niccolo R1)', ...
       'Two-seller pricing game; $|A|=5$, $s=5$ types/seller'}, ...
    'Interpreter', 'latex', 'FontSize', 12);
legend(contour_handles, N_labels, 'Location', 'northeast', ...
    'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;
set(gca, 'FontSize', 11);
hold off;

% Save
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'rate_landscape_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'overlay_R1.png'));
saveas(fig, fullfile(fig_dir, 'overlay_R1.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'overlay_R1.png'));

%% Helper for legend formatting
function s = commas(n)
    str = sprintf('%d', n);
    L   = length(str);
    s   = '';
    for i = 1:L
        s = [s, str(i)];
        rem = L - i;
        if rem > 0 && mod(rem, 3) == 0
            s = [s, ','];
        end
    end
end
