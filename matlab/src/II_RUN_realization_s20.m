%% II_RUN_realization_s20
%
%  Realization-conditional identified set with FINER cost grid: s = 20.
%  Cost grid: linspace(0, 6, 20), spacing 6/19 ≈ 0.316.
%
%  Data-only spec only (BCCE LP skipped: it's mis-specified for persistent
%  types and prohibitively slow at s=20).  Reports BOTH max-Kappa and
%  per-candidate-Kappa eps so we can see which is sharper.
%
%  Truths chosen to match the cost positions from the s=5 run:
%    - symmetric center: index 11 (cost 3.158, near 3.0)
%    - asymmetric:       index (6, 15) (cost (1.579, 4.421), near (1.5, 4.5))
%    - corner:           index (1, 20) (cost (0.0, 6.0), exact)

clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(20260430 + 20);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 20;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

fprintf('\n=== Realization-conditional, s = %d ===\n', s);
fprintf('Cost grid (first 10): [%s]\n', sprintf('%.3f ', cfg.type_space{1,1}(1:10)));
fprintf('Cost grid (last 10) : [%s]\n', sprintf('%.3f ', cfg.type_space{1,1}(11:20)));

%% Per-type Kappa
Kappa_per_type = zeros(s, NPlayers);
for j = 1:NPlayers
    for t = 1:s
        Pi_jt = cfg.Pi(:, t, j);
        Kappa_per_type(t, j) = max(Pi_jt) - min(Pi_jt);
    end
end
Kappa_max = max(Kappa_per_type(:));
fprintf('Kappa range across types: [%.4f, %.4f] (max = %.4f)\n', ...
    min(Kappa_per_type(:)), max(Kappa_per_type(:)), Kappa_max);

%% Truths
truths = [11, 11;     % symmetric center
           6, 15;     % asymmetric (low, high) ~ s=5's (2, 4)
           1, 20];    % corner extremes
n_truths = size(truths, 1);

maxiters_values = [250000, 500000, 1000000, 2000000, 4000000];
n_iters = numel(maxiters_values);
alpha_set = 0.05;

%% Storage
results = struct();
results.cost_grid = cfg.type_space{1,1};
results.truths = truths;
results.maxiters_values = maxiters_values;
results.s = s;
results.Kappa_per_type = Kappa_per_type;

results.R1_dataonly = nan(n_truths, n_iters, s, s);
results.R2_dataonly = nan(n_truths, n_iters, s, s);
results.feas_data_maxK = false(n_truths, n_iters, s, s);
results.feas_data_perK = false(n_truths, n_iters, s, s);
results.eps_maxK_full = nan(n_truths, n_iters);
results.eps_perK = nan(n_truths, n_iters, s, NPlayers);
results.distY_all = cell(n_truths, n_iters);
results.run_log = cell(n_truths, n_iters);

%% Main loop
t_outer = tic;
for tr = 1:n_truths
    t_star_idx = truths(tr, :);
    t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                    cfg.type_space{2,1}(t_star_idx(2))];
    fprintf('\n========================================\n');
    fprintf('TRUTH %d/%d: idx (%d, %d) -> cost (%.3f, %.3f)\n', ...
            tr, n_truths, t_star_idx(1), t_star_idx(2), ...
            t_star_costs(1), t_star_costs(2));

    for n_idx = 1:n_iters
        N = maxiters_values(n_idx);
        fprintf('\n--- N = %s ---\n', commas(N));

        %% Learn
        t_learn = tic;
        distY = df.sim.learn_fixed(cfg, N, t_star_idx);
        t_learn_val = toc(t_learn);
        results.distY_all{tr, n_idx} = distY;
        fprintf('  Learn: %.1fs.  m_N support: %d/%d action profiles.\n', ...
                t_learn_val, sum(distY > 0), cfg.NActPr);

        %% Eps values
        % max-K eps (player 1 and 2 share same Kappa_max here)
        eps_maxK_full = Kappa_max * sqrt(2 * log(cfg.NAct) / N) / alpha_set;
        results.eps_maxK_full(tr, n_idx) = eps_maxK_full;
        % per-K eps: 1 per type per player
        for j = 1:NPlayers
            for t = 1:s
                results.eps_perK(tr, n_idx, t, j) = ...
                    Kappa_per_type(t, j) * sqrt(2 * log(cfg.NAct) / N) / alpha_set;
            end
        end

        %% Per-candidate regret + feasibility
        t_lp = tic;
        n_feas_max = 0;
        n_feas_per = 0;
        for i = 1:s
            for j = 1:s
                cand_costs = [cfg.type_space{1,1}(i), cfg.type_space{2,1}(j)];
                [R1, R2] = df.solvers.compute_dataonly_regret(cfg, distY, ...
                    cand_costs(1), cand_costs(2));
                results.R1_dataonly(tr, n_idx, i, j) = R1;
                results.R2_dataonly(tr, n_idx, i, j) = R2;

                feas_max = (R1 <= eps_maxK_full) && (R2 <= eps_maxK_full);
                results.feas_data_maxK(tr, n_idx, i, j) = feas_max;
                if feas_max, n_feas_max = n_feas_max + 1; end

                eps_1 = results.eps_perK(tr, n_idx, i, 1);
                eps_2 = results.eps_perK(tr, n_idx, j, 2);
                feas_per = (R1 <= eps_1) && (R2 <= eps_2);
                results.feas_data_perK(tr, n_idx, i, j) = feas_per;
                if feas_per, n_feas_per = n_feas_per + 1; end
            end
        end
        t_lp_val = toc(t_lp);
        fprintf('  Regret eval (%dx%d = %d candidates): %.1fs\n', s, s, s*s, t_lp_val);
        fprintf('    max-K feasible: %3d/%d, eps_maxK = %.4f\n', n_feas_max, s*s, eps_maxK_full);
        fprintf('    per-K feasible: %3d/%d\n', n_feas_per, s*s);

        results.run_log{tr, n_idx} = struct('t_learn', t_learn_val, 't_lp', t_lp_val, ...
                                            'n_feas_max', n_feas_max, ...
                                            'n_feas_per', n_feas_per);
    end
end
fprintf('\n=== Total wall time: %.1f min ===\n', toc(t_outer) / 60);

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'realization_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_realization_s20.mat'), '-struct', 'results');
fprintf('Saved %s\n', fullfile(out_dir, 'results_realization_s20.mat'));

%% Heatmaps per truth: max-K (top) vs per-K (bottom)
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'realization_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
cmap_panel = [0.92 0.92 0.92; 0.20 0.55 0.85];
cost_grid = cfg.type_space{1,1};

for tr = 1:n_truths
    t_star_idx = truths(tr, :);
    t_star_costs = [cost_grid(t_star_idx(1)), cost_grid(t_star_idx(2))];

    fig = figure('Color', 'w', 'Position', [60 60 1700 700]);
    for n_idx = 1:n_iters
        subplot(2, n_iters, n_idx);
        feas = squeeze(results.feas_data_maxK(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 14, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'max-$\kappa$', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        n_feas = nnz(feas);
        title(sprintf('$N = %s$ (%d/%d)', commas(maxiters_values(n_idx)), n_feas, s*s), ...
              'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 7);

        subplot(2, n_iters, n_iters + n_idx);
        feas = squeeze(results.feas_data_perK(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 14, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'per-$\kappa$', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        xlabel('$t_2$', 'Interpreter', 'latex', 'FontSize', 11);
        n_feas = nnz(feas);
        title(sprintf('(%d/%d)', n_feas, s*s), 'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 7);
    end
    sgtitle(sprintf('s=%d data-only ID set: truth $(%d,%d)$, costs $(%.2f, %.2f)$', ...
                    s, t_star_idx(1), t_star_idx(2), t_star_costs(1), t_star_costs(2)), ...
            'Interpreter', 'latex', 'FontSize', 13);
    fname = sprintf('s20_truth_%d_%d', t_star_idx(1), t_star_idx(2));
    saveas(fig, fullfile(fig_dir, [fname '.png']));
    saveas(fig, fullfile(fig_dir, [fname '.pdf']));
    fprintf('Saved %s\n', fname);
end

%% Cardinality vs N
fig = figure('Color', 'w', 'Position', [80 80 1100 600]);
hold on;
truth_colors = [0.20 0.55 0.85; 0.85 0.45 0.10; 0.45 0.20 0.65];
truth_labels = arrayfun(@(k) sprintf('truth $(%d,%d)$', truths(k,1), truths(k,2)), ...
                        1:n_truths, 'UniformOutput', false);
hh = gobjects(n_truths * 2, 1);
ll = strings(n_truths * 2, 1);
for tr = 1:n_truths
    maxK = arrayfun(@(k) results.run_log{tr, k}.n_feas_max, 1:n_iters);
    perK = arrayfun(@(k) results.run_log{tr, k}.n_feas_per, 1:n_iters);
    h1 = plot(maxiters_values, maxK, '-o', 'LineWidth', 1.8, ...
              'Color', truth_colors(tr,:), 'MarkerFaceColor', truth_colors(tr,:), ...
              'MarkerSize', 8);
    h2 = plot(maxiters_values, perK, '--s', 'LineWidth', 1.8, ...
              'Color', truth_colors(tr,:), 'MarkerFaceColor', 'w', 'MarkerSize', 8);
    hh(2*tr - 1) = h1; hh(2*tr) = h2;
    ll(2*tr - 1) = sprintf('%s, max-$\\kappa$', truth_labels{tr});
    ll(2*tr)     = sprintf('%s, per-$\\kappa$', truth_labels{tr});
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('$N$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$|\Lambda_N|$ (out of 400)', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('s=%d data-only: max-$\\kappa$ vs per-$\\kappa$ identified-set cardinality', s), ...
      'Interpreter', 'latex', 'FontSize', 12);
legend(hh, ll, 'Interpreter', 'latex', 'Location', 'eastoutside', 'FontSize', 10);
grid on; box on;
saveas(fig, fullfile(fig_dir, 's20_cardinality_vs_N.png'));
saveas(fig, fullfile(fig_dir, 's20_cardinality_vs_N.pdf'));
fprintf('Saved s20_cardinality_vs_N\n');

fprintf('\n=== Done. ===\n');

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
