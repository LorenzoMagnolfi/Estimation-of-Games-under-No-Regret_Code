%% II_RUN_realization_serious
%
%  Serious characterization of the realization-conditional identified set.
%
%  Three axes:
%    (1) Truth (t1*, t2*): {(3,3) symmetric, (2,4) asymmetric, (1,5) corner}
%    (2) Sample size N: {250k, 500k, 1M, 2M, 4M}
%    (3) LP form:
%          (a) BCCE LP  -- existing slice-equality + obedience-everywhere LP,
%                          per-cell radius eps_R1 / sqrt(s).
%          (b) Data-only -- max-regret check per player at candidate cost,
%                           full Hedge bound eps_R1 (NST-style; see
%                           litreview/NST_2015.md in paper repo).
%
%  One trajectory per (truth, N).  Variance bands deferred to Tier 2.

clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(20260429 + 1);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

fprintf('\n=== Realization-conditional serious run ===\n');
fprintf('Cost grid: [%s]\n', sprintf('%.3f ', cfg.type_space{1,1}));

%% Configuration
truths = [3, 3;     % symmetric center
          2, 4;     % asymmetric off-diagonal
          1, 5];    % corner (cost extremes)
n_truths = size(truths, 1);

maxiters_values = [250000, 500000, 1000000, 2000000, 4000000];
n_iters = numel(maxiters_values);

alpha_set = 0.05;
switch_eps = 10;  % Niccolo R1, full feedback

%% Constants for LP construction
s2 = s^2;
NA = cfg.NActPr;
a_dim = cfg.NAct;
dim_u = NA - 1;
dineq_obed = NPlayers * s * a_dim;

bmarg_uniform = ones(s2, 1) / s2;
marg_per_player = ones(s, 1) / s;
pi_realized = 1 / s2;

% Kronecker index (i, j) -> realized_idx
kron_idx = @(i, j) (j - 1) * s + i;

%% Storage
results = struct();
results.cost_grid = cfg.type_space{1,1};
results.truths = truths;
results.maxiters_values = maxiters_values;
results.s = s;

% (truth_idx, N_idx, t1_cand, t2_cand)
results.VV_bcce     = nan(n_truths, n_iters, s, s);
results.feas_bcce   = false(n_truths, n_iters, s, s);
results.R1_dataonly = nan(n_truths, n_iters, s, s);
results.R2_dataonly = nan(n_truths, n_iters, s, s);
results.feas_data   = false(n_truths, n_iters, s, s);

results.eps_R1_per_cell = nan(n_truths, n_iters);   % BCCE LP eps (per-cell, scalar summary)
results.eps_R1_full     = nan(n_truths, n_iters);   % data-only eps (full Hedge bound)

results.distY_all  = cell(n_truths, n_iters);
results.run_log    = cell(n_truths, n_iters);

%% Main loop
t_outer = tic;
for tr = 1:n_truths
    t_star_idx = truths(tr, :);
    t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                    cfg.type_space{2,1}(t_star_idx(2))];
    fprintf('\n========================================\n');
    fprintf('TRUTH %d/%d: (t1*, t2*) = (%d, %d), costs (%.3f, %.3f)\n', ...
            tr, n_truths, t_star_idx(1), t_star_idx(2), ...
            t_star_costs(1), t_star_costs(2));

    for n_idx = 1:n_iters
        M = maxiters_values(n_idx);
        fprintf('\n--- N = %s ---\n', commas(M));

        %% (i) DGP
        t_learn = tic;
        distY = df.sim.learn_fixed(cfg, M, t_star_idx);
        t_learn_val = toc(t_learn);
        results.distY_all{tr, n_idx} = distY;
        fprintf('  Learn: %.1fs.  m_N support: %d/%d action profiles.\n', ...
                t_learn_val, sum(distY > 0), NA);

        %% (ii) Eps values
        eps_vec = df.solvers.compute_epsilon(cfg, M, alpha_set, switch_eps);  % 1 x s
        % Data-only: full Hedge bound (no sqrt(phi) factor)
        % eps_R1_full per player = max_t Kappa(t) * sqrt(2 log|A| / N) / alpha
        eps_R1_full = max(eps_vec);
        % BCCE LP: eps_R1 / sqrt(s) per cell (uniform marg)
        eps_R1_per_cell = eps_R1_full / sqrt(s);
        results.eps_R1_full(tr, n_idx) = eps_R1_full;
        results.eps_R1_per_cell(tr, n_idx) = eps_R1_per_cell;

        eps_fin_bcce = repmat(sqrt(marg_per_player)', 1, NPlayers * a_dim) .* ...
                       repmat(eps_vec, 1, NPlayers * a_dim);

        %% (iii) Loop over candidates
        t_lp = tic;
        n_feas_bcce = 0;
        n_feas_data = 0;
        for i = 1:s
            for j = 1:s
                cand_costs = [cfg.type_space{1,1}(i), cfg.type_space{2,1}(j)];

                %% (a) BCCE LP
                realized_idx = kron_idx(i, j);
                cstr = df.solvers.build_constraints_realization( ...
                    cfg.type_space, cfg.action_space, cfg.Pi, realized_idx);
                c = [zeros(1, dim_u), ...
                     pi_realized * distY', ...
                     bmarg_uniform', ...
                     1, ...
                     eps_fin_bcce]';
                [VV, ~] = df.solvers.solve_socp_cvx(cstr, c, 'sedumi', 'default');
                results.VV_bcce(tr, n_idx, i, j) = VV;
                bcce_feas = (VV <= 1e-6);
                results.feas_bcce(tr, n_idx, i, j) = bcce_feas;
                if bcce_feas, n_feas_bcce = n_feas_bcce + 1; end

                %% (b) Data-only
                [R1, R2] = df.solvers.compute_dataonly_regret(cfg, distY, ...
                    cand_costs(1), cand_costs(2));
                results.R1_dataonly(tr, n_idx, i, j) = R1;
                results.R2_dataonly(tr, n_idx, i, j) = R2;
                data_feas = (R1 <= eps_R1_full) && (R2 <= eps_R1_full);
                results.feas_data(tr, n_idx, i, j) = data_feas;
                if data_feas, n_feas_data = n_feas_data + 1; end
            end
        end
        t_lp_val = toc(t_lp);
        fprintf('  LP grid (5x5 = 25 candidates): %.1fs\n', t_lp_val);
        fprintf('    BCCE feasible:      %2d/25,   eps_R1/sqrt(s) = %.4f\n', ...
                n_feas_bcce, eps_R1_per_cell);
        fprintf('    Data-only feasible: %2d/25,   eps_R1         = %.4f\n', ...
                n_feas_data, eps_R1_full);

        results.run_log{tr, n_idx} = struct('t_learn', t_learn_val, 't_lp', t_lp_val, ...
                                            'n_feas_bcce', n_feas_bcce, ...
                                            'n_feas_data', n_feas_data);
    end
end
fprintf('\n=== Total wall time: %.1f min ===\n', toc(t_outer) / 60);

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'realization_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_realization_serious.mat'), '-struct', 'results');
fprintf('Saved results to %s\n', fullfile(out_dir, 'results_realization_serious.mat'));

%% Figures: per-truth heatmap, 2 rows (BCCE / data-only) x 5 cols (N)
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'realization_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
cmap_panel = [0.92 0.92 0.92; 0.20 0.55 0.85];

for tr = 1:n_truths
    t_star_idx = truths(tr, :);
    t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                    cfg.type_space{2,1}(t_star_idx(2))];

    fig = figure('Color', 'w', 'Position', [60 60 1600 700]);
    for n_idx = 1:n_iters
        % Top row: BCCE LP
        subplot(2, n_iters, n_idx);
        feas = squeeze(results.feas_bcce(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        set(gca, 'XTick', 1:s, 'YTick', 1:s);
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 16, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'BCCE LP', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        title(sprintf('$N = %s$ \\quad (%d/25)', ...
              commas(maxiters_values(n_idx)), ...
              results.run_log{tr, n_idx}.n_feas_bcce), ...
              'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 8);

        % Bottom row: data-only
        subplot(2, n_iters, n_iters + n_idx);
        feas = squeeze(results.feas_data(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        set(gca, 'XTick', 1:s, 'YTick', 1:s);
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 16, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'Data-only', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        xlabel('$t_2$', 'Interpreter', 'latex', 'FontSize', 11);
        title(sprintf('(%d/25)', results.run_log{tr, n_idx}.n_feas_data), ...
              'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 8);
    end

    sgtitle(sprintf('Realization-conditional identified set: truth $(t_1^*, t_2^*) = (%d, %d)$, costs $(%.2f, %.2f)$', ...
                    t_star_idx(1), t_star_idx(2), ...
                    t_star_costs(1), t_star_costs(2)), ...
            'Interpreter', 'latex', 'FontSize', 13);

    fname = sprintf('serious_truth_%d_%d', t_star_idx(1), t_star_idx(2));
    saveas(fig, fullfile(fig_dir, [fname '.png']));
    saveas(fig, fullfile(fig_dir, [fname '.pdf']));
    fprintf('Saved %s.{png,pdf}\n', fname);
end

%% Summary: cardinality of Lambda_N vs N (log-log), all truths, both LP variants
fig_summary = figure('Color', 'w', 'Position', [80 80 900 600]);
hold on;
truth_colors = [0.20 0.55 0.85;   % center: blue
                0.85 0.45 0.10;   % asym: orange
                0.45 0.20 0.65];  % corner: purple
truth_labels = arrayfun(@(k) sprintf('truth $(%d,%d)$', truths(k,1), truths(k,2)), ...
                        1:n_truths, 'UniformOutput', false);

handles_legend = gobjects(n_truths * 2, 1);
labels_legend  = strings(n_truths * 2, 1);

for tr = 1:n_truths
    bcce_card = arrayfun(@(k) results.run_log{tr, k}.n_feas_bcce, 1:n_iters);
    data_card = arrayfun(@(k) results.run_log{tr, k}.n_feas_data, 1:n_iters);

    h1 = plot(maxiters_values, bcce_card, '-o', 'LineWidth', 1.8, ...
              'Color', truth_colors(tr, :), 'MarkerFaceColor', truth_colors(tr, :), ...
              'MarkerSize', 8);
    h2 = plot(maxiters_values, data_card, '--s', 'LineWidth', 1.8, ...
              'Color', truth_colors(tr, :), 'MarkerFaceColor', 'w', ...
              'MarkerSize', 8);

    handles_legend(2*tr - 1) = h1;
    handles_legend(2*tr)     = h2;
    labels_legend(2*tr - 1)  = sprintf('%s, BCCE LP', truth_labels{tr});
    labels_legend(2*tr)      = sprintf('%s, data-only', truth_labels{tr});
end

set(gca, 'XScale', 'log', 'YScale', 'linear', 'FontSize', 11);
xlabel('$N$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$|\Lambda_N|$ \quad (out of 25)', 'Interpreter', 'latex', 'FontSize', 14);
title('Identified-set cardinality vs.\ $N$', 'Interpreter', 'latex', 'FontSize', 13);
legend(handles_legend, labels_legend, 'Interpreter', 'latex', 'Location', 'eastoutside', 'FontSize', 10);
grid on; box on;
ylim([0 26]);

saveas(fig_summary, fullfile(fig_dir, 'serious_cardinality_vs_N.png'));
saveas(fig_summary, fullfile(fig_dir, 'serious_cardinality_vs_N.pdf'));
fprintf('Saved serious_cardinality_vs_N.{png,pdf}\n');

fprintf('\n=== Done. ===\n');

%% Helper
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
