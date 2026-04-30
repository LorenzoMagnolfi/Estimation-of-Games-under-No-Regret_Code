%% II_REPROCESS_realization_perK
%
%  Re-process II_RUN_realization_serious results with PER-CANDIDATE Kappa
%  (instead of max-Kappa over types).  Loads m_N and the per-candidate
%  regret values from the saved .mat; recomputes feasibility under the
%  candidate-specific eps; saves new feasibility tensor and a comparison
%  cardinality figure.

clear all; clc; close all;

%% Load
paths = df_repo_paths();
in_path = fullfile(paths.matlab_root, 'output', 'realization_pilot', ...
    'results_realization_serious.mat');
R = load(in_path);

%% Rebuild cfg (Pi tensor needed for per-type Kappa)
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = R.s;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Per-type Kappa for each player
%   K_jt = max(Pi(:, t, j)) - min(Pi(:, t, j))
Kappa_per_type = zeros(s, NPlayers);
for j = 1:NPlayers
    for t = 1:s
        Pi_jt = cfg.Pi(:, t, j);
        Kappa_per_type(t, j) = max(Pi_jt) - min(Pi_jt);
    end
end
fprintf('Per-type Kappa:\n');
fprintf('  player 1: [%s]\n', sprintf('%.4f ', Kappa_per_type(:,1)));
fprintf('  player 2: [%s]\n', sprintf('%.4f ', Kappa_per_type(:,2)));

%% Recompute feasibility with candidate-specific eps
n_truths = size(R.truths, 1);
n_iters = numel(R.maxiters_values);
alpha_set = 0.05;

R.eps_perK = nan(n_truths, n_iters, s, NPlayers);
R.feas_data_perK = false(n_truths, n_iters, s, s);

for tr = 1:n_truths
    for n_idx = 1:n_iters
        N = R.maxiters_values(n_idx);
        for j = 1:NPlayers
            for t = 1:s
                R.eps_perK(tr, n_idx, t, j) = ...
                    Kappa_per_type(t, j) * sqrt(2 * log(cfg.NAct) / N) / alpha_set;
            end
        end
        for i = 1:s
            for j = 1:s
                R1 = R.R1_dataonly(tr, n_idx, i, j);
                R2 = R.R2_dataonly(tr, n_idx, i, j);
                eps_1 = R.eps_perK(tr, n_idx, i, 1);
                eps_2 = R.eps_perK(tr, n_idx, j, 2);
                R.feas_data_perK(tr, n_idx, i, j) = (R1 <= eps_1) && (R2 <= eps_2);
            end
        end
    end
end

%% Save
save(in_path, '-struct', 'R');
fprintf('\nUpdated %s with feas_data_perK and eps_perK.\n', in_path);

%% Print summary
fprintf('\nSetCardinality (data-only, max-K vs per-K):\n');
fprintf('  truth        | N         | maxK | perK\n');
fprintf('  -------------+-----------+------+------\n');
for tr = 1:n_truths
    t1 = R.truths(tr,1); t2 = R.truths(tr,2);
    for n_idx = 1:n_iters
        N = R.maxiters_values(n_idx);
        maxK = nnz(squeeze(R.feas_data(tr, n_idx, :, :)));
        perK = nnz(squeeze(R.feas_data_perK(tr, n_idx, :, :)));
        fprintf('  (%d, %d)        | %-9s | %4d | %4d\n', t1, t2, ...
            sprintf('%.0e', N), maxK, perK);
    end
end

%% Cardinality vs N figure: max-K vs per-K side by side
fig = figure('Color', 'w', 'Position', [80 80 1300 600]);

truth_colors = [0.20 0.55 0.85;
                0.85 0.45 0.10;
                0.45 0.20 0.65];

for col = 1:2
    subplot(1, 2, col);
    hold on;
    handles_legend = gobjects(n_truths * 2, 1);
    labels_legend  = strings(n_truths * 2, 1);
    for tr = 1:n_truths
        t1 = R.truths(tr,1); t2 = R.truths(tr,2);
        bcce_card = arrayfun(@(k) nnz(squeeze(R.feas_bcce(tr, k, :, :))), 1:n_iters);
        if col == 1
            data_card = arrayfun(@(k) nnz(squeeze(R.feas_data(tr, k, :, :))), 1:n_iters);
            label_data = sprintf('truth $(%d,%d)$, data-only (max-$\\kappa$)', t1, t2);
        else
            data_card = arrayfun(@(k) nnz(squeeze(R.feas_data_perK(tr, k, :, :))), 1:n_iters);
            label_data = sprintf('truth $(%d,%d)$, data-only (per-$\\kappa$)', t1, t2);
        end
        h1 = plot(R.maxiters_values, bcce_card, '-o', 'LineWidth', 1.8, ...
                  'Color', truth_colors(tr, :), 'MarkerFaceColor', truth_colors(tr, :), ...
                  'MarkerSize', 8);
        h2 = plot(R.maxiters_values, data_card, '--s', 'LineWidth', 1.8, ...
                  'Color', truth_colors(tr, :), 'MarkerFaceColor', 'w', ...
                  'MarkerSize', 8);
        handles_legend(2*tr - 1) = h1;
        handles_legend(2*tr) = h2;
        labels_legend(2*tr - 1) = sprintf('truth $(%d,%d)$, BCCE LP', t1, t2);
        labels_legend(2*tr) = label_data;
    end
    set(gca, 'XScale', 'log', 'FontSize', 10);
    xlabel('$N$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$|\Lambda_N|$ (out of 25)', 'Interpreter', 'latex', 'FontSize', 12);
    if col == 1
        title('max-$\kappa$ data-only', 'Interpreter', 'latex', 'FontSize', 12);
    else
        title('per-candidate-$\kappa$ data-only', 'Interpreter', 'latex', 'FontSize', 12);
    end
    legend(handles_legend, labels_legend, 'Interpreter', 'latex', ...
        'Location', 'eastoutside', 'FontSize', 8);
    grid on; box on;
    ylim([0 26]);
end

sgtitle('Identified-set cardinality: max-$\kappa$ vs.\ per-candidate-$\kappa$ data-only', ...
        'Interpreter', 'latex', 'FontSize', 13);

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'realization_pilot');
saveas(fig, fullfile(fig_dir, 'serious_cardinality_perK_compare.png'));
saveas(fig, fullfile(fig_dir, 'serious_cardinality_perK_compare.pdf'));
fprintf('Saved comparison figure.\n');

%% Also: per-truth heatmap with per-K version
for tr = 1:n_truths
    t_star_idx = R.truths(tr, :);
    t_star_costs = [cfg.type_space{1,1}(t_star_idx(1)), ...
                    cfg.type_space{2,1}(t_star_idx(2))];

    fig = figure('Color', 'w', 'Position', [60 60 1600 700]);
    cmap_panel = [0.92 0.92 0.92; 0.20 0.55 0.85];
    for n_idx = 1:n_iters
        subplot(2, n_iters, n_idx);
        feas = squeeze(R.feas_data(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        set(gca, 'XTick', 1:s, 'YTick', 1:s);
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 16, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'max-$\kappa$', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        n_feas = sum(feas(:));
        title(sprintf('$N = %s$ (%d/25)', commas(R.maxiters_values(n_idx)), n_feas), ...
              'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 8);

        subplot(2, n_iters, n_iters + n_idx);
        feas = squeeze(R.feas_data_perK(tr, n_idx, :, :));
        imagesc(double(feas));
        colormap(gca, cmap_panel); caxis([0 1]);
        set(gca, 'YDir', 'normal'); axis equal tight;
        set(gca, 'XTick', 1:s, 'YTick', 1:s);
        hold on;
        plot(t_star_idx(2), t_star_idx(1), 'p', 'MarkerSize', 16, ...
             'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.95 0.85 0.20], ...
             'LineWidth', 1.2);
        hold off;
        if n_idx == 1, ylabel({'per-$\kappa$', '$t_1$'}, 'Interpreter', 'latex', 'FontSize', 11); end
        xlabel('$t_2$', 'Interpreter', 'latex', 'FontSize', 11);
        n_feas = sum(feas(:));
        title(sprintf('(%d/25)', n_feas), 'Interpreter', 'latex', 'FontSize', 10);
        set(gca, 'FontSize', 8);
    end
    sgtitle(sprintf('Data-only ID set, max-$\\kappa$ vs per-$\\kappa$: truth $(t_1^*, t_2^*) = (%d, %d)$, costs $(%.2f, %.2f)$', ...
                    t_star_idx(1), t_star_idx(2), t_star_costs(1), t_star_costs(2)), ...
            'Interpreter', 'latex', 'FontSize', 13);

    fname = sprintf('serious_truth_%d_%d_perK_compare', t_star_idx(1), t_star_idx(2));
    saveas(fig, fullfile(fig_dir, [fname '.png']));
    saveas(fig, fullfile(fig_dir, [fname '.pdf']));
    fprintf('Saved %s.\n', fname);
end

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
