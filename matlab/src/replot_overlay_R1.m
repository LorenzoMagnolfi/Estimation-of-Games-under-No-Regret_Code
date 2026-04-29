%% replot_overlay_R1 — re-render v1 overlay with hardcoded axis crop.
clear all; close all;
paths = df_repo_paths();
in_path = fullfile(paths.matlab_root, 'output', 'rate_landscape_pilot', 'results_R1.mat');
R = load(in_path);

mu_true = 3; sg_true = 1;
xlim_use = [2.0, 4.0];
ylim_use = [0.0, 4.5];

cmap = [0.70 0.78 0.92;  0.40 0.55 0.85;  0.18 0.32 0.65;  0.04 0.10 0.38];
n_iters = numel(R.maxiters_values);

fig = figure('Color', 'w', 'Position', [100 100 760 580]);
hold on;
contour_handles = gobjects(n_iters, 1);
N_labels = strings(n_iters, 1);

for k = 1:n_iters
    distpars = squeeze(R.distpars_all(k, :, :));
    VV = R.VV_all(k, :);
    nV = R.NGridV + 1;
    nM = R.NGridM + 1;
    mu_grid = reshape(distpars(:, 1), nV, nM);
    sg_grid = reshape(distpars(:, 2), nV, nM);
    VV_grid = reshape(VV, nV, nM);

    % gridparamV/M each prepend an out-of-order "1" entry that breaks
    % contour() because the grid is non-monotonic at row/col 1.
    % Drop (1,:) and (:,1) to recover a clean 100x100 monotonic grid.
    mu_grid = mu_grid(2:end, 2:end);
    sg_grid = sg_grid(2:end, 2:end);
    VV_grid = VV_grid(2:end, 2:end);

    feasible = VV_grid <= 1e-6;

    [~, h] = contour(mu_grid, sg_grid, double(feasible), [0.5 0.5], ...
        'LineColor', cmap(k, :), 'LineWidth', 2.4);
    contour_handles(k) = h;
    N_labels(k) = sprintf('N = %s', commas(R.maxiters_values(k)));
end

plot(mu_true, sg_true, 'p', 'MarkerSize', 16, ...
    'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.85 0.15 0.15], ...
    'LineWidth', 1.2);
text(mu_true + 0.04, sg_true + 0.18, '$(\mu^*,\sigma^{2*})$', ...
    'Interpreter', 'latex', 'FontSize', 13);

xlim(xlim_use); ylim(ylim_use);
xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 14);
title({'95\% confidence regions, corrected adversarial + full-feedback radius (R1)', ...
       'Two-seller pricing game; $|A|=5$, $s=5$ types/seller'}, ...
    'Interpreter', 'latex', 'FontSize', 12);
legend(contour_handles, N_labels, 'Location', 'northeast', ...
    'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;
set(gca, 'FontSize', 11);
hold off;

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'rate_landscape_pilot');
saveas(fig, fullfile(fig_dir, 'overlay_R1_cropped.png'));
saveas(fig, fullfile(fig_dir, 'overlay_R1_cropped.pdf'));
fprintf('Saved cropped figure to %s\n', fullfile(fig_dir, 'overlay_R1_cropped.png'));

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
