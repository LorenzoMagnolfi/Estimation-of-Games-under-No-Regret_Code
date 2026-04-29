%% II_RUN_rate_landscape_pilot_v2
%
%  v2 of the pilot: re-solve at the four sample sizes both WITH and WITHOUT
%  the finite-sample consistency slack from Niccolo's note (coordinatewise
%  Hoeffding box version of his Option 2; alpha = 0.05 split 0.9 R / 0.1 C).
%
%  Reuses v1 trajectories from results_R1.mat if available (no re-learning).
%
%  Outputs a single 2-panel figure:
%    Panel A: obedience corrected (R1), exact consistency.   = v1 baseline
%    Panel B: obedience corrected (R1) + consistency slack.

clear all; clc; close all;

%% Setup (matches v1)
paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Load v1 trajectories if available
v1_path = fullfile(paths.matlab_root, 'output', 'rate_landscape_pilot', 'results_R1.mat');
if exist(v1_path, 'file')
    fprintf('Loading v1 trajectories from %s ...\n', v1_path);
    v1 = load(v1_path);
    cached_distY = v1.distY_time_all;
    have_v1 = true;
else
    fprintf('No v1 cache found; will run pass A from scratch.\n');
    cached_distY = {};
    have_v1 = false;
end

%% Pass A: switch_eps=10, exact consistency  (= v1 baseline)
if have_v1
    fprintf('\n=== Pass A: loaded from v1 cache (no recompute) ===\n');
    results_A = v1;
else
    stage_opts_A = struct();
    stage_opts_A.maxiters_values = [500000, 1000000, 2000000, 4000000];
    stage_opts_A.NGridV = 100;
    stage_opts_A.NGridM = 100;
    stage_opts_A.alpha_set = 0.05;
    stage_opts_A.switch_eps = 10;
    stage_opts_A.backend = 'fast';
    stage_opts_A.adaptive = true;
    stage_opts_A.consistency_slack_kind = 'none';

    fprintf('\n=== Pass A: switch_eps=10, exact consistency ===\n');
    results_A = df.stages.run_stage_ii(cfg, stage_opts_A);
    cached_distY = results_A.distY_time_all;
end

%% Pass B: switch_eps=10 + consistency slack (Option 2 / box, 0.9/0.1 split)
stage_opts_B = struct();
stage_opts_B.maxiters_values = [500000, 1000000, 2000000, 4000000];
stage_opts_B.NGridV = 100;
stage_opts_B.NGridM = 100;
stage_opts_B.alpha_set = 0.05;
stage_opts_B.alpha_R = 0.9 * 0.05;
stage_opts_B.alpha_C = 0.1 * 0.05;
stage_opts_B.switch_eps = 10;
stage_opts_B.backend = 'fast';
stage_opts_B.adaptive = true;
stage_opts_B.consistency_slack_kind = 'box';
stage_opts_B.precomputed_distY = cached_distY;

fprintf('\n=== Pass B: switch_eps=10 + consistency slack (box, alpha_R=0.045, alpha_C=0.005) ===\n');
t_total = tic;
results_B = df.stages.run_stage_ii(cfg, stage_opts_B);
fprintf('Pass B wall time: %.1f s\n', toc(t_total));

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'rate_landscape_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_R1_no_slack.mat'),  '-struct', 'results_A');
save(fullfile(out_dir, 'results_R1_box_slack.mat'), '-struct', 'results_B');
fprintf('Saved both passes to %s\n', out_dir);

%% 2-panel overlay figure
fig = figure('Color', 'w', 'Position', [100 100 1300 560]);

cmap = [0.70 0.78 0.92;  0.40 0.55 0.85;  0.18 0.32 0.65;  0.04 0.10 0.38];
n_iters = numel(results_A.maxiters_values);

panel_results = {results_A, results_B};
panel_titles  = {
    'A.\ Obedience corrected (R1), exact consistency';
    'B.\ Obedience corrected (R1) + consistency slack ($\alpha_R{=}0.045$, $\alpha_C{=}0.005$)'
};

% Hardcoded, common visualization window — keeps both panels comparable and
% avoids contour() clipping/edge artifacts.
xlim_use = [2.0, 4.0];
ylim_use = [0.0, 4.5];

for p = 1:2
    subplot(1, 2, p);
    hold on;
    R = panel_results{p};
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
        % contour() (non-monotonic at row/col 1).  Drop (1,:) and (:,1).
        mu_grid = mu_grid(2:end, 2:end);
        sg_grid = sg_grid(2:end, 2:end);
        VV_grid = VV_grid(2:end, 2:end);

        feasible = VV_grid <= 1e-6;
        [~, h] = contour(mu_grid, sg_grid, double(feasible), [0.5 0.5], ...
            'LineColor', cmap(k, :), 'LineWidth', 2.2);
        contour_handles(k) = h;
        N_labels(k) = sprintf('N = %s', commas(R.maxiters_values(k)));
    end
    plot(mu(1), sigma2(1, 1), 'p', 'MarkerSize', 16, ...
        'MarkerEdgeColor', [0.6 0 0], 'MarkerFaceColor', [0.85 0.15 0.15], ...
        'LineWidth', 1.2);
    text(mu(1) + 0.04, sigma2(1, 1) + 0.18, '$(\mu^*,\sigma^{2*})$', ...
        'Interpreter', 'latex', 'FontSize', 12);
    xlim(xlim_use); ylim(ylim_use);
    xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 13);
    title(panel_titles{p}, 'Interpreter', 'latex', 'FontSize', 11);
    legend(contour_handles, N_labels, 'Location', 'northeast', ...
        'Interpreter', 'latex', 'FontSize', 10);
    grid on; box on;
    set(gca, 'FontSize', 10);
    hold off;
end

sgtitle('95\% confidence regions: rate correction vs.\ rate + consistency slack', ...
    'Interpreter', 'latex', 'FontSize', 13);

% Save
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'rate_landscape_pilot');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'overlay_2panel.png'));
saveas(fig, fullfile(fig_dir, 'overlay_2panel.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'overlay_2panel.png'));

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
