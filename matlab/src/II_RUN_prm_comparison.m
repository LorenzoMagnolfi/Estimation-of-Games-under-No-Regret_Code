%% II_RUN_prm_comparison — R1.1.c: Confidence sets under proxy-regret matching
%
% Runs both regret matching (full feedback) and proxy-regret matching
% (bandit feedback) for the baseline s=5 case, generating side-by-side
% identified set comparisons. Responds to R1.1.c: "I would also be
% interested in seeing confidence sets under proxy-regret matching."
%
% Uses parametric (mu, sigma) grid for direct comparison with existing
% figures. Nonparametric variant can be added after validation.

%% Setup
clear all; clc; close all;

paths = df_repo_paths();
rng(12345);

% Game parameters
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Shared options
maxiters_values = [500000, 1000000, 2000000, 4000000];
alpha_set = 0.05;
switch_eps = 1;

% Grid: use the nonparametric grid for apples-to-apples
grid_opts = struct('K_local', 100, 'K_global', 800, 'K_spiky', 100);

stage_opts_base = struct();
stage_opts_base.maxiters_values = maxiters_values;
stage_opts_base.alpha_set = alpha_set;
stage_opts_base.switch_eps = switch_eps;
stage_opts_base.backend = 'fast';
stage_opts_base.K_local  = grid_opts.K_local;
stage_opts_base.K_global = grid_opts.K_global;
stage_opts_base.K_spiky  = grid_opts.K_spiky;
stage_opts_base.spike_mult = 3;
stage_opts_base.n_adjacent = 4;

%% Run 1: Standard regret matching (full feedback)
fprintf('\n========== Regret Matching (full feedback) ==========\n');
rng(12345);  % same seed for comparability
stage_opts_rm = stage_opts_base;
stage_opts_rm.learning_style = 'rm';
results_rm = df.stages.run_stage_ii_nonparam(cfg, stage_opts_rm);

%% Run 2: Proxy-regret matching (bandit feedback)
fprintf('\n========== Proxy-Regret Matching (bandit feedback) ==========\n');
rng(12345);  % same seed for comparability
stage_opts_prm = stage_opts_base;
stage_opts_prm.learning_style = 'prm';
results_prm = df.stages.run_stage_ii_nonparam(cfg, stage_opts_prm);

%% Compare and plot side-by-side
for mi = 1:numel(maxiters_values)
    iteration_k = maxiters_values(mi) / 1000;

    VV_rm = results_rm.VV_all(mi, :);
    VV_prm = results_prm.VV_all(mi, :);
    distpars = squeeze(results_rm.distpars_all(mi, :, :));

    id_rm  = (VV_rm  <= 1e-12);
    id_prm = (VV_prm <= 1e-12);

    fprintf('iter=%dk: RM identified %d/%d (%.1f%%), PRM identified %d/%d (%.1f%%)\n', ...
        iteration_k, sum(id_rm), numel(id_rm), 100*mean(id_rm), ...
        sum(id_prm), numel(id_prm), 100*mean(id_prm));

    % SVM for both
    mu_range = [min(distpars(:,1)), max(distpars(:,1))];
    sig_range = [min(distpars(:,2)), max(distpars(:,2))];
    halton_ranges = [mu_range(1), mu_range(2), sig_range(1), sig_range(2)];
    svm_opts = struct('quality', 'draft');

    if sum(id_rm) >= 3
        [label_rm, PGx_rm, ~] = df.report.classify_identified_set(...
            distpars, id_rm(:), halton_ranges, svm_opts);
    end
    if sum(id_prm) >= 3
        [label_prm, PGx_prm, ~] = df.report.classify_identified_set(...
            distpars, id_prm(:), halton_ranges, svm_opts);
    end

    % Side-by-side figure
    figure('Position', [100, 100, 1400, 550]);

    % Left panel: RM
    subplot(1, 2, 1);
    if sum(id_rm) >= 3
        hold on;
        gscatter(PGx_rm(:,1), PGx_rm(:,2), label_rm(:), 'wc', '.', 15);
        plot(3, 1, '.', 'MarkerSize', 25, 'Color', 'k');
        xlabel('$\mu$', 'Interpreter', 'latex');
        ylabel('$\sigma$', 'Interpreter', 'latex');
        title(sprintf('Regret Matching | %dk iters', iteration_k), 'Interpreter', 'none');
        set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 14, 'Box', 'off');
    else
        text(0.5, 0.5, 'Too few identified', 'HorizontalAlignment', 'center');
        title(sprintf('Regret Matching | %dk iters', iteration_k), 'Interpreter', 'none');
    end

    % Right panel: PRM
    subplot(1, 2, 2);
    if sum(id_prm) >= 3
        hold on;
        gscatter(PGx_prm(:,1), PGx_prm(:,2), label_prm(:), 'wc', '.', 15);
        plot(3, 1, '.', 'MarkerSize', 25, 'Color', 'k');
        xlabel('$\mu$', 'Interpreter', 'latex');
        ylabel('$\sigma$', 'Interpreter', 'latex');
        title(sprintf('Proxy-Regret Matching | %dk iters', iteration_k), 'Interpreter', 'none');
        set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 14, 'Box', 'off');
    else
        text(0.5, 0.5, 'Too few identified', 'HorizontalAlignment', 'center');
        title(sprintf('Proxy-Regret Matching | %dk iters', iteration_k), 'Interpreter', 'none');
    end

    saveas(gcf, fullfile(paths.figures_ii, ...
        sprintf('prm_comparison_s%d_%dk.png', s, iteration_k)));
    saveas(gcf, fullfile(paths.figures_ii, ...
        sprintf('prm_comparison_s%d_%dk.eps', s, iteration_k)), 'epsc');
    close;
end

%% Save
save(fullfile(paths.artifacts, 'prm_comparison_s5.mat'), ...
    'results_rm', 'results_prm', '-v7.3');

fprintf('\n========== PRM comparison complete ==========\n');
