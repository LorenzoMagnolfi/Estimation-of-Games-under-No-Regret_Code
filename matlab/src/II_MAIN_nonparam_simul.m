%% II_MAIN_nonparam_simul — Stage II nonparametric simulation identification
%
% Nonparametric variant of Stage II. Searches over direct probability mass
% vectors on the type support instead of (mu, sigma) parameterization.
% Refactored from Bo Feng's original version to use df.* infrastructure.
%
% See: df.stages.run_stage_ii_nonparam (orchestrator)
%      df.report.build_nonparam_grid (grid construction)

%% Setup
clear all; clc; close all;

paths = df_repo_paths();
rng(12345);

% Game parameters (same as parametric Stage II)
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Run nonparametric Stage II
stage_opts = struct();
stage_opts.maxiters_values = [500000, 1000000, 2000000, 4000000];
stage_opts.alpha_set = 0.05;
stage_opts.switch_eps = 1;

% Grid parameters (match Bo Feng's setup)
stage_opts.K_local  = 1000;
stage_opts.K_global = 9000;
stage_opts.K_spiky  = 200;
stage_opts.spike_mult = 3;
stage_opts.n_adjacent = 4;

results = df.stages.run_stage_ii_nonparam(cfg, stage_opts);

%% Plot identified sets
for maxiter_index = 1:numel(results.maxiters_values)
    VV = results.VV_all(maxiter_index, :);
    distpars = squeeze(results.distpars_all(maxiter_index, :, :));

    id_set_index = (VV <= 1e-12);
    ddpars = repmat(distpars, 1, numel(results.alpha_set), 1)';

    % Scatter plot of evaluated grid (no SVM — nonparametric grid is irregular)
    figure; hold on;
    gscatter(ddpars(1,:)', ddpars(2,:)', id_set_index, 'kc', '.', 10);
    plot(3, 1, '.', 'MarkerSize', 25, 'color', 'k');
    xlabel('$\mu$', 'Interpreter', 'latex');
    ylabel('$\sigma$', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex');
    set(gca, 'FontName', 'cmr10', 'FontSize', 17, 'Box', 'off');

    iteration_k = results.maxiters_values(maxiter_index) / 1000;
    title(sprintf('Nonparametric grid (N=%d) | s=%d, iter=%dk', ...
        size(ddpars,2), s, iteration_k), 'Interpreter', 'none');

    saveas(gcf, fullfile(paths.figures_ii, ...
        sprintf('nonparam_FullGrid_s%d_%dk.png', s, iteration_k)));
    saveas(gcf, fullfile(paths.figures_ii, ...
        sprintf('nonparam_FullGrid_s%d_%dk.eps', s, iteration_k)), 'epsc');
    close;
end

fprintf('Nonparametric Stage II complete.\n');
