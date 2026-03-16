%% II_MAIN_simul — Stage II: Simulation Identification
%
%  Thin wrapper: delegates compute to df.stages.run_stage_ii,
%  reporting to df.report.classify_identified_set / plot_identified_set.
clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Compute
stage_opts = struct();
stage_opts.maxiters_values = [500000, 1000000, 2000000, 4000000];
stage_opts.NGridV = 100;
stage_opts.NGridM = 100;
stage_opts.alpha_set = 0.05;
stage_opts.switch_eps = 1;

results = df.stages.run_stage_ii(cfg, stage_opts);

%% Fixtures
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end

for k = 1:numel(results.maxiters_values)
    maxiters = results.maxiters_values(k);
    distY_time = results.distY_time_all{k};
    action_distribution = distY_time;
    VV = results.VV_all(k, :);
    distpars = squeeze(results.distpars_all(k, :, :));
    distribution_parameters = results.distribution_parameters{k};
    switch_eps = results.switch_eps;
    save(fullfile(fixture_dir, sprintf('stage_ii_iter_%dk.mat', maxiters/1000)), ...
        'distY_time', 'action_distribution', 'VV', 'distpars', 'maxiters', ...
        'distribution_parameters', 'switch_eps');
end
VV_all = results.VV_all;
distribution_parameters_all = results.distpars_all;
maxiters_values = results.maxiters_values;
save(fullfile(fixture_dir, 'stage_ii_solver_all.mat'), ...
    'VV_all', 'distribution_parameters_all', 'maxiters_values');

%% Reporting
num_alpha = numel(results.alpha_set);

for k = 1:numel(results.maxiters_values)
    VV = results.VV_all(k, :);
    distpars = squeeze(results.distpars_all(k, :, :));
    id_set_index = (VV <= 1e-12);
    ddpars = repmat(distpars, 1, num_alpha, 1)';

    % Halton ranges vary by iteration count
    if k == 1
        halton_ranges = [7.5, 10];
    elseif k == 2
        halton_ranges = [7, 6];
    else
        halton_ranges = [6.5, 3.5];
    end

    [label, PGx] = df.report.classify_identified_set(ddpars', id_set_index, halton_ranges);

    plot_opts = struct();
    plot_opts.true_param = [3, 1];
    plot_opts.hover_offset = (k == 1 || k == 3) * 0.9 + (k == 2 || k == 4) * 1.2;
    if k == 1
        plot_opts.yticks = [0 4 8];
    elseif k == 2
        plot_opts.yticks = [0 2 4 6];
    else
        plot_opts.yticks = [0 2 4];
    end
    if k == 1
        plot_opts.legend_ncol = 1;
    else
        plot_opts.legend_ncol = 2;
    end
    plot_opts.scatter_colors = 'wc';
    plot_opts.save_path = fullfile(paths.figures_ii, sprintf('IdSet_simul_%dk', 500*2^(k-1)));

    df.report.plot_identified_set(PGx, label, ddpars', id_set_index, plot_opts);
end

fprintf('Simulations completed for all maxiters values.\n');
