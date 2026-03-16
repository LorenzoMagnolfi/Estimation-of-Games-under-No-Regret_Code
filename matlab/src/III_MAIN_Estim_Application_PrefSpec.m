%% III_MAIN_Estim_Application_PrefSpec — Stage III: Application Identification
%
%  Thin wrapper: delegates compute to df.stages.run_stage_iii,
%  reporting to df.report.classify_identified_set / plot_identified_set / write_tables.
clear all; clc; close all;

%% Setup
tic;
paths = df_repo_paths();
rng(1);

%% Compute
stage_opts = struct();
stage_opts.Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
stage_opts.Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');
stage_opts.players = 1:2;
stage_opts.NGridV = 100;
stage_opts.NGridM = 100;
stage_opts.n_types = 5;
stage_opts.epsilon_grid = [0.05];
stage_opts.n_draws = 200;
stage_opts.n_samples = 2000;
stage_opts.rng_seed = 12345;

results = df.stages.run_stage_iii(stage_opts);

%% Fixtures
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
maxvals = results.maxvals;
epsilon_grid = results.epsilon_grid;
players = results.players;
NGridV = stage_opts.NGridV;
NGridM = stage_opts.NGridM;
n_types = results.n_types;
save(fullfile(fixture_dir, 'stage_iii_solver_raw.mat'), ...
    'maxvals', 'epsilon_grid', 'players', 'NGridV', 'NGridM', 'n_types');

for idx = 1:numel(results.players)
    iii = results.players(idx);
    VV = results.player(idx).VV;
    id_set_index = results.player(idx).id_set_index;
    ddpars = results.player(idx).ddpars;
    cost_stats = results.player(idx).cost_stats;
    tot_cost_stats = results.player(idx).tot_cost_stats;
    id_set_points = results.player(idx).id_set_points;
    min_mu = results.player(idx).min_mu;
    max_mu = results.player(idx).max_mu;
    min_sigma = results.player(idx).min_sigma;
    max_sigma = results.player(idx).max_sigma;
    save(fullfile(fixture_dir, sprintf('stage_iii_player_%d.mat', iii)), ...
        'VV', 'id_set_index', 'ddpars', 'cost_stats', 'tot_cost_stats', ...
        'id_set_points', 'min_mu', 'max_mu', 'min_sigma', 'max_sigma');
end

%% Reporting: identified set plots
for idx = 1:numel(results.players)
    iii = results.players(idx);
    ddpars = results.player(idx).ddpars;
    id_set_index = results.player(idx).id_set_index;

    % Data-adaptive Halton ranges
    Xtrain = ddpars;
    halton_ranges = [...
        -abs(min(Xtrain(:,1)))*1.1, abs(min(Xtrain(:,1))-max(Xtrain(:,1)))*1.2 - abs(min(Xtrain(:,1)))*1.1, ...
        -abs(min(Xtrain(:,2)))*1.1, abs(min(Xtrain(:,2))-max(Xtrain(:,2)))*1.2 - abs(min(Xtrain(:,2)))*1.1];

    [label, PGx] = df.report.classify_identified_set(ddpars, id_set_index, halton_ranges);

    plot_opts = struct();
    plot_opts.xlabel_str = ['$\mu_' num2str(iii) '$'];
    plot_opts.ylabel_str = ['$\sigma_' num2str(iii) '$'];
    plot_opts.hover_offset = 80;
    plot_opts.scatter_colors = 'wb';
    plot_opts.legend_loc = 'northwest';
    plot_opts.font_size = 18;
    plot_opts.save_path = fullfile(paths.figures_iii, strcat('Estim_Pl_', num2str(iii)));

    df.report.plot_identified_set(PGx, label, ddpars, id_set_index, plot_opts);
end

%% Reporting: tables
data_paths = struct('data', paths.data);
df.report.write_tables(results, paths, data_paths);

toc;
