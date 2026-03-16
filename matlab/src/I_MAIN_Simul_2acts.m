%% I_MAIN_Simul_2acts — Stage I: Polytope + Learning Convergence
%
%  Thin wrapper: delegates compute to df.stages.run_stage_i,
%  plotting to existing draw* functions.
clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(1);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 20;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Compute
stage_opts = struct();
stage_opts.maxiters = 10000;
stage_opts.conf_set = [0.1, 0.025, 0.05];
stage_opts.switch_eps = 3;
stage_opts.rng_seed_learn = 11111;

results = df.stages.run_stage_i(cfg, stage_opts);

%% Fixtures
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
for jj = 1:numel(results.conf_set)
    VP = results.VP{jj};
    conf2 = results.conf_set(jj);
    switch_eps = results.switch_eps;
    maxiters = results.maxiters;
    save(fullfile(fixture_dir, sprintf('stage_i_polytope_conf%d.mat', round(conf2*1000))), ...
        'VP', 'conf2', 'switch_eps', 'maxiters');
end

%% Reporting — native MATLAB (no MPT3 dependency)
for jj = 1:numel(results.conf_set)
    conf2 = results.conf_set(jj);
    plotname = fullfile(paths.figures_i, strcat('BCCE_set_', num2str(rem(conf2,1)*1e3)));
    df.report.plot_polytope(results.VP{jj}, plotname);
    close;
end

numdists = 1000;
df.report.plot_convergence(cfg, fullfile(paths.figures_i, 'Learning2A_s20'), ...
    numdists, results.maxiters, 1, results.actions, results.VP{end});
close;

drawRegrets_per_period(cfg, fullfile(paths.figures_i, 'RegretsPerPeriodPlot'), ...
    results.regret_per_period);
close;
