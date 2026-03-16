%% IV_MAIN_Emp_Distrib_Regrets — Stage IV: Bootstrap Regrets + Identification
%
%  Thin wrapper: delegates compute to df.stages.run_stage_iv,
%  reporting to df.report.plot_regret_histogram / classify_identified_set /
%  plot_identified_set.
clear all; clc; close all;

%% Setup
paths = df_repo_paths();
rng(1);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Compute
stage_opts = struct();
stage_opts.B = 500;
stage_opts.maxiters = 100000;
stage_opts.NGridV = 100;
stage_opts.NGridM = 100;
stage_opts.alpha_set = 0.05;
stage_opts.Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
stage_opts.Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');

results = df.stages.run_stage_iv(cfg, stage_opts);

%% Save artifacts
save(fullfile(paths.artifacts_iv, 'final_regret.mat'), '-struct', 'results', 'final_regret');
save(fullfile(paths.artifacts_iv, 'ratio1_Pl1.mat'), '-struct', 'results', 'ratio1_Pl1');
save(fullfile(paths.artifacts_iv, 'ratio2_Pl1.mat'), '-struct', 'results', 'ratio2_Pl1');
save(fullfile(paths.artifacts_iv, 'ratio1_Pl2.mat'), '-struct', 'results', 'ratio1_Pl2');
save(fullfile(paths.artifacts_iv, 'ratio2_Pl2.mat'), '-struct', 'results', 'ratio2_Pl2');

%% Fixtures
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end

final_regret = results.final_regret;
Pl1_regret = results.Pl1_regret;
Pl2_regret = results.Pl2_regret;
distY_time = results.distY_time;
B = results.B;
maxiters = results.maxiters;
Nobs_Pl1 = results.Nobs_Pl1;
Nobs_Pl2 = results.Nobs_Pl2;
save(fullfile(fixture_dir, 'stage_iv_bootstrap.mat'), ...
    'final_regret', 'Pl1_regret', 'Pl2_regret', 'distY_time', ...
    'B', 'maxiters', 'Nobs_Pl1', 'Nobs_Pl2');

avg_exp_regret2 = results.avg_exp_regret2;
avg_th_regret = results.avg_th_regret;
ExpectedRegretComp = results.ExpectedRegretComp;
regret_95perc = results.regret_95perc;
ratio1_Pl1 = results.ratio1_Pl1;
ratio2_Pl1 = results.ratio2_Pl1;
ratio1_Pl2 = results.ratio1_Pl2;
ratio2_Pl2 = results.ratio2_Pl2;
save(fullfile(fixture_dir, 'stage_iv_regret_comparison.mat'), ...
    'avg_exp_regret2', 'avg_th_regret', 'ExpectedRegretComp', ...
    'regret_95perc', 'ratio1_Pl1', 'ratio2_Pl1', 'ratio1_Pl2', 'ratio2_Pl2');

VV = results.VV;
id_set_index = results.id_set_index;
ddpars = results.ddpars;
distpars = results.distpars;
ExpRegr_pass = results.ExpRegr_pass;
save(fullfile(fixture_dir, 'stage_iv_identification.mat'), ...
    'VV', 'id_set_index', 'ddpars', 'distpars', 'ExpRegr_pass');

%% Reporting: regret histogram
df.report.plot_regret_histogram(results, cfg, ...
    struct('save_path', fullfile(paths.figures_iv, 'Modified_Exp_regr_comp')));

%% Reporting: identification plot
halton_ranges = [5, 3];
filter_fn = @(PGx) (PGx(:,1) > 2) & (PGx(:,1) < 4) & (PGx(:,2) > 0.5) & (PGx(:,2) < 1.5);
classify_opts = struct('filter', filter_fn);

[label, PGx] = df.report.classify_identified_set(results.ddpars', results.id_set_index, ...
    halton_ranges, classify_opts);

plot_opts = struct();
plot_opts.true_param = [3, 1];
plot_opts.hover_offset = 0.6;
plot_opts.yticks = [0.2 0.6 1 1.4 1.8];
plot_opts.legend_loc = 'north';
plot_opts.legend_ncol = 2;
plot_opts.font_size = 30;
plot_opts.marker_size = 40;
plot_opts.save_path = fullfile(paths.figures_iv, 'IdSet_simul_exp_regr');

df.report.plot_identified_set(PGx, label, results.ddpars', results.id_set_index, plot_opts);
