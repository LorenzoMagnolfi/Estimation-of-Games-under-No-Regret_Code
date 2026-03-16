%% run_stage_ii_fast.m — Stage II (500k) with fast backend
%
% Runs the 500k identified set computation using:
%   coneprog + adaptive grid + parfor
% Then generates the identified set figure for comparison with paper.

clear all; clc; close all;

this_file = mfilename('fullpath');
test_dir  = fileparts(this_file);
matlab_root = fileparts(test_dir);
src_dir   = fullfile(matlab_root, 'src');
addpath(src_dir);

paths = df_repo_paths();

%% Setup (identical to II_MAIN_simul)
rng(12345);
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Run Stage II — fast backend, 500k only
stage_opts = struct();
stage_opts.maxiters_values = [500000];
stage_opts.NGridV = 100;
stage_opts.NGridM = 100;
stage_opts.alpha_set = 0.05;
stage_opts.switch_eps = 1;
stage_opts.backend = 'fast';
stage_opts.use_parfor = false;
stage_opts.adaptive = false;  % Full grid for production-quality results

fprintf('=== Stage II (500k) — FAST backend (full grid, CVX+SeDuMi) ===\n');
t_total = tic;
results = df.stages.run_stage_ii(cfg, stage_opts);
fprintf('=== Total time: %.1fs ===\n\n', toc(t_total));

%% Generate identified set figure
VV = results.VV_all(1, :);
distpars = squeeze(results.distpars_all(1, :, :));
id_set_index = (VV <= 1e-12);
num_alpha = 1;
ddpars = repmat(distpars, 1, num_alpha, 1)';

halton_ranges = [7.5, 10];  % k=1 (500k)
fprintf('Classifying identified set (SVM + Halton)...\n');
[label, PGx] = df.report.classify_identified_set(ddpars', id_set_index, halton_ranges);

plot_opts = struct();
plot_opts.true_param = [3, 1];
plot_opts.hover_offset = 0.9;
plot_opts.yticks = [0 4 8];
plot_opts.legend_ncol = 1;
plot_opts.scatter_colors = 'wc';

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'part_ii');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
plot_opts.save_path = fullfile(fig_dir, 'IdSet_simul_500k_fullgrid');

df.report.plot_identified_set(PGx, label, ddpars', id_set_index, plot_opts);

n_identified = sum(id_set_index);
fprintf('\nIdentified set: %d/%d points (%.1f%%)\n', ...
    n_identified, numel(id_set_index), 100*n_identified/numel(id_set_index));
fprintf('Figure saved to: %s\n', plot_opts.save_path);

%% Save results
save(fullfile(test_dir, 'stage_ii_500k_fast.mat'), 'results');
fprintf('Results saved.\n');
