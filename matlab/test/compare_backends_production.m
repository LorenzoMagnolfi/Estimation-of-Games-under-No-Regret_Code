%% compare_backends_production.m — Run Stage II (500k) with coneprog, compare to CVX
%
% Runs the 500k-iteration case at FULL production grid (101x101 = 10201)
% using coneprog backend. Then runs CVX+SeDuMi on a subsample to validate.
% Finally generates the identified set figure for visual comparison with paper.

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
cfg.learning_style = 'rm';

type_space   = cfg.type_space;
action_space = cfg.action_space;
Pi           = cfg.Pi;

NGridV = 100; NGridM = 100;
switch_eps = 1;
confid = 0.05;

%% Learning (500k)
maxiters = 500000;
fprintf('=== Learning (maxiters=%dk) ===\n', maxiters/1000);
t_learn = tic;
[distY_time, ~] = learn_mod(cfg, 1, maxiters, maxiters, 1, 1, 1, 1);
fprintf('  Done: %.1fs\n\n', toc(t_learn));
action_distribution = distY_time;

%% Grid (101x101, k=1 variance range)
gridparamV = [1; linspace(0.15, sigma2(1,1)*10, NGridV)'];
gridparamM = [1; linspace(0.55, mu(1,1)*0.5, NGridM)'];
nV = numel(gridparamV);
nM = numel(gridparamM);
NGrid = nV * nM;

[distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV);

fprintf('Grid: %dx%d = %d points\n\n', nV, nM, NGrid);

%% Precompute constraint data and objective vectors
cstr = df.solvers.build_constraints(type_space, action_space, Pi);
eps_vec = df.solvers.compute_epsilon(cfg, maxiters, confid, switch_eps);

s2 = cstr.s2; NAg = cstr.NAg; a_dim = cstr.a; dim_u = cstr.NA - 1;
T_sorted = cstr.T_sorted;

Psi = zeros(s2, NGrid);
marg_distrib_grid = zeros(s, NGrid);
for nd = 1:NGrid
    Psi(:,nd) = mvnpdf(T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
    mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
    sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
    md = normpdf(type_space{1,1}, mu_val, sqrt(sg_val));
    marg_distrib_grid(:,nd) = md / sum(md);
end
Psi = Psi ./ sum(Psi, 1);

c_all = zeros(size(cstr.B_EQ, 2), NGrid);
for nd = 1:NGrid
    bmarg = Psi(:,nd);
    eps_fin = repmat(sqrt(marg_distrib_grid(:,nd))', 1, NAg*a_dim) .* repmat(eps_vec, 1, NAg*a_dim);
    c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', bmarg', 1, eps_fin]';
end

%% =========================================================================
%  RUN 1: coneprog (full grid, serial)
%  =========================================================================
fprintf('=== coneprog: full grid (%d points) ===\n', NGrid);
g_coneprog = zeros(NGrid, 1);
t_cp = tic;
for nd = 1:NGrid
    [g_coneprog(nd), ~] = df.solvers.solve_socp_coneprog(cstr, c_all(:,nd));
    if mod(nd, 1000) == 0
        elapsed = toc(t_cp);
        fprintf('  %d/%d (%.1fs, ETA %.0fs)\n', nd, NGrid, elapsed, elapsed/nd*(NGrid-nd));
    end
end
time_cp = toc(t_cp);
fprintf('  Done: %d points in %.1fs (%.3fs/solve)\n\n', NGrid, time_cp, time_cp/NGrid);

%% =========================================================================
%  RUN 2: CVX+SeDuMi subsample validation
%  =========================================================================
% Randomly sample 200 points (mix of feasible and infeasible)
rng(42);
sample_idx = sort(randperm(NGrid, min(200, NGrid)));

fprintf('=== CVX+SeDuMi: %d-point subsample validation ===\n', numel(sample_idx));
g_cvx_sub = zeros(numel(sample_idx), 1);
t_cvx = tic;
for ii = 1:numel(sample_idx)
    nd = sample_idx(ii);
    [g_cvx_sub(ii), ~] = df.solvers.solve_socp_cvx(cstr, c_all(:,nd), 'sedumi', 'default');
    if mod(ii, 50) == 0
        fprintf('  %d/%d\n', ii, numel(sample_idx));
    end
end
time_cvx = toc(t_cvx);
fprintf('  Done: %.1fs\n\n', time_cvx);

%% =========================================================================
%  COMPARISON
%  =========================================================================
g_cp_sub = g_coneprog(sample_idx);
feas_cp  = g_cp_sub <= 1e-6;
feas_cvx = g_cvx_sub <= 1e-6;
agree = sum(feas_cp == feas_cvx);
disagree_idx = find(feas_cp ~= feas_cvx);

fprintf('=== FEASIBILITY COMPARISON (%d points) ===\n', numel(sample_idx));
fprintf('  Agree:    %d/%d (%.1f%%)\n', agree, numel(sample_idx), 100*agree/numel(sample_idx));
fprintf('  Disagree: %d\n', numel(disagree_idx));
if ~isempty(disagree_idx)
    fprintf('  Disagreement details:\n');
    for ii = 1:min(10, numel(disagree_idx))
        di = disagree_idx(ii);
        fprintf('    Grid %d: coneprog=%.4e, CVX=%.4e\n', ...
            sample_idx(di), g_cp_sub(di), g_cvx_sub(di));
    end
end

% Identified set sizes
n_id_cp  = sum(g_coneprog <= 1e-6);
n_id_cvx = sum(feas_cvx);
fprintf('\n  coneprog: %d/%d identified (%.1f%%)\n', n_id_cp, NGrid, 100*n_id_cp/NGrid);
fprintf('  CVX sub:  %d/%d identified (%.1f%%)\n', n_id_cvx, numel(sample_idx), 100*n_id_cvx/numel(sample_idx));

%% =========================================================================
%  FIGURE: Identified set (coneprog) — compare with paper Figure X
%  =========================================================================
fprintf('\n=== Generating identified set figure ===\n');

VV = g_coneprog';
id_set_index = (VV <= 1e-12);
num_alpha = 1;
ddpars = repmat(distpars, 1, num_alpha, 1)';

halton_ranges = [7.5, 10];  % k=1 (500k)
[label, PGx] = df.report.classify_identified_set(ddpars', id_set_index, halton_ranges);

plot_opts = struct();
plot_opts.true_param = [3, 1];
plot_opts.hover_offset = 0.9;
plot_opts.yticks = [0 4 8];
plot_opts.legend_ncol = 1;
plot_opts.scatter_colors = 'wc';

% Save to output directory
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'part_ii');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
plot_opts.save_path = fullfile(fig_dir, 'IdSet_simul_500k_coneprog');

df.report.plot_identified_set(PGx, label, ddpars', id_set_index, plot_opts);
fprintf('  Figure saved to: %s\n', plot_opts.save_path);

%% Save raw results for later comparison
save(fullfile(test_dir, 'stage_ii_500k_coneprog.mat'), ...
    'g_coneprog', 'g_cvx_sub', 'sample_idx', 'distpars', ...
    'distribution_parameters', 'time_cp', 'time_cvx');
fprintf('\n=== DONE ===\n');
fprintf('Total time: %.1fs (learning) + %.1fs (coneprog) + %.1fs (CVX sub) = %.1fs\n', ...
    toc(t_learn), time_cp, time_cvx, toc(t_learn) + time_cp + time_cvx);
