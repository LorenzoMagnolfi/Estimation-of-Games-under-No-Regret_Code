%% benchmark_solvers.m — Compare SOCP solver backends and grid strategies
%
% Benchmarks at fixture scale (20x20 grid, 5 actions/player, s=5 types):
%   1. CVX + SeDuMi   (current baseline)
%   2. coneprog        (MATLAB native, no CVX overhead)
%   3. Direct SeDuMi   (sedumi() call, no CVX overhead)
%   4. Adaptive grid    (solve only near boundary)
%   5. parfor scaling   (parallel workers)

clear all; clc; close all;

this_file = mfilename('fullpath');
test_dir  = fileparts(this_file);
matlab_root = fileparts(test_dir);
src_dir   = fullfile(matlab_root, 'src');
addpath(src_dir);

%% Setup: replicate fixture Stage II exactly
rng(12345);
NPlayers = 2;
alpha    = -(1/3);
actions_vec = [4;5;6;7;8];
mu    = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s     = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
cfg.learning_style = 'rm';

type_space   = cfg.type_space;
action_space = cfg.action_space;
Pi           = cfg.Pi;

% Learning
maxiters = 5000;
numdst_t = 1;
fprintf('Learning (maxiters=%d)...', maxiters);
t_learn = tic;
[distY_time, ~] = learn_mod(cfg, 1, maxiters, maxiters, numdst_t, numdst_t, 1, 1);
fprintf(' %.1fs\n', toc(t_learn));
action_distribution = distY_time;

% Grid (20x20)
NGridV = 20; NGridM = 20;
gridparamV = [1; linspace(0.15, sigma2(1,1)*10, NGridV)'];
gridparamM = [1; linspace(0.55, mu(1,1)*0.5, NGridM)'];
NGrid = numel(gridparamV) * numel(gridparamM);
nV = numel(gridparamV);
nM = numel(gridparamM);

% Build distribution_parameters (same indexing as run_fixtures)
distribution_parameters = cell(3, NGrid);
for ind1 = 1:nM
    for ind2 = 1:nV
        nd = (ind1-1)*nM + ind2;
        distribution_parameters{1,nd} = 'Normal';
        distribution_parameters{2,nd} = gridparamM(ind1)*mu';
        distribution_parameters{3,nd} = gridparamV(ind2)*sigma2';
    end
end

% Epsilon
switch_eps = 1;
confid = 0.05;
eps_vec = df.solvers.compute_epsilon(cfg, maxiters, confid, switch_eps);
eps_info = struct('mode', 'switch', 'eps', eps_vec);

% Build constraints ONCE (shared across all backends)
cstr = df.solvers.build_constraints(type_space, action_space, Pi);

% Precompute Psi, marg_distrib, and all objective vectors
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

% Precompute all objective vectors
c_all = zeros(size(cstr.B_EQ, 2), NGrid);
for nd = 1:NGrid
    bmarg = Psi(:,nd);
    eps_fin = repmat(sqrt(marg_distrib_grid(:,nd))', 1, NAg*a_dim) .* repmat(eps_vec, 1, NAg*a_dim);
    c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', bmarg', 1, eps_fin]';
end

fprintf('\nProblem dimensions: n=%d, eq=%d, ineq=%d, soc=%d\n', ...
    size(c_all,1), size(cstr.B_EQ,1), size(cstr.B_INEQ,1), size(cstr.Mat_NLC,1));
fprintf('Grid points: %d (%dx%d)\n\n', NGrid, nV, nM);

%% =========================================================================
%  BENCHMARK 1: CVX + SeDuMi (current baseline)
%  =========================================================================
fprintf('=== Benchmark 1: CVX + SeDuMi (baseline) ===\n');
N_bench = min(50, NGrid);  % first 50 points
g_cvx = zeros(N_bench, 1);
t1 = tic;
for nd = 1:N_bench
    [g_cvx(nd), ~] = df.solvers.solve_socp_cvx(cstr, c_all(:,nd), 'sedumi', 'default');
end
time_cvx = toc(t1);
fprintf('  %d solves in %.2fs (%.4fs/solve)\n\n', N_bench, time_cvx, time_cvx/N_bench);

%% =========================================================================
%  BENCHMARK 2: coneprog (MATLAB native)
%  =========================================================================
fprintf('=== Benchmark 2: coneprog (MATLAB native) ===\n');
g_coneprog = zeros(N_bench, 1);
t2 = tic;
for nd = 1:N_bench
    [g_coneprog(nd), ~] = df.solvers.solve_socp_coneprog(cstr, c_all(:,nd));
end
time_coneprog = toc(t2);
fprintf('  %d solves in %.2fs (%.4fs/solve)\n', N_bench, time_coneprog, time_coneprog/N_bench);

% Agreement check
agree_cp = sum((g_cvx <= 1e-6) == (g_coneprog <= 1e-6));
diff_cp = max(abs(g_cvx - g_coneprog));
fprintf('  Agreement: %d/%d, Max obj diff: %.2e\n\n', agree_cp, N_bench, diff_cp);

%% =========================================================================
%  BENCHMARK 3: Direct SeDuMi (bypass CVX)
%  =========================================================================
fprintf('=== Benchmark 3: Direct SeDuMi (no CVX) ===\n');
try
    fprintf('  Converting to SeDuMi form...');
    t_conv = tic;
    sedumi_data = df.solvers.socp_to_sedumi(cstr);
    fprintf(' %.2fs\n', toc(t_conv));

    g_sedumi = zeros(N_bench, 1);
    t3 = tic;
    for nd = 1:N_bench
        [g_sedumi(nd), ~] = df.solvers.solve_socp_sedumi(sedumi_data, c_all(:,nd));
    end
    time_sedumi = toc(t3);
    fprintf('  %d solves in %.2fs (%.4fs/solve)\n', N_bench, time_sedumi, time_sedumi/N_bench);

    agree_sd = sum((g_cvx <= 1e-6) == (g_sedumi <= 1e-6));
    diff_sd = max(abs(g_cvx - g_sedumi));
    fprintf('  Agreement: %d/%d, Max obj diff: %.2e\n\n', agree_sd, N_bench, diff_sd);
    sedumi_ok = true;
catch ME
    fprintf('  FAILED: %s\n\n', ME.message);
    time_sedumi = Inf;
    sedumi_ok = false;
end

%% =========================================================================
%  BENCHMARK 4: Full grid (coneprog) for ground truth
%  =========================================================================
fprintf('=== Benchmark 4: Full grid (coneprog, %d points) ===\n', NGrid);
g_full = zeros(NGrid, 1);
t4 = tic;
for nd = 1:NGrid
    [g_full(nd), ~] = df.solvers.solve_socp_coneprog(cstr, c_all(:,nd));
    if mod(nd, 100) == 0
        fprintf('  %d/%d (%.1fs)\n', nd, NGrid, toc(t4));
    end
end
time_full = toc(t4);
fprintf('  %d solves in %.1fs (%.4fs/solve)\n\n', NGrid, time_full, time_full/NGrid);

%% =========================================================================
%  BENCHMARK 5: Adaptive grid
%  =========================================================================
fprintf('=== Benchmark 5: Adaptive grid ===\n');
t5 = tic;
[g_adaptive, n_solved, boundary_idx] = df.solvers.solve_grid_adaptive( ...
    cstr, c_all, nV, nM);
time_adapt = toc(t5);
fprintf('  %d/%d solves in %.1fs (%.1f%% of grid)\n', ...
    n_solved, NGrid, time_adapt, 100*n_solved/NGrid);

% Compare identified sets
is_full  = g_full <= 1e-6;
is_adapt = g_adaptive <= 1e-6;
mismatch = sum(is_full ~= is_adapt);
fprintf('  Identified set mismatch: %d/%d points\n\n', mismatch, NGrid);

%% =========================================================================
%  BENCHMARK 6: parfor scaling
%  =========================================================================
fprintf('=== Benchmark 6: parfor scaling ===\n');
if license('test', 'Distrib_Computing_Toolbox')
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('  Starting parallel pool...\n');
        pool = parpool('local');
    end
    nworkers = pool.NumWorkers;

    g_par = zeros(NGrid, 1);
    t6 = tic;
    parfor nd = 1:NGrid
        [g_par(nd), ~] = df.solvers.solve_socp_coneprog(cstr, c_all(:,nd));
    end
    time_par = toc(t6);
    fprintf('  parfor (%d workers): %.1fs (%.1fx vs serial)\n', ...
        nworkers, time_par, time_full/time_par);
    diff_par = max(abs(g_full - g_par));
    fprintf('  Max diff vs serial: %.2e\n\n', diff_par);
    parfor_ok = true;
else
    fprintf('  Parallel Computing Toolbox not available.\n\n');
    time_par = Inf;
    nworkers = 1;
    parfor_ok = false;
end

%% =========================================================================
%  SUMMARY
%  =========================================================================
fprintf('\n========================================\n');
fprintf('  SOLVER BENCHMARK SUMMARY\n');
fprintf('========================================\n\n');
fprintf('%-28s %8s %10s %8s\n', 'Backend', 'Time(s)', 'Per-solve', 'Speedup');
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-28s %8.2f %10.4f %8s\n', 'CVX + SeDuMi (50pts)', time_cvx, time_cvx/N_bench, '1.0x');
fprintf('%-28s %8.2f %10.4f %8.1fx\n', 'coneprog (50pts)', time_coneprog, time_coneprog/N_bench, time_cvx/time_coneprog);
if sedumi_ok
    fprintf('%-28s %8.2f %10.4f %8.1fx\n', 'Direct SeDuMi (50pts)', time_sedumi, time_sedumi/N_bench, time_cvx/time_sedumi);
end
fprintf('%-28s %8.1f %10.4f %8s\n', sprintf('coneprog full (%dpts)', NGrid), time_full, time_full/NGrid, '-');
fprintf('%-28s %8.1f %10s %8.1fx\n', sprintf('Adaptive (%d/%d)', n_solved, NGrid), ...
    time_adapt, '-', time_full/time_adapt);
if parfor_ok
    fprintf('%-28s %8.1f %10s %8.1fx\n', sprintf('parfor %dw (%dpts)', nworkers, NGrid), ...
        time_par, '-', time_full/time_par);
end

fprintf('\n--- Extrapolation to production (101x101 = 10201 grid) ---\n');
rate_cp = time_coneprog / N_bench;
if sedumi_ok
    rate_best = min(rate_cp, time_sedumi/N_bench);
    best_name = 'coneprog';
    if time_sedumi/N_bench < rate_cp, best_name = 'Direct SeDuMi'; end
else
    rate_best = rate_cp;
    best_name = 'coneprog';
end
pct_boundary = n_solved / NGrid;

fprintf('  Fastest per-solve (%s): %.4fs\n', best_name, rate_best);
fprintf('  Full grid serial:    %6.0fs (%4.1f min)\n', rate_best*10201, rate_best*10201/60);
if parfor_ok
    fprintf('  Full grid parfor:    %6.0fs (%4.1f min)\n', rate_best*10201/nworkers, rate_best*10201/nworkers/60);
end
fprintf('  Adaptive serial:     %6.0fs (%4.1f min)\n', rate_best*10201*pct_boundary, rate_best*10201*pct_boundary/60);
if parfor_ok
    fprintf('  Adaptive + parfor:   %6.0fs (%4.1f min)\n', ...
        rate_best*10201*pct_boundary/nworkers, rate_best*10201*pct_boundary/nworkers/60);
end
fprintf('  Current (CVX+SeDuMi): %5.0fs (%4.1f min)\n', (time_cvx/N_bench)*10201, (time_cvx/N_bench)*10201/60);
