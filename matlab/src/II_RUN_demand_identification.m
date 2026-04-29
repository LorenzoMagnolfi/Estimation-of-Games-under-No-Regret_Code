%% II_RUN_demand_identification.m
%  R1.1.d exercise: joint identification of demand parameter eta and cost
%  distribution (mu, sigma).
%
%  For each candidate eta on a grid, rebuilds Pi(eta), BCCE constraints,
%  K(eta), epsilon(eta), and tests all (mu,sigma) candidates. The identified
%  set is a collection of (eta, mu, sigma) triples.
%
%  Also computes worst-case K bound (no demand knowledge) for comparison.
%
%  Usage: run from matlab/src/ with CVX and SeDuMi on path.

clear; clc; close all;
rng(12345);

%% ======== Configuration ========
eta_true   = -1/3;                        % true DGP demand parameter
eta_grid   = linspace(-1.0, -0.05, 15);   % broad search range
actions_vec = [4; 5; 6; 7; 8];
mu_true    = [3; 3];
sigma2_true = eye(2);
s_val      = 5;                           % type-space size
T          = 4000000;                     % learning horizon
confid     = 0.05;
switch_eps = 1;

% (mu,sigma) grid — reduced for fast pass (15 eta × ~256 = ~3840 total SOCPs)
NGridM = 15;
NGridV = 15;
gridparamM = [1; linspace(0.55, mu_true(1)*0.5, NGridM)'];
gridparamV = [1; linspace(0.15, sigma2_true(1,1)*3.5, NGridV)'];

%% ======== Paths ========
addpath(fileparts(mfilename('fullpath')));
paths = df_repo_paths();
fig_dir = fullfile(paths.output, 'figures', 'part_ii');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ======== Step 1: Simulate learning under TRUE DGP ========
fprintf('=== R1.1.d: Joint demand-cost identification ===\n');
fprintf('True eta = %.4f, s = %d, T = %dk\n\n', eta_true, s_val, T/1000);

cfg_true = df.setup.game_simulation(2, eta_true, actions_vec, mu_true, sigma2_true, s_val);
fprintf('[DGP] Learning (T=%dk)... ', T/1000);
t0 = tic;
[distY_time, ~] = learn_mod(cfg_true, 1, T, T, 1, 1, 1, 1);
action_distribution = distY_time;
fprintf('%.1fs\n', toc(t0));

%% ======== Step 2: Build (mu,sigma) grid ONCE ========
[distpars, distribution_parameters] = df.report.build_param_grid(...
    mu_true, sigma2_true, gridparamM, gridparamV, 'Both');
NGrid = size(distpars, 1);
fprintf('[Grid] %d (mu,sigma) candidates\n\n', NGrid);

%% ======== Step 3: Loop over eta candidates ========
n_eta = numel(eta_grid);
VV_by_eta = zeros(n_eta, NGrid);
K_by_eta = zeros(n_eta, s_val);
eps_by_eta = zeros(n_eta, s_val);
n_identified = zeros(n_eta, 1);
timing_eta = zeros(n_eta, 1);

for ei = 1:n_eta
    eta_cand = eta_grid(ei);
    t_eta = tic;
    fprintf('[eta %2d/%d] eta=%.4f: ', ei, n_eta, eta_cand);

    % Build game config for this candidate eta
    cfg_test = df.setup.game_simulation(2, eta_cand, actions_vec, mu_true, sigma2_true, s_val);

    % Build BCCE constraints for this Pi(eta)
    cstr = df.solvers.build_constraints(cfg_test.type_space, cfg_test.action_space, cfg_test.Pi);
    dim_u = cstr.NA - 1;
    a_dim = cstr.a;
    NAg = cstr.NAg;
    s2 = cstr.s2;
    T_sorted = cstr.T_sorted;

    % Compute K and epsilon for this eta
    eps_vec = df.solvers.compute_epsilon(cfg_test, T, confid, switch_eps);
    K_vec = max(cfg_test.Pi(:,:,1)) - min(cfg_test.Pi(:,:,1));
    K_by_eta(ei, :) = K_vec;
    eps_by_eta(ei, :) = eps_vec;

    % Build Psi and marginal distributions for all (mu,sigma) candidates
    Psi = zeros(s2, NGrid);
    marg_distrib = zeros(s_val, NGrid);
    type_space = cfg_test.type_space;
    for nd = 1:NGrid
        Psi(:,nd) = mvnpdf(T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
        mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
        sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
        md = normpdf(type_space{1,1}, mu_val, sqrt(sg_val));
        marg_distrib(:,nd) = md / sum(md);
    end
    Psi = Psi ./ sum(Psi, 1);

    % Build all objective vectors
    n_vars = size(cstr.B_EQ, 2);
    c_all = zeros(n_vars, NGrid);
    for nd = 1:NGrid
        bmarg = Psi(:,nd);
        eps_fin = repmat(sqrt(marg_distrib(:,nd))', 1, NAg*a_dim) .* ...
                  repmat(eps_vec, 1, NAg*a_dim);
        c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', ...
                       bmarg', 1, eps_fin]';
    end

    % Batch solve
    cvx_opts = struct('verbose', false, 'solver', 'sedumi');
    [VV, ~] = df.solvers.solve_grid_cvx(cstr, c_all, cvx_opts);

    VV_by_eta(ei, :) = VV(:)';
    n_identified(ei) = sum(VV <= 1e-12);
    timing_eta(ei) = toc(t_eta);
    fprintf('%d/%d identified (%.1f%%), %.1fs\n', ...
        n_identified(ei), NGrid, 100*n_identified(ei)/NGrid, timing_eta(ei));
end

%% ======== Step 4: Worst-case K bound ========
fprintf('\n[Worst-case] Computing K_max bound (no demand knowledge)...\n');
% Since sale prob in [0,1], max utility = 1 * (max_price - min_cost)
% For our game: max_price = 8, min possible cost ~ type_space min
% K_worst = max over all (a,t) of |u(a,t)| = max(actions) - min(types) for the profit
% More precisely: K_worst(t) = max_a [1*(a_i - t_i)] - min_a [0*(a_i - t_i)]
%                             = max(actions) - t_i   (since min is 0)
% But K = max_a u - min_a u over ALL action PROFILES, and sale prob affects both.
% Conservative: K_worst = max(actions_vec) (assuming sale prob = 1 on best action)
K_worstcase = max(actions_vec);
eps_worstcase = K_worstcase * sqrt(log(length(actions_vec))) / (confid * sqrt(T));
fprintf('  K_worst = %.2f, eps_worst = %.6f\n', K_worstcase, eps_worstcase);

% Run worst-case identification (use true eta cfg but inflate epsilon)
cfg_wc = cfg_true;
% Build constraints with true Pi (structure doesn't change, only epsilon is wider)
cstr_wc = df.solvers.build_constraints(cfg_wc.type_space, cfg_wc.action_space, cfg_wc.Pi);
dim_u_wc = cstr_wc.NA - 1;
a_dim_wc = cstr_wc.a;
NAg_wc = cstr_wc.NAg;
s2_wc = cstr_wc.s2;

% Recompute objectives with worst-case epsilon
Psi_wc = zeros(s2_wc, NGrid);
marg_wc = zeros(s_val, NGrid);
for nd = 1:NGrid
    Psi_wc(:,nd) = mvnpdf(cstr_wc.T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
    mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
    sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
    md = normpdf(cfg_wc.type_space{1,1}, mu_val, sqrt(sg_val));
    marg_wc(:,nd) = md / sum(md);
end
Psi_wc = Psi_wc ./ sum(Psi_wc, 1);

eps_wc_vec = repmat(eps_worstcase, 1, s_val);
c_all_wc = zeros(size(cstr_wc.B_EQ, 2), NGrid);
for nd = 1:NGrid
    bmarg = Psi_wc(:,nd);
    eps_fin = repmat(sqrt(marg_wc(:,nd))', 1, NAg_wc*a_dim_wc) .* ...
              repmat(eps_wc_vec, 1, NAg_wc*a_dim_wc);
    c_all_wc(:,nd) = [zeros(1, dim_u_wc), action_distribution(:,1)', ...
                      bmarg', 1, eps_fin]';
end

cvx_opts_wc = struct('verbose', false, 'solver', 'sedumi');
[VV_wc, ~] = df.solvers.solve_grid_cvx(cstr_wc, c_all_wc, cvx_opts_wc);
n_id_wc = sum(VV_wc <= 1e-12);
fprintf('  Worst-case: %d/%d identified (%.1f%%)\n', n_id_wc, NGrid, 100*n_id_wc/NGrid);

%% ======== Step 5: Results summary ========
fprintf('\n=== Results Summary ===\n');
fprintf('eta_grid: [%.2f, %.2f], %d points\n', eta_grid(1), eta_grid(end), n_eta);
fprintf('True eta = %.4f\n\n', eta_true);

% Identified eta range
eta_identified = eta_grid(n_identified > 0);
if ~isempty(eta_identified)
    fprintf('Identified eta range: [%.4f, %.4f]\n', min(eta_identified), max(eta_identified));
else
    fprintf('WARNING: no eta value has any identified (mu,sigma) pairs!\n');
end

fprintf('\nPer-eta results:\n');
fprintf('  %8s  %8s  %8s\n', 'eta', 'n_id', 'pct');
for ei = 1:n_eta
    fprintf('  %8.4f  %8d  %7.1f%%\n', eta_grid(ei), n_identified(ei), 100*n_identified(ei)/NGrid);
end
fprintf('\n  Worst-case (no demand info): %d identified (%.1f%%)\n', n_id_wc, 100*n_id_wc/NGrid);

%% ======== Step 6: Profile plot ========
fig1 = figure('Position', [100 100 700 400]);
bar(eta_grid, 100 * n_identified / NGrid, 0.7, 'FaceColor', [0.3 0.6 0.9]);
hold on;
yline(100 * n_id_wc / NGrid, 'r--', 'LineWidth', 1.5);
xline(eta_true, 'k-', 'LineWidth', 1.5);
xlabel('$\eta$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Identified (\%)', 'Interpreter', 'latex', 'FontSize', 14);
legend({'Joint (per $\eta$)', 'Worst-case $K$', 'True $\eta$'}, ...
    'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
title('R1.1.d: Joint demand-cost identification', 'FontSize', 13);
set(gca, 'FontSize', 12);
hold off;

print(fig1, fullfile(fig_dir, 'demand_id_profile'), '-depsc');
print(fig1, fullfile(fig_dir, 'demand_id_profile'), '-dpng', '-r150');
fprintf('\nSaved: demand_id_profile.{eps,png}\n');

%% ======== Step 7: Comparison SVM figures ========
% Find closest eta to true value
[~, idx_true] = min(abs(eta_grid - eta_true));

% Three panels: (a) known eta, (b) worst-case K, (c) joint (profiled)
% Panel (a): known eta = true
id_known = VV_by_eta(idx_true, :) <= 1e-12;
% Panel (b): worst-case
id_wc = VV_wc <= 1e-12;
% Panel (c): joint projection = union of identified sets across all eta
id_joint = any(VV_by_eta <= 1e-12, 1);

panels = {id_known, id_wc, id_joint};
panel_titles = {'Known $\eta$', 'Worst-case $K$', 'Joint (profiled $\eta$)'};

fig2 = figure('Position', [100 100 1200 400]);
for pp = 1:3
    subplot(1, 3, pp);
    id_idx = panels{pp};
    if sum(id_idx) >= 3
        % SVM classification
        [label, PGx, ~] = df.report.classify_identified_set(distpars, id_idx(:), ...
            [max(distpars(:,1))*1.1, max(distpars(:,2))*1.1]);
        scatter(PGx(label==1,1), PGx(label==1,2), 1, [0.6 0.9 1], 'filled');
        hold on;
    end
    scatter(distpars(id_idx,1), distpars(id_idx,2), 15, 'b', 'filled');
    hold on;
    scatter(distpars(~id_idx,1), distpars(~id_idx,2), 8, [0.8 0.8 0.8], 'filled');
    plot(mu_true(1), sigma2_true(1,1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 13);
    title(panel_titles{pp}, 'Interpreter', 'latex', 'FontSize', 13);
    hold off;
end
sgtitle(sprintf('R1.1.d: Identified set comparison ($s=%d$, $T=%dk$)', s_val, T/1000), ...
    'Interpreter', 'latex', 'FontSize', 14);

print(fig2, fullfile(fig_dir, 'demand_id_comparison'), '-depsc');
print(fig2, fullfile(fig_dir, 'demand_id_comparison'), '-dpng', '-r150');
fprintf('Saved: demand_id_comparison.{eps,png}\n');

%% ======== Save workspace ========
save(fullfile(paths.output, 'artifacts', 'demand_identification.mat'), ...
    'eta_grid', 'eta_true', 'VV_by_eta', 'VV_wc', 'distpars', ...
    'distribution_parameters', 'n_identified', 'n_id_wc', ...
    'K_by_eta', 'eps_by_eta', 'K_worstcase', 'eps_worstcase', ...
    'timing_eta', 'T', 's_val', 'confid', 'NGrid');

fprintf('\n=== Done. Total time: %.1f min ===\n', sum(timing_eta)/60);
