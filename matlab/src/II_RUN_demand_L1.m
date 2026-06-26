%% II_RUN_demand_L1
%  Joint demand-and-cost identification (S5, R1.1.d) with the paper-exact BHC-L1
%  consistency set. For each candidate demand parameter eta on a grid, rebuild
%  Pi(eta), the BCCE constraints, the obedience radius eps(eta) at alpha_R, and the
%  L1 consistency radius at alpha_C, then test all (mu,sigma) cost candidates. The
%  full-feedback regret-matching play is generated once under the true DGP
%  (eta_0 = -1/3). Saves identified counts per eta; NO figures (regenerate locally).

clear; clc; rng(12345);
eta_true    = -1/3;
eta_grid    = linspace(-1.0, -0.05, 15);
actions_vec = [4; 5; 6; 7; 8];
mu_true     = [3; 3];
sigma2_true = eye(2);
s_val       = 5;
T           = 4000000;
alpha_set   = 0.05;
alpha_R     = 0.9 * alpha_set;   % 0.045  obedience budget
alpha_C     = 0.1 * alpha_set;   % 0.005  consistency budget
switch_eps  = 10;
IDTOL       = 1e-8;

paths = df_repo_paths();

NGridM = 15; NGridV = 15;
gridparamM = [1; linspace(0.55, mu_true(1)*0.5, NGridM)'];
gridparamV = [1; linspace(0.15, sigma2_true(1,1)*3.5, NGridV)'];

% Learn under the TRUE DGP (full-feedback regret matching)
cfg_true = df.setup.game_simulation(2, eta_true, actions_vec, mu_true, sigma2_true, s_val);
fprintf('[DGP] Learning (T=%dk)... ', T/1000);
t0 = tic; [distY_time, ~] = learn_mod(cfg_true, 1, T, T, 1, 1, 1, 1);
action_distribution = distY_time; fprintf('%.1fs\n', toc(t0));

% (mu,sigma) candidate grid (built once)
[distpars, distribution_parameters] = df.report.build_param_grid( ...
    mu_true, sigma2_true, gridparamM, gridparamV, 'Both');
NGrid = size(distpars, 1);

n_eta = numel(eta_grid);
VV_by_eta = zeros(n_eta, NGrid);
n_identified = zeros(n_eta, 1);
K_by_eta = zeros(n_eta, s_val);

for ei = 1:n_eta
    eta_cand = eta_grid(ei);
    cfg_test = df.setup.game_simulation(2, eta_cand, actions_vec, mu_true, sigma2_true, s_val);
    cstr = df.solvers.build_constraints(cfg_test.type_space, cfg_test.action_space, cfg_test.Pi);
    dim_u = cstr.NA - 1; a_dim = cstr.a; NAg = cstr.NAg; s2 = cstr.s2; T_sorted = cstr.T_sorted;
    cons_idx = dim_u + cstr.NA + (1:s2);

    eps_vec = df.solvers.compute_epsilon(cfg_test, T, alpha_R, switch_eps);
    r_L1 = df.solvers.compute_consistency_slack(s2, alpha_C, T, 'L1');
    K_by_eta(ei, :) = max(cfg_test.Pi(:,:,1)) - min(cfg_test.Pi(:,:,1));

    Psi = zeros(s2, NGrid); marg = zeros(s_val, NGrid);
    for nd = 1:NGrid
        Psi(:, nd) = mvnpdf(T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
        mv = distribution_parameters{2,nd}; mv = mv(1);
        sv = distribution_parameters{3,nd}; sv = sv(1,1);
        md = normpdf(cfg_test.type_space{1,1}, mv, sqrt(sv)); marg(:, nd) = md / sum(md);
    end
    Psi = Psi ./ sum(Psi, 1);

    c_all = zeros(size(cstr.B_EQ, 2), NGrid);
    for nd = 1:NGrid
        bmarg = Psi(:, nd);
        eps_fin = repmat(sqrt(marg(:,nd))', 1, NAg*a_dim) .* repmat(eps_vec, 1, NAg*a_dim);
        c_all(:, nd) = [zeros(1, dim_u), action_distribution(:,1)', bmarg', 1, eps_fin]';
    end

    cvx_opts = struct('verbose', false, 'solver', 'sedumi', ...
        'cons_l1_r', r_L1, 'cons_l1_idx', cons_idx);
    [VV, ~] = df.solvers.solve_grid_cvx(cstr, c_all, cvx_opts);
    VV_by_eta(ei, :) = VV(:)';
    n_identified(ei) = nnz(VV <= IDTOL);
    fprintf('[eta %2d/%d] eta=%.3f: %d/%d identified\n', ei, n_eta, eta_cand, n_identified(ei), NGrid);
end

eta_identified = eta_grid(n_identified > 0);
summary = table(eta_grid(:), n_identified(:), 'VariableNames', {'eta', 'n_identified'});
fprintf('\n=== Demand (S5) + L1: identified (mu,sigma) per eta (IDTOL=%.0e) ===\n', IDTOL);
disp(summary);
if ~isempty(eta_identified)
    fprintf('Identified eta range: [%.4f, %.4f]; true eta = %.4f\n', ...
        min(eta_identified), max(eta_identified), eta_true);
end

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'demand_L1.csv'));
save(fullfile(paths.artifacts, 'demand_L1.mat'), ...
    'eta_grid', 'eta_true', 'VV_by_eta', 'n_identified', 'distpars', 'K_by_eta', ...
    'T', 's_val', 'alpha_R', 'alpha_C', 'switch_eps', 'IDTOL', 'NGrid', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'demand_L1.mat'));
fprintf('========== Demand L1 complete ==========\n');
