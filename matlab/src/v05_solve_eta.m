function [VV, dp] = v05_solve_eta(eta_cand, gM, gV, T, alpha_R, alpha_C, code, action_distribution, lab)
% V05_SOLVE_ETA  Stage II feasibility over a (mu, sigma^2) grid at a candidate demand slope, certified calibration.
%  Rebuilds the payoff table and the BCCE constraints at eta_cand, the L1 consistency radius at T, and the per-candidate
%  V05 tolerances (code 14 tailF, 15 tailB, 16 momentF, 17 momentB; the tail codes use the latent-count ceiling), then
%  solves the direct SeDuMi lane over the grid. gM and gV are multiplier vectors (first entry 1 = truth) of lab.mu_true(1)
%  and lab.sigma2_true(1,1); lab also carries actions_vec, s_val, IDTOL. Returns VV (NGrid x 1) and dp (NGrid x 2: mu,
%  sigma^2). Used by II_RUN_v05_demand_c3 and II_SMOKE_v05b.
cfg_test = df.setup.game_simulation(2, eta_cand, lab.actions_vec, lab.mu_true, lab.sigma2_true, lab.s_val);
cstr = df.solvers.build_constraints(cfg_test.type_space, cfg_test.action_space, cfg_test.Pi);
dim_u = cstr.NA - 1; a_dim = cstr.a; NAg = cstr.NAg; s2 = cstr.s2; T_sorted = cstr.T_sorted;
cons_idx = dim_u + cstr.NA + (1:s2);
r_L1 = df.solvers.compute_consistency_slack(s2, alpha_C, T, 'L1');
[dp, distribution_parameters] = df.report.build_param_grid(lab.mu_true, lab.sigma2_true, gM, gV, 'Both');
NGrid = size(dp, 1); Psi = zeros(s2, NGrid); marg = zeros(lab.s_val, NGrid);
for nd = 1:NGrid
    Psi(:, nd) = mvnpdf(T_sorted, distribution_parameters{2, nd}, distribution_parameters{3, nd});
    mv = distribution_parameters{2, nd}; mv = mv(1); sv = distribution_parameters{3, nd}; sv = sv(1, 1);
    md = normpdf(cfg_test.type_space{1, 1}, mv, sqrt(sv)); marg(:, nd) = md / sum(md);
end
Psi = Psi ./ sum(Psi, 1);
c_all = zeros(size(cstr.B_EQ, 2), NGrid);
for nd = 1:NGrid
    eps_t = df.solvers.compute_epsilon_v05(cfg_test, T, marg(:, nd)', r_L1, code, alpha_R);
    c_all(:, nd) = [zeros(1, dim_u), action_distribution(:, 1)', Psi(:, nd)', 1, repmat(eps_t, 1, NAg * a_dim)]';
end
gopts = struct('verbose', false, 'solver', 'sedumi', 'cons_l1_r', r_L1, 'cons_l1_idx', cons_idx, 'use_parfor', true);
[VV, ~, ~] = df.solvers.solve_grid_sedumi(cstr, c_all, gopts);
VV = VV(:);
end
