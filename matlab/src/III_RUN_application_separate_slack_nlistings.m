%% III_RUN_application_separate_slack_nlistings
%
%  (b.light) per Application_State_Expansion_Plan.md.
%
%  Same as III_RUN_application_separate_slack.m, but replaces T (number of
%  listing-day observations) with n_listings (number of independent
%  listings) in BOTH the regret eps and the consistency-slack r_N.
%
%  Motivation: R1.4.e — "you use the same listing on a different day as a
%  new observation… the iid assumption would then be violated."  The
%  Python data audit shows 97% of within-listing transitions are at the
%  same dollar price; intracluster correlation rho_w ~ 0.79 (Seller 1)
%  and 0.74 (Seller 2).  Design effect ~7-10.  The honest finite-sample
%  correction replaces T with n_listings (the count of independent
%  pricing decisions) wherever T enters concentration bounds.
%
%  Hard-coded n_listings (computed by python audit on
%  data/intermediate/price_res_5bins_15_sellers.dta):
%    Seller 1: n_listings = 1512  (T = 18293)
%    Seller 2: n_listings = 1695  (T = 14724)
%
%  Compares baseline (no slack), separate-slack at T, and separate-slack
%  at n_listings, on the same 50x50 (mu, sigma^2) grid.

clear all; clc; close all;

paths = df_repo_paths();
rng(20260430);

Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');
players      = 1:2;
NGridV       = 50;
NGridM       = 50;
n_types      = 5;
alpha_set    = 0.05;
alpha_C      = 0.005;
alpha_R      = alpha_set - alpha_C;
switch_eps   = 9;

% n_listings per seller (precomputed via python audit)
n_listings_per_seller = [1512, 1695];

NGrid = NGridV * NGridM;

results = struct();
results.players   = players;
results.NGridV    = NGridV;
results.NGridM    = NGridM;
results.alpha_set = alpha_set;
results.alpha_C   = alpha_C;
results.alpha_R   = alpha_R;
results.n_listings_per_seller = n_listings_per_seller;

for idx = 1:numel(players)
    iii = players(idx);
    nL  = n_listings_per_seller(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d (separate-slack @ n_listings)\n', iii);
    fprintf('========================================\n');

    %% Build cfg
    cfg = df.setup.game_application(iii, Dist_file, Prob_file, n_types);
    T   = cfg.maxiters;
    s   = n_types;
    NAct = cfg.NAct;
    fprintf('  T (listing-days) = %d, n_listings = %d, s = %d, |A| = %d\n', T, nL, s, NAct);
    fprintf('  effective N = %d (DEFF ~ %.1f)\n', nL, T/nL);

    action_distribution = cfg.distrib';

    %% Grid (matches Identification_Pricing_Game_ApplicationL)
    mu = cfg.mu; sigma2 = cfg.sigma2;
    gridparamV = linspace(0.1 * sigma2(1,1), sigma2(1,1) * 5, NGridV)';
    gridparamM = linspace(mu(1,1) * 4, mu(1,1) * 0.25, NGridM)';
    [distpars, distribution_parameters] = df.report.build_param_grid(mu(1,1), sigma2, gridparamM, gridparamV);

    %% Eps + slack with n_listings (the (b.light) substitution)
    kwargs       = struct('mode', 'application', 'marg_mean', cfg.marg_mean, 's_override', s);
    eps_baseline = df.solvers.compute_epsilon(cfg, nL, alpha_set, switch_eps, kwargs);
    eps_regret_R = df.solvers.compute_epsilon(cfg, nL, alpha_R,   switch_eps, kwargs);

    r_N = sqrt(log(2 * s / alpha_C) / (2 * nL));
    fprintf('  r_N (box, alpha_C=%.4f, N=n_listings) = %.5f\n', alpha_C, r_N);
    fprintf('  eps_baseline = [%s]\n', sprintf('%.4f ', eps_baseline));
    fprintf('  eps_regret_R = [%s]\n', sprintf('%.4f ', eps_regret_R));

    %% Marginal-mode opponent distribution
    marg_act_I  = kron(eye(NAct), ones(1, NAct)) * action_distribution;
    marg_act_II = kron(ones(1, NAct), eye(NAct)) * action_distribution;

    %% Build constraints once for each LP variant.
    cstr_baseline = df.solvers.build_constraints_marginal_slack( ...
        cfg.type_space, cfg.action_space, cfg.Pi, marg_act_II, 0);
    cstr_slack    = df.solvers.build_constraints_marginal_slack( ...
        cfg.type_space, cfg.action_space, cfg.Pi, marg_act_II, r_N);

    dim_u    = cstr_baseline.Nactions - 1;
    a_dim    = cstr_baseline.Nactions;
    s_app    = cstr_baseline.s;

    %% Compute prior (Psi) across the grid
    NGrid_lambda = size(distribution_parameters, 2);
    Psi = zeros(s_app, NGrid_lambda);
    for nd = 1:NGrid_lambda
        mu_v = distribution_parameters{2,nd}; mu_v = mu_v(1);
        sg_v = distribution_parameters{3,nd}; sg_v = sg_v(1,1);
        md = pdf(distribution_parameters{1,nd}, cfg.type_space{1,1}, mu_v, sg_v);
        Psi(:,nd) = md / sum(md);
    end

    %% Solve baseline at n_listings (no slack)
    fprintf('\n  [Baseline @ n_listings] solving %d grid points (no slack) ...\n', NGrid_lambda);
    t_b = tic;
    VV_baseline = nan(NGrid_lambda, 1);
    for nd = 1:NGrid_lambda
        bmarg = Psi(:, nd);
        eps_fin_baseline = repmat(eps_baseline, 1, a_dim);
        c = [zeros(1, dim_u), marg_act_I(:,1)', bmarg', 1, eps_fin_baseline]';
        [VV_baseline(nd), ~] = df.solvers.solve_socp_cvx(cstr_baseline, c, 'sedumi', 'default');
    end
    fprintf('  [Baseline @ n_listings] done in %.1fs\n', toc(t_b));
    id_baseline = (VV_baseline <= 1e-12);

    %% Solve separate-slack @ n_listings
    fprintf('\n  [Separate-slack @ n_listings] solving %d grid points (alpha_R=%.3f, alpha_C=%.3f) ...\n', ...
        NGrid_lambda, alpha_R, alpha_C);
    t_s = tic;
    VV_slack = nan(NGrid_lambda, 1);
    for nd = 1:NGrid_lambda
        bmarg = Psi(:, nd);
        eps_fin_R = repmat(eps_regret_R, 1, a_dim);
        c = [zeros(1, dim_u), marg_act_I(:,1)', 1, ...
             (bmarg + r_N)', (-bmarg + r_N)', eps_fin_R]';
        [VV_slack(nd), ~] = df.solvers.solve_socp_cvx(cstr_slack, c, 'sedumi', 'default');
    end
    fprintf('  [Separate-slack @ n_listings] done in %.1fs\n', toc(t_s));
    id_slack = (VV_slack <= 1e-12);

    %% Report intervals
    fprintf('\n  Baseline @ n_listings (no slack):     %4d/%4d feasible', sum(id_baseline), NGrid_lambda);
    if any(id_baseline)
        ddpb = distpars(id_baseline, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddpb(:,1)), max(ddpb(:,1)), min(ddpb(:,2)), max(ddpb(:,2)));
    end
    fprintf('\n');
    fprintf('  Separate-slack @ n_listings:          %4d/%4d feasible', sum(id_slack), NGrid_lambda);
    if any(id_slack)
        ddps = distpars(id_slack, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddps(:,1)), max(ddps(:,1)), min(ddps(:,2)), max(ddps(:,2)));
    end
    fprintf('\n');

    %% Pack
    results.seller(idx).player_id    = iii;
    results.seller(idx).T            = T;
    results.seller(idx).n_listings   = nL;
    results.seller(idx).distpars     = distpars;
    results.seller(idx).VV_baseline  = VV_baseline;
    results.seller(idx).VV_slack     = VV_slack;
    results.seller(idx).id_baseline  = id_baseline;
    results.seller(idx).id_slack     = id_slack;
    results.seller(idx).eps_baseline = eps_baseline;
    results.seller(idx).eps_regret_R = eps_regret_R;
    results.seller(idx).r_N          = r_N;
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_slack');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_application_separate_slack_nlistings.mat'), '-struct', 'results');
fprintf('\nSaved %s\n', fullfile(out_dir, 'results_application_separate_slack_nlistings.mat'));

%% Compact tabular comparison vs the existing T-based separate-slack run.
T_path = fullfile(out_dir, 'results_application_separate_slack.mat');
if exist(T_path, 'file') == 2
    T_res = load(T_path);
    fprintf('\n=== Comparison: T (listing-days) vs n_listings ===\n');
    fprintf('%-10s %-30s %-30s\n', 'Seller', 'Separate-slack @ T', 'Separate-slack @ n_listings');
    for idx = 1:numel(players)
        sd_T = T_res.seller(idx);
        sd_N = results.seller(idx);
        if any(sd_T.id_slack)
            ddT = sd_T.distpars(sd_T.id_slack, :);
            txt_T = sprintf('mu in [%.1f, %.1f] (%d cells)', ...
                min(ddT(:,1)), max(ddT(:,1)), sum(sd_T.id_slack));
        else
            txt_T = 'EMPTY';
        end
        if any(sd_N.id_slack)
            ddN = sd_N.distpars(sd_N.id_slack, :);
            txt_N = sprintf('mu in [%.1f, %.1f] (%d cells)', ...
                min(ddN(:,1)), max(ddN(:,1)), sum(sd_N.id_slack));
        else
            txt_N = 'EMPTY';
        end
        fprintf('  %-8d %-30s %-30s\n', sd_N.player_id, txt_T, txt_N);
    end
end

fprintf('\n=== Done. ===\n');
