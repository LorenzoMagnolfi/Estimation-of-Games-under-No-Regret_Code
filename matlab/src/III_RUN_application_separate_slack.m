%% III_RUN_application_separate_slack
%
%  Parametric Stage III with the PROPER separate-slack consistency relaxation
%  per Niccolò's note Sec. 6.2.  Unlike III_RUN_application_slack.m (which
%  absorbs the consistency term into the obedience radius — provably
%  conservative), this version moves the consistency slack into the LP itself
%  as inequality relaxations.
%
%  Per type t in T_i:
%      pi(t) - r_N  <=  sum_a nu(a, t)  <=  pi(t) + r_N
%  where r_N = sqrt(log(2 s / alpha_C) / (2 T))  (Hoeffding box).
%
%  Compare to baseline (no slack) and absorbed-eps version side by side.

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

NGrid = NGridV * NGridM;

results = struct();
results.players   = players;
results.NGridV    = NGridV;
results.NGridM    = NGridM;
results.alpha_set = alpha_set;
results.alpha_C   = alpha_C;
results.alpha_R   = alpha_R;

for idx = 1:numel(players)
    iii = players(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d (separate-slack)\n', iii);
    fprintf('========================================\n');

    %% Build cfg
    cfg = df.setup.game_application(iii, Dist_file, Prob_file, n_types);
    T   = cfg.maxiters;
    s   = n_types;
    NAct = cfg.NAct;
    fprintf('  T = %d, s = %d, |A| = %d\n', T, s, NAct);

    action_distribution = cfg.distrib';

    %% Grid (matches Identification_Pricing_Game_ApplicationL)
    mu = cfg.mu; sigma2 = cfg.sigma2;
    gridparamV = linspace(0.1 * sigma2(1,1), sigma2(1,1) * 5, NGridV)';
    gridparamM = linspace(mu(1,1) * 4, mu(1,1) * 0.25, NGridM)';
    [distpars, distribution_parameters] = df.report.build_param_grid(mu(1,1), sigma2, gridparamM, gridparamV);

    %% Eps + slack
    kwargs       = struct('mode', 'application', 'marg_mean', cfg.marg_mean, 's_override', s);
    eps_baseline = df.solvers.compute_epsilon(cfg, T, alpha_set, switch_eps, kwargs);
    eps_regret_R = df.solvers.compute_epsilon(cfg, T, alpha_R,   switch_eps, kwargs);

    r_N = sqrt(log(2 * s / alpha_C) / (2 * T));
    fprintf('  r_N (box, alpha_C=%.4f) = %.5f\n', alpha_C, r_N);
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

    %% Compute prior (Psi) and per-player marginal across the grid
    NGrid_lambda = size(distribution_parameters, 2);
    Psi = zeros(s_app, NGrid_lambda);
    for nd = 1:NGrid_lambda
        mu_v = distribution_parameters{2,nd}; mu_v = mu_v(1);
        sg_v = distribution_parameters{3,nd}; sg_v = sg_v(1,1);
        md = pdf(distribution_parameters{1,nd}, cfg.type_space{1,1}, mu_v, sg_v);
        Psi(:,nd) = md / sum(md);
    end

    %% Solve baseline: existing eps, no slack
    fprintf('\n  [Baseline] solving %d grid points (no slack, eps=baseline) ...\n', NGrid_lambda);
    t_b = tic;
    VV_baseline = nan(NGrid_lambda, 1);
    for nd = 1:NGrid_lambda
        bmarg = Psi(:, nd);
        eps_fin_baseline = repmat(eps_baseline, 1, a_dim);
        c = [zeros(1, dim_u), marg_act_I(:,1)', bmarg', 1, eps_fin_baseline]';
        [VV_baseline(nd), ~] = df.solvers.solve_socp_cvx(cstr_baseline, c, 'sedumi', 'default');
    end
    fprintf('  [Baseline] done in %.1fs\n', toc(t_b));
    id_baseline = (VV_baseline <= 1e-12);

    %% Solve separate-slack: regret eps with alpha_R, consistency in LP
    fprintf('\n  [Separate-slack] solving %d grid points (alpha_R=%.3f, alpha_C=%.3f) ...\n', ...
        NGrid_lambda, alpha_R, alpha_C);
    t_s = tic;
    VV_slack = nan(NGrid_lambda, 1);
    for nd = 1:NGrid_lambda
        bmarg = Psi(:, nd);
        eps_fin_R = repmat(eps_regret_R, 1, a_dim);
        % c_all (slack form): [u, action_dist, 1, bmarg+r_N, -bmarg+r_N, eps_fin]'
        c = [zeros(1, dim_u), marg_act_I(:,1)', 1, ...
             (bmarg + r_N)', (-bmarg + r_N)', eps_fin_R]';
        [VV_slack(nd), ~] = df.solvers.solve_socp_cvx(cstr_slack, c, 'sedumi', 'default');
    end
    fprintf('  [Separate-slack] done in %.1fs\n', toc(t_s));
    id_slack = (VV_slack <= 1e-12);

    %% Report intervals
    fprintf('\n  Baseline (no slack):       %4d/%4d feasible', sum(id_baseline), NGrid_lambda);
    if any(id_baseline)
        ddpb = distpars(id_baseline, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddpb(:,1)), max(ddpb(:,1)), min(ddpb(:,2)), max(ddpb(:,2)));
    end
    fprintf('\n');
    fprintf('  Separate-slack:            %4d/%4d feasible', sum(id_slack), NGrid_lambda);
    if any(id_slack)
        ddps = distpars(id_slack, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddps(:,1)), max(ddps(:,1)), min(ddps(:,2)), max(ddps(:,2)));
    end
    fprintf('\n');

    %% Pack
    results.seller(idx).player_id   = iii;
    results.seller(idx).T           = T;
    results.seller(idx).distpars    = distpars;
    results.seller(idx).VV_baseline = VV_baseline;
    results.seller(idx).VV_slack    = VV_slack;
    results.seller(idx).id_baseline = id_baseline;
    results.seller(idx).id_slack    = id_slack;
    results.seller(idx).eps_baseline = eps_baseline;
    results.seller(idx).eps_regret_R = eps_regret_R;
    results.seller(idx).r_N         = r_N;
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_slack');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_application_separate_slack.mat'), '-struct', 'results');
fprintf('\nSaved %s\n', fullfile(out_dir, 'results_application_separate_slack.mat'));

%% Figure: identified sets baseline vs separate-slack vs (load) absorbed-eps
fig = figure('Color', 'w', 'Position', [60 60 800 * numel(players) 600]);

% Try to load absorbed results for overlay
absorbed_path = fullfile(out_dir, 'results_application_slack.mat');
have_absorbed = exist(absorbed_path, 'file') == 2;
if have_absorbed
    A = load(absorbed_path);
end

for idx = 1:numel(players)
    subplot(1, numel(players), idx);
    sd = results.seller(idx);
    hold on;

    if have_absorbed && idx <= numel(A.seller)
        ad = A.seller(idx);
        if any(ad.id_slack)
            scatter(ad.distpars(ad.id_slack, 1), ad.distpars(ad.id_slack, 2), 35, ...
                [0.85 0.55 0.10], 'filled', 'MarkerFaceAlpha', 0.30, ...
                'DisplayName', 'absorbed-eps slack');
        end
    end

    if any(sd.id_slack)
        scatter(sd.distpars(sd.id_slack, 1), sd.distpars(sd.id_slack, 2), 25, ...
            [0.10 0.65 0.40], 'filled', 'MarkerFaceAlpha', 0.55, ...
            'DisplayName', 'separate-slack (proper)');
    end
    if any(sd.id_baseline)
        scatter(sd.distpars(sd.id_baseline, 1), sd.distpars(sd.id_baseline, 2), 18, ...
            [0.15 0.50 0.85], 'filled', 'DisplayName', 'baseline (no slack)');
    end

    xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 13);
    title(sprintf('Seller %d ($T=%d$)', sd.player_id, sd.T), ...
        'Interpreter', 'latex', 'FontSize', 12);
    legend('Interpreter', 'none', 'Location', 'best', 'FontSize', 9);
    grid on; box on;
end
sgtitle('Application Stage III: baseline vs absorbed vs separate-slack', ...
    'Interpreter', 'none', 'FontSize', 13);

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'application_slack');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'application_separate_slack_id_sets.png'));
saveas(fig, fullfile(fig_dir, 'application_separate_slack_id_sets.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'application_separate_slack_id_sets.png'));

fprintf('\n=== Done. ===\n');
