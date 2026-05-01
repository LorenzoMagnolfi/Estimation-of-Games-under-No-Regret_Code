%% III_RUN_application_slack
%
%  Parametric Stage III (varying-cost, BCCE LP, marginal mode) with
%  CONSISTENCY-CONVERGENCE adjustment to the obedience radius.
%
%  Per Niccolò's note (Sec. 6.2): with separate-slack, alpha = alpha_R + alpha_C.
%  The "absorbed" version here uses
%      eps_total(t) = eps_regret(t; alpha_R) + K(t) * r_N(alpha_C, T)
%  where r_N is the box (Hoeffding per-coord) consistency slack.
%  Niccolò flags this as suboptimal vs separate-slack LP relaxation, but
%  it's a fast first-pass that uses the existing LP infrastructure.
%
%  Compares baseline (switch_eps=9, no slack) against consistency-inflated eps.

clear all; clc; close all;

paths = df_repo_paths();
rng(20260430);

Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');
players      = 1:2;       % top-2 sellers (matches existing III_MAIN_Estim_Application_PrefSpec)
NGridV       = 50;        % halved from production 100 for first-pass speed
NGridM       = 50;
n_types      = 5;
alpha_set    = 0.05;
alpha_C      = 0.005;     % consistency budget (10% of alpha)
alpha_R      = alpha_set - alpha_C;
switch_eps   = 9;

NGrid = NGridV * NGridM;

results = struct();
results.players = players;
results.NGridV = NGridV;
results.NGridM = NGridM;
results.alpha_set = alpha_set;
results.alpha_C = alpha_C;
results.alpha_R = alpha_R;

for idx = 1:numel(players)
    iii = players(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d\n', iii);
    fprintf('========================================\n');

    %% Build per-player config
    cfg = df.setup.game_application(iii, Dist_file, Prob_file, n_types);
    T   = cfg.maxiters;
    s   = n_types;
    NAct = cfg.NAct;
    fprintf('  T = %d, s = %d, |A| = %d\n', T, s, NAct);

    action_distribution = cfg.distrib';

    %% Build (mu, sigma^2) grid (matches Identification_Pricing_Game_ApplicationL)
    mu = cfg.mu; sigma2 = cfg.sigma2;
    gridparamV = linspace(0.1 * sigma2(1,1), sigma2(1,1) * 5, NGridV)';
    gridparamM = linspace(mu(1,1) * 4, mu(1,1) * 0.25, NGridM)';
    [distpars, distribution_parameters] = df.report.build_param_grid(mu(1,1), sigma2, gridparamM, gridparamV);

    %% Eps choices
    kwargs = struct('mode', 'application', 'marg_mean', cfg.marg_mean, 's_override', s);
    eps_baseline = df.solvers.compute_epsilon(cfg, T, alpha_set, switch_eps, kwargs);
    eps_regret_R = df.solvers.compute_epsilon(cfg, T, alpha_R,  switch_eps, kwargs);

    % Box consistency slack (per Niccolò Sec. 6.2 with Hoeffding per-coord).
    %   r_N = sqrt(log(2 s / alpha_C) / (2 T))
    r_N = sqrt(log(2 * s / alpha_C) / (2 * T));

    % Per-type Kappa (max-min payoff range over actions, at each type).
    K_per_type = max(cfg.Pi(:,:,1)) - min(cfg.Pi(:,:,1));   % 1 x s

    eps_consistency = K_per_type * r_N;
    eps_total       = eps_regret_R + eps_consistency;

    fprintf('  eps_baseline (switch=9, alpha=%.3f)   = [%s]\n', alpha_set, sprintf('%.4f ', eps_baseline));
    fprintf('  eps_regret_R (switch=9, alpha_R=%.3f) = [%s]\n', alpha_R,   sprintf('%.4f ', eps_regret_R));
    fprintf('  K_per_type                                = [%s]\n', sprintf('%.3f ', K_per_type));
    fprintf('  r_N (box, alpha_C=%.4f, T=%d)         = %.5f\n', alpha_C, T, r_N);
    fprintf('  eps_consistency = K * r_N                = [%s]\n', sprintf('%.4f ', eps_consistency));
    fprintf('  eps_total = eps_regret_R + eps_consistency = [%s]\n', sprintf('%.4f ', eps_total));
    fprintf('  Mean inflation factor (total / baseline)  = %.2f\n', mean(eps_total ./ eps_baseline));

    %% Marginal-mode LP setup (mirrors ComputeBCCE_eps_ApplicationL)
    marg_act_I  = kron(eye(NAct), ones(1, NAct)) * action_distribution;
    marg_act_II = kron(ones(1, NAct), eye(NAct)) * action_distribution;
    opts_lp = struct('marginal', true, 'switch_eps', switch_eps, ...
        'solver', 'sedumi', 'precision', 'default', ...
        'marg_act_distrib_I',  marg_act_I, ...
        'marg_act_distrib_II', marg_act_II);

    %% Baseline solve (existing eps)
    fprintf('\n  [Baseline] solving %d grid points with switch_eps=9, alpha=%.3f ...\n', NGrid, alpha_set);
    t_b = tic;
    g_baseline = df.solvers.solve_bcce(cfg.type_space, cfg.action_space, action_distribution, ...
        cfg.Pi, distribution_parameters, cfg.Pi, struct('mode','switch','eps',eps_baseline), opts_lp);
    fprintf('  [Baseline] done in %.1fs\n', toc(t_b));
    VV_baseline = squeeze(g_baseline);
    id_baseline = (VV_baseline <= 1e-12);

    %% Slack-inflated solve
    fprintf('\n  [Slack] solving %d grid points with eps_total ...\n', NGrid);
    t_s = tic;
    g_slack = df.solvers.solve_bcce(cfg.type_space, cfg.action_space, action_distribution, ...
        cfg.Pi, distribution_parameters, cfg.Pi, struct('mode','switch','eps',eps_total), opts_lp);
    fprintf('  [Slack] done in %.1fs\n', toc(t_s));
    VV_slack = squeeze(g_slack);
    id_slack = (VV_slack <= 1e-12);

    %% Report intervals
    fprintf('\n  Baseline (no slack):     %4d/%4d feasible', sum(id_baseline), NGrid);
    if any(id_baseline)
        ddpb = distpars(id_baseline, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddpb(:,1)), max(ddpb(:,1)), min(ddpb(:,2)), max(ddpb(:,2)));
    end
    fprintf('\n');
    fprintf('  With consistency slack:  %4d/%4d feasible', sum(id_slack), NGrid);
    if any(id_slack)
        ddps = distpars(id_slack, :);
        fprintf(',  mu in [%.2f, %.2f],  sigma^2 in [%.4f, %.4f]', ...
            min(ddps(:,1)), max(ddps(:,1)), min(ddps(:,2)), max(ddps(:,2)));
    end
    fprintf('\n');

    %% Pack
    results.seller(idx).player_id        = iii;
    results.seller(idx).T                = T;
    results.seller(idx).distpars         = distpars;
    results.seller(idx).VV_baseline      = VV_baseline;
    results.seller(idx).VV_slack         = VV_slack;
    results.seller(idx).id_baseline      = id_baseline;
    results.seller(idx).id_slack         = id_slack;
    results.seller(idx).eps_baseline     = eps_baseline;
    results.seller(idx).eps_regret_R     = eps_regret_R;
    results.seller(idx).eps_consistency  = eps_consistency;
    results.seller(idx).eps_total        = eps_total;
    results.seller(idx).K_per_type       = K_per_type;
    results.seller(idx).r_N              = r_N;
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_slack');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_application_slack.mat'), '-struct', 'results');
fprintf('\nSaved %s\n', fullfile(out_dir, 'results_application_slack.mat'));

%% Figure: identified sets baseline vs slack
fig = figure('Color', 'w', 'Position', [60 60 800 * numel(players) 600]);
for idx = 1:numel(players)
    subplot(1, numel(players), idx);
    sd = results.seller(idx);
    hold on;
    if any(sd.id_slack)
        scatter(sd.distpars(sd.id_slack, 1), sd.distpars(sd.id_slack, 2), 30, ...
            [0.85 0.55 0.10], 'filled', 'MarkerFaceAlpha', 0.45);
    end
    if any(sd.id_baseline)
        scatter(sd.distpars(sd.id_baseline, 1), sd.distpars(sd.id_baseline, 2), 18, ...
            [0.15 0.50 0.85], 'filled');
    end
    xlabel('$\mu$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\sigma^2$', 'Interpreter', 'latex', 'FontSize', 13);
    title(sprintf('Seller %d ($T=%d$)', sd.player_id, sd.T), ...
        'Interpreter', 'latex', 'FontSize', 12);
    legend({'with consistency slack', 'baseline (no slack)'}, ...
        'Interpreter', 'latex', 'Location', 'best', 'FontSize', 10);
    grid on; box on;
end
sgtitle('Application Stage III: identified sets — baseline vs consistency-inflated $\varepsilon$', ...
    'Interpreter', 'latex', 'FontSize', 13);

fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'application_slack');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(fig, fullfile(fig_dir, 'application_slack_id_sets.png'));
saveas(fig, fullfile(fig_dir, 'application_slack_id_sets.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'application_slack_id_sets.png'));

fprintf('\n=== Done. ===\n');
