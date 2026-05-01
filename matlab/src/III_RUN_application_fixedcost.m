%% III_RUN_application_fixedcost
%
%  Application-stage fixed-cost (data-only / NST-style) identification.
%
%  Per seller, identifies the seller's marginal cost as the set of candidate
%  costs c such that the regret of the seller's empirical action distribution
%  m_N under c is bounded by the per-candidate Hedge bound.
%
%  Differs from III_MAIN_Estim_Application_PrefSpec (parametric (mu, sigma^2)
%  identification) by treating each seller as having a single fixed cost
%  realization rather than a cost distribution.  Conceptually closer to NST
%  2015 regret-rationalizability — see JPE Revision/litreview/NST_2015.md
%  in the paper repo.
%
%  Output: per-seller, identified-cost interval and grid feasibility.

% Allow caller to set aggregation_mode = 'avg' (default) or 'minprice'.
% Caller sets it in workspace before calling this script; we leave it alone.
if ~exist('aggregation_mode', 'var')
    aggregation_mode = 'avg';
end
clc; close all;

%% Setup
paths = df_repo_paths();
rng(20260430);

if strcmp(aggregation_mode, 'minprice')
    Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1_minprice.xlsx');
    Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1_minprice.xlsx');
    file_suffix = '_minprice';
    fprintf('\n[Aggregation mode: MIN-OF-OTHERS competing price]\n');
else
    aggregation_mode = 'avg';
    Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
    Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');
    file_suffix = '';
    fprintf('\n[Aggregation mode: AVG-OF-OTHERS competing price]\n');
end
players = 1:2;          % match parametric spec; expand to 1:15 for full set
n_grid_cost = 200;      % candidate cost grid resolution per seller
alpha_set = 0.05;
% Type space (n_types) only used by game_application to build the (cfg) struct;
% irrelevant to the data-only test, which uses the empirical action distribution
% directly.
n_types_dummy = 5;

%% Per-seller loop
results = struct();
results.players = players;
results.n_grid_cost = n_grid_cost;
results.alpha_set = alpha_set;

for idx = 1:numel(players)
    player_id = players(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d\n', player_id);

    % Load seller config (gives action grid, sale prob, m_N, T)
    cfg = df.setup.game_application(player_id, Dist_file, Prob_file, n_types_dummy);
    T = cfg.maxiters;
    AA = cfg.AA;
    P_l = AA(1);
    P_h = AA(end);
    fprintf('  T = %d, prices = [%s]\n', T, sprintf('%.2f ', AA));
    fprintf('  m_N support: %d/%d profiles\n', sum(cfg.distrib > 0), cfg.NActPr);

    %% Cost candidate grid
    % In residual-price units, prices span [P_l, P_h].  The seller's marginal
    % cost (also in residual-price units) plausibly extends from somewhat
    % below P_l down to ~2*P_l or even further.  Use a wide range so the
    % R(c) minimum is interior.
    diff_p_l = P_h - P_l;
    cost_lo_grid = P_l - 1.0 * diff_p_l;   % well below the lowest price
    cost_hi_grid = P_h + 0.0 * diff_p_l;   % up to the highest price
    cost_grid = linspace(cost_lo_grid, cost_hi_grid, n_grid_cost)';

    %% Per-candidate regret + eps + feasibility
    R           = nan(n_grid_cost, 1);
    kappa       = nan(n_grid_cost, 1);    % payoff range over (a, a_-i) at c
    Phi_filt    = nan(n_grid_cost, 1);    % gap-dep, with 0.1*K threshold filter
    Phi_full    = nan(n_grid_cost, 1);    % gap-dep, without threshold (Niccolo's true formula)
    eps_R1      = nan(n_grid_cost, 1);    % adversarial Hedge
    eps_R3_filt = nan(n_grid_cost, 1);    % stoch full, gap-dep, with filter (existing)
    eps_R3_full = nan(n_grid_cost, 1);    % stoch full, gap-dep, no filter
    eps_Hoeff   = nan(n_grid_cost, 1);    % stoch full, Hoeffding (no gap dep), 1/sqrt(N)
    feas_R1      = false(n_grid_cost, 1);
    feas_R3_filt = false(n_grid_cost, 1);
    feas_R3_full = false(n_grid_cost, 1);
    feas_Hoeff   = false(n_grid_cost, 1);

    log_factor_Hoeff = sqrt(log(2 * cfg.NAct / alpha_set) / (2 * T));

    t_loop = tic;
    for k = 1:n_grid_cost
        c = cost_grid(k);

        % Regret of m_N at candidate cost c
        [Rk, Kk] = df.solvers.compute_dataonly_regret_app(cfg, c);
        R(k)     = Rk;
        kappa(k) = Kk;

        % Compute mu_hat per own-action under candidate cost
        Pi_c    = cfg.prob .* (cfg.A(:, 1) - c);                     % NActPr x 1
        JL_c    = kron(eye(cfg.NAct), cfg.marg_mean') * Pi_c;        % NAct x 1
        Delta_c = abs(max(JL_c) - JL_c);                             % gaps from optimal
        K_c     = max(Delta_c);                                      % payoff range under c

        % Phi WITH filter (matches switch_eps==9 numerical convention)
        if K_c > 0
            nontrivial = Delta_c > 0.1 * K_c;
            Phi_filt_c = K_c * sum(1 ./ Delta_c(nontrivial));
        else
            Phi_filt_c = 0;
        end

        % Phi WITHOUT filter (Niccolò's literal formula: any positive gap)
        if K_c > 0
            positive = Delta_c > 1e-12 * K_c;   % numerical zero only
            Phi_full_c = K_c * sum(1 ./ Delta_c(positive));
        else
            Phi_full_c = 0;
        end

        Phi_filt(k) = Phi_filt_c;
        Phi_full(k) = Phi_full_c;

        % Four eps options
        eps_R1(k)      = Kk * sqrt(2 * log(cfg.NAct) / T) / alpha_set;     % adversarial
        eps_R3_filt(k) = Phi_filt_c / (alpha_set * T);                     % stoch gap-dep, with filter
        eps_R3_full(k) = Phi_full_c / (alpha_set * T);                     % stoch gap-dep, no filter
        eps_Hoeff(k)   = K_c * log_factor_Hoeff;                           % stoch Hoeffding, no gap dep

        feas_R1(k)      = (Rk <= eps_R1(k));
        feas_R3_filt(k) = (Rk <= eps_R3_filt(k));
        feas_R3_full(k) = (Rk <= eps_R3_full(k));
        feas_Hoeff(k)   = (Rk <= eps_Hoeff(k));
    end
    fprintf('  Regret eval (%d candidates): %.2fs\n', n_grid_cost, toc(t_loop));

    %% Summary
    fmt_interval = @(feas) ifempty_str(any(feas), ...
        sprintf('[%.2f, %.2f]', min(cost_grid(feas)), max(cost_grid(feas))), '[empty]');
    fprintf('  R1 (adversarial Hedge)        : %3d/%d, c in %s\n', ...
        sum(feas_R1), n_grid_cost, fmt_interval(feas_R1));
    fprintf('  R3 (stoch gap-dep, w/ filter) : %3d/%d, c in %s\n', ...
        sum(feas_R3_filt), n_grid_cost, fmt_interval(feas_R3_filt));
    fprintf('  R3-full (stoch gap-dep, none) : %3d/%d, c in %s\n', ...
        sum(feas_R3_full), n_grid_cost, fmt_interval(feas_R3_full));
    fprintf('  Hoeff (stoch, no gap dep)     : %3d/%d, c in %s\n', ...
        sum(feas_Hoeff), n_grid_cost, fmt_interval(feas_Hoeff));

    % Identified intervals helper
    [cost_lo_R1, cost_hi_R1]           = interval_of(feas_R1, cost_grid);
    [cost_lo_R3_filt, cost_hi_R3_filt] = interval_of(feas_R3_filt, cost_grid);
    [cost_lo_R3_full, cost_hi_R3_full] = interval_of(feas_R3_full, cost_grid);
    [cost_lo_Hoeff, cost_hi_Hoeff]     = interval_of(feas_Hoeff, cost_grid);

    % Argmin of R(c) — best-fit cost, point summary
    [~, idx_argmin] = min(R);
    cost_argmin = cost_grid(idx_argmin);
    R_min = R(idx_argmin);
    fprintf('  argmin R(c) = %.3f (R = %.4f)\n', cost_argmin, R_min);

    %% Pack
    results.seller(idx).player_id    = player_id;
    results.seller(idx).T            = T;
    results.seller(idx).cfg_AA       = AA;
    results.seller(idx).cost_grid    = cost_grid;
    results.seller(idx).R            = R;
    results.seller(idx).kappa        = kappa;
    results.seller(idx).Phi_filt     = Phi_filt;
    results.seller(idx).Phi_full     = Phi_full;
    results.seller(idx).eps_R1       = eps_R1;
    results.seller(idx).eps_R3_filt  = eps_R3_filt;
    results.seller(idx).eps_R3_full  = eps_R3_full;
    results.seller(idx).eps_Hoeff    = eps_Hoeff;
    results.seller(idx).feas_R1      = feas_R1;
    results.seller(idx).feas_R3_filt = feas_R3_filt;
    results.seller(idx).feas_R3_full = feas_R3_full;
    results.seller(idx).feas_Hoeff   = feas_Hoeff;
    results.seller(idx).cost_lo_R1      = cost_lo_R1;       results.seller(idx).cost_hi_R1      = cost_hi_R1;
    results.seller(idx).cost_lo_R3_filt = cost_lo_R3_filt;  results.seller(idx).cost_hi_R3_filt = cost_hi_R3_filt;
    results.seller(idx).cost_lo_R3_full = cost_lo_R3_full;  results.seller(idx).cost_hi_R3_full = cost_hi_R3_full;
    results.seller(idx).cost_lo_Hoeff   = cost_lo_Hoeff;    results.seller(idx).cost_hi_Hoeff   = cost_hi_Hoeff;
    results.seller(idx).cost_argmin = cost_argmin;
    results.seller(idx).R_min       = R_min;
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_fixedcost');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
results.aggregation_mode = aggregation_mode;
out_path = fullfile(out_dir, ['results_application_fixedcost' file_suffix '.mat']);
save(out_path, '-struct', 'results');
fprintf('\nSaved %s\n', out_path);

%% Per-seller diagnostic figure: regret vs candidate cost, with eps overlay
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'application_fixedcost');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

n_sellers = numel(players);
ncols = min(2, n_sellers);
nrows = ceil(n_sellers / ncols);

fig = figure('Color', 'w', 'Position', [60 60 900*ncols 500*nrows]);
for idx = 1:n_sellers
    subplot(nrows, ncols, idx);
    s = results.seller(idx);
    hold on;
    % Plot R(c) and four eps lines on a log y-axis (huge range)
    h_R   = plot(s.cost_grid, s.R,            '-',  'LineWidth', 2.4, 'Color', [0.20 0.55 0.85]);
    h_e1  = plot(s.cost_grid, s.eps_R1,       ':',  'LineWidth', 1.5, 'Color', [0.50 0.50 0.50]);
    h_e3f = plot(s.cost_grid, s.eps_R3_filt,  '--', 'LineWidth', 1.5, 'Color', [0.85 0.30 0.10]);
    h_e3F = plot(s.cost_grid, s.eps_R3_full,  '--', 'LineWidth', 1.5, 'Color', [0.85 0.55 0.10]);
    h_eH  = plot(s.cost_grid, s.eps_Hoeff,    '--', 'LineWidth', 1.8, 'Color', [0.10 0.60 0.30]);

    % Mark argmin of R(c)
    plot(s.cost_argmin, s.R_min, 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', [0.95 0.85 0.20], 'MarkerEdgeColor', [0.6 0 0], 'LineWidth', 1.2);

    set(gca, 'YScale', 'log');
    ylim_floor = max(min([s.eps_R3_filt; s.eps_R3_full]) * 0.5, 1e-4);
    ylim([ylim_floor, max([s.R; s.eps_R1]) * 1.5]);

    xlabel('candidate $c$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('regret / $\varepsilon$ (log)', 'Interpreter', 'latex', 'FontSize', 12);
    title(sprintf('Seller %d ($T = %d$): $\\arg\\min R = %.2f$, $R_{\\min} = %.3f$', ...
        s.player_id, s.T, s.cost_argmin, s.R_min), ...
        'Interpreter', 'latex', 'FontSize', 11);
    legend([h_R, h_e1, h_e3f, h_e3F, h_eH], ...
        {'$R(c)$', ...
         '$\varepsilon_{R1}$ (adv. Hedge)', ...
         '$\varepsilon_{R3,\mathrm{filt}}$ (stoch gap-dep, $\Delta>0.1K$)', ...
         '$\varepsilon_{R3,\mathrm{full}}$ (stoch gap-dep, $\Delta>0$)', ...
         '$\varepsilon_{\mathrm{Hoeff}}$ (stoch, no gap dep)'}, ...
        'Interpreter', 'latex', 'Location', 'south', 'FontSize', 8);
    grid on; box on;
    set(gca, 'FontSize', 10);
    hold off;
end
sgtitle('Application fixed-cost: $R(c)$ vs four $\varepsilon$-radii (log-scale)', ...
    'Interpreter', 'latex', 'FontSize', 13);
saveas(fig, fullfile(fig_dir, ['application_fixedcost_R_vs_cost' file_suffix '.png']));
saveas(fig, fullfile(fig_dir, ['application_fixedcost_R_vs_cost' file_suffix '.pdf']));
fprintf('Saved diagnostic figure.\n');

fprintf('\n=== Done. ===\n');

function v = ifempty(x, fallback)
    if isempty(x) || (isscalar(x) && isnan(x)), v = fallback; else, v = x; end
end

function s = ifempty_str(cond, str_yes, str_no)
    if cond, s = str_yes; else, s = str_no; end
end

function [lo, hi] = interval_of(feas, cost_grid)
    if any(feas)
        lo = min(cost_grid(feas));
        hi = max(cost_grid(feas));
    else
        lo = NaN; hi = NaN;
    end
end
