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
n_grid_cost = 1000;     % candidate cost grid resolution per seller (firmed up: 1000)
alpha_set = 0.05;
alpha_psi = 0.005;      % first-stage budget for psi-hat coverage (rest goes to regret)
alpha_R   = alpha_set - alpha_psi;
I_inf     = numel(players);   % # sellers needing simultaneous coverage (=2 for top-2)
L_offline = 80000;             % offline sample size for psi-hat (all-sellers data ~250k;
                              %  top-15 ~80k; conservative pick = 80k)
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
    % In residual-price units, prices span [P_l, P_h].  Extended on both ends
    % so identified intervals are interior at the firmed-up resolution.
    diff_p_l = P_h - P_l;
    cost_lo_grid = P_l - 1.5 * diff_p_l;   % well below the lowest price
    cost_hi_grid = P_h + 1.0 * diff_p_l;   % above the highest price
    cost_grid = linspace(cost_lo_grid, cost_hi_grid, n_grid_cost)';

    %% Per-candidate regret + eps + feasibility
    R              = nan(n_grid_cost, 1);
    kappa          = nan(n_grid_cost, 1);    % K^path: joint-payoff range over (a, a_-i) at c
    K_psi          = nan(n_grid_cost, 1);    % K^psi: range of expected payoffs JL over a
    Phi_filt       = nan(n_grid_cost, 1);    % gap-dep, with 0.1*K threshold filter
    Phi_full       = nan(n_grid_cost, 1);    % gap-dep, no threshold (numerical-zero only) — UNSTABLE
    Phi_stab       = nan(n_grid_cost, 1);    % gap-dep, threshold 1e-3*K — stable against gap singularities
    eps_R1         = nan(n_grid_cost, 1);    % adversarial Hedge: K*sqrt(2 ln|A|/T)/alpha
    eps_RM         = nan(n_grid_cost, 1);    % RM full-feedback (Niccolò Note 2): K*sqrt((|A|-1)/T)/alpha
    eps_RM_psi     = nan(n_grid_cost, 1);    % RM with K^psi (Note 2 §5.2)
    eps_RM_sim     = nan(n_grid_cost, 1);    % RM with simultaneous coverage (Note 2 §3): factor |I_inf|
    eps_R3_filt    = nan(n_grid_cost, 1);    % stoch full, gap-dep, with 0.1K filter
    eps_R3_full    = nan(n_grid_cost, 1);    % stoch full, gap-dep, no filter (Note 1 §9 literal)
    eps_R3_stab    = nan(n_grid_cost, 1);    % stoch full, gap-dep, 1e-3K filter (HEADLINE)
    eps_R3_cap     = nan(n_grid_cost, 1);    % min(R3_full, R1_Hedge) — cap at adversarial bound
    eps_R3_full_fs = nan(n_grid_cost, 1);    % R3-full + first-stage psi-hat term (Note 1 §10)
    eps_Hoeff      = nan(n_grid_cost, 1);    % stoch full, Hoeffding (no gap dep), 1/sqrt(N)
    feas_R1         = false(n_grid_cost, 1);
    feas_RM         = false(n_grid_cost, 1);
    feas_RM_psi     = false(n_grid_cost, 1);
    feas_RM_sim     = false(n_grid_cost, 1);
    feas_R3_filt    = false(n_grid_cost, 1);
    feas_R3_full    = false(n_grid_cost, 1);
    feas_R3_stab    = false(n_grid_cost, 1);
    feas_R3_cap     = false(n_grid_cost, 1);
    feas_R3_full_fs = false(n_grid_cost, 1);
    feas_Hoeff      = false(n_grid_cost, 1);

    log_factor_Hoeff = sqrt(log(2 * cfg.NAct / alpha_set) / (2 * T));

    % First-stage Hoeffding-L1 slack on psi-hat (Note 1 §10, Weissman et al. 2003):
    %   delta_psi = sqrt(2 * log((2^S - 2)/alpha_psi) / L)  where S = |comp states|
    % Conservative L1 multinomial concentration; alpha-budget alpha_psi.
    S_psi = cfg.NAct;     % # of competitor states (= 5 here)
    delta_psi = sqrt(2 * log(max(2^S_psi - 2, 2) / alpha_psi) / L_offline);
    fprintf('  alpha_R = %.4f, alpha_psi = %.4f; L_offline = %d, delta_psi = %.5f\n', ...
        alpha_R, alpha_psi, L_offline, delta_psi);

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

        % Phi WITHOUT filter (Niccolò's literal formula: any positive gap) — UNSTABLE
        if K_c > 0
            positive = Delta_c > 1e-12 * K_c;   % numerical zero only
            Phi_full_c = K_c * sum(1 ./ Delta_c(positive));
        else
            Phi_full_c = 0;
        end

        % Phi STABILIZED with small threshold 1e-3 * K (HEADLINE)
        % Excludes near-tie numerical artifacts but keeps real gaps.
        if K_c > 0
            stable = Delta_c > 1e-3 * K_c;
            Phi_stab_c = K_c * sum(1 ./ Delta_c(stable));
        else
            Phi_stab_c = 0;
        end

        Phi_filt(k) = Phi_filt_c;
        Phi_full(k) = Phi_full_c;
        Phi_stab(k) = Phi_stab_c;
        K_psi(k)    = K_c;

        % All eps options (cf. Niccolò Notes 1+2; alpha = alpha_R + alpha_psi):
        %   R1 (Hedge):            K^path * sqrt(2 ln |A|/T) / alpha_R
        %   RM (Note 2 §6):         K^path * sqrt((|A|-1)/T) / alpha_R           [seller-by-seller]
        %   RM_psi (Note 2 §5.2):   K^psi  * sqrt((|A|-1)/T) / alpha_R           [if RM updates with E_psi]
        %   RM_sim (Note 2 §3):     |I_inf| * K^path * sqrt((|A|-1)/T) / alpha_R  [simultaneous on |I| sellers]
        %   R3-full (Note 1 §9):    Phi_full / (alpha_R * T)                     [HEADLINE under stoch full]
        %   R3-full+fs (Note 1 §10): R3-full + K^psi * delta_psi                  [adds first-stage psi-hat]
        %   Hoeff:                 K^psi * sqrt(log(2|A|/alpha)/(2T))            [stoch, no gap-dep]
        eps_R1(k)         = Kk * sqrt(2 * log(cfg.NAct) / T) / alpha_R;
        eps_RM(k)         = Kk * sqrt((cfg.NAct - 1) / T) / alpha_R;
        eps_RM_psi(k)     = K_c * sqrt((cfg.NAct - 1) / T) / alpha_R;
        eps_RM_sim(k)     = I_inf * Kk * sqrt((cfg.NAct - 1) / T) / alpha_R;
        eps_R3_filt(k)    = Phi_filt_c / (alpha_R * T);
        eps_R3_full(k)    = Phi_full_c / (alpha_R * T);
        eps_R3_stab(k)    = Phi_stab_c / (alpha_R * T);
        eps_R3_cap(k)     = min(eps_R3_full(k), eps_R1(k));
        eps_R3_full_fs(k) = eps_R3_full(k) + K_c * delta_psi;
        eps_Hoeff(k)      = K_c * log_factor_Hoeff;

        feas_R1(k)         = (Rk <= eps_R1(k));
        feas_RM(k)         = (Rk <= eps_RM(k));
        feas_RM_psi(k)     = (Rk <= eps_RM_psi(k));
        feas_RM_sim(k)     = (Rk <= eps_RM_sim(k));
        feas_R3_filt(k)    = (Rk <= eps_R3_filt(k));
        feas_R3_full(k)    = (Rk <= eps_R3_full(k));
        feas_R3_stab(k)    = (Rk <= eps_R3_stab(k));
        feas_R3_cap(k)     = (Rk <= eps_R3_cap(k));
        feas_R3_full_fs(k) = (Rk <= eps_R3_full_fs(k));
        feas_Hoeff(k)      = (Rk <= eps_Hoeff(k));
    end
    fprintf('  Regret eval (%d candidates): %.2fs\n', n_grid_cost, toc(t_loop));

    %% Summary
    fmt_interval = @(feas) ifempty_str(any(feas), ...
        sprintf('[%.2f, %.2f]', min(cost_grid(feas)), max(cost_grid(feas))), '[empty]');
    fprintf('  R1 (adv. Hedge,  K^path)         : %4d/%d, c in %s\n', ...
        sum(feas_R1), n_grid_cost, fmt_interval(feas_R1));
    fprintf('  RM (Note 2, K^path, sb-s)        : %4d/%d, c in %s\n', ...
        sum(feas_RM), n_grid_cost, fmt_interval(feas_RM));
    fprintf('  RM (Note 2, K^psi,  sb-s)        : %4d/%d, c in %s\n', ...
        sum(feas_RM_psi), n_grid_cost, fmt_interval(feas_RM_psi));
    fprintf('  RM (Note 2, K^path, simul |I|=%d) : %4d/%d, c in %s\n', ...
        I_inf, sum(feas_RM_sim), n_grid_cost, fmt_interval(feas_RM_sim));
    fprintf('  R3-stab (gap-dep, 1e-3K filter, HEADLINE) : %4d/%d, c in %s\n', ...
        sum(feas_R3_stab), n_grid_cost, fmt_interval(feas_R3_stab));
    fprintf('  R3-cap (min(R3_full, Hedge))     : %4d/%d, c in %s\n', ...
        sum(feas_R3_cap), n_grid_cost, fmt_interval(feas_R3_cap));
    fprintf('  R3-full (numerical zero only)    : %4d/%d, c in %s\n', ...
        sum(feas_R3_full), n_grid_cost, fmt_interval(feas_R3_full));
    fprintf('  R3-full + first-stage (§10)      : %4d/%d, c in %s\n', ...
        sum(feas_R3_full_fs), n_grid_cost, fmt_interval(feas_R3_full_fs));
    fprintf('  R3-filt (gap-dep, 0.1K filter)   : %4d/%d, c in %s\n', ...
        sum(feas_R3_filt), n_grid_cost, fmt_interval(feas_R3_filt));
    fprintf('  Hoeff (stoch, no gap-dep)        : %4d/%d, c in %s\n', ...
        sum(feas_Hoeff), n_grid_cost, fmt_interval(feas_Hoeff));

    % Identified intervals helper
    [cost_lo_R1, cost_hi_R1]                 = interval_of(feas_R1, cost_grid);
    [cost_lo_RM, cost_hi_RM]                 = interval_of(feas_RM, cost_grid);
    [cost_lo_RM_psi, cost_hi_RM_psi]         = interval_of(feas_RM_psi, cost_grid);
    [cost_lo_RM_sim, cost_hi_RM_sim]         = interval_of(feas_RM_sim, cost_grid);
    [cost_lo_R3_filt, cost_hi_R3_filt]       = interval_of(feas_R3_filt, cost_grid);
    [cost_lo_R3_full, cost_hi_R3_full]       = interval_of(feas_R3_full, cost_grid);
    [cost_lo_R3_stab, cost_hi_R3_stab]       = interval_of(feas_R3_stab, cost_grid);
    [cost_lo_R3_cap,  cost_hi_R3_cap]        = interval_of(feas_R3_cap,  cost_grid);
    [cost_lo_R3_full_fs, cost_hi_R3_full_fs] = interval_of(feas_R3_full_fs, cost_grid);
    [cost_lo_Hoeff, cost_hi_Hoeff]           = interval_of(feas_Hoeff, cost_grid);

    % Argmin of R(c) — best-fit cost, point summary
    [~, idx_argmin] = min(R);
    cost_argmin = cost_grid(idx_argmin);
    R_min = R(idx_argmin);
    fprintf('  argmin R(c) = %.3f (R = %.4f)\n', cost_argmin, R_min);

    %% Pack
    results.seller(idx).player_id       = player_id;
    results.seller(idx).T               = T;
    results.seller(idx).cfg_AA          = AA;
    results.seller(idx).cost_grid       = cost_grid;
    results.seller(idx).R               = R;
    results.seller(idx).kappa           = kappa;
    results.seller(idx).K_psi           = K_psi;
    results.seller(idx).Phi_filt        = Phi_filt;
    results.seller(idx).Phi_full        = Phi_full;
    results.seller(idx).Phi_stab        = Phi_stab;
    results.seller(idx).delta_psi       = delta_psi;
    results.seller(idx).eps_R1          = eps_R1;
    results.seller(idx).eps_RM          = eps_RM;
    results.seller(idx).eps_RM_psi      = eps_RM_psi;
    results.seller(idx).eps_RM_sim      = eps_RM_sim;
    results.seller(idx).eps_R3_filt     = eps_R3_filt;
    results.seller(idx).eps_R3_full     = eps_R3_full;
    results.seller(idx).eps_R3_stab     = eps_R3_stab;
    results.seller(idx).eps_R3_cap      = eps_R3_cap;
    results.seller(idx).eps_R3_full_fs  = eps_R3_full_fs;
    results.seller(idx).eps_Hoeff       = eps_Hoeff;
    results.seller(idx).feas_R1         = feas_R1;
    results.seller(idx).feas_RM         = feas_RM;
    results.seller(idx).feas_RM_psi     = feas_RM_psi;
    results.seller(idx).feas_RM_sim     = feas_RM_sim;
    results.seller(idx).feas_R3_filt    = feas_R3_filt;
    results.seller(idx).feas_R3_full    = feas_R3_full;
    results.seller(idx).feas_R3_stab    = feas_R3_stab;
    results.seller(idx).feas_R3_cap     = feas_R3_cap;
    results.seller(idx).feas_R3_full_fs = feas_R3_full_fs;
    results.seller(idx).feas_Hoeff      = feas_Hoeff;
    results.seller(idx).cost_lo_R1         = cost_lo_R1;         results.seller(idx).cost_hi_R1         = cost_hi_R1;
    results.seller(idx).cost_lo_RM         = cost_lo_RM;         results.seller(idx).cost_hi_RM         = cost_hi_RM;
    results.seller(idx).cost_lo_RM_psi     = cost_lo_RM_psi;     results.seller(idx).cost_hi_RM_psi     = cost_hi_RM_psi;
    results.seller(idx).cost_lo_RM_sim     = cost_lo_RM_sim;     results.seller(idx).cost_hi_RM_sim     = cost_hi_RM_sim;
    results.seller(idx).cost_lo_R3_filt    = cost_lo_R3_filt;    results.seller(idx).cost_hi_R3_filt    = cost_hi_R3_filt;
    results.seller(idx).cost_lo_R3_full    = cost_lo_R3_full;    results.seller(idx).cost_hi_R3_full    = cost_hi_R3_full;
    results.seller(idx).cost_lo_R3_stab    = cost_lo_R3_stab;    results.seller(idx).cost_hi_R3_stab    = cost_hi_R3_stab;
    results.seller(idx).cost_lo_R3_cap     = cost_lo_R3_cap;     results.seller(idx).cost_hi_R3_cap     = cost_hi_R3_cap;
    results.seller(idx).cost_lo_R3_full_fs = cost_lo_R3_full_fs; results.seller(idx).cost_hi_R3_full_fs = cost_hi_R3_full_fs;
    results.seller(idx).cost_lo_Hoeff      = cost_lo_Hoeff;      results.seller(idx).cost_hi_Hoeff      = cost_hi_Hoeff;
    results.seller(idx).cost_argmin = cost_argmin;
    results.seller(idx).R_min       = R_min;
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_fixedcost');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
results.aggregation_mode = aggregation_mode;
results.alpha_R    = alpha_R;
results.alpha_psi  = alpha_psi;
results.I_inf      = I_inf;
results.L_offline  = L_offline;
out_path = fullfile(out_dir, ['results_application_fixedcost' file_suffix '.mat']);
save(out_path, '-struct', 'results');
fprintf('\nSaved %s\n', out_path);

%% Headline summary table — for response letter
fprintf('\n========== HEADLINE SUMMARY (alpha=%.3f, alpha_R=%.3f, alpha_psi=%.3f, |I_inf|=%d) ==========\n', ...
    alpha_set, alpha_R, alpha_psi, I_inf);
fprintf('%-7s %-32s %-25s %-7s\n', 'Seller', 'Spec', 'Identified set [c]', 'Range');
fprintf('%-7s %-32s %-25s %-7s\n', '------', '----', '------------------', '-----');
fmt_iv = @(lo, hi) ifempty_str(~isnan(lo), sprintf('[%.2f, %.2f]', lo, hi), 'EMPTY');
fmt_rg = @(lo, hi) ifempty_str(~isnan(lo), sprintf('%.2f', hi - lo), '-');
for idx = 1:numel(players)
    s = results.seller(idx);
    pid = s.player_id;
    fprintf('%-7d %-34s %-25s %-7s\n', pid, 'R3-stab (HEADLINE, 1/N w/ thresh)', fmt_iv(s.cost_lo_R3_stab,    s.cost_hi_R3_stab),    fmt_rg(s.cost_lo_R3_stab,    s.cost_hi_R3_stab));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'R3-cap (= min(R3_full, Hedge))',   fmt_iv(s.cost_lo_R3_cap,     s.cost_hi_R3_cap),     fmt_rg(s.cost_lo_R3_cap,     s.cost_hi_R3_cap));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'R3-full (Niccolò literal)',         fmt_iv(s.cost_lo_R3_full,    s.cost_hi_R3_full),    fmt_rg(s.cost_lo_R3_full,    s.cost_hi_R3_full));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'R3-full + first-stage',             fmt_iv(s.cost_lo_R3_full_fs, s.cost_hi_R3_full_fs), fmt_rg(s.cost_lo_R3_full_fs, s.cost_hi_R3_full_fs));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'RM (Note 2, K^path, sb-s)',         fmt_iv(s.cost_lo_RM,         s.cost_hi_RM),         fmt_rg(s.cost_lo_RM,         s.cost_hi_RM));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'RM K^psi (Note 2 §5.2)',            fmt_iv(s.cost_lo_RM_psi,     s.cost_hi_RM_psi),     fmt_rg(s.cost_lo_RM_psi,     s.cost_hi_RM_psi));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  sprintf('RM simul |I|=%d', I_inf),   fmt_iv(s.cost_lo_RM_sim,     s.cost_hi_RM_sim),     fmt_rg(s.cost_lo_RM_sim,     s.cost_hi_RM_sim));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'R1 Hedge (adv. full)',              fmt_iv(s.cost_lo_R1,         s.cost_hi_R1),         fmt_rg(s.cost_lo_R1,         s.cost_hi_R1));
    fprintf('%-7s %-34s %-25s %-7s\n', '',  'Hoeff (no gap-dep)',                fmt_iv(s.cost_lo_Hoeff,      s.cost_hi_Hoeff),      fmt_rg(s.cost_lo_Hoeff,      s.cost_hi_Hoeff));
    fprintf('%-7s %-34s c=%.2f, R_min=%.4f\n', '', 'argmin R(c) (point summary)', s.cost_argmin, s.R_min);
    fprintf('\n');
end

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
    % Plot R(c) and the key eps lines on log y (skip RM_psi, RM_sim, R3_filt for clarity)
    h_R   = plot(s.cost_grid, s.R,               '-',  'LineWidth', 2.4, 'Color', [0.20 0.55 0.85]);
    h_e1  = plot(s.cost_grid, s.eps_R1,          ':',  'LineWidth', 1.5, 'Color', [0.50 0.50 0.50]);
    h_eRM = plot(s.cost_grid, s.eps_RM,          ':',  'LineWidth', 1.5, 'Color', [0.30 0.30 0.55]);
    h_e3F = plot(s.cost_grid, s.eps_R3_full,     '--', 'LineWidth', 2.0, 'Color', [0.85 0.30 0.10]);
    h_e3Ffs = plot(s.cost_grid, s.eps_R3_full_fs,'-.', 'LineWidth', 1.7, 'Color', [0.85 0.55 0.10]);
    h_eH  = plot(s.cost_grid, s.eps_Hoeff,       '--', 'LineWidth', 1.5, 'Color', [0.10 0.60 0.30]);

    % Mark argmin of R(c)
    plot(s.cost_argmin, s.R_min, 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', [0.95 0.85 0.20], 'MarkerEdgeColor', [0.6 0 0], 'LineWidth', 1.2);

    set(gca, 'YScale', 'log');
    ylim_floor = max(min(s.eps_R3_full) * 0.5, 1e-4);
    ylim([ylim_floor, max([s.R; s.eps_RM]) * 1.5]);

    xlabel('candidate $c$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('regret / $\varepsilon$ (log)', 'Interpreter', 'latex', 'FontSize', 12);
    title(sprintf('Seller %d ($T = %d$): $\\arg\\min R = %.2f$, $R_{\\min} = %.3f$', ...
        s.player_id, s.T, s.cost_argmin, s.R_min), ...
        'Interpreter', 'latex', 'FontSize', 11);
    legend([h_R, h_e1, h_eRM, h_e3F, h_e3Ffs, h_eH], ...
        {'$R(c)$', ...
         '$\varepsilon_{R1}$ (Hedge, adv full)', ...
         '$\varepsilon_{RM}$ (Note~2, RM full)', ...
         '$\varepsilon_{R3,\mathrm{full}}$ (Note~1 \S9, HEADLINE)', ...
         '$\varepsilon_{R3,\mathrm{full}}+ \mathrm{fs}$ (Note~1 \S10)', ...
         '$\varepsilon_{\mathrm{Hoeff}}$ (no gap-dep)'}, ...
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
