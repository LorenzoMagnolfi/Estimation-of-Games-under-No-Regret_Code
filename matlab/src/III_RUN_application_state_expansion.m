%% III_RUN_application_state_expansion
%
%  Path C1.A from Application_State_Expansion_Plan.md.
%
%  Fixed-cost spec with state-conditional regret on a RICHER STATE:
%      s_rich = (comp_bin, nc_bin)   |s_rich| = 5 x 4 = 20
%  where nc_bin is a 4-bin quartile of num_competitor.
%
%  Period payoff per cell:
%      π(a, s_rich; c) = (price_a - c) · q(a, s_rich)
%
%  Per-state external regret:
%      R_{s_rich}(c) = max_{a'} Σ_a m_N(a | s_rich) [π(a', s_rich; c) - π(a, s_rich; c)]
%
%  Test: max_{s_rich} R_{s_rich}(c) ≤ ε_{s_rich}(c) for all rich states.
%  Two ε flavors: R1 (Hedge adversarial) and Hoeffding stochastic.
%
%  Inputs (built by python/build_rich_state_data.py):
%      sale_probability_5bins_res1_rich.xlsx
%      SellerDistribution_15_sellers_res1_rich.xlsx

clear all; clc; close all;

paths = df_repo_paths();
rng(20260430);

Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1_rich.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1_rich.xlsx');
Dist_file_orig = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');  % for action prices

players      = 1:2;
n_grid_cost  = 200;
alpha_set    = 0.05;
NAct         = 5;
N_NC_BINS    = 4;
N_RICH       = NAct * N_NC_BINS;     % 20

% n_listings per seller (precomputed via python audit on .dta).  Used for
% the (b.light)-style fixed-cost variant that treats listings (not
% listing-days) as the independent unit of observation.
n_listings_per_seller = [1512, 1695];

results = struct();
results.players      = players;
results.n_grid_cost  = n_grid_cost;
results.alpha_set    = alpha_set;
results.NAct         = NAct;
results.N_NC_BINS    = N_NC_BINS;
results.N_RICH       = N_RICH;

%% Load action prices (median) from the original SellerDistribution file
opts_act = spreadsheetImportOptions("NumVariables", NAct);
opts_act.Sheet      = "Actions_Median";
opts_act.DataRange  = "A2:E2";
opts_act.VariableTypes = repmat({'double'}, 1, NAct);
actions = table2array(readtable(Dist_file_orig, opts_act, "UseExcel", false));
fprintf('Action prices (median per self bin): [%s]\n', sprintf('%.2f ', actions));

%% Load rich-state sale probability (Allsellers sheet)
% Columns: self_net_price_bins_1, comp_net_price_bins_1, nc_bin, n_obs, Sale_Prob
% (all 0-indexed for the bin columns)
opts_sp = spreadsheetImportOptions("NumVariables", 5);
opts_sp.Sheet     = "Allsellers";
opts_sp.DataRange = "A2:E101";   % up to 100 cells max
opts_sp.VariableNames = {'self_bin','comp_bin','nc_bin','n_obs','Sale_Prob'};
opts_sp.VariableTypes = {'double','double','double','double','double'};
sp_tbl = readtable(Prob_file, opts_sp, "UseExcel", false);
sp_tbl = sp_tbl(~isnan(sp_tbl.Sale_Prob), :);
fprintf('Loaded %d sale-prob cells (Allsellers, rich)\n', height(sp_tbl));

% Build q_rich: (NAct, N_RICH) matrix of sale probabilities
%   rich_state index r = comp_bin * N_NC_BINS + nc_bin (0-indexed)
q_rich = nan(NAct, N_RICH);
for ii = 1:height(sp_tbl)
    self0 = sp_tbl.self_bin(ii);   % 0..4
    comp0 = sp_tbl.comp_bin(ii);   % 0..4
    nc0   = sp_tbl.nc_bin(ii);     % 0..3
    r0    = comp0 * N_NC_BINS + nc0;          % 0..19
    q_rich(self0+1, r0+1) = sp_tbl.Sale_Prob(ii);
end
% Cells with no obs -> set q to 0 (sale never happens, payoff zero from that)
q_rich(isnan(q_rich)) = 0;
fprintf('q_rich populated cells: %d / %d\n', sum(~isnan(q_rich(:)) & q_rich(:) > 0), NAct * N_RICH);
fprintf('q_rich range: [%.4f, %.4f]\n', min(q_rich(:)), max(q_rich(:)));

for idx = 1:numel(players)
    iii = players(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d (rich-state, fixed-cost)\n', iii);
    fprintf('========================================\n');

    %% Load seller's rich-state cumulative time-average distribution (last row)
    sheet = sprintf('Seller_%d', iii);
    % First, find T (number of rows) by reading the whole sheet
    [NUM_, ~, ~] = xlsread(Dist_file, sheet);
    T = size(NUM_, 1);    % NB: header row not counted in NUM
    fprintf('  T = %d (listing-day obs)\n', T);

    % Now read just the last row.  Columns are TimeAverage_<self>_<rich>
    n_cols = NAct * N_RICH;     % 100
    opts = spreadsheetImportOptions("NumVariables", n_cols);
    opts.Sheet      = sheet;
    opts.DataRange  = sprintf('A%d:%s%d', T+1, xlcolumn(n_cols), T+1);
    opts.VariableTypes = repmat({'double'}, 1, n_cols);
    last_row = table2array(readtable(Dist_file, opts, "UseExcel", false));

    % Reshape to NAct x N_RICH: distrib_rich(self0+1, r0+1) = TimeAverage_{self0}_{r0}
    distrib_rich = reshape(last_row, [N_RICH, NAct])';
    % Sanity check: total should be ~1
    if abs(sum(distrib_rich(:)) - 1) > 1e-3
        warning('distrib_rich does not sum to 1: %.4f', sum(distrib_rich(:)));
    end

    %% Compute m_N(s_rich) = sum over self of joint dist
    %  m_N(a | s_rich) = distrib_rich(a, s_rich) / m_N(s_rich)
    m_rich = sum(distrib_rich, 1)';                 % N_RICH x 1
    N_s    = T * m_rich;                            % per-cell sample @ T (listing-day)
    nL     = n_listings_per_seller(idx);
    N_s_eff = nL * m_rich;                          % per-cell sample @ n_listings (b.light)
    m_a_given_s = distrib_rich;
    for r = 1:N_RICH
        if m_rich(r) > 1e-12
            m_a_given_s(:, r) = m_a_given_s(:, r) / m_rich(r);
        else
            m_a_given_s(:, r) = NaN;
        end
    end
    fprintf('  m_rich (state distribution) min=%.4f, median=%.4f, max=%.4f\n', ...
        min(m_rich), median(m_rich), max(m_rich));
    fprintf('  N_s @ T  (listing-day)       min=%.0f,  median=%.0f,  max=%.0f\n', ...
        min(N_s), median(N_s), max(N_s));
    fprintf('  N_s @ n_listings (b.light)   min=%.0f,  median=%.0f,  max=%.0f  (DEFF=%.1f)\n', ...
        min(N_s_eff), median(N_s_eff), max(N_s_eff), T/nL);

    %% Cost candidate grid: extended on both ends so upper/lower bounds are interior
    P_l = actions(1); P_h = actions(end);
    diff_p = P_h - P_l;
    cost_grid = linspace(P_l - 1.5 * diff_p, P_h + 1.0 * diff_p, n_grid_cost)';

    %% Per-candidate, per-rich-state regret + per-state eps options
    %  We compute eps under TWO sample-size assumptions:
    %    (i)  N_s     = T * m_rich         (listing-day iid)
    %    (ii) N_s_eff = n_listings * m_rich (listing-as-unit, b.light)
    R_max         = nan(n_grid_cost, 1);
    R_per_state   = nan(n_grid_cost, N_RICH);
    K_per_state_c = nan(n_grid_cost, N_RICH);

    eps_R1_state         = nan(n_grid_cost, N_RICH);   % @ T
    eps_Hoeff_state      = nan(n_grid_cost, N_RICH);
    eps_R1_state_nL      = nan(n_grid_cost, N_RICH);   % @ n_listings (b.light)
    eps_Hoeff_state_nL   = nan(n_grid_cost, N_RICH);

    feas_R1            = false(n_grid_cost, 1);
    feas_Hoeff         = false(n_grid_cost, 1);
    feas_R1_nL         = false(n_grid_cost, 1);
    feas_Hoeff_nL      = false(n_grid_cost, 1);

    log_factor_Hoeff_perN = sqrt(log(2 * NAct / alpha_set) / 2);

    t_loop = tic;
    for k = 1:n_grid_cost
        c = cost_grid(k);

        for r = 1:N_RICH
            if N_s(r) < 1 || any(isnan(m_a_given_s(:, r)))
                continue
            end

            % Per-state expected payoffs per own action
            mu_per_action = q_rich(:, r) .* (actions(:) - c);

            mean_payoff = sum(m_a_given_s(:, r) .* mu_per_action);
            R_per_state(k, r) = max(mu_per_action) - mean_payoff;

            K_s_c = max(mu_per_action) - min(mu_per_action);
            K_per_state_c(k, r) = K_s_c;

            eps_R1_state(k, r)    = K_s_c * sqrt(2 * log(NAct) / N_s(r)) / alpha_set;
            eps_Hoeff_state(k, r) = K_s_c * log_factor_Hoeff_perN / sqrt(N_s(r));

            if N_s_eff(r) >= 1
                eps_R1_state_nL(k, r)    = K_s_c * sqrt(2 * log(NAct) / N_s_eff(r)) / alpha_set;
                eps_Hoeff_state_nL(k, r) = K_s_c * log_factor_Hoeff_perN / sqrt(N_s_eff(r));
            end
        end

        R_max(k) = max(R_per_state(k, :), [], 'omitnan');

        valid = ~isnan(R_per_state(k, :)) & ~isnan(eps_R1_state(k, :));
        feas_R1(k)    = all(R_per_state(k, valid) <= eps_R1_state(k, valid));
        feas_Hoeff(k) = all(R_per_state(k, valid) <= eps_Hoeff_state(k, valid));

        valid_nL = ~isnan(R_per_state(k, :)) & ~isnan(eps_R1_state_nL(k, :));
        feas_R1_nL(k)    = all(R_per_state(k, valid_nL) <= eps_R1_state_nL(k, valid_nL));
        feas_Hoeff_nL(k) = all(R_per_state(k, valid_nL) <= eps_Hoeff_state_nL(k, valid_nL));
    end
    fprintf('  Loop time: %.2fs\n', toc(t_loop));

    %% Summary
    fmt_iv = @(feas) ifempty_str(any(feas), ...
        sprintf('[%.2f, %.2f]', min(cost_grid(feas)), max(cost_grid(feas))), '[empty]');
    fprintf('\n  --- @ T (listing-day) ---\n');
    fprintf('  R1 state-cond rich  (Hedge):    %3d/%d feasible, c in %s\n', ...
        sum(feas_R1), n_grid_cost, fmt_iv(feas_R1));
    fprintf('  Hoeff state-cond rich (stoch):  %3d/%d feasible, c in %s\n', ...
        sum(feas_Hoeff), n_grid_cost, fmt_iv(feas_Hoeff));

    fprintf('\n  --- @ n_listings (b.light) ---\n');
    fprintf('  R1 state-cond rich  (Hedge):    %3d/%d feasible, c in %s\n', ...
        sum(feas_R1_nL), n_grid_cost, fmt_iv(feas_R1_nL));
    fprintf('  Hoeff state-cond rich (stoch):  %3d/%d feasible, c in %s\n', ...
        sum(feas_Hoeff_nL), n_grid_cost, fmt_iv(feas_Hoeff_nL));

    [~, k_argmin] = min(R_max);
    fprintf('\n  argmin max_s R_s(c) = %.3f (R_max = %.4f)\n', cost_grid(k_argmin), R_max(k_argmin));

    %% Pack
    results.seller(idx).player_id      = iii;
    results.seller(idx).T              = T;
    results.seller(idx).n_listings     = nL;
    results.seller(idx).cost_grid      = cost_grid;
    results.seller(idx).m_rich         = m_rich;
    results.seller(idx).N_s            = N_s;
    results.seller(idx).N_s_eff        = N_s_eff;
    results.seller(idx).distrib_rich   = distrib_rich;
    results.seller(idx).R_max          = R_max;
    results.seller(idx).R_per_state    = R_per_state;
    results.seller(idx).K_per_state_c  = K_per_state_c;
    results.seller(idx).eps_R1_state         = eps_R1_state;
    results.seller(idx).eps_Hoeff_state      = eps_Hoeff_state;
    results.seller(idx).eps_R1_state_nL      = eps_R1_state_nL;
    results.seller(idx).eps_Hoeff_state_nL   = eps_Hoeff_state_nL;
    results.seller(idx).feas_R1        = feas_R1;
    results.seller(idx).feas_Hoeff     = feas_Hoeff;
    results.seller(idx).feas_R1_nL     = feas_R1_nL;
    results.seller(idx).feas_Hoeff_nL  = feas_Hoeff_nL;
    results.seller(idx).cost_argmin    = cost_grid(k_argmin);
    results.seller(idx).R_min          = R_max(k_argmin);
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_state_expansion');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_state_expansion.mat'), '-struct', 'results');
fprintf('\nSaved %s\n', fullfile(out_dir, 'results_state_expansion.mat'));

%% Compact comparison vs Path B' (state-conditional, comp_bin only)
B_path = fullfile(paths.matlab_root, 'output', 'application_state_conditional', ...
    'results_state_conditional.mat');
if exist(B_path, 'file') == 2
    Bp = load(B_path);
end

iv_str = @(grid, feas) ifempty_str(any(feas), ...
    sprintf('[%.1f,%.1f] (%d/%d)', min(grid(feas)), max(grid(feas)), sum(feas), numel(grid)), ...
    'EMPTY');

fprintf('\n=== Path C1.A: 4-variant identification (R1 Hedge / Hoeffding x T / n_listings) ===\n');
fprintf('%-7s %-22s %-22s %-22s %-22s\n', 'Seller', 'R1 @ T', 'Hoeff @ T', 'R1 @ n_listings', 'Hoeff @ n_listings');
for idx = 1:numel(players)
    sC = results.seller(idx);
    fprintf('  %-5d %-22s %-22s %-22s %-22s\n', sC.player_id, ...
        iv_str(sC.cost_grid, sC.feas_R1), ...
        iv_str(sC.cost_grid, sC.feas_Hoeff), ...
        iv_str(sC.cost_grid, sC.feas_R1_nL), ...
        iv_str(sC.cost_grid, sC.feas_Hoeff_nL));
end

if exist('Bp', 'var')
    fprintf('\nPath B'' (comp_bin only) reference: ');
    for idx = 1:numel(players)
        sB = Bp.seller(idx);
        fprintf('S%d=%s  ', sB.player_id, iv_str(sB.cost_grid, sB.feas_R1));
    end
    fprintf('\n');
end

fprintf('\n=== Done. ===\n');

function s = ifempty_str(cond, str_yes, str_no)
    if cond, s = str_yes; else, s = str_no; end
end

function col = xlcolumn(n)
% xlcolumn(n) -> Excel column letter for a 1-based column index
    n = n - 1;
    if n < 26
        col = char('A' + n);
    else
        col = [char('A' + floor(n/26) - 1), char('A' + mod(n, 26))];
    end
end
