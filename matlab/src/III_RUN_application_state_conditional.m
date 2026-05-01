%% III_RUN_application_state_conditional
%
%  Path B': fixed-cost spec with PER-STATE external regret.  See
%  JPE Revision/Application_DoF_Register.md (Path B' section).
%
%  For each candidate cost c and each competitor state s ∈ {1,...,5}:
%      R_s(c) = max_{a' ∈ A}  Σ_a m_N(a | comp=s) * [π(a', s; c) - π(a, s; c)]
%
%  Test: max_s R_s(c) ≤ eps_state(s, c)  for each s.
%
%  Per-state Hedge eps options:
%      R1 adversarial Hedge:    K(s,c) · √(2 ln|A| / N_s) / α
%      Hoeffding stochastic:    K(s,c) · √(log(2|A|/α) / (2 N_s))
%
%  No LP — closed-form per (c, s, a').  Microseconds per candidate.
%
%  Compares to:
%   - existing fixed-cost (state-marginalized) test (rejects everything)
%   - parametric Stage III intervals (in companion runs)

clear all; clc; close all;

paths = df_repo_paths();
rng(20260430);

Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');
players      = 1:2;             % focal sellers
n_grid_cost  = 200;
alpha_set    = 0.05;
n_types_dummy = 5;              % cfg.type_space dimension; not used substantively

results = struct();
results.players = players;
results.n_grid_cost = n_grid_cost;
results.alpha_set = alpha_set;

for idx = 1:numel(players)
    iii = players(idx);
    fprintf('\n========================================\n');
    fprintf('SELLER %d (state-conditional fixed-cost)\n', iii);
    fprintf('========================================\n');

    cfg = df.setup.game_application(iii, Dist_file, Prob_file, n_types_dummy);
    T   = cfg.maxiters;
    A   = cfg.A;       % NActPr × 2  joint action profiles
    AA  = cfg.AA;      % NAct × 1    own actions
    prob = cfg.prob;   % NActPr × 1  sale prob per joint profile
    distrib = cfg.distrib(:);  % NActPr × 1  m_N over joint profiles
    NActPr = size(A, 1);
    NAct   = size(AA, 1);
    P_l = AA(1); P_h = AA(end);

    fprintf('  T = %d, |A| = %d, prices = [%s]\n', T, NAct, sprintf('%.2f ', AA));

    %% Build action-profile lookup: idx_lookup(self_idx, comp_idx) -> joint profile row
    idx_lookup = zeros(NAct, NAct);
    for k = 1:NActPr
        a1_idx = find(AA == A(k, 1), 1);
        a2_idx = find(AA == A(k, 2), 1);
        idx_lookup(a1_idx, a2_idx) = k;
    end

    %% Per-state competitor probability and conditional m_N(a | s)
    % m_N(comp=s) = Σ_a m_N(a, s)
    % m_N(a | s) = m_N(a, s) / m_N(comp=s)
    m_comp = zeros(NAct, 1);
    m_a_given_s = zeros(NAct, NAct);    % rows = own action, cols = comp state
    for s_idx = 1:NAct
        for self_idx = 1:NAct
            prof = idx_lookup(self_idx, s_idx);
            m_a_given_s(self_idx, s_idx) = distrib(prof);
        end
        m_comp(s_idx) = sum(m_a_given_s(:, s_idx));
        if m_comp(s_idx) > 0
            m_a_given_s(:, s_idx) = m_a_given_s(:, s_idx) / m_comp(s_idx);
        end
    end
    N_s = T * m_comp;
    fprintf('  m_comp distribution: [%s]\n', sprintf('%.4f ', m_comp));
    fprintf('  N_s (per state):     [%s]\n', sprintf('%6.0f ', N_s));

    %% Cost candidate grid (wide, matching previous fixedcost runs)
    diff_p = P_h - P_l;
    cost_grid = linspace(P_l - 1.0 * diff_p, P_h + 0.0 * diff_p, n_grid_cost)';

    %% Per-candidate, per-state regret + per-state eps options
    R_max         = nan(n_grid_cost, 1);            % max over states of R_s(c)
    R_per_state   = nan(n_grid_cost, NAct);         % R_s(c) for each s
    K_per_state_c = nan(n_grid_cost, NAct);         % κ(s, c) per (c, s)

    eps_R1_state    = nan(n_grid_cost, NAct);
    eps_Hoeff_state = nan(n_grid_cost, NAct);
    feas_R1         = false(n_grid_cost, 1);
    feas_Hoeff      = false(n_grid_cost, 1);

    log_factor_Hoeff_perN = sqrt(log(2 * NAct / alpha_set) / 2);   % factor; divide by √N_s per state

    t_loop = tic;
    for k = 1:n_grid_cost
        c = cost_grid(k);

        for s_idx = 1:NAct
            if N_s(s_idx) < 1
                % no obs at this state — skip
                R_per_state(k, s_idx)   = NaN;
                K_per_state_c(k, s_idx) = NaN;
                continue
            end

            % Compute per-state expected payoffs for each own action.
            mu_per_action = zeros(NAct, 1);
            for self_idx = 1:NAct
                prof = idx_lookup(self_idx, s_idx);
                a_self = AA(self_idx);
                mu_per_action(self_idx) = prob(prof) * (a_self - c);
            end

            % R_s(c) = max_{a'} mu(a') - Σ_a m_N(a|s) mu(a)
            mean_payoff = sum(m_a_given_s(:, s_idx) .* mu_per_action);
            R_s_c = max(mu_per_action) - mean_payoff;
            R_per_state(k, s_idx) = R_s_c;

            K_s_c = max(mu_per_action) - min(mu_per_action);
            K_per_state_c(k, s_idx) = K_s_c;

            eps_R1_state(k, s_idx)    = K_s_c * sqrt(2 * log(NAct) / N_s(s_idx)) / alpha_set;
            eps_Hoeff_state(k, s_idx) = K_s_c * log_factor_Hoeff_perN / sqrt(N_s(s_idx));
        end

        R_max(k) = max(R_per_state(k, :), [], 'omitnan');

        % Per-state pass: all states must satisfy
        valid_states = ~isnan(R_per_state(k, :)) & ~isnan(eps_R1_state(k, :));
        feas_R1(k)    = all(R_per_state(k, valid_states) <= eps_R1_state(k, valid_states));
        feas_Hoeff(k) = all(R_per_state(k, valid_states) <= eps_Hoeff_state(k, valid_states));
    end
    fprintf('  Loop time: %.2fs\n', toc(t_loop));

    %% Summary
    fmt_interval = @(feas) ifempty_str(any(feas), ...
        sprintf('[%.2f, %.2f]', min(cost_grid(feas)), max(cost_grid(feas))), '[empty]');
    fprintf('\n  R1 state-cond (adv. Hedge):  %3d/%d feasible, c in %s\n', ...
        sum(feas_R1), n_grid_cost, fmt_interval(feas_R1));
    fprintf('  Hoeff state-cond (stoch):    %3d/%d feasible, c in %s\n', ...
        sum(feas_Hoeff), n_grid_cost, fmt_interval(feas_Hoeff));

    [~, k_argmin] = min(R_max);
    fprintf('  argmin max_s R_s(c) = %.3f (R_max = %.4f)\n', cost_grid(k_argmin), R_max(k_argmin));

    %% Pack
    results.seller(idx).player_id      = iii;
    results.seller(idx).T              = T;
    results.seller(idx).cost_grid      = cost_grid;
    results.seller(idx).m_comp         = m_comp;
    results.seller(idx).N_s            = N_s;
    results.seller(idx).R_max          = R_max;
    results.seller(idx).R_per_state    = R_per_state;
    results.seller(idx).K_per_state_c  = K_per_state_c;
    results.seller(idx).eps_R1_state    = eps_R1_state;
    results.seller(idx).eps_Hoeff_state = eps_Hoeff_state;
    results.seller(idx).feas_R1        = feas_R1;
    results.seller(idx).feas_Hoeff     = feas_Hoeff;
    results.seller(idx).cost_argmin    = cost_grid(k_argmin);
    results.seller(idx).R_min          = R_max(k_argmin);
end

%% Save
out_dir = fullfile(paths.matlab_root, 'output', 'application_state_conditional');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_state_conditional.mat'), '-struct', 'results');
fprintf('\nSaved %s\n', fullfile(out_dir, 'results_state_conditional.mat'));

%% Figure: R_max(c) and eps_state(c) per seller; identified intervals marked
fig_dir = fullfile(paths.matlab_root, 'output', 'figures', 'application_state_conditional');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
n_pl = numel(players);
fig = figure('Color', 'w', 'Position', [60 60 850 * n_pl 600]);
for idx = 1:n_pl
    subplot(1, n_pl, idx);
    sd = results.seller(idx);
    hold on;

    % R_max(c) = the BINDING per-state regret
    h_Rmax = plot(sd.cost_grid, sd.R_max, '-', 'LineWidth', 2.4, 'Color', [0.20 0.55 0.85]);

    % "Effective" eps for the binding state at each c: eps at the state where R is max
    [~, binding_state] = max(sd.R_per_state, [], 2);
    eps_R1_binding   = arrayfun(@(k) sd.eps_R1_state(k, binding_state(k)),    1:n_grid_cost)';
    eps_Hoeff_binding = arrayfun(@(k) sd.eps_Hoeff_state(k, binding_state(k)), 1:n_grid_cost)';
    h_e1 = plot(sd.cost_grid, eps_R1_binding,    '--', 'LineWidth', 1.8, 'Color', [0.85 0.30 0.10]);
    h_eH = plot(sd.cost_grid, eps_Hoeff_binding, '--', 'LineWidth', 1.8, 'Color', [0.10 0.60 0.30]);

    plot(sd.cost_argmin, sd.R_min, 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', [0.95 0.85 0.20], 'MarkerEdgeColor', [0.6 0 0], 'LineWidth', 1.2);

    % Mark identified intervals via xline
    if any(sd.feas_R1)
        xline(min(sd.cost_grid(sd.feas_R1)), 'Color', [0.85 0.30 0.10], 'LineWidth', 1.5, 'LineStyle', '-.');
        xline(max(sd.cost_grid(sd.feas_R1)), 'Color', [0.85 0.30 0.10], 'LineWidth', 1.5, 'LineStyle', '-.');
    end
    if any(sd.feas_Hoeff)
        xline(min(sd.cost_grid(sd.feas_Hoeff)), 'Color', [0.10 0.60 0.30], 'LineWidth', 1.5, 'LineStyle', ':');
        xline(max(sd.cost_grid(sd.feas_Hoeff)), 'Color', [0.10 0.60 0.30], 'LineWidth', 1.5, 'LineStyle', ':');
    end

    set(gca, 'YScale', 'log');
    ylim_floor = max(min(eps_Hoeff_binding) * 0.5, 1e-3);
    ylim([ylim_floor, max([sd.R_max; eps_R1_binding]) * 1.5]);

    xlabel('candidate $c$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$\max_s R_s(c)$ / $\varepsilon_{\mathrm{state}}$ (binding state)', 'Interpreter', 'latex', 'FontSize', 11);
    title(sprintf('Seller %d ($T=%d$): argmin %.2f, $R_{\\min}=%.3f$', ...
        sd.player_id, sd.T, sd.cost_argmin, sd.R_min), ...
        'Interpreter', 'latex', 'FontSize', 11);
    legend([h_Rmax, h_e1, h_eH], ...
        {'$\max_s R_s(c)$', ...
         '$\varepsilon_{R1,\mathrm{state}}$ (binding)', ...
         '$\varepsilon_{\mathrm{Hoeff,state}}$ (binding)'}, ...
        'Interpreter', 'latex', 'Location', 'best', 'FontSize', 9);
    grid on; box on;
    hold off;
end
sgtitle('Application Path B$^\prime$: state-conditional fixed-cost identification', ...
    'Interpreter', 'latex', 'FontSize', 13);
saveas(fig, fullfile(fig_dir, 'state_conditional_R_vs_cost.png'));
saveas(fig, fullfile(fig_dir, 'state_conditional_R_vs_cost.pdf'));
fprintf('Saved figure to %s\n', fullfile(fig_dir, 'state_conditional_R_vs_cost.png'));

fprintf('\n=== Done. ===\n');

function s = ifempty_str(cond, str_yes, str_no)
    if cond, s = str_yes; else, s = str_no; end
end
