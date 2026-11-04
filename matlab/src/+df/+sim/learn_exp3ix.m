function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn_exp3ix(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN_EXP3IX  Contextual EXP3-IX in the stabilized bandit family B.
%
%   Same signature and outputs as df.sim.learn. One learner copy per (player, type)
%   with a local clock. The learner observes only its realized profit. At the s-th
%   visit of type t, player j normalizes the realized profit u to a loss
%       ell = (max Pi(:,t,j) - u) / K_t  in [0, 1],
%   with K_t = max Pi(:,t,j) - min Pi(:,t,j) the known width of the type's payoff
%   interval, forms the implicit-exploration estimate
%       ell_hat(a) = ell * 1{a = A_s} / (p(a) + gamma_s),
%   and mixes p proportional to exp(-eta_s * sum of past ell_hat) from a uniform
%   prior, with
%       eta_s = sqrt(ln|A| / (|A| s)),   gamma_s = eta_s / 2.
%   This is the family of Niccolo's V05 manuscript (2026-11-02, Appendix "A
%   Stabilized Bandit Family", eqs. s4IX, s4banditupdate, s4banditbands) with
%   schedule bands a_- = a_+ = 1, b_- = b_+ = 1/2 (so A = 3), C_pr = ln|A|, L = K_t.
%
%   The first N periods are uniformly random play, as in df.sim.learn (Stage II uses
%   N = 1); with zero estimates the family's first visit is also uniform, and the
%   period-1 loss updates the estimate, so the run is the family's initialized run.
%   Full counterfactual payoffs are accumulated for the regret outputs only; the
%   learner never uses them.
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space, .sigma2, .mu, .Pi

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
Pi        = cfg.Pi;

nAct = size(AA, 1);
nProf = size(A, 1);
s = size(type_space{1,1}, 1);
lnA = log(nAct);

type_vals = cell(NPlayers, 1);
for j = 1:NPlayers
    type_vals{j} = type_space{j,1};
end

if NPlayers == 2
    action_profile_map = zeros(nAct, nAct);
    for k = 1:nProf
        a1_idx = find(AA == A(k,1), 1);
        a2_idx = find(AA == A(k,2), 1);
        action_profile_map(a1_idx, a2_idx) = k;
    end
    use_2d_map = true;
else
    use_2d_map = false;
end

record_times = round(M * (1:numdst_t) / numdst_t);
record_map = zeros(N + M, 1);
for k = 1:numdst_t
    t_rec = record_times(k);
    if t_rec >= 1 && t_rec <= N + M
        record_map(t_rec) = k;
    end
end
record_times_obs = round(M_obs * (1:numdst_t_obs) / numdst_t_obs);
obs_start = N + M - M_obs;
record_map_obs = zeros(N + M, 1);
for k = 1:numdst_t_obs
    t_abs = obs_start + record_times_obs(k);
    if t_abs >= 1 && t_abs <= N + M
        record_map_obs(t_abs) = k;
    end
end

sigma2 = cfg.sigma2;
mu_val = cfg.mu;
if NPlayers == 2 && size(sigma2, 1) == 2
    combs = [kron(ones(size(type_vals{2})), type_vals{1}), ...
             kron(type_vals{2}, ones(size(type_vals{1})))];
    md1 = pdf('normal', type_vals{1}, mu_val(1), sqrt(sigma2(1,1))); md1 = md1 / sum(md1);
    md2 = pdf('normal', type_vals{2}, mu_val(2), sqrt(sigma2(2,2))); md2 = md2 / sum(md2);
    joint_distrib = kron(md2, md1);
    n_combs = size(combs, 1);
    combs_type_idx = zeros(n_combs, NPlayers);
    for k = 1:n_combs
        combs_type_idx(k, 1) = find(type_vals{1} == combs(k, 1), 1);
        combs_type_idx(k, 2) = find(type_vals{2} == combs(k, 2), 1);
    end
    comb_indices = (1:n_combs)';
end

% Known payoff interval per (type, player): [umin, umax] of width K_t.
umax = zeros(s, NPlayers); Kt = zeros(s, NPlayers);
for j = 1:NPlayers
    umax(:, j) = max(Pi(:, :, j), [], 1)';
    Kt(:, j) = (max(Pi(:, :, j), [], 1) - min(Pi(:, :, j), [], 1))';
end
Kt = max(Kt, 1e-12);

cum_util    = zeros(s, NPlayers);
cum_cf_util = zeros(nAct, s, NPlayers);      % diagnostics only
Shat        = zeros(nAct, s, NPlayers);      % cumulative IX loss estimates
visits      = zeros(s, NPlayers);

distY_time     = zeros(nProf, numdst_t);
distY_time_obs = zeros(nProf, numdst_t_obs);
action_counts  = zeros(nProf, 1);
final_regret   = zeros(s, NPlayers);
Pl1_EmpRegr    = zeros(s, 1);
Pl2_EmpRegr    = zeros(s, 1);

exp_alpha_AA = exp(alpha_val * AA);
p_cur = zeros(nAct, NPlayers); a_cur = zeros(1, NPlayers); gam_cur = zeros(1, NPlayers);

for t = 1:(N + M)
    if NPlayers == 2 && size(sigma2, 1) == 2
        mc_idx = randsample(comb_indices, 1, true, joint_distrib);
        mc_draw = combs(mc_idx, :);
        type_indices = combs_type_idx(mc_idx, :);
    else
        mc_draw = marginal_cost_draws_v4_new(cfg, type_space, 1);
        type_indices = zeros(1, NPlayers);
        for j = 1:NPlayers
            type_indices(j) = find(type_vals{j} == mc_draw(j), 1);
        end
    end

    actions = zeros(1, NPlayers);
    for j = 1:NPlayers
        type_j = type_indices(j);
        visits(type_j, j) = visits(type_j, j) + 1;
        s_loc = visits(type_j, j);
        eta = sqrt(lnA / (nAct * s_loc));
        gam = eta / 2;
        if t <= N
            p = ones(nAct, 1) / nAct;                          % random play phase
        else
            logits = -eta * Shat(:, type_j, j);
            logits = logits - max(logits);
            p = exp(logits); p = p / sum(p);
        end
        u = rand;
        a_idx = find(cumsum(p) >= u, 1);
        if isempty(a_idx), a_idx = nAct; end
        actions(j) = AA(a_idx);
        p_cur(:, j) = p; a_cur(j) = a_idx; gam_cur(j) = gam;
    end

    exp_alpha_actions = exp(alpha_val * actions);
    sum_exp_all = sum(exp_alpha_actions);
    for j = 1:NPlayers
        type_j = type_indices(j);
        denom_own = 1 + sum_exp_all;
        cp_own = exp_alpha_actions(j) / denom_own;
        utility = cp_own * (actions(j) - mc_draw(j));
        cum_util(type_j, j) = cum_util(type_j, j) + utility;
        % Bandit update: loss of the chosen action only, implicit-exploration estimate.
        loss = (umax(type_j, j) - utility) / Kt(type_j, j);
        loss = min(max(loss, 0), 1);
        a_idx = a_cur(j);
        Shat(a_idx, type_j, j) = Shat(a_idx, type_j, j) + loss / (p_cur(a_idx, j) + gam_cur(j));
        % Diagnostics: full counterfactual vector (not observed by the learner).
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        denom_cf = sum_others + exp_alpha_AA;
        cp_cf = exp_alpha_AA ./ denom_cf;
        cf_utilities = cp_cf .* (AA - mc_draw(j));
        cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + cf_utilities;
    end

    if use_2d_map
        a1_idx = find(AA == actions(1), 1);
        a2_idx = find(AA == actions(2), 1);
        prof_idx = action_profile_map(a1_idx, a2_idx);
    else
        prof_idx = find(all(A == actions, 2), 1);
    end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;

    col = record_map(t);
    if col > 0, distY_time(:, col) = action_counts / t; end
    col_obs = record_map_obs(t);
    if col_obs > 0, distY_time_obs(:, col_obs) = action_counts / t; end

    if t == N + M
        final_regret = local_regret(cum_util, cum_cf_util, t);
    elseif t == N + Nobs_Pl1
        R = local_regret(cum_util, cum_cf_util, t); Pl1_EmpRegr = R(:, 1);
    elseif t == N + Nobs_Pl2
        R = local_regret(cum_util, cum_cf_util, t); Pl2_EmpRegr = R(:, 2);
    end
end
end

function R = local_regret(cum_util, cum_cf_util, t)
    best = squeeze(max(cum_cf_util, [], 1));
    if isvector(best) && size(cum_util, 2) == 1, best = best(:); end
    R = max((best - cum_util) / t, 0);
end
