function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn_hedge(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN_HEDGE  Contextual Hedge in the full-feedback regularization family F.
%
%   Same signature and outputs as df.sim.learn (regret matching). One learner copy
%   per (player, type) with a local clock. At the s-th visit of type t, player j
%   plays the mixture
%       p(b) proportional to exp( eta_s * S_t(b) / K_t ),   eta_s = sqrt(ln|A| / s),
%   where S_t is the cumulative counterfactual (full-feedback) payoff vector over the
%   previous s-1 visits of that type and K_t = max Pi(:,t,j) - min Pi(:,t,j) is the
%   payoff span of the type. This is follow-the-regularized-leader with relative
%   entropy to the uniform prior (B_reg = ln|A|, sigma = 1) and step multiplier c = 1,
%   a member of the family in Niccolo's V05 manuscript (2026-11-02, Appendix "A
%   Full-Feedback Regularization Family", eq. s4ftrl); its constant is
%   c_F = (1/c + c/4) sqrt(B_reg / sigma) = 1.25 sqrt(ln|A|). Centering the payoff
%   vector by its midrange changes no mixture, so it is omitted.
%
%   The first N periods are uniformly random play, as in df.sim.learn (Stage II
%   uses N = 1). With S_t = 0 the family's first visit is also uniform, so the run is
%   the family's initialized run. Counterfactual payoffs are accumulated from period
%   1 for every realized type, exactly as in df.sim.learn, so the regret outputs
%   are the paper's positive-part average regrets at the same clock.
%
%   Type draws use the same randsample call as df.sim.learn, so for a given RNG
%   seed the cost sequence equals the regret-matching run's; action draws use
%   inverse-cdf sampling.
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space, .sigma2, .mu, .Pi

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
Pi        = cfg.Pi;
% sale-probability scale q (sparse-outcome exhibit): profits and the normalizer Kt (from the
% q-scaled Pi) must carry the same q, so that the step multiplier stays c = 1 (finding I4,
% Niccolò, 4 November 2026; before this fix the effective multiplier was 1/q).
q_scale = 1; if isfield(cfg, 'sale_scale') && ~isempty(cfg.sale_scale), q_scale = cfg.sale_scale; end

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

% Payoff span per (type, player): the family's normalization constant K_t.
Kt = zeros(s, NPlayers);
for j = 1:NPlayers
    Kt(:, j) = (max(Pi(:, :, j), [], 1) - min(Pi(:, :, j), [], 1))';
end
Kt = max(Kt, 1e-12);

cum_util    = zeros(s, NPlayers);
cum_cf_util = zeros(nAct, s, NPlayers);
visits      = zeros(s, NPlayers);           % local clock per (type, player)

distY_time     = zeros(nProf, numdst_t);
distY_time_obs = zeros(nProf, numdst_t_obs);
action_counts  = zeros(nProf, 1);
final_regret   = zeros(s, NPlayers);
Pl1_EmpRegr    = zeros(s, 1);
Pl2_EmpRegr    = zeros(s, 1);

exp_alpha_AA = exp(alpha_val * AA);

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
        if t <= N
            a_idx = randi(nAct);                               % random play phase
        else
            s_loc = visits(type_j, j);
            eta = sqrt(lnA / s_loc);
            logits = eta * cum_cf_util(:, type_j, j) / Kt(type_j, j);
            logits = logits - max(logits);
            p = exp(logits); p = p / sum(p);
            u = rand;
            a_idx = find(cumsum(p) >= u, 1);
            if isempty(a_idx), a_idx = nAct; end
        end
        actions(j) = AA(a_idx);
    end

    exp_alpha_actions = exp(alpha_val * actions);
    sum_exp_all = sum(exp_alpha_actions);
    for j = 1:NPlayers
        type_j = type_indices(j);
        denom_own = 1 + sum_exp_all;
        cp_own = exp_alpha_actions(j) / denom_own;
        utility = q_scale * cp_own * (actions(j) - mc_draw(j));
        cum_util(type_j, j) = cum_util(type_j, j) + utility;
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        denom_cf = sum_others + exp_alpha_AA;
        cp_cf = exp_alpha_AA ./ denom_cf;
        cf_utilities = q_scale * cp_cf .* (AA - mc_draw(j));
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
% Positive part of the average gain of the best fixed action, per (type, player),
% averaged over the total horizon t (the paper's R_N(i, t_i)).
    best = squeeze(max(cum_cf_util, [], 1));       % s x NPlayers
    if isvector(best) && size(cum_util, 2) == 1, best = best(:); end
    R = max((best - cum_util) / t, 0);
end
