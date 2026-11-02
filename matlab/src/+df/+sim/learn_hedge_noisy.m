function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn_hedge_noisy(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN_HEDGE_NOISY  Contextual Hedge on REALIZED potential profits (sparse-outcome exhibit).
%
%   Same signature and outputs as df.sim.learn_hedge. The structural payoff is the
%   expected profit u(b) = q cp(b, a_{-j}) (b - c), with q = cfg.sale_scale (default 1)
%   a sale-probability scale that makes outcomes sparse (cfg.Pi must already be scaled
%   by q by the caller). Each period the simulation draws, for EVERY own price b, an
%   independent potential sale 1{sale_b} ~ Bernoulli(q cp(b, a_{-j})) and the learner
%   observes the full potential-feedback vector Y(b) = 1{sale_b} (b - c), in cash
%   units. This satisfies the noisy-feedback bridge of V05 (Appendix "Noisy Reward
%   Feedback") by construction: E[Y(b) | pre-noise history] = u(b) for every b, the
%   potential vector is independent of the private action draw, and all coordinates
%   lie in the known interval [-(c-4)_+, (8-c)_+] of width L_t = (8-c)_+ + (c-4)_+.
%   The learner is Hedge (family F) on Y with payoffs scaled by L_t and step
%   sqrt(ln|A| / s) on the type's local clock. Regret outputs use the structural
%   (expected-profit) payoffs, which the learner never sees.
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space, .sigma2, .mu, .Pi,
%   and optionally .sale_scale.

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
q = 1; if isfield(cfg, 'sale_scale') && ~isempty(cfg.sale_scale), q = cfg.sale_scale; end

nAct = size(AA, 1); nProf = size(A, 1); s = size(type_space{1,1}, 1); lnA = log(nAct);
type_vals = cell(NPlayers, 1);
for j = 1:NPlayers, type_vals{j} = type_space{j,1}; end
if NPlayers == 2
    action_profile_map = zeros(nAct, nAct);
    for k = 1:nProf
        action_profile_map(find(AA == A(k,1), 1), find(AA == A(k,2), 1)) = k;
    end
    use_2d_map = true;
else
    use_2d_map = false;
end
record_times = round(M * (1:numdst_t) / numdst_t); record_map = zeros(N + M, 1);
for k = 1:numdst_t, t_rec = record_times(k); if t_rec >= 1 && t_rec <= N + M, record_map(t_rec) = k; end, end
record_times_obs = round(M_obs * (1:numdst_t_obs) / numdst_t_obs); obs_start = N + M - M_obs; record_map_obs = zeros(N + M, 1);
for k = 1:numdst_t_obs, t_abs = obs_start + record_times_obs(k); if t_abs >= 1 && t_abs <= N + M, record_map_obs(t_abs) = k; end, end
sigma2 = cfg.sigma2; mu_val = cfg.mu;
combs = [kron(ones(size(type_vals{2})), type_vals{1}), kron(type_vals{2}, ones(size(type_vals{1})))];
md1 = pdf('normal', type_vals{1}, mu_val(1), sigma2(1,1)); md1 = md1 / sum(md1);
md2 = pdf('normal', type_vals{2}, mu_val(2), sigma2(2,2)); md2 = md2 / sum(md2);
joint_distrib = kron(md2, md1); n_combs = size(combs, 1);
combs_type_idx = zeros(n_combs, NPlayers);
for k = 1:n_combs
    combs_type_idx(k, 1) = find(type_vals{1} == combs(k, 1), 1);
    combs_type_idx(k, 2) = find(type_vals{2} == combs(k, 2), 1);
end
comb_indices = (1:n_combs)';
% Feedback interval width per (type, player): realized profit in [-(c-4)_+, (8-c)_+]
Lt = zeros(s, NPlayers);
for j = 1:NPlayers, c = type_vals{j}; Lt(:, j) = max(max(AA) - c, 0) + max(c - min(AA), 0); end
Lt = max(Lt, 1e-12);

cum_util = zeros(s, NPlayers); cum_cf_util = zeros(nAct, s, NPlayers);   % structural (for regret outputs)
cum_Y = zeros(nAct, s, NPlayers);                                         % learner's cumulative feedback vectors
visits = zeros(s, NPlayers);
distY_time = zeros(nProf, numdst_t); distY_time_obs = zeros(nProf, numdst_t_obs); action_counts = zeros(nProf, 1);
final_regret = zeros(s, NPlayers); Pl1_EmpRegr = zeros(s, 1); Pl2_EmpRegr = zeros(s, 1);
exp_alpha_AA = exp(alpha_val * AA);

for t = 1:(N + M)
    mc_idx = randsample(comb_indices, 1, true, joint_distrib);
    mc_draw = combs(mc_idx, :); type_indices = combs_type_idx(mc_idx, :);
    actions = zeros(1, NPlayers);
    for j = 1:NPlayers
        type_j = type_indices(j); visits(type_j, j) = visits(type_j, j) + 1;
        if t <= N
            a_idx = randi(nAct);
        else
            eta = sqrt(lnA / visits(type_j, j));
            logits = eta * cum_Y(:, type_j, j) / Lt(type_j, j); logits = logits - max(logits);
            p = exp(logits); p = p / sum(p);
            a_idx = find(cumsum(p) >= rand, 1); if isempty(a_idx), a_idx = nAct; end
        end
        actions(j) = AA(a_idx);
    end
    exp_alpha_actions = exp(alpha_val * actions); sum_exp_all = sum(exp_alpha_actions);
    for j = 1:NPlayers
        type_j = type_indices(j);
        cp_own = exp_alpha_actions(j) / (1 + sum_exp_all);
        cum_util(type_j, j) = cum_util(type_j, j) + q * cp_own * (actions(j) - mc_draw(j));
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        cp_cf = exp_alpha_AA ./ (sum_others + exp_alpha_AA);                  % sale probability at every own price
        cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + q * cp_cf .* (AA - mc_draw(j));
        sale = rand(nAct, 1) < q * cp_cf;                                      % potential sales, one per own price
        cum_Y(:, type_j, j) = cum_Y(:, type_j, j) + sale .* (AA - mc_draw(j));
    end
    if use_2d_map, prof_idx = action_profile_map(find(AA == actions(1), 1), find(AA == actions(2), 1));
    else, prof_idx = find(all(A == actions, 2), 1); end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;
    col = record_map(t); if col > 0, distY_time(:, col) = action_counts / t; end
    col_obs = record_map_obs(t); if col_obs > 0, distY_time_obs(:, col_obs) = action_counts / t; end
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
