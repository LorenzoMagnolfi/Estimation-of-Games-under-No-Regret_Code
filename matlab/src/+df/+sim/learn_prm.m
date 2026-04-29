function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn_prm(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN_PRM  Proxy-regret matching simulation (bandit feedback).
%
%   Drop-in replacement for df.sim.learn with identical signature and outputs.
%   Implements proxy-regret matching (Hart & Mas-Colell, 2001): each player
%   only observes their realized profit (bandit feedback) and constructs
%   proxy regrets via importance weighting.
%
%   The proxy regret for player j, type t, action a is updated each round as:
%     proxy_R(a, t, j) += [(u_n / gamma(a_n, t)) * 1{a_n = a} - u_n] * 1{t_n = t}
%   where gamma(a, t) is the current mixing probability, u_n is realized utility,
%   and a_n is the action actually chosen.
%
%   Key differences from standard regret matching (learn.m):
%     1. No counterfactual utility computation — only realized payoff observed
%     2. Importance-weighted proxy regrets replace standard regrets
%     3. Mixing probabilities must be bounded away from 0 (exploration floor)
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space, .sigma2, .mu

% Exploration floor: ensures mixing probs bounded away from 0
% (required for importance-weighted estimator to be well-defined)
GAMMA_FLOOR = 0.01;

% Unpack cfg fields
A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;

nAct = size(AA, 1);
nProf = size(A, 1);
s = size(type_space{1,1}, 1);

% Type-to-index lookup
type_vals = cell(NPlayers, 1);
for j = 1:NPlayers
    type_vals{j} = type_space{j,1};
end

% Action-profile-to-index lookup (2-player optimized)
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

% Recording times
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

% Marginal distribution
sigma2 = cfg.sigma2;
mu_val = cfg.mu;

if NPlayers == 2 && size(sigma2, 1) == 2
    combs = [kron(ones(size(type_vals{2})), type_vals{1}), ...
             kron(type_vals{2}, ones(size(type_vals{1})))];
    md1 = pdf('normal', type_vals{1}, mu_val(1), sigma2(1,1));
    md1 = md1 / sum(md1);
    md2 = pdf('normal', type_vals{2}, mu_val(2), sigma2(2,2));
    md2 = md2 / sum(md2);
    joint_distrib = kron(md2, md1);
    n_combs = size(combs, 1);

    combs_type_idx = zeros(n_combs, NPlayers);
    for k = 1:n_combs
        combs_type_idx(k, 1) = find(type_vals{1} == combs(k, 1), 1);
        combs_type_idx(k, 2) = find(type_vals{2} == combs(k, 2), 1);
    end
    comb_indices = (1:n_combs)';
end

%% Initialize proxy regrets (cumulative)
cum_proxy_regret = zeros(nAct, s, NPlayers);

% Current mixing probabilities (tracked for importance weighting)
gamma = ones(nAct, s, NPlayers) / nAct;  % uniform initially

%% Initialize output trackers
distY_time     = zeros(nProf, numdst_t);
distY_time_obs = zeros(nProf, numdst_t_obs);
action_counts  = zeros(nProf, 1);
final_regret   = zeros(s, NPlayers);
Pl1_EmpRegr    = zeros(s, 1);
Pl2_EmpRegr    = zeros(s, 1);

%% Precompute
exp_alpha_AA = exp(alpha_val * AA);

%% Main simulation loop
for t = 1:(N + M)

    % Draw types
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

    % Choose actions
    if t <= N
        % Random play phase
        action_indices = randi(nProf, 1, 1);
        actions = A(action_indices, :);
    else
        % Proxy-regret matching phase
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            % Get mixing probabilities from proxy regrets
            proxy_reg = max(cum_proxy_regret(:, type_indices(j), j), 0);
            if sum(proxy_reg) == 0
                gamma_j = ones(nAct, 1) / nAct;
            else
                gamma_j = proxy_reg / sum(proxy_reg);
            end

            % Apply exploration floor
            gamma_j = (1 - nAct * GAMMA_FLOOR) * gamma_j + GAMMA_FLOOR;
            gamma(:, type_indices(j), j) = gamma_j;

            % Sample action
            actions(j) = randsample(AA, 1, true, gamma_j);
        end
    end

    % Compute realized utilities (bandit feedback: each player observes ONLY their own)
    exp_alpha_actions = exp(alpha_val * actions);
    sum_exp_all = sum(exp_alpha_actions);

    for j = 1:NPlayers
        type_j = type_indices(j);

        % Realized utility
        denom_own = 1 + sum_exp_all;
        cp_own = exp_alpha_actions(j) / denom_own;
        utility = cp_own * (actions(j) - mc_draw(j));

        % Update proxy regrets (bandit feedback, importance weighted)
        % For each alternative action a:
        %   proxy_R(a) += (u / gamma(a_chosen)) * 1{a_chosen = a} - u
        % This is unbiased for the true regret
        if t > N
            gamma_j = gamma(:, type_j, j);
            a_chosen_idx = find(AA == actions(j), 1);

            for a = 1:nAct
                if a == a_chosen_idx
                    % Importance-weighted reward for chosen action
                    cum_proxy_regret(a, type_j, j) = cum_proxy_regret(a, type_j, j) + ...
                        utility / gamma_j(a) - utility;
                else
                    % Unchosen actions: subtract realized utility
                    cum_proxy_regret(a, type_j, j) = cum_proxy_regret(a, type_j, j) - utility;
                end
            end
        end
    end

    % Update action counts
    if use_2d_map
        a1_idx = find(AA == actions(1), 1);
        a2_idx = find(AA == actions(2), 1);
        prof_idx = action_profile_map(a1_idx, a2_idx);
    else
        prof_idx = find(all(A == actions, 2), 1);
    end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;

    % Record distributions
    col = record_map(t);
    if col > 0
        distY_time(:, col) = action_counts / t;
    end

    col_obs = record_map_obs(t);
    if col_obs > 0
        distY_time_obs(:, col_obs) = action_counts / t;
    end

    % Final regrets
    if t == N + M
        for j = 1:NPlayers
            for type_j = 1:s
                final_regret(type_j, j) = max(max(cum_proxy_regret(:, type_j, j), 0)) / (N + M);
            end
        end
    elseif t == N + Nobs_Pl1
        for type_j = 1:s
            Pl1_EmpRegr(type_j) = max(max(cum_proxy_regret(:, type_j, 1), 0)) / t;
        end
    elseif t == N + Nobs_Pl2
        for type_j = 1:s
            Pl2_EmpRegr(type_j) = max(max(cum_proxy_regret(:, type_j, 2), 0)) / t;
        end
    end

end

end
