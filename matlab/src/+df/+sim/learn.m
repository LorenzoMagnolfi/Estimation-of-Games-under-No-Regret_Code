function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN  Optimized regret-matching learning simulation.
%
%   [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
%       df.sim.learn(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
%
%   Drop-in replacement for learn_mod.m with identical outputs but faster
%   execution.  Three key optimizations:
%
%     1. Sufficient statistics: track cumulative sums (cum_util, cum_cf_util)
%        instead of running averages.  Only the realized type is updated per
%        iteration, eliminating the O(s) decay loop for non-realized types.
%        Averages are derived as cum/t only when needed for regret matching.
%
%     2. Precomputed index lookups: type-to-index map and action-profile-to-
%        index map replace per-iteration find() / linear-search calls.
%
%     3. Vectorized counterfactual utility: all per-action choice_prob calls
%        are replaced with a single vectorized computation per player.
%
%   The RNG call sequence is identical to learn_mod, so outputs match to
%   machine precision given the same RNG state.
%
%   cfg must contain: .A, .AA, .NPlayers, .alpha, .type_space,
%                     .learning_style, .sigma2, .mu

% Unpack cfg fields
A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;

nAct = size(AA, 1);         % number of individual actions
nProf = size(A, 1);         % number of joint action profiles
s = size(type_space{1,1}, 1);

%% Precompute index lookup tables

% Type-to-index: for each player, map type value -> index in type_space
% Use a tolerance-based lookup via exact float comparison (matches original)
type_vals = cell(NPlayers, 1);
for j = 1:NPlayers
    type_vals{j} = type_space{j,1};
end

% Action-profile-to-index: map (a1, a2, ...) -> row index in A
% Build a lookup using a containers.Map with a string key
% For 2-player games with small action sets, a 2D array is faster
if NPlayers == 2
    % Build 2D lookup: action_idx(a1_idx, a2_idx) -> profile index
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

%% Precompute recording times
% distY_time recording times (relative to start of simulation)
record_times = round(M * (1:numdst_t) / numdst_t);
record_map = zeros(N + M, 1);  % 0 = no recording; >0 = column index
for k = 1:numdst_t
    t_rec = record_times(k);
    if t_rec >= 1 && t_rec <= N + M
        record_map(t_rec) = k;
    end
end

% distY_time_obs recording times (relative to start of obs window)
record_times_obs = round(M_obs * (1:numdst_t_obs) / numdst_t_obs);
obs_start = N + M - M_obs;  % absolute time when obs window starts
record_map_obs = zeros(N + M, 1);
for k = 1:numdst_t_obs
    t_abs = obs_start + record_times_obs(k);
    if t_abs >= 1 && t_abs <= N + M
        record_map_obs(t_abs) = k;
    end
end

%% Precompute marginal distribution for type draws
% (inlined from marginal_cost_draws_v4_new for efficiency)
sigma2 = cfg.sigma2;
mu_val = cfg.mu;

if NPlayers == 2 && size(sigma2, 1) == 2
    % 2D case: build joint type distribution
    combs = [kron(ones(size(type_vals{2})), type_vals{1}), ...
             kron(type_vals{2}, ones(size(type_vals{1})))];
    md1 = pdf('normal', type_vals{1}, mu_val(1), sigma2(1,1));
    md1 = md1 / sum(md1);
    md2 = pdf('normal', type_vals{2}, mu_val(2), sigma2(2,2));
    md2 = md2 / sum(md2);
    joint_distrib = kron(md2, md1);
    n_combs = size(combs, 1);

    % Precompute type index for each combination
    combs_type_idx = zeros(n_combs, NPlayers);
    for k = 1:n_combs
        combs_type_idx(k, 1) = find(type_vals{1} == combs(k, 1), 1);
        combs_type_idx(k, 2) = find(type_vals{2} == combs(k, 2), 1);
    end
    comb_indices = (1:n_combs)';
end

%% Initialize cumulative statistics (sufficient statistics)
cum_util    = zeros(s, NPlayers);      % cumulative utility per (type, player)
cum_cf_util = zeros(nAct, s, NPlayers); % cumulative counterfactual utility

%% Initialize output trackers
distY_time     = zeros(nProf, numdst_t);
distY_time_obs = zeros(nProf, numdst_t_obs);
action_counts  = zeros(nProf, 1);
final_regret   = zeros(s, NPlayers);
Pl1_EmpRegr    = zeros(s, 1);
Pl2_EmpRegr    = zeros(s, 1);

%% Precompute exp(alpha * AA) for vectorized choice_prob
exp_alpha_AA = exp(alpha_val * AA);  % nAct x 1

%% Main simulation loop
for t = 1:(N + M)

    %--- Draw marginal costs (preserves RNG sequence of marginal_cost_draws_v4_new) ---
    if NPlayers == 2 && size(sigma2, 1) == 2
        mc_idx = randsample(comb_indices, 1, true, joint_distrib);
        mc_draw = combs(mc_idx, :);
        type_indices = combs_type_idx(mc_idx, :);
    else
        % Generic: call original function (preserves RNG)
        mc_draw = marginal_cost_draws_v4_new(cfg, type_space, 1);
        type_indices = zeros(1, NPlayers);
        for j = 1:NPlayers
            type_indices(j) = find(type_vals{j} == mc_draw(j), 1);
        end
    end

    %--- Choose actions ---
    if t <= N
        % Random play phase (preserves RNG: single randi call)
        action_indices = randi(nProf, 1, 1);
        actions = A(action_indices, :);
    else
        % Regret matching phase
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            % Derive averages from cumulative sums (only for this call)
            U_n_j = cum_util(:, j) / t;
            Avg_cf_j = cum_cf_util(:, :, j) / t;
            [actions(j), ~] = regret_matching_mod(AA, U_n_j, Avg_cf_j, type_indices(j));
        end
    end

    %--- Update cumulative statistics (only for realized types) ---
    % Precompute shared quantities for choice_prob vectorization
    exp_alpha_actions = exp(alpha_val * actions);  % 1 x NPlayers
    sum_exp_all = sum(exp_alpha_actions);          % scalar

    for j = 1:NPlayers
        type_j = type_indices(j);

        % Actual utility: choice_prob(actions(j), actions, j, alpha) * (actions(j) - mc_draw(j))
        % Vectorized: denom = 1 + sum_exp_all + exp(alpha*a) - exp(alpha*actions(j))
        %                    = 1 + sum_exp_all  (since a = actions(j), the terms cancel)
        %           Wait, no: denom = 1 + sum(exp(alpha*b),2) + exp(alpha*a) - exp(alpha*b(:,j))
        %           With a=actions(j), b=actions (1-row):
        %             = 1 + sum_exp_all + exp(alpha*actions(j)) - exp(alpha*actions(j))
        %             = 1 + sum_exp_all
        denom_own = 1 + sum_exp_all;
        cp_own = exp_alpha_actions(j) / denom_own;
        utility = cp_own * (actions(j) - mc_draw(j));

        cum_util(type_j, j) = cum_util(type_j, j) + utility;

        % Counterfactual utilities: for each alternative action a_k in AA
        % denom_k = 1 + sum_exp_all + exp(alpha*a_k) - exp(alpha*actions(j))
        %         = 1 + sum_exp_all - exp(alpha*actions(j)) + exp(alpha*a_k)
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);  % scalar
        denom_cf = sum_others + exp_alpha_AA;                  % nAct x 1
        cp_cf = exp_alpha_AA ./ denom_cf;                      % nAct x 1
        cf_utilities = cp_cf .* (AA - mc_draw(j));             % nAct x 1

        cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + cf_utilities;
    end

    %--- Update action counts ---
    if use_2d_map
        a1_idx = find(AA == actions(1), 1);
        a2_idx = find(AA == actions(2), 1);
        prof_idx = action_profile_map(a1_idx, a2_idx);
    else
        prof_idx = find(all(A == actions, 2), 1);
    end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;

    %--- Record distributions at precomputed times ---
    col = record_map(t);
    if col > 0
        distY_time(:, col) = action_counts / t;
    end

    col_obs = record_map_obs(t);
    if col_obs > 0
        distY_time_obs(:, col_obs) = action_counts / t;
    end

    %--- Compute final regrets at specified times ---
    if t == N + M
        U_n_final = cum_util / t;
        Avg_cf_final = cum_cf_util / t;
        for j = 1:NPlayers
            for type_j = 1:s
                [~, final_regret(type_j, j)] = regret_matching_mod(AA, ...
                    U_n_final(:, j), Avg_cf_final(:, :, j), type_j);
            end
        end
    elseif t == N + Nobs_Pl1
        U_n_t = cum_util / t;
        Avg_cf_t = cum_cf_util / t;
        for type_j = 1:s
            [~, Pl1_EmpRegr(type_j)] = regret_matching_mod(AA, ...
                U_n_t(:, 1), Avg_cf_t(:, :, 1), type_j);
        end
    elseif t == N + Nobs_Pl2
        U_n_t = cum_util / t;
        Avg_cf_t = cum_cf_util / t;
        for type_j = 1:s
            [~, Pl2_EmpRegr(type_j)] = regret_matching_mod(AA, ...
                U_n_t(:, 2), Avg_cf_t(:, :, 2), type_j);
        end
    end

end

end
