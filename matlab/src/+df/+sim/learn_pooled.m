function [distY_time, distY_time_obs, final_regret_pooled, final_regret_type, ...
          Pl1_EmpRegr, Pl2_EmpRegr] = ...
    learn_pooled(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% DF.SIM.LEARN_POOLED  Pooled-regret (unconditional / strategic-normal-form) learner.
%
%   Companion to df.sim.learn (type-conditional regret matching). The ONLY
%   behavioral difference is the regret UNIT used to choose actions:
%
%     df.sim.learn   — CONTEXTUAL regret: each player tracks regret separately
%                      per realized cost type and plays a type-contingent mixed
%                      strategy. Limiting play is a Bayes coarse correlated
%                      equilibrium (BCCE) — the paper's identified object.
%
%     df.sim.learn_pooled — POOLED regret: each player tracks a SINGLE regret
%                      vector pooled across all periods regardless of the
%                      realized type, and plays one (non-type-contingent) mixed
%                      strategy. This is standard external (unconditional)
%                      no-regret; limiting play is a coarse correlated
%                      equilibrium of the strategic normal form.
%
%   This isolates the "regret unit" axis (type-by-type vs pooled-across-types)
%   that the R2 discussion turns on (see the theory memo, items 2 and 4: the
%   pooled-vs-contextual distinction, NOT conditional independence, is the
%   likely correct framing of the HST comparison). It is the empirical
%   complement to that theory check.
%
%   To keep the two runs directly comparable, the RNG draw sequence is IDENTICAL
%   to df.sim.learn: the same type draw (randsample), the same random-phase
%   action (randi), and exactly one action draw per player per RM period
%   (randsample inside regret_matching_mod). Only the mixing weights pA differ,
%   so the two learners see the same cost-draw stream.
%
%   Extra outputs vs df.sim.learn:
%     final_regret_pooled — (1 x NPlayers) pooled (unconditional) final regret;
%                           ->0 as the pooled learner has no pooled regret.
%     final_regret_type   — (s x NPlayers) type-conditional final regret OF THE
%                           POOLED PLAY; generically bounded away from 0 — the
%                           pooled learner does NOT satisfy contextual no-regret.
%   run_stage_ii only consumes the first output (distY_time), so the extended
%   signature is backward compatible with the learn_mod_* call convention.

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

%% Precompute recording times
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

%% Precompute marginal distribution for type draws
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

%% Initialize cumulative statistics
% Pooled accumulators DRIVE the firm's decisions (unconditional regret).
cum_util_pool    = zeros(1, NPlayers);       % cumulative realized utility, pooled
cum_cf_util_pool = zeros(nAct, NPlayers);    % cumulative counterfactual, pooled

% Per-type accumulators are tracked ONLY for the end-of-run regret-unit
% diagnostic (the type-conditional regret of the pooled play). They never
% affect play.
cum_util    = zeros(s, NPlayers);
cum_cf_util = zeros(nAct, s, NPlayers);

%% Initialize output trackers
distY_time     = zeros(nProf, numdst_t);
distY_time_obs = zeros(nProf, numdst_t_obs);
action_counts  = zeros(nProf, 1);
final_regret_pooled = zeros(1, NPlayers);
final_regret_type   = zeros(s, NPlayers);
Pl1_EmpRegr    = zeros(s, 1);
Pl2_EmpRegr    = zeros(s, 1);

%% Precompute exp(alpha * AA)
exp_alpha_AA = exp(alpha_val * AA);

%% Main simulation loop
for t = 1:(N + M)

    %--- Draw marginal costs (identical RNG to df.sim.learn) ---
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

    %--- Choose actions ---
    if t <= N
        % Random play phase (identical RNG: single randi call)
        action_indices = randi(nProf, 1, 1);
        actions = A(action_indices, :);
    else
        % POOLED regret matching phase: no type conditioning. One randsample
        % per player (same RNG draw count as the contextual learner).
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            U_pool_j  = cum_util_pool(j) / t;          % scalar
            Avg_cf_j  = cum_cf_util_pool(:, j) / t;    % nAct x 1
            [actions(j), ~] = regret_matching_mod(AA, U_pool_j, Avg_cf_j, 1);
        end
    end

    %--- Update statistics ---
    exp_alpha_actions = exp(alpha_val * actions);
    sum_exp_all = sum(exp_alpha_actions);

    for j = 1:NPlayers
        type_j = type_indices(j);

        denom_own = 1 + sum_exp_all;
        cp_own = exp_alpha_actions(j) / denom_own;
        utility = cp_own * (actions(j) - mc_draw(j));

        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        denom_cf = sum_others + exp_alpha_AA;
        cp_cf = exp_alpha_AA ./ denom_cf;
        cf_utilities = cp_cf .* (AA - mc_draw(j));

        % Pooled accumulators (drive decisions): sum over ALL types.
        cum_util_pool(j)      = cum_util_pool(j) + utility;
        cum_cf_util_pool(:, j) = cum_cf_util_pool(:, j) + cf_utilities;

        % Per-type accumulators (diagnostic only): indexed by realized type.
        cum_util(type_j, j)       = cum_util(type_j, j) + utility;
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

    %--- Record distributions ---
    col = record_map(t);
    if col > 0
        distY_time(:, col) = action_counts / t;
    end
    col_obs = record_map_obs(t);
    if col_obs > 0
        distY_time_obs(:, col_obs) = action_counts / t;
    end

    %--- Final regrets (pooled and type-conditional of the pooled play) ---
    if t == N + M
        for j = 1:NPlayers
            % Pooled final regret (the quantity the pooled learner minimizes).
            U_pool_final  = cum_util_pool(j) / t;
            cf_pool_final = cum_cf_util_pool(:, j) / t;
            [~, final_regret_pooled(j)] = regret_matching_mod(AA, U_pool_final, cf_pool_final, 1);

            % Type-conditional regret OF THE POOLED PLAY (generically > 0).
            U_n_final    = cum_util(:, j) / t;
            Avg_cf_final = cum_cf_util(:, :, j) / t;
            for type_j = 1:s
                [~, final_regret_type(type_j, j)] = regret_matching_mod(AA, ...
                    U_n_final, Avg_cf_final, type_j);
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
