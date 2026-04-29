function distY = learn_fixed(cfg, M, type_indices)
% DF.SIM.LEARN_FIXED  Regret-matching with FIXED type realizations.
%
%   distY = df.sim.learn_fixed(cfg, M, type_indices)
%
%   Unlike df.sim.learn (which redraws (t_1, t_2) every period from the
%   joint type prior), this routine FIXES (t_1*, t_2*) at the start and
%   runs M rounds of regret matching with these types throughout.
%
%   Use case: realization-conditional identification (App. SIM-RC).
%   The empirical action distribution is conditional on the realized
%   type pair, m_N(a_1, a_2 | t_1*, t_2*).
%
%   Inputs:
%     cfg          - game config struct from df.setup.game_simulation
%     M            - number of regret-matching rounds (after 1 random round)
%     type_indices - 1 x NPlayers vector of fixed type indices in 1..s
%
%   Output:
%     distY - (NActPr x 1) empirical action distribution after 1+M rounds

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;

nAct  = size(AA, 1);
nProf = size(A, 1);
s     = size(type_space{1,1}, 1);

% Realized cost values (one per player)
mc_draw = zeros(1, NPlayers);
for j = 1:NPlayers
    mc_draw(j) = type_space{j,1}(type_indices(j));
end

% Action-profile lookup (2-player path)
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

% Cumulative statistics (only realized type updated)
cum_util    = zeros(s, NPlayers);
cum_cf_util = zeros(nAct, s, NPlayers);

action_counts = zeros(nProf, 1);
exp_alpha_AA = exp(alpha_val * AA);

% Total rounds: 1 random + M regret-matching
total_rounds = 1 + M;

for t = 1:total_rounds

    % --- Choose actions ---
    if t == 1
        action_indices = randi(nProf, 1, 1);
        actions = A(action_indices, :);
    else
        actions = zeros(1, NPlayers);
        for j = 1:NPlayers
            U_n_j     = cum_util(:, j) / t;
            Avg_cf_j  = cum_cf_util(:, :, j) / t;
            [actions(j), ~] = regret_matching_mod(AA, U_n_j, Avg_cf_j, type_indices(j));
        end
    end

    % --- Update cumulative stats (vectorized choice probs) ---
    exp_alpha_actions = exp(alpha_val * actions);
    sum_exp_all = sum(exp_alpha_actions);

    for j = 1:NPlayers
        type_j = type_indices(j);

        cp_own = exp_alpha_actions(j) / (1 + sum_exp_all);
        utility = cp_own * (actions(j) - mc_draw(j));
        cum_util(type_j, j) = cum_util(type_j, j) + utility;

        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        denom_cf = sum_others + exp_alpha_AA;
        cp_cf    = exp_alpha_AA ./ denom_cf;
        cf_utilities = cp_cf .* (AA - mc_draw(j));
        cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + cf_utilities;
    end

    % --- Action profile count ---
    if use_2d_map
        a1_idx = find(AA == actions(1), 1);
        a2_idx = find(AA == actions(2), 1);
        prof_idx = action_profile_map(a1_idx, a2_idx);
    else
        prof_idx = find(all(A == actions, 2), 1);
    end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;
end

distY = action_counts / total_rounds;

end
