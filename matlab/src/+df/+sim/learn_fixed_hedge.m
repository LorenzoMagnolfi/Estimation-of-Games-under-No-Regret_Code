function [distY, final_regret] = learn_fixed_hedge(cfg, M, type_indices)
% DF.SIM.LEARN_FIXED_HEDGE  Hedge (family F) with FIXED type realizations.
%
%   [distY, final_regret] = df.sim.learn_fixed_hedge(cfg, M, type_indices)
%
%   Realization-conditional counterpart of df.sim.learn_hedge: the cost pair
%   (t_1*, t_2*) is fixed for the whole run, so each player has a single context
%   visited every period. One uniformly random period, then M periods in which
%   player j mixes p proportional to exp(eta_s S(b) / K_j), eta_s = sqrt(ln|A| / s),
%   with S the cumulative counterfactual payoff vector and K_j the span of the
%   player's payoff at its realized cost (the family's normalization). This is
%   follow-the-regularized-leader with relative entropy to the uniform prior and
%   step multiplier c = 1 (c_F = 1.25 sqrt(ln|A|)); the first period is also
%   uniform, so the run is the family's initialized run. Replaces the regret
%   matching of df.sim.learn_fixed for the certified calibrations.
%
%   Outputs: distY (NActPr x 1) empirical action distribution after 1 + M periods;
%   final_regret (1 x NPlayers) positive-part average regret of each player.

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
Pi        = cfg.Pi;

nAct  = size(AA, 1);
nProf = size(A, 1);
lnA = log(nAct);

mc_draw = zeros(1, NPlayers); Kj = zeros(1, NPlayers);
for j = 1:NPlayers
    mc_draw(j) = type_space{j,1}(type_indices(j));
    Kj(j) = max(Pi(:, type_indices(j), j)) - min(Pi(:, type_indices(j), j));
end
Kj = max(Kj, 1e-12);

if NPlayers == 2
    action_profile_map = zeros(nAct, nAct);
    for k = 1:nProf
        a1_idx = find(AA == A(k,1), 1);
        a2_idx = find(AA == A(k,2), 1);
        action_profile_map(a1_idx, a2_idx) = k;
    end
end

cum_util = zeros(1, NPlayers);
cum_cf   = zeros(nAct, NPlayers);
action_counts = zeros(nProf, 1);
exp_alpha_AA = exp(alpha_val * AA);
T = 1 + M;
for t = 1:T
    actions = zeros(1, NPlayers);
    for j = 1:NPlayers
        if t == 1
            a_idx = randi(nAct);
        else
            eta = sqrt(lnA / t);                       % s-th visit of the single context, s = t
            logits = eta * cum_cf(:, j) / Kj(j);
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
        cp_own = exp_alpha_actions(j) / (1 + sum_exp_all);
        cum_util(j) = cum_util(j) + cp_own * (actions(j) - mc_draw(j));
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        cp_cf = exp_alpha_AA ./ (sum_others + exp_alpha_AA);
        cum_cf(:, j) = cum_cf(:, j) + cp_cf .* (AA - mc_draw(j));
    end
    if NPlayers == 2
        prof_idx = action_profile_map(find(AA == actions(1), 1), find(AA == actions(2), 1));
    else
        prof_idx = find(all(A == actions, 2), 1);
    end
    action_counts(prof_idx) = action_counts(prof_idx) + 1;
end
distY = action_counts / T;
final_regret = max((max(cum_cf, [], 1) - cum_util) / T, 0);
end
