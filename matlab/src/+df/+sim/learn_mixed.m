function [distY, final_regret, info] = learn_mixed(cfg, N, M, opp_rule, opp_param)
% DF.SIM.LEARN_MIXED  One Hedge learner (player 1, family F) against a non-learning rival (player 2).
%
%   [distY, final_regret, info] = df.sim.learn_mixed(cfg, N, M, opp_rule, opp_param)
%
%   Player 1 runs the contextual Hedge of df.sim.learn_hedge (one copy per type, local
%   clock, step sqrt(ln|A|/s), payoffs normalized by the type's span). Player 2 follows a
%   fixed rule that is not a no-regret algorithm:
%     'constant'  posts the price opp_param every period, whatever its cost;
%     'myopic'    best-responds to player 1's previous-period price given its own realized
%                 cost (best-response dynamics; the first period is uniform).
%   Types are redrawn every period from the joint prior, as in df.sim.learn_hedge, and
%   the first N periods are uniformly random for both players (Stage II uses N = 1).
%   Counterfactual payoffs are accumulated for BOTH players from period 1, so
%   final_regret (s x 2) reports the positive-part average regret of the learner and of
%   the non-learner at the same clock; the second column is the exhibit's diagnostic.
%
%   Outputs: distY (nProf x 1) empirical action-profile distribution after N + M periods;
%   final_regret (s x 2); info with the rule, the parameter and player 2's action
%   frequencies (nAct x 1).
%
%   SIM-MIXED exhibit (R1.2, R1.4.f): the learner's own obedience rows identify its set
%   whether or not the rival learns; the joint rows (both players modeled as learners)
%   are mis-specified for the rival. Pair with opts.eps_player_cap in run_stage_ii.

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
Pi        = cfg.Pi;
if NPlayers ~= 2
    error('learn_mixed:players', 'learn_mixed is written for the two-player lab.');
end
nAct = size(AA, 1); nProf = size(A, 1); s = size(type_space{1,1}, 1); lnA = log(nAct);
type_vals = {type_space{1,1}, type_space{2,1}};
action_profile_map = zeros(nAct, nAct);
for k = 1:nProf
    action_profile_map(find(AA == A(k,1), 1), find(AA == A(k,2), 1)) = k;
end
combs = [kron(ones(size(type_vals{2})), type_vals{1}), kron(type_vals{2}, ones(size(type_vals{1})))];
md1 = pdf('normal', type_vals{1}, cfg.mu(1), sqrt(cfg.sigma2(1,1))); md1 = md1 / sum(md1);
md2 = pdf('normal', type_vals{2}, cfg.mu(2), sqrt(cfg.sigma2(2,2))); md2 = md2 / sum(md2);
joint_distrib = kron(md2, md1);
n_combs = size(combs, 1); combs_type_idx = zeros(n_combs, 2);
for k = 1:n_combs
    combs_type_idx(k, 1) = find(type_vals{1} == combs(k, 1), 1);
    combs_type_idx(k, 2) = find(type_vals{2} == combs(k, 2), 1);
end
comb_indices = (1:n_combs)';
Kt = zeros(s, NPlayers);
for j = 1:NPlayers
    Kt(:, j) = (max(Pi(:, :, j), [], 1) - min(Pi(:, :, j), [], 1))';
end
Kt = max(Kt, 1e-12);
cum_util = zeros(s, NPlayers); cum_cf_util = zeros(nAct, s, NPlayers); visits = zeros(s, NPlayers);
action_counts = zeros(nProf, 1); opp_counts = zeros(nAct, 1);
exp_alpha_AA = exp(alpha_val * AA);
opp_rule = lower(opp_rule);
switch opp_rule
    case 'constant'
        a2_const = find(abs(AA - opp_param) < 1e-9, 1);
        if isempty(a2_const)
            error('learn_mixed:price', 'constant price %g is not on the action grid', opp_param);
        end
    case 'myopic'
        a2_const = NaN;
    otherwise
        error('learn_mixed:rule', 'opp_rule must be ''constant'' or ''myopic''');
end
p1_prev = AA(randi(nAct));
for t = 1:(N + M)
    mc_idx = randsample(comb_indices, 1, true, joint_distrib);
    mc_draw = combs(mc_idx, :); type_indices = combs_type_idx(mc_idx, :);
    % player 1: Hedge
    type_1 = type_indices(1); visits(type_1, 1) = visits(type_1, 1) + 1;
    if t <= N
        a1_idx = randi(nAct);
    else
        eta = sqrt(lnA / visits(type_1, 1));
        logits = eta * cum_cf_util(:, type_1, 1) / Kt(type_1, 1); logits = logits - max(logits);
        p = exp(logits); p = p / sum(p);
        a1_idx = find(cumsum(p) >= rand, 1);
        if isempty(a1_idx), a1_idx = nAct; end
    end
    % player 2: fixed rule
    type_2 = type_indices(2); visits(type_2, 2) = visits(type_2, 2) + 1;
    if t <= N
        a2_idx = randi(nAct);
    else
        switch opp_rule
            case 'constant'
                a2_idx = a2_const;
            case 'myopic'
                cp = exp_alpha_AA ./ (1 + exp(alpha_val * p1_prev) + exp_alpha_AA);
                [~, a2_idx] = max(cp .* (AA - mc_draw(2)));
        end
    end
    actions = [AA(a1_idx), AA(a2_idx)];
    exp_alpha_actions = exp(alpha_val * actions); sum_exp_all = sum(exp_alpha_actions);
    for j = 1:NPlayers
        type_j = type_indices(j);
        cp_own = exp_alpha_actions(j) / (1 + sum_exp_all);
        cum_util(type_j, j) = cum_util(type_j, j) + cp_own * (actions(j) - mc_draw(j));
        sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
        cp_cf = exp_alpha_AA ./ (sum_others + exp_alpha_AA);
        cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + cp_cf .* (AA - mc_draw(j));
    end
    prof_idx = action_profile_map(a1_idx, a2_idx);
    action_counts(prof_idx) = action_counts(prof_idx) + 1;
    opp_counts(a2_idx) = opp_counts(a2_idx) + 1;
    p1_prev = actions(1);
end
distY = action_counts / (N + M);
best = squeeze(max(cum_cf_util, [], 1));
if isvector(best) && size(cum_util, 2) == 1, best = best(:); end
final_regret = max((best - cum_util) / (N + M), 0);
info = struct('opp_rule', opp_rule, 'opp_param', opp_param, 'opp_action_freq', opp_counts / (N + M));
end
