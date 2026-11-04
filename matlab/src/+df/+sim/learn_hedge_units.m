function [distY, final_regret, U] = learn_hedge_units(cfg, N_total, dur_rule, dur_param)
% DF.SIM.LEARN_HEDGE_UNITS  Contextual Hedge (family F) with persistent types in units of predetermined or endogenous duration.
%
%   [distY, final_regret, U] = df.sim.learn_hedge_units(cfg, N_total, dur_rule, dur_param)
%
%   The lab's two sellers meet in a sequence of units j = 1, 2, ... Each unit draws one cost
%   pair from the joint prior and keeps it for L_j periods; the total horizon is N_total
%   (the last unit is truncated). Both players run the per-type Hedge copies of
%   df.sim.learn_hedge, carried across units (not reset), so each copy's family-F
%   guarantee holds at its own visit count whatever the visiting pattern.
%   Durations:
%     'geometric'  L_j = 1 + Geometric with mean dur_param, drawn before play and
%                  independent of the cost draws (predetermined; V05's independent-unit
%                  composition applies with J_eff = (sum L_j)^2 / sum L_j^2);
%     'sale'       the unit ends at the first sale by either seller (one logit sale draw
%                  per period at the posted prices); durations are then endogenous to the
%                  costs, the condition of the lemma fails, and the pooled type weights
%                  differ from the prior (dur_param unused).
%   Period 1 of the run is uniform play (the family's initialized run).
%
%   Outputs: distY (nProf x 1) empirical action-profile distribution over N_total periods;
%   final_regret (s x 2) positive-part average regret per (type, player) at N_total; U with
%   J (units), L (durations), J_eff, w (s^2 x 1 realized duration-weighted type-pair
%   frequencies, in the Kronecker order of the Stage II prior: player 1 fastest),
%   p_lambda (prior in the same order), L1 (||w - p_lambda||_1), mean_L_by_pair (s x s:
%   rows player 1's type, columns player 2's), mean_L, dur_rule, dur_param.

A         = cfg.A;
AA        = cfg.AA;
alpha_val = cfg.alpha;
type_space = cfg.type_space;
Pi        = cfg.Pi;
NPlayers  = 2;
if cfg.NPlayers ~= 2
    error('learn_hedge_units:players', 'learn_hedge_units is written for the two-player lab.');
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
joint_distrib = kron(md2, md1); n_combs = size(combs, 1);
combs_type_idx = zeros(n_combs, 2);
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
action_counts = zeros(nProf, 1); exp_alpha_AA = exp(alpha_val * AA);
pair_periods = zeros(n_combs, 1); pair_units = zeros(n_combs, 1);
dur_rule = lower(dur_rule);
endogenous = strcmp(dur_rule, 'sale');
if ~endogenous && ~strcmp(dur_rule, 'geometric')
    error('learn_hedge_units:rule', 'dur_rule must be ''geometric'' or ''sale''');
end
L_list = zeros(ceil(N_total / 2) + 1, 1); J = 0;
t = 0;
while t < N_total
    J = J + 1;
    mc_idx = randsample(comb_indices, 1, true, joint_distrib);
    mc_draw = combs(mc_idx, :); type_indices = combs_type_idx(mc_idx, :);
    if endogenous
        L_j = Inf;
    else
        L_j = 1 + floor(log(rand) / log(1 - 1 / dur_param));
    end
    L_j = min(L_j, N_total - t);                     % truncation at the horizon
    ell = 0; ended = false;
    while ~ended
        t = t + 1; ell = ell + 1;
        a_idx = zeros(1, 2);
        for j = 1:NPlayers
            type_j = type_indices(j); visits(type_j, j) = visits(type_j, j) + 1;
            if t == 1
                a_idx(j) = randi(nAct);
            else
                eta = sqrt(lnA / visits(type_j, j));
                logits = eta * cum_cf_util(:, type_j, j) / Kt(type_j, j); logits = logits - max(logits);
                p = exp(logits); p = p / sum(p);
                ai = find(cumsum(p) >= rand, 1);
                if isempty(ai), ai = nAct; end
                a_idx(j) = ai;
            end
        end
        actions = [AA(a_idx(1)), AA(a_idx(2))];
        exp_alpha_actions = exp(alpha_val * actions); sum_exp_all = sum(exp_alpha_actions);
        for j = 1:NPlayers
            type_j = type_indices(j);
            cp_own = exp_alpha_actions(j) / (1 + sum_exp_all);
            cum_util(type_j, j) = cum_util(type_j, j) + cp_own * (actions(j) - mc_draw(j));
            sum_others = 1 + sum_exp_all - exp_alpha_actions(j);
            cp_cf = exp_alpha_AA ./ (sum_others + exp_alpha_AA);
            cum_cf_util(:, type_j, j) = cum_cf_util(:, type_j, j) + cp_cf .* (AA - mc_draw(j));
        end
        prof_idx = action_profile_map(a_idx(1), a_idx(2));
        action_counts(prof_idx) = action_counts(prof_idx) + 1;
        if endogenous
            cp = exp_alpha_actions / (1 + sum_exp_all);     % sale probabilities of the two sellers
            ended = rand < sum(cp);
        end
        if ell >= L_j || t >= N_total, ended = true; end
    end
    if J > numel(L_list), L_list = [L_list; zeros(numel(L_list), 1)]; end %#ok<AGROW>
    L_list(J) = ell;
    pair_periods(mc_idx) = pair_periods(mc_idx) + ell; pair_units(mc_idx) = pair_units(mc_idx) + 1;
end
L = L_list(1:J);
distY = action_counts / N_total;
best = squeeze(max(cum_cf_util, [], 1));
if isvector(best) && size(cum_util, 2) == 1, best = best(:); end
final_regret = max((best - cum_util) / N_total, 0);
w = pair_periods / N_total;
U = struct('J', J, 'L', L, 'J_eff', sum(L)^2 / sum(L.^2), 'w', w, 'p_lambda', joint_distrib(:), ...
    'L1', sum(abs(w - joint_distrib(:))), 'mean_L_by_pair', reshape(pair_periods ./ max(pair_units, 1), s, s), ...
    'mean_L', mean(L), 'dur_rule', dur_rule, 'dur_param', dur_param);
end
