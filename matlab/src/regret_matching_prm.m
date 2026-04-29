function [opt_action, R_n] = regret_matching_prm(Act, cum_proxy_regret, type_index)
% REGRET_MATCHING_PRM  Proxy-regret matching (bandit feedback variant).
%
%   [opt_action, R_n] = regret_matching_prm(Act, cum_proxy_regret, type_index)
%
%   Implements the proxy-regret matching procedure of Hart and Mas-Colell
%   (2001, "A Reinforcement Procedure Leading to Correlated Equilibrium").
%   Unlike standard regret matching which requires full feedback (observation
%   of counterfactual payoffs for unchosen actions), proxy-regret matching
%   operates under bandit feedback: each player only observes their own
%   realized profit.
%
%   The proxy regret for action a given type t is maintained as a running sum:
%     proxy_R_n(a, t) += (1/gamma_n(a_n, t)) * u_n * 1{a_n = a} - u_n
%   where gamma_n(a, t) is the mixing probability used to choose a at time n,
%   u_n is the realized utility, and a_n is the chosen action.
%
%   Inputs:
%     Act               — (nAct x 1) action vector
%     cum_proxy_regret  — (nAct x s) cumulative proxy regrets
%     type_index        — scalar index of current type
%
%   Outputs:
%     opt_action — chosen action (scalar)
%     R_n        — max proxy regret for this type (scalar)

% Positive part of proxy regrets
regret = max(cum_proxy_regret(:, type_index), 0);
R_n = max(regret);

% Action probabilities proportional to positive proxy regrets
pA = regret;
if sum(pA) == 0
    pA = ones(size(pA)) / length(pA);
else
    pA = pA / sum(pA);
end

% Choose action
opt_action = randsample(Act, 1, true, pA);

end
