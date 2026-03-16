function [opt_action, R_n] = regret_matching_mod(Act, U_n_j, Avg_cf_util_j, type_index)

% Calculate regret
regret = max(Avg_cf_util_j(:, type_index) - U_n_j(type_index), 0);
R_n = max(max(Avg_cf_util_j(:, type_index)) - U_n_j(type_index), 0);

% Calculate action probabilities
pA = regret;
if sum(pA) == 0
    pA = ones(size(pA)) / length(pA);
else
    pA = pA / sum(pA);
end

% Choose optimal action
opt_action = randsample(Act, 1, true, pA);

end