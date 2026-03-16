function [mc_draws] = marginal_cost_draws_v4_new(type_space, draws)
global sigma2 mu

% Generate marginal distribution as in v5
marg_distrib = pdf('normal', type_space{1,1}, mu(1), sigma2(1,1));
marg_distrib = marg_distrib ./ sum(marg_distrib, 1);

% For 1-dimensional case
if size(sigma2, 1) == 1
    mc_draws = randsample(type_space{1,1}, draws, true, marg_distrib);

% For 2-dimensional case
elseif size(sigma2, 1) == 2
    % All Possible Type Combinations (Using Kronecker Product) 
    combs = [kron(ones(size(type_space{2,1})), type_space{1,1}), kron(type_space{2,1}, ones(size(type_space{1,1})))];
    
    % Joint distribution (assuming independence)
    marg_distrib2 = pdf('normal', type_space{2,1}, mu(2), sigma2(2,2));
    marg_distrib2 = marg_distrib2 ./ sum(marg_distrib2, 1);
    joint_distrib = kron(marg_distrib2, marg_distrib);
    
    % Random Draws from Marginal Cost Combinations
    mc_inds = randsample(1:size(combs,1), draws, true, joint_distrib);
    mc_draws = combs(mc_inds, :);

% For independently distributed marginal costs in greater than two dimensions
elseif sigma2 == diag(diag(sigma2)) 
    mc_draws = zeros(draws, size(sigma2, 1));
    for ind = 1:size(sigma2, 1)
        marg_distrib_i = pdf('normal', type_space{ind,1}, mu(ind), sigma2(ind,ind));
        marg_distrib_i = marg_distrib_i ./ sum(marg_distrib_i, 1);
        mc_draws(:, ind) = randsample(type_space{ind,1}, draws, true, marg_distrib_i);
    end

% For arbitrary dependence in multiple dimensions
else
    error('Arbitrary dependence in multiple dimensions not implemented in this version');
end
end