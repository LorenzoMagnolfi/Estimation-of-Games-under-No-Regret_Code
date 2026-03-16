function [outputs] = ComputeBCCE_eps_ApplicationL(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean,cfg)
% ComputeBCCE_eps_ApplicationL  Thin wrapper → df.solvers.solve_bcce (marginal mode).
%
% Preserved for backward compatibility.  All work delegated to solve_bcce.

% Dimensions
NAg = size(type_space, 1);
NA_i = zeros(NAg, 1);
for ind = 1:NAg
    NA_i(ind) = size(action_space{ind,1}, 1);
end
s = size(type_space{1,1}, 1);

% Compute marginal action distributions
marg_act_distrib_I  = kron(eye(NA_i(1)), ones(1, NA_i(1))) * action_distribution;
marg_act_distrib_II = kron(ones(1, NA_i(1)), eye(NA_i(1))) * action_distribution;

% Compute epsilon (application/distrib mode)
kwargs = struct('mode', 'application', 'marg_mean', marg_mean, 's_override', s);
eps_vec = df.solvers.compute_epsilon(cfg, T, confid, switch_eps, kwargs);

eps_info = struct('mode', 'switch', 'eps', eps_vec);
opts = struct('marginal', true, 'switch_eps', switch_eps, ...
    'solver', 'sedumi', ...
    'marg_act_distrib_I', marg_act_distrib_I, ...
    'marg_act_distrib_II', marg_act_distrib_II);

g = df.solvers.solve_bcce(type_space, action_space, action_distribution, ...
    payoff_parameters, distribution_parameters, Pi, eps_info, opts);

outputs{1,1} = g;
end
