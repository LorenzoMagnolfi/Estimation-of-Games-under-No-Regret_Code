function [outputs] = ComputeBCCE_eps(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,cfg)
% ComputeBCCE_eps  Thin wrapper → df.solvers.solve_bcce (joint mode, epsilon from switch).
%
% Preserved for backward compatibility.  All work delegated to solve_bcce.

% Compute epsilon
eps_vec = df.solvers.compute_epsilon(cfg, T, confid, switch_eps);

eps_info = struct('mode', 'switch', 'eps', eps_vec);
opts = struct('marginal', false, 'switch_eps', switch_eps);

g = df.solvers.solve_bcce(type_space, action_space, action_distribution, ...
    payoff_parameters, distribution_parameters, Pi, eps_info, opts);

outputs{1,1} = g;
end
