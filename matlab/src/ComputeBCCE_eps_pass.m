function [outputs] = ComputeBCCE_eps_pass(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,ExpRegr_pass)
% ComputeBCCE_eps_pass  Thin wrapper → df.solvers.solve_bcce (joint mode, epsilon passed directly).
%
% Preserved for backward compatibility.  All work delegated to solve_bcce.

eps_info = struct('mode', 'pass', 'ExpRegr_pass', ExpRegr_pass);
opts = struct('marginal', false, 'switch_eps', switch_eps);

g = df.solvers.solve_bcce(type_space, action_space, action_distribution, ...
    payoff_parameters, distribution_parameters, Pi, eps_info, opts);

outputs{1,1} = g;
end
