function eps = epsilon_switch_distrib(maxiters, conf, switch_eps, marg_mean, s, cfg)
% epsilon_switch_distrib  Compute epsilon bound (application mode).
%
%   eps = epsilon_switch_distrib(maxiters, conf, switch_eps, marg_mean, s, cfg)
%
%   cfg must contain: .Pi, .NAct, .NPlayers
%   Delegates to df.solvers.compute_epsilon in application mode.

kwargs = struct('mode', 'application', 'marg_mean', marg_mean, 's_override', s);
eps = df.solvers.compute_epsilon(cfg, maxiters, conf, switch_eps, kwargs);

end
