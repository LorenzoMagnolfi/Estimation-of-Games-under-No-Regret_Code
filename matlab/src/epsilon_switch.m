function eps = epsilon_switch(maxiters, conf, switch_eps, cfg)
% epsilon_switch  Compute epsilon bound (simulation mode).
%
%   eps = epsilon_switch(maxiters, conf, switch_eps, cfg)
%
%   cfg must contain: .Pi, .NAct, .NPlayers, .s
%   Delegates to df.solvers.compute_epsilon in simulation mode.

eps = df.solvers.compute_epsilon(cfg, maxiters, conf, switch_eps);

end
