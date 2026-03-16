function VP = find_polytope_switch(maxiters, conf2, switch_eps, cfg)
% FIND_POLYTOPE_SWITCH  Thin wrapper → df.solvers.solve_polytope_lp.
%
%   Preserved for backward compatibility.  AMPL dependency eliminated;
%   now uses native MATLAB linprog.

VP = df.solvers.solve_polytope_lp(cfg, maxiters, conf2, switch_eps);

end
