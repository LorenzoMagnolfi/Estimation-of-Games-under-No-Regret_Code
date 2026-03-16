function [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)
% learn_mod  Regret-matching learning simulation (thin wrapper).
%
%   Delegates to df.sim.learn.  See that function for details.
%
%   [distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
%       learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2)

[distY_time, distY_time_obs, final_regret, Pl1_EmpRegr, Pl2_EmpRegr] = ...
    df.sim.learn(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);

end
