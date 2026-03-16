function [outputs] = Identification_Pricing_Game_ApplicationL(epsilon_grid,Dist_file,Prob_file,players,NGridV,NGridM,n_types)
% Identification_Pricing_Game_ApplicationL  Application-stage identification.
%
%   No globals. Per-player setup delegated to df.setup.game_application.
%   Grid construction delegated to df.report.build_param_grid.

NPlayers = size(players,2);

%% DEAL W/ EPS/CONVERGENCE RATE
switch_eps = 9;
alpha_set = [0.05];
confid = alpha_set(1);

% Preallocate
n_epsilon = size(epsilon_grid,2);

for iii=1:NPlayers

% Build per-player config (replaces all inline setup + globals)
cfg = df.setup.game_application(iii, Dist_file, Prob_file, n_types);

T = cfg.maxiters;
type_space = cfg.type_space;
action_space = cfg.action_space;
Pi = cfg.Pi;
marg_mean = cfg.marg_mean;
mu = cfg.mu;
sigma2 = cfg.sigma2;

payoff_parameters = Pi;
action_distribution = cfg.distrib';

for kkk = 1:n_epsilon

% Grid construction via shared module
gridparamV = linspace(0.1*sigma2(1,1), sigma2(1,1)*5, NGridV)';
gridparamM = linspace(mu(1,1)*4, mu(1,1)*0.25, NGridM)';

% Pass mu(1,1) as scalar to trigger application mode in build_param_grid
[distpars, distribution_parameters] = df.report.build_param_grid(mu(1,1), sigma2, gridparamM, gridparamV);

%% Solver
outs = ComputeBCCE_eps_ApplicationL(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean,cfg);
maxvals = cell2mat(outs);
VV = squeeze(maxvals);

outputs{iii,kkk} = [distpars,VV];

end
end

end
