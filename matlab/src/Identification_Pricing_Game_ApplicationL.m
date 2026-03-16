function [outputs] = Identification_Pricing_Game_ApplicationL(epsilon_grid,Dist_file,Prob_file,players,NGridV,NGridM,n_types)
% Identification_Pricing_Game_ApplicationL  Application-stage identification.
%
%   No globals. Per-player setup delegated to df.setup.game_application.

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
s = cfg.s;
A = cfg.A;
NAct = cfg.NAct;
NActPr = cfg.NActPr;
marg_mean = cfg.marg_mean;
mu = cfg.mu;
sigma2 = cfg.sigma2;

NPlayer = 2;

payoff_parameters = Pi;

%distribution of data q^N
action_distribution = cfg.distrib';

%epsilon
% NEW interpretation of the eps as percent...
for kkk = 1:n_epsilon
eps = epsilon_grid(kkk);

% grid parameters
NGrid = NGridV*NGridM;

% plot_param = 'Variance';
% plot_param = 'Mean';
plot_param = 'Both';


% Marginal Cost Parameters (already set from cfg)

gridparamV = [linspace(0.1*sigma2(1,1),sigma2(1,1)*5,NGridV)'];
gridparamM = [linspace(mu(1,1)*4,mu(1,1)*0.25,NGridM)'];

if strcmp(plot_param,'Both') ~= 1

for ind = 1:NGrid
    distribution_parameters{1,ind} = 'Normal';
    if strcmp(plot_param,'Mean')
        distribution_parameters{2,ind} = Stepsize*mu'+Stepsize*(ind-1)*mu';
        distribution_parameters{3,ind} = sigma2';
        distpars(ind,1) = Stepsize*mu(1,1)+Stepsize*(ind-1)*mu(1,1);
    elseif strcmp(plot_param,'Variance')
        distribution_parameters{2,ind} = mu';
        distribution_parameters{3,ind} = gridparamV(ind)*sigma2';
        distpars(ind,1) = gridparamV(ind)*sigma2(1,1);
    end
end

else
    for ind1 = 1:NGridM
        for ind2 = 1:NGridV
            distribution_parameters{1,(ind1-1)*NGridM+ind2} = 'Normal';
            distribution_parameters{2,(ind1-1)*NGridM+ind2} = gridparamM(ind1);
            distribution_parameters{3,(ind1-1)*NGridM+ind2} = gridparamV(ind2)*eye(NPlayer);
            distpars((ind1-1)*NGridM+ind2,:) = [gridparamM(ind1),gridparamV(ind2)];
        end
    end
end


%% Let's Play the Pricing Game

outs = ComputeBCCE_eps_ApplicationL(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean,cfg);
%Plots
maxvals = cell2mat(outs);
VV = squeeze(maxvals) ;


outputs{iii,kkk} = [distpars,VV];


end
end

end
