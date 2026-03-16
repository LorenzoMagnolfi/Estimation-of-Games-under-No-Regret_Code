function [outputs] = Identification_Pricing_Game_ApplicationL(epsilon_grid,Dist_file,Prob_file,players,NGridV,NGridM,n_types)

global NAct NPlayers A AA s Pi mu sigma2 type_space tps

NPlayers = size(players,2);

%% DEAL W/ EPS/CONVERGENCE RATE
switch_eps = 9;
alpha_set = [0.05];
confid = alpha_set(1);

% Preallocate
n_epsilon = size(epsilon_grid,2);

for iii=1:NPlayers

    
[distrib, actions, prob, maxiters] = get_player_data_5acts(iii, 'median', Dist_file, Prob_file);

T=maxiters;

% HOW TO TREAT GRID OF MARG COSTS?

P_l = actions(1);
P_h = actions(5);
diff_p = P_h-P_l;
mid = P_l+(1/2*diff_p);

    ub = P_h +0.25*diff_p;
    lb = P_l-3*diff_p;

%    ub = P_h + 1/10*diff_p;
%    lb = P_l-2/3*diff_p;

%    ub = P_h - 1/10*diff_p;
%    lb = P_l-1/2*diff_p;

% Type space
s=n_types;
type_space = cell(2,1);
type_space{1,1} = linspace(lb,ub,s)';
type_space{2,1} = linspace(lb,ub,s)';

% ACTION SPACE
% Players
NPlayer = 2;                                                                    % Number of agents
dim =NPlayer;

action_space={};

% Action Space (Finite)
for ind=1:NPlayer
     action_space{ind,1} = actions';                                                     % 3 actions!
end

AA = action_space{1,1};
AH = AA(2);
AL = AA(1);

A = allcomb(action_space{1,:},action_space{2,:});
NAct = size(action_space{1,:},1);
NActPr = size(A,1);

marg_mean =  kron(ones(1,NAct),eye(NAct))*distrib';

% HOW TO TREAT UTILIY FCN?

cp = zeros(NActPr,NPlayer); % Probability of selling for each action profile (row) and seller (column)
tps = [type_space{1,:},type_space{2,:}]; %matrix with types (row) individual (column)

r_sq = size(tps,1);

Pi = zeros(NActPr,r_sq,NPlayer); % Matrix that has - for each action profile (row) the corresponding payoff for each payoff type (column)

for j = 1:NPlayer
    for aa = 1:NActPr
        cp(aa,j) = prob(aa);
        for tt = 1:s
            Pi(aa,tt,j) = cp(aa,j).*(A(aa,j)-tps(tt,j));
        end
    end
end

% Alternative set of sale prob obtained 
% Ps_LL = 0.038;
% Ps_LH = 0.048;
% Ps_HL = 0.019;
% Ps_HH = 0.023;

%Ps = [Ps_LL,Ps_LH,Ps_HL,Ps_HH]';
payoff_parameters = Pi;

%% Params to choose:

%distribution of data q^N
action_distribution = distrib';

%epsilon
% NEW interpretation of the eps as percent...
for kkk = 1:n_epsilon
eps = epsilon_grid(kkk);

% grid parameters
NGrid = NGridV*NGridM;

% Stepsize = 0.3;         % spacing

% params for grid both
% NGridM = 10;
% NGridV = 20;

%% Initialize Parameters

% plot_param = 'Variance';
% plot_param = 'Mean';
plot_param = 'Both';


% Marginal Cost Parameters
%mu = 1.2*P_l*ones(dim,1);
mu = mid;
sigma2 = 0.33*diff_p*eye(NPlayer);

%gridparamV = [linspace(0.1*sigma2(1,1),sigma2(1,1)*8,NGridV)'];
%gridparamM = [linspace(mu(1,1)*5,mu(1,1)*0.25,NGridM)'];

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
        %distribution_parameters{3,ind} = linspace(0.1,sigma2*3,NGrid);
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

%outs = ComputeBCCE_eps_Application_II(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean);
outs = ComputeBCCE_eps_ApplicationL(type_space,action_space,action_distribution,payoff_parameters,distribution_parameters,T,confid,Pi,switch_eps,marg_mean);
%Plots
maxvals = cell2mat(outs);
VV = squeeze(maxvals) ;


outputs{iii,kkk} = [distpars,VV];


end
end

end
