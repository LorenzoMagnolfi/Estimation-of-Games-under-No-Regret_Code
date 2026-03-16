%% Clean Up
clear all; clc; close all;
clear global;

%% A: Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Globals

global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resolve repository-relative paths and keep optimization templates on path.
paths = df_repo_paths();

% Initialize random number generator to seed 54321
rng(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% How long do they play?

maxiters = 10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters of the Game

% Players
NPlayers = 2;    

% Demand parameter
alpha = -(1/3);

% Marginal Cost Parameters
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);                                                            % Variance of marginal costs (Currently independently distributed)

% Action Space (Finite)
for ind=1:NPlayers
%    action_space{ind,1} = linspace(3,10,10)'; %[2;3;4;5;6;7;8;9;10;11];                                                     % Possible Actions (Column Vector)
%     action_space{ind,1} = [5;7;9];                                                     % 3 actions!
     action_space{ind,1} = [4;8];                                                     % 2 actions!

end

AA = action_space{1,1};
AH = AA(2);
AL = AA(1);

A = allcomb(action_space{1,:},action_space{2,:});
NAct = size(action_space{1,:},1);
NActPr = size(A,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NEW Grid of Epsilons

% step_size = (1/8)*ones(NPlayers,1);                                                 % Step Size: governs number of possible types (currently identically spaced)
% 1/3.3 corresponds to s=20;
% 1/8 corresponds to s=50

s=20;

% This FUNCTION??
[type_space,marg_distrib] = marginal_cost_draws_v5(mu,sigma2,s);
% [type_space,marg_distrib] = marginal_cost_draws_v3(mu,sigma2,step_size);

% s = size(type_space{1,1},1);
s2 = s^2;

Egrid = type_space{1,1}';

% get type space JOINT

% define a dummy useful for reshaping 
AEnum = 1;
% initialize
T_sorted = type_space{1,1};
for ind=2:NPlayers
    AEnum = [size(T_sorted,1), AEnum];
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1) ) ...
    kron( ones(size(type_space{ind,1},1),1), T_sorted)]; 
end

% Initialize the matrix of distributions psi for all values of lambda
Psi = zeros(s2,1);
kk = 0 ;
for ii = 1:s
    for jj = 1:s
        kk = (ii-1)*s+jj;
        Psi(kk) = marg_distrib(ii)*marg_distrib(jj);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create Matrix of Utilities - for each combination of types, actions

cp = zeros(NActPr,NPlayers); % Probability of selling for each action profile (row) and seller (column)
tps = [type_space{1,:},type_space{2,:}]; %matrix with types (row) individual (column)

r_sq = size(tps,1);

Pi = zeros(NActPr,r_sq,NPlayers); % Matrix that has - for each action profile (row) the corresponding payoff for each payoff type (column)

for j = 1:NPlayers
    for aa = 1:NActPr
        cp(aa,j) = exp(alpha*A(aa,j))/(1 + sum(exp(alpha*A(aa,:)),2));
        for tt = 1:s
            Pi(aa,tt,j) = cp(aa,j).*(A(aa,j)-tps(tt,j));
        end
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare discretized set of directions u to find vertices of Polytope of Predictions
%% B: Polytope

conf_set = [0.1, 0.025, 0.05];
switch_eps = 3;

for jj = 1:length(conf_set)
    
conf2 =  conf_set(jj);

VP = find_polytope_switch(maxiters,conf2,switch_eps);

%% Fixture: save polytope vertices
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
save(fullfile(fixture_dir, sprintf('stage_i_polytope_conf%d.mat', round(conf2*1000))), ...
    'VP', 'conf2', 'switch_eps', 'maxiters');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% draw Polyhedra of BCE Predictions and of Simplex


plotname = fullfile(paths.figures_i, strcat('BCCE_set_', num2str(rem(conf2,1)*10^3)));
drawBCCE(plotname,VP);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% C: Learning

global learning_style

%learning_style = 'fp';                                                     % Learning style: 'fp' or 'rm'
learning_style = 'rm';                                                      % Learning style: 'fp' or 'rm'

% For Identification figures: 
Ntrain = 1;                                                                    % Training period: 'phase-in' period where players play actions uniformly at random
M = maxiters;                                                                  % Number of time periods (past the training period)
M_obs = maxiters;                                                               % How many observations the econometrician gets to see at the END of the sample

numdst_t = 2;                                                               % FULL DIstr with training period; now these are dist at different points in time!
numdst_t_obs = numdst_t;                                                           % NO TRAINING PERIOD! now these are dist at different points in time!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning algo
rng(11111);

[distY_time, distY_time_obs, actions, regret, type_inds] = learn(Ntrain,M,M_obs,numdst_t,numdst_t_obs);

% create regret matrix

regret_per_period = zeros(Ntrain+M,s,NPlayers);

for p = 1:NPlayers
    for jj = 2:Ntrain+M
        for ii = 1:s
            if type_inds(jj,ii,p) == 1
            regret_per_period(jj,ii,p) = regret(jj,p);
            else 
            regret_per_period(jj,ii,p) = regret_per_period(jj-1,ii,p)*(jj-1)/jj;    
            end
        end
    end
end


%% Convergence Figure

close


numdists = 1000;
filename = fullfile(paths.figures_i, 'Learning2A_s20');

drawConvergence(filename,numdists,M,Ntrain,actions,VP)

%% Plot regrets

filename22 = fullfile(paths.figures_i, 'RegretsPerPeriodPlot');
drawRegrets_per_period(filename22,regret_per_period)

%%
