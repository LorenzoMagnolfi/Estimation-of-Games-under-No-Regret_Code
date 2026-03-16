%% Clean Up
clear all; clc; close all;

%% A: Setup
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

% Build config struct (replaces all globals)
% Stage I uses 2-action game
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;8];          % 2 actions for polytope demo
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 20;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare discretized set of directions u to find vertices of Polytope of Predictions
%% B: Polytope

conf_set = [0.1, 0.025, 0.05];
switch_eps = 3;

for jj = 1:length(conf_set)

conf2 =  conf_set(jj);

VP = find_polytope_switch(maxiters,conf2,switch_eps,cfg);

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

cfg.learning_style = 'rm';

% For Identification figures:
Ntrain = 1;                                                                    % Training period: 'phase-in' period where players play actions uniformly at random
M = maxiters;                                                                  % Number of time periods (past the training period)
M_obs = maxiters;                                                               % How many observations the econometrician gets to see at the END of the sample

numdst_t = 2;                                                               % FULL DIstr with training period; now these are dist at different points in time!
numdst_t_obs = numdst_t;                                                           % NO TRAINING PERIOD! now these are dist at different points in time!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning algo
rng(11111);

[distY_time, distY_time_obs, actions, regret, type_inds] = learn(cfg,Ntrain,M,M_obs,numdst_t,numdst_t_obs);

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

drawConvergence(cfg, filename, numdists, M, Ntrain, actions, VP)

%% Plot regrets

filename22 = fullfile(paths.figures_i, 'RegretsPerPeriodPlot');
drawRegrets_per_period(cfg, filename22, regret_per_period)

%%
