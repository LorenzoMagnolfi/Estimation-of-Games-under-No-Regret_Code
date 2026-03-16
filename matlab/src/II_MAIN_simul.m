%% Clean Up
clear all; clc; close all;
clear global;

%diary II_simul

%% A: Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Globals

global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resolve repository-relative paths and keep optimization templates on path.
paths = df_repo_paths();

% Initialize random number generator to seed 54321
rng(12345);

% Define the maxiters values to loop over
maxiters_values = [500000, 1000000, 2000000, 4000000];
% maxiters_values = [500000];

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
%     action_space{ind,1} = linspace(3,10,5)';                                                     % Possible Actions (Column Vector)
     action_space{ind,1} = [4;5;6;7;8];                                                     % 3 actions!
%     action_space{ind,1} = [3;10];                                                     % 2 actions!

end

AA = action_space{1,1};
AH = AA(2);
AL = AA(1);

A = allcomb(action_space{1,:},action_space{2,:});
NAct = size(action_space{1,:},1);
NActPr = size(A,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NEW Grid of Epsilons

step_size = (1/8)*ones(NPlayers,1);                                                 % Step Size: governs number of possible types (currently identically spaced)
% 1/3.3 corresponds to s=20;
% 1/8 corresponds to s=50

s=5;

% This FUNCTION??
[type_space,marg_distrib] = marginal_cost_draws_v5(mu,sigma2,s);

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

%% C: Learning

global learning_style

%learning_style = 'fp';                                                     % Learning style: 'fp' or 'rm'
learning_style = 'rm';                                                      % Learning style: 'fp' or 'rm'
                                                        % How many observations the econometrician gets to see at the END of the sample

numdst_t = 1;                                                               % FULL DIstr with training period; now these are dist at different points in time!
numdst_t_obs = numdst_t;                                                           % NO TRAINING PERIOD! now these are dist at different points in time!

for maxiter_index = 1:length(maxiters_values)
    
maxiters = maxiters_values(maxiter_index);

% For Identification figures: 
N = 1;                                                                    % Training period: 'phase-in' period where players play actions uniformly at random
M = maxiters;                                                                  % Number of time periods (past the training period)
M_obs = maxiters;       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning algo

% [distY_time, distY_time_obs, actions, regret, type_inds] = learn(N,M,M_obs,numdst_t,numdst_t_obs);
[distY_time, ~] = learn_mod(N, M, M_obs, numdst_t, numdst_t_obs,1,1);
%toc 

%% Plot regrets

%{
conf = 0.05;

filename = 'RegretsPlot_eps';
drawRegrets_eps(filename,type_inds,regret,maxiters,conf)
%}
action_distribution = distY_time;

%%

% grid parameters
% NGrid = 100;             % how many candidate params
Stepsize = 0.05;         % spacing

NGridV = 100;
NGridM = 100;
NGrid = NGridV*NGridM;

%% Need a function for the params grid!

% Initialize Parameters
% Players
NPlayers = 2;                                                                    % Number of agents
% plot_param = 'Variance';
plot_param = 'Both';

% Now, construct a grid of CANDIDATE Distributional Parameters

    if maxiter_index == 1
        gridparamV = [1; linspace(0.15, sigma2(1,1)*10, NGridV)'];
    elseif maxiter_index == 2
        gridparamV = [1; linspace(0.15, sigma2(1,1)*6, NGridV)'];
    else % for 2 million and 4 million
        gridparamV = [1; linspace(0.15, sigma2(1,1)*3.5, NGridV)'];
    end
    gridparamM = [1;linspace(0.55,mu(1,1)*0.5,NGridM)'];

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
            distribution_parameters{2,(ind1-1)*NGridM+ind2} = gridparamM(ind1)*mu';
            distribution_parameters{3,(ind1-1)*NGridM+ind2} = gridparamV(ind2)*sigma2';
            distpars((ind1-1)*NGridM+ind2,:) = [gridparamM(ind1)*mu(1,1),gridparamV(ind2)*sigma2(1,1)];
        end
    end
end

distribution_parameters_all(maxiter_index,:,:) =  distpars;

switch_eps = 1;

alpha_set = [0.05];
% alpha_set = [0.05, 0.01];

numdist= size(action_distribution,2);
num_alpha = length(alpha_set);

%% Let's Play the Pricing Game

maxvals = zeros(numdist,num_alpha,NGrid);

for ii = 1:numdist
    
    T=ii*(maxiters/numdist);
    distrib = action_distribution(:,ii);
    
    for jj = 1:num_alpha
        
    confid = alpha_set(jj);       
    
    
    outs = ComputeBCCE_eps(type_space,action_space,distrib,alpha,distribution_parameters,maxiters,confid,Pi,switch_eps);

    maxvals(ii,jj,:) = cell2mat(outs);
    end
end

VV = squeeze(maxvals) ;
VV_all(maxiter_index,:) = VV;

%% Fixture: save per-iteration intermediates
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
save(fullfile(fixture_dir, sprintf('stage_ii_iter_%dk.mat', maxiters/1000)), ...
    'distY_time', 'action_distribution', 'VV', 'distpars', 'maxiters', ...
    'distribution_parameters', 'switch_eps');

end

%% Fixture: save consolidated solver outputs
save(fullfile(fixture_dir, 'stage_ii_solver_all.mat'), ...
    'VV_all', 'distribution_parameters_all', 'maxiters_values');

for maxiter_index = 1:length(maxiters_values)

%% Plots
%% Computing Identified Sets

VV = VV_all(maxiter_index,:);
distpars = squeeze(distribution_parameters_all(maxiter_index,:,:));

id_set_index = (VV<= 1e-12);
ddpars = repmat(distpars,1,num_alpha,1)';

%% Plot before SVM

%{
figure 
hold on
ss1 = gscatter(ddpars(1,:)',ddpars(2,:)',id_set_index,'green','.',20);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM for 2d plot

Xtrain = ddpars';
YTrain = id_set_index;

% SVM Grid:

NGx= 500000;
ppx= haltonset(2,'Skip',1e3,'Leap',1e2);
PG0x = net(ppx,NGx-1);

    if maxiter_index == 1
        PGx = [PG0x(:,1)*7.5, PG0x(:,2)*10];
    elseif maxiter_index == 2
        PGx = [PG0x(:,1)*7, PG0x(:,2)*6];
    else % for 2 million and 4 million
        PGx = [PG0x(:,1)*6.5, PG0x(:,2)*3.5];
    end

% SVMMod = fitcsvm(Xtrain,YTrain)

SVMMod = fitcsvm(Xtrain,YTrain, ...
    'KernelFunction','gaussian','Standardize',true,'KernelScale','auto');
label = predict(SVMMod,PGx);

%% Generate F

PG_IdSet = [PGx(label>0,:)];

figure 
hold on
s1 = gscatter(PGx(:,1),PGx(:,2),label(:), 'wc', '.', 20);
s2 = plot(3,1,'.','MarkerSize',25,'color','k');

xlabel('$\mu$','Interpreter','latex') 
ylabel('$\sigma$','Interpreter','latex') 

 % Extract identified set points
    % id_set_points = PGx(label == 1, :);
    id_set_points = ddpars(:, id_set_index);
    % Determine min and max along each axis
    % min_mu = min(id_set_points(:,1));
    % max_mu = max(id_set_points(:,1));
    % min_sigma = min(id_set_points(:,2));
    % max_sigma = max(id_set_points(:,2));
    min_mu = min(id_set_points(1,:));
    max_mu = max(id_set_points(1,:));
    min_sigma = min(id_set_points(2,:));
    max_sigma = max(id_set_points(2,:));
    
    if maxiter_index == 1 || maxiter_index ==3
    % Hovering line positions
    hover_offset = 0.9; % Adjust this value to control the hover height
    else
    hover_offset = 1.2; % Adjust this value to control the hover height
    end

    % Plot horizontal projection (mu range)
    plot([min_mu max_mu], [min_sigma min_sigma] - hover_offset, 'r-', 'LineWidth', 1.5);
    s3 =  plot(min_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r');
    plot(max_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r');
    
    % Plot vertical projection (sigma range)
    plot([min_mu min_mu] - hover_offset, [min_sigma max_sigma], 'g-', 'LineWidth', 1.5);
    s4 = plot(min_mu - hover_offset, max_sigma, 'ko', 'MarkerFaceColor', 'g');
    plot(min_mu - hover_offset, min_sigma, 'ko', 'MarkerFaceColor', 'g');

%lgnd = legend([s1(2) s2 s3 s4],'Confidence region','True parameter',...
%            strcat('$\mu \in [', num2str(min_mu, '%.2f'), ', ', num2str(max_mu, '%.2f'), ']$'),...
%            strcat('$\sigma \in [', num2str(min_sigma, '%.2f'), ', ', num2str(max_sigma, '%.2f'), ']$'),...
%            'Location','northeast','NumColumns',2);

if maxiter_index == 1
lgnd = legend([s3 s4],strcat('$\mu \in [', num2str(min_mu, '%.2f'), ', ', num2str(max_mu, '%.2f'), ']$'),...
            strcat('$\sigma \in [', num2str(min_sigma, '%.2f'), ', ', num2str(max_sigma, '%.2f'), ']$'),...
            'Location','northeast');
else
lgnd = legend([s3 s4],strcat('$\mu \in [', num2str(min_mu, '%.2f'), ', ', num2str(max_mu, '%.2f'), ']$'),...
            strcat('$\sigma \in [', num2str(min_sigma, '%.2f'), ', ', num2str(max_sigma, '%.2f'), ']$'),...
            'Location','northeast','NumColumns',2);
end

set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',17)

if maxiter_index == 1
yticks([0 4 8 ])
elseif maxiter_index == 2
yticks([0 2 4 6])
else
yticks([0 2 4])
end


set(gca,'TickLabelInterpreter', 'latex');
set(gca,...
            'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
         'FontName','cmr10',...
        'FontSize',17,...
        'Box','off');
    fig_base = fullfile(paths.figures_ii, sprintf('IdSet_simul_%dk', 500*2^(maxiter_index-1)));
    saveas(gcf, fig_base, 'epsc')
    saveas(gcf, fig_base, 'png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
figure; hold on;

mupars = reshape(ddpars(1,:),NGridV,NGridM);
mupars = mupars(1,2:end);
VV_mu = reshape(max(VV,0),NGridV,NGridM);
VV_mu = VV_mu(1,2:end);
hhhh = plot(mupars,VV_mu,'r','linewidth',5);

ymax = VV_mu;
ymax = max(VV_mu(VV_mu < 1));
ymin = - 0.1;
plot([3,3],[ymin,ymax],'color','black','linewidth',3);


set(gcf,'color','white');

%lgnd = legend([hhhh],'$g(\mu;\sigma_0)$','Location','northeast');
%set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',9)

xl=xlabel('$\mu$');
set(xl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

yl=ylabel('$g\big(\mu;\sigma_0\big)$') ;
set(yl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

set(gca,'TickLabelInterpreter', 'latex');
set(gca,...
            'Units','normalized',...
             'FontName','cmr10',...
       'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',24,...
        'Box','off');
    fig_base = fullfile(paths.figures_ii, sprintf('g_mu_%dk', maxiters/1000));
    saveas(gcf, fig_base, 'epsc')
    saveas(gcf, fig_base, 'png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; hold on;

sigpars = reshape(ddpars(2,:),NGridV,NGridM);
sigpars = sigpars(2:end,1);
VV_sig = reshape(max(VV,0),NGridV,NGridM);
VV_sig = VV_sig(2:end,1);
plot(sigpars,VV_sig,'g','linewidth',5);


ymax = VV_sig;
ymax = max(VV_sig(VV_sig < 1));
ymin = - 0.05;
plot([1,1],[ymin,ymax],'color','black','linewidth',3);


set(gcf,'color','white');

%lgnd = legend([hhhh],'$g(\mu;\sigma_0)$','Location','northeast');
%set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',9)

xl=xlabel('$\sigma$');
set(xl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

yl=ylabel('$g\big(\sigma;\mu_0\big)$') ;
set(yl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

set(gca,'TickLabelInterpreter', 'latex');
set(gca,...
            'Units','normalized',...
             'FontName','cmr10',...
       'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',24,...
        'Box','off');
    fig_base = fullfile(paths.figures_ii, sprintf('g_sig_%dk', maxiters/1000));
    saveas(gcf, fig_base, 'epsc')
    saveas(gcf, fig_base, 'png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%figure 
%hold on
%scatter3(Xtrain(:,1),Xtrain(:,2),VV,30, VV, 'filled')
%plot3(PG_IdSet(:,1),PG_IdSet(:,2),zeros(length(PG_IdSet(:,1))))
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%{

if numdist>1

id_set = [];
id_set_bds = zeros(numdist,num_alpha,2);

table = []
for i = 1:numdist
    for j=1:num_alpha
        id_set =  distpars(squeeze(id_set_index(i,j,:)));
        id_set_bds(i,j,:) = [min(id_set),max(id_set)];
        clear id_set
        
    end
end

table = [];

for jj = 1:num_alpha
    table = [table id_set_bds(:,jj,1) id_set_bds(:,jj,2)];
end

else
    
id_set = [];
id_set_bds = zeros(num_alpha,2);

table = []
for j=1:num_alpha
        id_set =  distpars(squeeze(id_set_index(j,:)));
        id_set_bds(j,:) = [min(id_set),max(id_set)];
        clear id_set
end

table = [];

for jj = 1:num_alpha
    table = [table id_set_bds(jj,1) id_set_bds(jj,2)];
end

end


    xlswrite(fullfile(paths.tables_ii, sprintf('Table1_%dk.xls', maxiters/1000)), round(table,2), 'PanelA')
    clear VV id_set_index ddpars SVMMod label PG_IdSet id_set_points
    clear min_mu max_mu min_sigma max_sigma
    clear mupars VV_mu sigpars VV_sig
    clear id_set id_set_bds table
end
%}

end
fprintf('Simulations completed for all maxiters values.\n');

