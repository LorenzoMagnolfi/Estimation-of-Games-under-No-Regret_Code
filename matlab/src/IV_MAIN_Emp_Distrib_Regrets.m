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

maxiters = 100000;

%% BOOTSTRAP

B = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters of the Game

% Build config struct (replaces all globals)
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);
s = 5;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

% Unpack what's needed locally
action_space = cfg.action_space;
type_space = cfg.type_space;
Pi = cfg.Pi;
marg_distrib = cfg.marg_distrib;
NAct = cfg.NAct;
A = cfg.A;

s2 = s^2;

%% Comparison of eps
%{
conf = 0.05;

eps1 = epsilon_switch(maxiters,conf,1,cfg);     % NEW eps linked to conv rate
eps0 = epsilon_switch(maxiters,conf,0,cfg);     % OLD eps linked to perc deviations

eps_fin1 = repmat(marg_distrib,1,NAct*NPlayers).*repmat(sqrt(marg_distrib),1,NAct*NPlayers).*repmat(eps1',1,NAct*NPlayers);
eps_fin0 = repmat(marg_distrib,1,NAct*NPlayers).*repmat(eps0',1,NAct*NPlayers);

ratio = mean(eps_fin1,2)./mean(eps_fin0,2);
%}
%% C: Learning

cfg.learning_style = 'rm';

% For Identification figures:
N = 1;                                                                    % Training period: 'phase-in' period where players play actions uniformly at random
M = maxiters;                                                                  % Number of time periods (past the training period)
M_obs = maxiters;                                                               % How many observations the econometrician gets to see at the END of the sample

numdst_t = 1;                                                               % FULL DIstr with training period; now these are dist at different points in time!
numdst_t_obs = numdst_t;                                                           % NO TRAINING PERIOD! now these are dist at different points in time!

% Load number of obs for players in application; to be used in Robustness
% spec
Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');

[~, ~, ~, Nobs_Pl1] = get_player_data_5acts(1, 'median', Dist_file, Prob_file);
[~, ~, ~, Nobs_Pl2] = get_player_data_5acts(2, 'median', Dist_file, Prob_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bootstrap Objects
final_regret = zeros(s,NPlayers,B);
Psi_emp = zeros(s,NPlayers,B);

Pl1_regret = zeros(s,B);
Pl2_regret = zeros(s,B);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Learning algo

for b = 1:B
[distY_time, distY_time_obs, fin, Pl1_EmpRegr, Pl2_EmpRegr] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
final_regret(:,:,b) = fin;

% Also save regrets for application, using # of obs for pl. 1 and 2;
Pl1_regret(:,b) = Pl1_EmpRegr;
Pl2_regret(:,b) = Pl2_EmpRegr;

%[distY_time, distY_time_obs, actions, regret, type_inds] = learn(N,M,M_obs,numdst_t,numdst_t_obs);

%{
for pl = 1:NPlayers
for jj = 1:s
% A vector of final regrets for each type/player/B
lreg  = regret(type_inds(:,jj,pl));
final_regret(jj,pl,b) = lreg(end);

% A psi frequency for each type/player/B
Psi_emp(jj,pl,b) = sum(type_inds(:,jj,pl))/maxiters;
end
end
%}

end

save(fullfile(paths.artifacts_iv, 'final_regret.mat'), "final_regret")

%% Fixture: save bootstrap learning outputs
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
save(fullfile(fixture_dir, 'stage_iv_bootstrap.mat'), ...
    'final_regret', 'Pl1_regret', 'Pl2_regret', 'distY_time', ...
    'B', 'maxiters', 'Nobs_Pl1', 'Nobs_Pl2');

% load("final_regret_10t.mat")
% load("Psi_emp_10t.mat")

%% COMPUTE THEORETICAL REGRETS
eps1 = epsilon_switch(maxiters,1,1,cfg).*0.05;     % NEW eps linked to conv rate
ExpectedRegretComp = sqrt(marg_distrib).*eps1';
avg_th_regret = sum(ExpectedRegretComp)/s;

% Now for parameters of application
epsPl1 = epsilon_switch(Nobs_Pl1,0.05,1,cfg).*0.05;     % NEW eps linked to conv rate
ExpectedRegretComp_Pl1 = sqrt(marg_distrib).*epsPl1';
aepsPl1 = NPlayers*s*ExpectedRegretComp_Pl1/0.05;

epsPl2 = epsilon_switch(Nobs_Pl2,0.05,1,cfg).*0.05;     % NEW eps linked to conv rate
ExpectedRegretComp_Pl2 = sqrt(marg_distrib).*epsPl1';
aepsPl2 = NPlayers*s*ExpectedRegretComp_Pl2/0.05;

%% Comparison

avg_exp_regret2 = mean(final_regret,3);
avg_exp_regret = mean(avg_exp_regret2(:,1));

ratio = avg_th_regret/avg_exp_regret;

regret_95perc = prctile(final_regret(:,:,:), 95, 3);
avg_regret_95perc = mean(regret_95perc);

regret_995perc = prctile(final_regret(:,:,:), 99.5, 3);
avg_regret_995perc = mean(regret_995perc);

ratio = sqrt(marg_distrib).*mean(epsilon_switch(maxiters,1,1,cfg))./avg_regret_95perc;

%% Comparisons for Application (by pl/type)

avg_exp_regret2_Pl1 = mean(Pl1_regret,2);
regret_95perc_Pl1 = prctile(Pl1_regret, 100*(1-(0.05/NPlayers*s)), 2);
ratio1_Pl1 = ExpectedRegretComp_Pl1./avg_exp_regret2_Pl1;
ratio2_Pl1 = aepsPl1./regret_95perc_Pl1;

avg_exp_regret2_Pl2 = mean(Pl2_regret,2);
regret_95perc_Pl2 = prctile(Pl2_regret, 100*(1-(0.05/NPlayers*s)), 2);
ratio1_Pl2 = ExpectedRegretComp_Pl2./avg_exp_regret2_Pl2;
ratio2_Pl2 = aepsPl2./regret_95perc_Pl2;

save(fullfile(paths.artifacts_iv, 'ratio1_Pl1.mat'), "ratio1_Pl1")
save(fullfile(paths.artifacts_iv, 'ratio2_Pl1.mat'), "ratio2_Pl1")
save(fullfile(paths.artifacts_iv, 'ratio1_Pl2.mat'), "ratio1_Pl2")
save(fullfile(paths.artifacts_iv, 'ratio2_Pl2.mat'), "ratio2_Pl2")

%% Fixture: save regret comparison outputs
save(fullfile(fixture_dir, 'stage_iv_regret_comparison.mat'), ...
    'avg_exp_regret2', 'avg_th_regret', 'ExpectedRegretComp', ...
    'regret_95perc', 'ratio1_Pl1', 'ratio2_Pl1', 'ratio1_Pl2', 'ratio2_Pl2');

%% Plot across all types
% Flatten the final_regret array across players and bootstrap iterations
all_regrets = reshape(final_regret, [], 1);

% Calculate various statistics
percentile_95 = prctile(all_regrets, 95);
mean_expected_regret = mean(ExpectedRegretComp);
average_empirical_regret = mean(all_regrets);
average_epsilon = mean(sqrt(marg_distrib).*mean(epsilon_switch(maxiters,1,1,cfg)));

% Create the figure
figure('Position', [100, 100, 800, 600]);

% Create histogram data
[counts, edges] = histcounts(all_regrets, 20, 'Normalization', 'probability');
centers = (edges(1:end-1) + edges(2:end)) / 2;

% Define the break point
break_start = max(percentile_95, mean_expected_regret) * 1.3;
break_end = average_epsilon * 0.98;
break_width = (break_end - break_start) / 50;  % Adjust this value to change the width of the break

% Use BreakXAxis to create the broken axis plot
h = BreakXAxis([centers average_epsilon], [counts 0.00001], break_start, break_end, break_width);
hold on;

% Recreate the histogram as bar plot
bar_width = centers(2) - centers(1);
h_bar = bar(centers, counts, bar_width*1000000, 'FaceColor', 'b', 'EdgeColor', 'b');

% Get the new x-axis limits after the break
new_xlim = xlim;

% Function to map original x-values to new x-values after the break
map_x = @(x) x - (x > break_start) * (break_end - break_start - break_width);

% Add vertical lines
l1 = line(map_x([average_empirical_regret average_empirical_regret]), ylim, 'Color', 'r', 'LineStyle', '-', 'LineWidth', 2);
l2 = line(map_x([percentile_95 percentile_95]), ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
l3 = line(map_x([mean_expected_regret mean_expected_regret]), ylim, 'Color', 'g', 'LineStyle', '-', 'LineWidth', 2);
l4 = line(map_x([average_epsilon average_epsilon]), ylim, 'Color', 'g', 'LineStyle', '--', 'LineWidth', 2);

% Add labels and title
xlabel('Regret', 'Interpreter', 'latex');
ylabel('Probability', 'Interpreter', 'latex');
%title('Distribution of Empirical Regrets and Theoretical Bounds', 'Interpreter', 'latex');

% Customize x-axis ticks
left_ticks = linspace(min(all_regrets), break_start, 3);
right_ticks = [break_end, average_epsilon];
all_ticks = [left_ticks, right_ticks];
mapped_ticks = map_x(all_ticks);

set(gca, 'XTick', mapped_ticks);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1e', x), all_ticks, 'UniformOutput', false));

% Add legend
lgnd = legend([h_bar, l1, l2, l3, l4], {'Empirical regrets', 'Average empirical regret', ...
               '$95^{th}$ percentile empirical regret', ...
               'Average worst-case expected regret', ...
               'Average $\varepsilon(i,t_i;\lambda)$'}, ...
       'Location', 'northeast');
set(lgnd, 'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 21)

% Adjust appearance
yticks([0 0.2 0.4 0.6 ])
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 21, 'Box', 'off');

% Adjust font size for better readability
set(findall(gcf, 'type', 'text'), 'FontSize', 21);

% Save the figure
fig_base = fullfile(paths.figures_iv, 'Modified_Exp_regr_comp');
saveas(gcf, fig_base, 'epsc');
saveas(gcf, fig_base, 'png');

%% NOW, Identification using an eps

%main change: have to pass eps to BCCE estim routine, modify function

%%
% HERE: Change, have it load a certain distY_time!
% Or perhaps: draw a new one?
action_distribution = distY_time;

%%

% grid parameters
% NGrid = 100;             % how many candidate params
Stepsize = 0.05;         % spacing

NGridV = 100;
NGridM =100;
NGrid = NGridV*NGridM;

%% Need a function for the params grid!

% Initialize Parameters
% Players
NPlayers = 2;    % Number of agents
% plot_param = 'Variance';
plot_param = 'Both';

% Now, construct a grid of CANDIDATE Distributional Parameters

% gridparam = linspace(0,Stepsize*sigma2(1,1)*NGrid,NGrid)'-Stepsize*sigma2(1,1)*NGrid/2;
%distpars =  gridparam + sigma2(1,1);

gridparamV = [1;linspace(0.15,sigma2(1,1)*2,NGridV)'];
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

switch_eps = 0;

alpha_set = [0.05];

numdist= size(action_distribution,2);
num_alpha = length(alpha_set);

%% Let's Play the Pricing Game

maxvals = zeros(numdist,num_alpha,NGrid);


for ii = 1:numdist

    T=ii*(maxiters/numdist);
    distrib = action_distribution(:,ii);

    for jj = 1:num_alpha

    confid = alpha_set(jj);

    regret_comp = avg_exp_regret2*(s*NPlayers);
    ExpRegr_pass = regret_comp./confid;

    outs = ComputeBCCE_eps_pass(type_space,action_space,distrib,alpha,distribution_parameters,T,confid,Pi,switch_eps,ExpRegr_pass);

    maxvals(ii,jj,:) = cell2mat(outs);
    end
end

%% Plots
%plot_Identified_Set;
%saveas(gcf,'Figure','epsc');

%% Computing Identified Sets
% num_plots = size(maxvals,3);

VV = squeeze(maxvals) ;
id_set_index = (VV<=1e-12);
ddpars = repmat(distpars,1,num_alpha,1)';

%% Fixture: save identification exercise outputs
save(fullfile(fixture_dir, 'stage_iv_identification.mat'), ...
    'VV', 'id_set_index', 'ddpars', 'distpars', 'ExpRegr_pass');

%% Plot before SVM
ss1 = gscatter(ddpars(1,:)',ddpars(2,:)',id_set_index,'green','.',20);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM for 2d plot

Xtrain = ddpars';
YTrain = id_set_index;

% SVM Grid:

NGx= 500000;
ppx= haltonset(2,'Skip',1e3,'Leap',1e2);
PG0x = net(ppx,NGx-1);
PGx =[PG0x(:,1)*5,PG0x(:,2)*3];

PGx = PGx(((PGx(:,1)>2) & (PGx(:,1)<4) & (PGx(:,2)>0.5) & (PGx(:,2)<1.5)),:);

% SVMMod = fitcsvm(Xtrain,YTrain)

SVMMod = fitcsvm(Xtrain,YTrain, ...
    'KernelFunction','gaussian','Standardize',true,'KernelScale','auto');
label = predict(SVMMod,PGx);

%% Generate F

PG_IdSet = [PGx(label>0,:)];

figure
hold on
s1 = gscatter(PGx(:,1),PGx(:,2),label(:), 'wc', '.', 40);
s2 = plot(3,1,'.','MarkerSize',40,'color','k');

xlabel('$\mu$','Interpreter','latex')
ylabel('$\sigma$','Interpreter','latex')

 % Extract identified set points
    id_set_points = PGx(label == 1, :);
    % Determine min and max along each axis
    min_mu = min(id_set_points(:,1));
    max_mu = max(id_set_points(:,1));
    min_sigma = min(id_set_points(:,2));
    max_sigma = max(id_set_points(:,2));

    % Hovering line positions
    hover_offset = 0.6; % Adjust this value to control the hover height

    % Plot horizontal projection (mu range)
    plot([min_mu max_mu], [min_sigma min_sigma] - hover_offset, 'r-', 'LineWidth', 1.5);
    s3 =  plot(min_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r','MarkerSize',10);
    plot(max_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r','MarkerSize',10);

    % Plot vertical projection (sigma range)
    plot([min_mu min_mu] - hover_offset, [min_sigma max_sigma], 'g-', 'LineWidth', 1.5);
    s4 = plot(min_mu - hover_offset, max_sigma, 'ko', 'MarkerFaceColor', 'g','MarkerSize',10);
    plot(min_mu - hover_offset, min_sigma, 'ko', 'MarkerFaceColor', 'g','MarkerSize',10);


    yticks([0.2 0.6 1 1.4 1.8])


lgnd = legend([s1(2) s2 s3 s4],'Confidence region','True parameter',...
            strcat('$\mu \in [', num2str(min_mu, '%.2f'), ', ', num2str(max_mu, '%.2f'), ']$'),...
            strcat('$\sigma \in [', num2str(min_sigma, '%.2f'), ', ', num2str(max_sigma, '%.2f'), ']$'),...
            'Location','north','NumColumns',2);
set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',30)

set(gca,'TickLabelInterpreter', 'latex');
set(gca,...
            'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
         'FontName','cmr10',...
        'FontSize',30,...
        'Box','off');
fig_base = fullfile(paths.figures_iv, 'IdSet_simul_exp_regr');
saveas(gcf, fig_base, 'epsc')
saveas(gcf, fig_base, 'png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
