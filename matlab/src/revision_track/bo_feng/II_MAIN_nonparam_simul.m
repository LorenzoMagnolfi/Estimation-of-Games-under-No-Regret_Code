%% Clean Up
clear all; clc; close all;
clear global;

%diary II_simul

%% A: Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define Globals

global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Navigate to Parent Folder 
% cd("A:\Dropbox\DynamicFoundations\Matlab_Code\LatestCode\")\
%cd("C:\Users\lorem\Dropbox\DynamicFoundations\Matlab_Code\LatestCode\")
% cd("C:\Users\DELL\Desktop\No Regret\replication package for matlab")
cd("/Users/bofeng/git_repos/estimation_of_games_under_no_regret/replication package for matlab")

% Initialize random number generator to seed 54321
rng(12345);

% Runtime tracking (start)
startTime = datetime('now');
tStart = tic;

% Define the maxiters values to loop over
maxiters_values = [500000, 1000000, 2000000, 4000000];

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

% Set s value manually - change this value as needed
s = 5; % s candidate values: 5, 10, 20, 50

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

% grid parameters - dynamic based on s value
if s == 5 || s == 10
    NGrid = 10000;           % 10k params for s=5 and s=10
    K_local = 1000;          % 1k local candidates
    K_global = 9000;         % 9k global candidates
elseif s == 20
    NGrid = 5000;            % 5k params for s=20
    K_local = 500;           % 500 local candidates
    K_global = 4500;         % 4.5k global candidates
end

NGridV = sqrt(NGrid);
NGridM = sqrt(NGrid);

%% Need a function for the params grid!

% Initialize Parameters
% Players
NPlayers = 2;                                                                    % Number of agents

% Construct grid of candidate marginal probability vectors (length s)
% Dynamic grid sizing based on s value
local_width = 20;        % larger width for perturbations: rand(s,1)/local_width

distribution_parameters = cell(1, NGrid);

% 1) Include the true marginal distribution (column vector s x 1)
distribution_parameters{1} = marg_distrib(:);

% 2) Local perturbations around the true marginal (indices 2..K_local)
for ind = 2:K_local
    candidate = marg_distrib(:) + rand(s,1) / local_width;
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

% 3) Global draws over the full simplex (Dirichlet(1) via exponential trick)
for ind = (K_local+1):NGrid
    w = -log(rand(s,1));
    candidate = w ./ sum(w);
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

% 4) Spiky distributions for better low-sigma coverage (200 candidates)
K_spiky = 200;
spike_multiplier = 3;  % Tunable hyperparameter
n_adjacent = 4;        % Number of adjacent points to keep on EACH side of peak

for ind = (NGrid+1):(NGrid+K_spiky)
    % Draw random numbers for each support point
    random_draws = rand(s,1);
    
    % Find the index with the highest draw (the peak)
    [~, peak_idx] = max(random_draws);
    
    % Initialize candidate with zeros
    candidate = zeros(s,1);
    
    % Set the peak value (multiplied by spike_multiplier)
    candidate(peak_idx) = random_draws(peak_idx) * spike_multiplier;
    
    % Keep n_adjacent points on each side at their random draws
    for offset = 1:n_adjacent
        % Left neighbor(s)
        if peak_idx - offset > 0
            candidate(peak_idx - offset) = random_draws(peak_idx - offset);
        end
        % Right neighbor(s)
        if peak_idx + offset <= s
            candidate(peak_idx + offset) = random_draws(peak_idx + offset);
        end
    end
    
    % Normalize to sum to 1
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

% Update NGrid to include spiky distributions
NGrid = NGrid + K_spiky;

% Reconstruct a matrix of distribution parameters [NGrid x 2]
% Column 1: Mean of Player 1's marginal cost distribution
% Column 2: Variance of Player 1's marginal cost distribution
distpars = zeros(NGrid, 2);
support_player1 = type_space{1,1}; % s x 1 support for Player 1
for ind = 1:NGrid
    p = distribution_parameters{ind}; % s x 1 probability vector
    p = p(:) ./ sum(p(:));            % ensure normalization
    mu1 = sum(p .* support_player1);
    var1 = sum(p .* (support_player1 - mu1).^2);
    distpars(ind,1) = mu1;
    distpars(ind,2) = var1;
end
distribution_parameters_all(maxiter_index,:,:) = distpars;

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


end



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
% Plot: Visualize full evaluated grid of parameters with membership coloring
figure; hold on
gscatter(ddpars(1,:)', ddpars(2,:)', id_set_index, 'kc', '.', 10);
plot(3,1,'.','MarkerSize',25,'color','k');
xlabel('$\mu$','Interpreter','latex') 
ylabel('$\sigma$','Interpreter','latex') 
set(gca,'TickLabelInterpreter','latex');
set(gca,'FontName','cmr10','FontSize',17,'Box','off');

% Add hyperparameter info to title and filename
iteration_value = 500*2^(maxiter_index-1);
hyperparam_info = sprintf('K=%d,mult=%.1f,nadj=%d', K_spiky, spike_multiplier, n_adjacent);
title(sprintf('Evaluated grid (N=%d, %s) | s=%d, iter=%dk', size(ddpars,2), hyperparam_info, s, iteration_value), 'Interpreter','none');

% Save with hyperparameter info in filename
safe_hyperparam = sprintf('K%d_mult%.1f_nadj%d', K_spiky, spike_multiplier, n_adjacent);
saveas(gcf, sprintf('nonparam_FullGrid_s%d_%dk_%s.png', s, iteration_value, safe_hyperparam));
saveas(gcf, sprintf('nonparam_FullGrid_s%d_%dk_%s.eps', s, iteration_value, safe_hyperparam), 'epsc');

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM CODE COMMENTED OUT - Only showing evaluated grid per professor's request
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SVM hyperparameter sweep without re-estimation
kernels = {'gaussian','linear'}; % extend as needed
kernelScales = {'auto', 1};
boxConstraints = [0.1, 1, 10];

for kk = 1:numel(kernels)
    kern = kernels{kk};
    for ss_ = 1:numel(kernelScales)
        ks = kernelScales{ss_};
        for bb = 1:numel(boxConstraints)
            bx = boxConstraints(bb);
            % Train and predict
            SVMMod = fitcsvm(Xtrain,YTrain, 'KernelFunction',kern, 'KernelScale',ks, 'Standardize',true, 'BoxConstraint',bx);
            label = predict(SVMMod,PGx);

            % Plot SVM prediction over PGx and overlay evaluated points
            figure; hold on
            s1 = gscatter(PGx(:,1),PGx(:,2),label(:), 'wc', '.', 20);
            s2 = plot(3,1,'.','MarkerSize',25,'color','k');
            scatter(ddpars(1,:)', ddpars(2,:)', 6, [0.6 0.6 0.6], '.');

            xlabel('$\mu$','Interpreter','latex') 
            ylabel('$\sigma$','Interpreter','latex') 

            % Extract identified set points
            id_set_points = ddpars(:, id_set_index);
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
            set(gca,'TickLabelInterpreter', 'latex');
            set(gca,'Units','normalized','FontUnits','points','FontWeight','normal','FontName','cmr10','FontSize',17,'Box','off');

            % SVM hyperparameter info in title and filenames
            iteration_value = 500*2^(maxiter_index-1);
            svm_info = sprintf('kernel=%s,scale=%s,box=%.3g', kern, mat2str(ks), bx);
            title(sprintf('Id set via SVM (%s) | s=%d, iter=%dk', svm_info, s, iteration_value), 'Interpreter','none');
            safe_info = regexprep(svm_info,'[^a-zA-Z0-9_=\-\.]','_');
            saveas(gcf, sprintf('nonparam_IdSet_simul_s%d_%dk_%s.png', s, iteration_value, safe_info));
            saveas(gcf, sprintf('nonparam_IdSet_simul_s%d_%dk_%s.eps', s, iteration_value, safe_info), 'epsc');
        end
    end
end

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

% Add text annotation showing s and iteration values
iteration_value = 500*2^(maxiter_index-1);
text_info = sprintf('$s = %d$, $\\mathrm{iterations} = %d$', s, iteration_value);
text(0.02, 0.98, text_info, 'Units', 'normalized', ...
     'Interpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 14, ...
     'BackgroundColor', 'white', 'EdgeColor', 'black', ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

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
%}


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
    saveas(gcf, sprintf('g_mu_%dk', maxiters/1000), 'epsc')
    saveas(gcf, sprintf('g_mu_%dk', maxiters/1000), 'png')

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
    saveas(gcf, sprintf('g_sig_%dk', maxiters/1000), 'epsc')
    saveas(gcf, sprintf('g_sig_%dk', maxiters/1000), 'png')

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


    xlswrite(sprintf('Table1_%dk.xls', maxiters/1000), round(table,2), 'PanelA')
    clear VV id_set_index ddpars SVMMod label PG_IdSet id_set_points
    clear min_mu max_mu min_sigma max_sigma
    clear mupars VV_mu sigpars VV_sig
    clear id_set id_set_bds table
end
%}

end
fprintf('Simulations completed for all maxiters values.\n');

% Runtime tracking (end)
elapsed = toc(tStart);
endTime = datetime('now');

% Format elapsed as HH:MM:SS robustly
elapsedHHMMSS = datestr(elapsed/(24*3600), 'HH:MM:SS');
startStr = datestr(startTime,'yyyy-mm-dd HH:MM:SS');
endStr = datestr(endTime,'yyyy-mm-dd HH:MM:SS');
stampStr = datestr(now,'yyyy-mm-dd HH:MM:SS');

summaryLine = sprintf('[%s] II_MAIN_nonparam_simul: s=%d, elapsed=%s (%.3fs), start=%s, end=%s\n', ...
    stampStr, s, elapsedHHMMSS, elapsed, startStr, endStr);

% Print to console
fprintf('%s', summaryLine);

% Append to runtime log file
logPath = 'runtime_log.txt';
fid = fopen(logPath,'a');
if fid ~= -1
    fprintf(fid, '%s', summaryLine);
    fclose(fid);
else
    warning('Could not open runtime_log.txt for appending: %s', logPath);
end

