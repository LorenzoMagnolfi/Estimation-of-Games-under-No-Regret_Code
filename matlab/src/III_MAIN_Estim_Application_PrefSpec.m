%% Clean Up
clear all; clc; close all;
clear global;

%%

tic

% global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resolve repository-relative paths and keep optimization templates on path.
paths = df_repo_paths();

% Initialize random number generator to seed
rng(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot & devise identified sets for each player

% three different kinds of plots

%% Start from this exercise: Fix epsilon, for each of the 4 players plot id set for 3 values of
% variance
Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');

epsilon_grid = [0.05];
n_epsilon = size(epsilon_grid,2);

% SELECT WHICH PLAYERS IN THE APPLICATION!
players = 1:2;
n_players = size(players,2);

% How many parameter values? (change for 2 dim)
NGridV = 100;
NGridM = 100;
NGrid = NGridV*NGridM;

% How fine is the grid for payoff types?
n_types = 5;

%[outputs] = Identification_Pricing_Game_Application_funct_5Prices(maxiters,epsilon_grid,Dist_file,Prob_file,players,NGridV,NGridM,n_types);
[outputs] = Identification_Pricing_Game_ApplicationL(epsilon_grid,Dist_file,Prob_file,players,NGridV,NGridM,n_types);
maxvals = cell2mat(outputs);

%% Fixture: save raw solver output
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end
save(fullfile(fixture_dir, 'stage_iii_solver_raw.mat'), ...
    'maxvals', 'epsilon_grid', 'players', 'NGridV', 'NGridM', 'n_types');

% Initialize table for marginal cost statistics
mc_stats_table = cell2table(cell(2, 2), 'VariableNames', {'Residual_Mean', 'Residual_SD'}, ...
                             'RowNames', arrayfun(@(x) sprintf('Seller %d', x), 1:2, 'UniformOutput', false));

for iii=players


maxvals_iii = maxvals(NGrid*(iii-1)+1:NGrid*iii,:);

VV = maxvals_iii(:,3) ;
id_set_index = (VV<=1e-12);
ddpars = maxvals_iii(:,1:2);

%gscatter(ddpars(:,1),ddpars(:,2),id_set_index)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM for 2d plot

Xtrain = ddpars;
YTrain = id_set_index;

% SVM Grid:

NGx= 500000;
ppx= haltonset(2,'Skip',1e3,'Leap',1e2);
PG0x = net(ppx,NGx-1);
PGx =[PG0x(:,1)*abs((min(Xtrain(:,1))-max(Xtrain(:,1))))*1.2-abs(min(Xtrain(:,1)))*1.1,PG0x(:,2)*abs((min(Xtrain(:,2))-max(Xtrain(:,2))))*1.2-abs(min(Xtrain(:,2)))*1.1];


SVMMod = fitcsvm(Xtrain,YTrain, ...
    'KernelFunction','gaussian','Standardize',true,'KernelScale','auto');
label = predict(SVMMod,PGx);


%% Generate F
figure
hold on

s1 = gscatter(PGx(:,1), PGx(:,2), label(:), 'wb', '.', 20);
xlabel(['$\mu_' num2str(iii) '$'],'Interpreter','latex') 
ylabel(['$\sigma_' num2str(iii) '$'],'Interpreter','latex') 

 % Extract identified set points
    % id_set_points = PGx(label == 1, :);
    id_set_points = ddpars(id_set_index, :);
    % Determine min and max along each axis
    % min_mu = min(id_set_points(:,1));
    % max_mu = max(id_set_points(:,1));
    % min_sigma = min(id_set_points(:,2));
    % max_sigma = max(id_set_points(:,2));
    min_mu = min(id_set_points(:,1));
    max_mu = max(id_set_points(:,1));
    min_sigma = min(id_set_points(:,2));
    max_sigma = max(id_set_points(:,2));

    % Hovering line positions
    hover_offset = 80; % Adjust this value to control the hover height
    
    % Plot horizontal projection (mu range)
    plot([min_mu max_mu], [min_sigma min_sigma] - hover_offset, 'r-', 'LineWidth', 1.5);
    s2 =  plot(min_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r');
    plot(max_mu, min_sigma - hover_offset, 'ko', 'MarkerFaceColor', 'r');
    
    % Plot vertical projection (sigma range)
    plot([min_mu min_mu] - hover_offset, [min_sigma max_sigma], 'g-', 'LineWidth', 1.5);
    s3 = plot(min_mu - hover_offset, max_sigma, 'ko', 'MarkerFaceColor', 'g');
    plot(min_mu - hover_offset, min_sigma, 'ko', 'MarkerFaceColor', 'g');
     

% Update legend
lgnd = legend([s2 s3],...
            strcat(['$\mu_' num2str(iii) ' \in [', num2str(min_mu, '%.1f'), ', ', num2str(max_mu, '%.1f'), ']$']),...
            strcat(['$\sigma_' num2str(iii) ' \in [', num2str(min_sigma, '%.1f'), ', ', num2str(max_sigma, '%.1f'), ']$']),...
      'Location','northwest');
set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',18)

    set(gca,'TickLabelInterpreter', 'latex');
    set(gca,...
        'Units','normalized',...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontName','cmr10',...
        'FontSize',18,...
       'Box','off');

fig_base = fullfile(paths.figures_iii, strcat('Estim_Pl_', num2str(iii)));
saveas(gcf, fig_base, 'epsc')
saveas(gcf, fig_base, 'png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Draw from distrib of cost, find avg marginal cost draw

% Load sample of player-specific reference prices;
% Learn Num periods
[ref_prices_all,~,~]= xlsread(fullfile(paths.data, 'Ref_Price_for_Device_Day.xlsx'), strcat('Seller_',num2str(iii)));

% Load actions
[distrib, actions, ~, period] = get_player_data_5acts(iii, 'mean', Dist_file, Prob_file);

% Initialize arrays to store results
n_draws = 200;
n_samples = 2000;
cost_stats = zeros(n_draws, 5); % mean, median, p25, p75, sd
tot_cost_stats = zeros(n_draws, 5);
avg_prices = zeros(n_draws, 1);
avg_markups = zeros(n_draws, 1);

% Get the identified set
id_set = ddpars(id_set_index, :);

rng(12345)

% Draw parameters from the identified set
n_id_set = sum(id_set_index);
random_indices = randi(n_id_set, n_draws, 1);
drawn_params = id_set(random_indices, :);

for theta = 1:n_draws
    mu = drawn_params(theta, 1);
    sigma2 = drawn_params(theta, 2);
    
    % Get support and marginal distribution of costs

    P_l = actions(1);
    P_h = actions(5);
    diff_p = P_h-P_l;
    mid = P_l+(1/2*diff_p);

    ub = P_h +0.25*diff_p;
    lb = P_l-3*diff_p;

    % Type space
    s=n_types;
    type_space = cell(1,1);
    type_space{1,1} = linspace(lb,ub,s)';
    
    marg_distrib = pdf('Normal',type_space{1,1},mu,sigma2);
    marg_distrib = marg_distrib./sum(marg_distrib,1);

    % Generate samples from the distribution
    cumulative_dist = cumsum(marg_distrib);
    cum_dist_refp = (1:period+1)/(period+1);
    marginal_costs = zeros(n_samples, 1);
    marginal_costs_tot = zeros(n_samples,1);
    for i = 1:n_samples
        r = rand();
        r2 = rand();
        idx = find(cumulative_dist >= r, 1, 'first');
        marginal_costs(i) = type_space{1}(idx);
        idx2 = find(cum_dist_refp >= r2, 1, 'first');
        marginal_costs_tot(i) = type_space{1}(idx)+ref_prices_all(idx2);
    end
    
    % Compute statistics
    cost_stats(theta, :) = [mean(marginal_costs), median(marginal_costs), ...
                            prctile(marginal_costs, 25), prctile(marginal_costs, 75), ...
                            std(marginal_costs)];
    tot_cost_stats(theta,:) = [mean(marginal_costs_tot), median(marginal_costs_tot), ...
                            prctile(marginal_costs_tot, 25), prctile(marginal_costs_tot, 75), ...
                            std(marginal_costs_tot)];

    % Compute average price
    avg_prices(theta) = sum(actions * (distrib*(kron(eye(5),ones(5,1))))');
    
    % Compute average markup
    avg_markups(theta) = avg_prices(theta) - mean(marginal_costs);
end

% Compute intervals
cost_resid_intervals = [min(cost_stats); max(cost_stats)];
tot_cost_intervals = [min(tot_cost_stats); max(tot_cost_stats)];
tot_cost_sd(iii,:) = tot_cost_intervals(:,end)';


    % Create tables with consistent data types
    cost_table = array2table(cost_resid_intervals', 'VariableNames', {'Min', 'Max'}, ...
        'RowNames', {'Mean', 'Median', 'P25', 'P75', 'SD'});
    total_cost_table = array2table(tot_cost_intervals', 'VariableNames', {'Min', 'Max'}, ...
        'RowNames', {'Mean', 'Median', 'P25', 'P75', 'SD'});

    % Convert numeric data to cell arrays for consistency
    cost_table.Min = num2cell(cost_table.Min);
    cost_table.Max = num2cell(cost_table.Max);
    total_cost_table.Min = num2cell(total_cost_table.Min);
    total_cost_table.Max = num2cell(total_cost_table.Max);

        % Populate the marginal cost statistics table
   if iii <= 4
        mc_stats_table{iii, 1} = {sprintf('[%.1f, %.1f]', min(cost_stats(:, 1)), max(cost_stats(:, 1)))};
        mc_stats_table{iii, 2} = {sprintf('[%.1f, %.1f]', min(cost_stats(:, 5)), max(cost_stats(:, 5)))};
    end

    avg_ref_prices(iii) = mean(ref_prices_all);
    sd_ref_prices(iii) = std(ref_prices_all);

    %% Fixture: save per-player identification + cost stats
    save(fullfile(fixture_dir, sprintf('stage_iii_player_%d.mat', iii)), ...
        'VV', 'id_set_index', 'ddpars', 'cost_stats', 'tot_cost_stats', ...
        'id_set_points', 'min_mu', 'max_mu', 'min_sigma', 'max_sigma');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now add average ref prices to get cost statistics

% Load Swappa reference price data
swappa_Ref_data = readtable(fullfile(paths.data, 'RefPrice.csv'));
swappa_Ref_data_sold = readtable(fullfile(paths.data, 'RefSoldPrice.csv'));

% Calculate average reference price for sellers 1-4
% avg_ref_prices = mean(table2array(swappa_Ref_data(1:4, 3:end)), 2, 'omitnan');


avg_ref_prices_sold = mean(table2array(swappa_Ref_data_sold(1:4, 3:end)), 2, 'omitnan');

% Update mc_stats_table
mc_stats_table.RefPrice_Mean = cell(2,1);
mc_stats_table.RefPrice_SD = cell(2,1);
mc_stats_table.MarginalCost_Mean = cell(2,1);
mc_stats_table.MarginalCost_SD = cell(2,1);
mc_stats_table = mc_stats_table(:, {'Residual_Mean', 'Residual_SD', 'MarginalCost_Mean', 'MarginalCost_SD', 'RefPrice_Mean', 'RefPrice_SD'});
mc_stats_table.Properties.VariableNames = {'Residual_Mean', 'Residual_SD', 'MarginalCost_Mean', 'MarginalCost_SD', 'RefPrice_Mean', 'RefPrice_SD'};


for i = 1:2
    mc_stats_table.RefPrice_Mean{i} = sprintf('%.1f', avg_ref_prices(i));
    mc_stats_table.RefPrice_SD{i} = sprintf('%.1f', sd_ref_prices(i));
    residual_min = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, '[', ','));
    residual_max = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, ',', ']'));
    mc_min = residual_min + avg_ref_prices(i);
    mc_max = residual_max + avg_ref_prices(i);
    mc_stats_table.MarginalCost_Mean{i} = sprintf('[%.1f, %.1f]', mc_min, mc_max);
    mc_stats_table.MarginalCost_SD{i} = sprintf('[%.1f, %.1f]', tot_cost_sd(i,1), tot_cost_sd(i,2));
end

% Write updated table to Excel
updated_stats_xlsx = fullfile(paths.tables_iii, 'Updated_Marginal_Cost_Statistics.xlsx');
if isfile(updated_stats_xlsx)
    delete(updated_stats_xlsx);
end
writetable(mc_stats_table, updated_stats_xlsx, 'WriteRowNames', true);

% Generate LaTeX for updated marginal cost statistics table
updated_mc_stats_latex = table2latex(mc_stats_table, 'Updated Marginal Cost Statistics', 'tab:updated_mc_stats');

% Write updated LaTeX table to file
fid = fopen(fullfile(paths.tables_iii, 'updated_marginal_cost_statistics.tex'), 'w');
fprintf(fid, '%s', updated_mc_stats_latex);
fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Gazelle data, prices and markups!

% Load necessary data
swappa_data = readtable(fullfile(paths.data, 'SwappaListingPrice.csv'));
gazelle_data = readtable(fullfile(paths.data, 'Gazelle_data.csv'));

% Define devices of interest
devices = {'Average',  'iPhone_13_128GB_Mint', ...
           'iPhone_12_64GB_Good', 'iPhone_11_64GB_Good'};

% Initialize new markup table
new_markup_table = cell2table(cell(4, length(devices)), 'VariableNames', devices, ...
    'RowNames', {'Swappa Price', 'Seller 1 Markup', 'Seller 2 Markup', ...
                    'Gazelle Markup'});
%                 'Seller 3 Markup', 'Seller 4 Markup', 'Gazelle Markup'});

% Fill Swappa Price row
for i = 1:length(devices)
    new_markup_table{1, devices{i}} = {num2str(mean(swappa_data{1:4, devices{i}}, 'omitnan'))};
end

% Fill Seller Markup rows
for i = 1:2
    for j = 1:length(devices)
        swappa_price = swappa_data{i, devices{j}};
        residual_min = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, '[', ','));
        residual_max = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, ',', ']'));
        %ref_price = swappa_Ref_data_sold{i, devices{j}};
        ref_price = swappa_Ref_data{i, devices{j}};
        markup_min = swappa_price - (residual_max + ref_price);
        markup_max = swappa_price - (residual_min + ref_price);
        new_markup_table{i+1, devices{j}} = {sprintf('[%.1f, %.1f]', markup_min, markup_max)};
    end
end

% Fill Gazelle Markup row
for i = 1:length(devices)
    sell_price = gazelle_data{1, devices{i}};
    buy_price = gazelle_data{2, devices{i}};
    markup_min = sell_price - buy_price*1.3;
    markup_max = sell_price - buy_price*1.1;
    new_markup_table{4, devices{i}} = {sprintf('[%.1f, %.1f]', markup_min, markup_max)};
end

% Write new markup table to Excel
writetable(new_markup_table, fullfile(paths.tables_iii, 'New_Markup_Table.xlsx'), 'WriteRowNames', true);

% Generate LaTeX for new markup table
new_markup_latex = table2latex(new_markup_table, 'New Markup Table', 'tab:new_markup');

% Write new LaTeX table to file
fid = fopen(fullfile(paths.tables_iii, 'new_markup_table.tex'), 'w');
fprintf(fid, '%s', new_markup_latex);
fclose(fid);

%%
