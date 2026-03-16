function write_tables(results, paths, data_paths)
% DF.REPORT.WRITE_TABLES  Write Stage III cost and markup tables (xlsx + tex).
%
%   df.report.write_tables(results, paths, data_paths)
%
%   Merges residual cost statistics with reference prices and Gazelle data,
%   then writes Excel and LaTeX tables. Extracted from III_MAIN lines 253-355.
%
%   Inputs:
%     results    — struct from df.stages.run_stage_iii
%     paths      — paths struct from df.io.repo_paths
%     data_paths — struct with .data field (path to data directory)

mc_stats_table = results.mc_stats_table;
avg_ref_prices = results.avg_ref_prices;
sd_ref_prices = results.sd_ref_prices;
tot_cost_sd = results.tot_cost_sd;
players = results.players;

%% Load reference data
swappa_Ref_data = readtable(fullfile(data_paths.data, 'RefPrice.csv'));
swappa_Ref_data_sold = readtable(fullfile(data_paths.data, 'RefSoldPrice.csv'));

%% Update mc_stats_table with reference prices and total costs
mc_stats_table.RefPrice_Mean = cell(size(mc_stats_table, 1), 1);
mc_stats_table.RefPrice_SD = cell(size(mc_stats_table, 1), 1);
mc_stats_table.MarginalCost_Mean = cell(size(mc_stats_table, 1), 1);
mc_stats_table.MarginalCost_SD = cell(size(mc_stats_table, 1), 1);
mc_stats_table = mc_stats_table(:, {'Residual_Mean', 'Residual_SD', ...
    'MarginalCost_Mean', 'MarginalCost_SD', 'RefPrice_Mean', 'RefPrice_SD'});

for i = 1:numel(players)
    iii = players(i);
    mc_stats_table.RefPrice_Mean{i} = sprintf('%.1f', avg_ref_prices(iii));
    mc_stats_table.RefPrice_SD{i} = sprintf('%.1f', sd_ref_prices(iii));
    residual_min = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, '[', ','));
    residual_max = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, ',', ']'));
    mc_min = residual_min + avg_ref_prices(iii);
    mc_max = residual_max + avg_ref_prices(iii);
    mc_stats_table.MarginalCost_Mean{i} = sprintf('[%.1f, %.1f]', mc_min, mc_max);
    mc_stats_table.MarginalCost_SD{i} = sprintf('[%.1f, %.1f]', tot_cost_sd(iii, 1), tot_cost_sd(iii, 2));
end

%% Write updated stats
updated_stats_xlsx = fullfile(paths.tables_iii, 'Updated_Marginal_Cost_Statistics.xlsx');
if isfile(updated_stats_xlsx)
    delete(updated_stats_xlsx);
end
writetable(mc_stats_table, updated_stats_xlsx, 'WriteRowNames', true);

updated_mc_stats_latex = table2latex(mc_stats_table, 'Updated Marginal Cost Statistics', 'tab:updated_mc_stats');
fid = fopen(fullfile(paths.tables_iii, 'updated_marginal_cost_statistics.tex'), 'w');
fprintf(fid, '%s', updated_mc_stats_latex);
fclose(fid);

%% Markup table (Swappa + Gazelle)
swappa_data = readtable(fullfile(data_paths.data, 'SwappaListingPrice.csv'));
gazelle_data = readtable(fullfile(data_paths.data, 'Gazelle_data.csv'));

devices = {'Average', 'iPhone_13_128GB_Mint', ...
    'iPhone_12_64GB_Good', 'iPhone_11_64GB_Good'};

new_markup_table = cell2table(cell(numel(players) + 2, numel(devices)), ...
    'VariableNames', devices, ...
    'RowNames', [{'Swappa Price'}, ...
        arrayfun(@(x) sprintf('Seller %d Markup', x), players, 'UniformOutput', false), ...
        {'Gazelle Markup'}]);

% Swappa Price row
for i = 1:numel(devices)
    new_markup_table{1, devices{i}} = {num2str(mean(swappa_data{1:4, devices{i}}, 'omitnan'))};
end

% Seller Markup rows
for i = 1:numel(players)
    for j = 1:numel(devices)
        swappa_price = swappa_data{players(i), devices{j}};
        residual_min = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, '[', ','));
        residual_max = str2double(extractBetween(mc_stats_table.Residual_Mean{i}, ',', ']'));
        ref_price = swappa_Ref_data{players(i), devices{j}};
        markup_min = swappa_price - (residual_max + ref_price);
        markup_max = swappa_price - (residual_min + ref_price);
        new_markup_table{i + 1, devices{j}} = {sprintf('[%.1f, %.1f]', markup_min, markup_max)};
    end
end

% Gazelle Markup row
for i = 1:numel(devices)
    sell_price = gazelle_data{1, devices{i}};
    buy_price = gazelle_data{2, devices{i}};
    markup_min = sell_price - buy_price * 1.3;
    markup_max = sell_price - buy_price * 1.1;
    new_markup_table{end, devices{i}} = {sprintf('[%.1f, %.1f]', markup_min, markup_max)};
end

writetable(new_markup_table, fullfile(paths.tables_iii, 'New_Markup_Table.xlsx'), 'WriteRowNames', true);

new_markup_latex = table2latex(new_markup_table, 'New Markup Table', 'tab:new_markup');
fid = fopen(fullfile(paths.tables_iii, 'new_markup_table.tex'), 'w');
fprintf(fid, '%s', new_markup_latex);
fclose(fid);

end
