function [distribution, actions, prob, period] = get_player_data_5acts(player_number, action_type, data_file, Prob_file)

% Local override for the R1.4.g robustness exercise.
% Matches the maintained function but allows the sale-probability sheet
% to be selected via the DF_SALEPROB_SHEET environment variable.

%% Get Distribution
[NUM,~,~] = xlsread(data_file, strcat('Seller_', num2str(player_number)));
period = size(NUM,1) - 1;

opts = spreadsheetImportOptions("NumVariables", 25);
opts.Sheet = ['Seller_' num2str(player_number)];
opts.DataRange = ['A' num2str(period + 1) ':Y' num2str(period + 1)];
opts.VariableNames = ["self0_comp0", "self0_comp1", "self0_comp2", "self0_comp3", "self0_comp4", ...
                      "self1_comp0", "self1_comp1", "self1_comp2", "self1_comp3", "self1_comp4", ...
                      "self2_comp0", "self2_comp1", "self2_comp2", "self2_comp3", "self2_comp4", ...
                      "self3_comp0", "self3_comp1", "self3_comp2", "self3_comp3", "self3_comp4", ...
                      "self4_comp0", "self4_comp1", "self4_comp2", "self4_comp3", "self4_comp4"];
opts.VariableTypes = ["double", "double", "double", "double", "double", ...
                      "double", "double", "double", "double", "double", ...
                      "double", "double", "double", "double", "double", ...
                      "double", "double", "double", "double", "double", ...
                      "double", "double", "double", "double", "double"];

distribution = readtable(data_file, opts, "UseExcel", false);
distribution = table2array(distribution);
clear opts

%% Get Actions
opts = spreadsheetImportOptions("NumVariables", 5);

if strcmpi(action_type, "mean")
    opts.Sheet = "Actions_Mean";
elseif strcmpi(action_type, "median")
    opts.Sheet = "Actions_Median";
else
    opts.Sheet = "Actions_Mean";
end

opts.DataRange = "A2:E2";
opts.VariableNames = ["SelfPrice0", "SelfPrice1", "SelfPrice2", "SelfPrice3", "SelfPrice4"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];

actions = readtable(data_file, opts, "UseExcel", false);
actions = table2array(actions);

%% Get Probabilities
opts = spreadsheetImportOptions("NumVariables", 1);
opts.DataRange = "C2:C26";

sheet_name = getenv('DF_SALEPROB_SHEET');
if isempty(sheet_name)
    sheet_name = 'Allsellers';
end
opts.Sheet = sheet_name;

opts.VariableNames = ["Sale_prob"];
opts.VariableTypes = ["double"];

prob = readtable(Prob_file, opts, "UseExcel", false);
prob = table2array(prob);
clear opts

end
