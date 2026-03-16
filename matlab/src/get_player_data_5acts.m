function [distribution, actions, prob, period] = get_player_data_5acts(player_number, action_type, data_file, Prob_file)

%==========================================================================
%FUNCTION: get_player_data
%AUTHOR(S): Jonathan Becker, Lorenzo Magnolfi
%DATE: September 2021
%--------------------------------------------------------------------------
%DESCRIPTION: Gets the time-average distribution of strategies and values
%of the actions for a given player at a given period from a file containing
%time-averages for all players.
%--------------------------------------------------------------------------
%INPUTS: 
% 1. player_number | integer
% - Number of the player 
% 2. period | integer
% - Period of play 
% 3. data_file | string
% - Full path of data file
% 4. action_type | string
% - Statistic to use as action: either "mean" or "median"
%OUTPUTS:
% 1. distribution | 1x4 double matrix
% - Time average fraction of LL, LH, HL, and HH strategies
% 2. actions | 1x2 double matrix
% - Prices constituting low and high actions
%--------------------------------------------------------------------------
%NOTE(S):
% - Assumes that the first row is the variable names
% - Requires that sheets be names "Seller_#"
% - If an improper action type is specified, assume "mean"
%==========================================================================

%% Get Distribution
% Learn Num periods
[NUM,TXT,RAW]= xlsread(data_file,strcat('Seller_',num2str(player_number)));
[period]= size(NUM,1)-1;

% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 25);

% Specify sheet and range
opts.Sheet = ['Seller_' num2str(player_number)];
opts.DataRange = ['A' num2str(period+1) ':Y' num2str(period+1)];

% Specify column names and types
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

% Import the data
distribution = readtable(data_file, opts, "UseExcel", false);
distribution = table2array(distribution);

% Clear temporary variables
clear opts


%% Get Actions
% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 5);

% Specify sheet and range
if strcmpi(action_type,"mean")
    opts.Sheet = "Actions_Mean";
elseif strcmpi(action_type,"median")
    opts.Sheet = "Actions_Median";
else
    opts.Sheet = "Actions_Mean";
end
opts.DataRange = "A2:E2";

% Specify column names and types
opts.VariableNames = ["SelfPrice0", "SelfPrice1", "SelfPrice2", "SelfPrice3", "SelfPrice4"];
opts.VariableTypes = ["double", "double","double", "double","double"];

% Import the data and convert
actions = readtable(data_file, opts, "UseExcel", false);
actions = table2array(actions);

%% Get Probabilities

% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 1);
opts.DataRange = "C2:C26";
%opts.Sheet = ['4980sellers'];
opts.Sheet = ['Allsellers'];

opts.VariableNames = ["Sale_prob"];
opts.VariableTypes = ["double"];

prob = readtable(Prob_file, opts, "UseExcel", false);
prob = table2array(prob);

% Clear temporary variables
clear opts
