function results = run_stage_iii(opts)
% DF.STAGES.RUN_STAGE_III  Stage III: application identification + cost statistics.
%
%   results = df.stages.run_stage_iii(opts)
%
%   Runs the application-stage identification for each player, then
%   computes Monte Carlo cost statistics from the identified set.
%
%   Inputs:
%     opts — struct with fields:
%       .Dist_file   — path to seller distribution xlsx
%       .Prob_file   — path to sale probability xlsx
%       .players     — player indices (default: 1:2)
%       .NGridV      — variance grid size (default: 100)
%       .NGridM      — mean grid size (default: 100)
%       .n_types     — number of cost types (default: 5)
%       .epsilon_grid — epsilon values (default: 0.05)
%       .n_draws     — Monte Carlo draws from identified set (default: 200)
%       .n_samples   — samples per draw for cost statistics (default: 2000)
%       .rng_seed    — RNG seed for cost sampling (default: 12345)
%
%   Outputs:
%     results — struct with fields:
%       .maxvals         — raw solver output
%       .player(iii)     — per-player struct with:
%           .VV, .id_set_index, .ddpars
%           .id_set_points, .min_mu, .max_mu, .min_sigma, .max_sigma
%           .cost_stats, .tot_cost_stats
%       .mc_stats_table  — summary table
%       .avg_ref_prices, .sd_ref_prices

if ~isfield(opts, 'players'),      opts.players = 1:2; end
if ~isfield(opts, 'NGridV'),       opts.NGridV = 100; end
if ~isfield(opts, 'NGridM'),       opts.NGridM = 100; end
if ~isfield(opts, 'n_types'),      opts.n_types = 5; end
if ~isfield(opts, 'epsilon_grid'), opts.epsilon_grid = 0.05; end
if ~isfield(opts, 'n_draws'),      opts.n_draws = 200; end
if ~isfield(opts, 'n_samples'),    opts.n_samples = 2000; end
if ~isfield(opts, 'rng_seed'),     opts.rng_seed = 12345; end

NGrid = opts.NGridV * opts.NGridM;
n_players = numel(opts.players);

%% Run identification (via existing ApplicationL wrapper)
[outputs] = Identification_Pricing_Game_ApplicationL(...
    opts.epsilon_grid, opts.Dist_file, opts.Prob_file, ...
    opts.players, opts.NGridV, opts.NGridM, opts.n_types);
maxvals = cell2mat(outputs);

results.maxvals = maxvals;
results.epsilon_grid = opts.epsilon_grid;

%% Per-player analysis
avg_ref_prices = zeros(1, max(opts.players));
sd_ref_prices = zeros(1, max(opts.players));
tot_cost_sd = zeros(max(opts.players), 2);

mc_stats_table = cell2table(cell(n_players, 2), ...
    'VariableNames', {'Residual_Mean', 'Residual_SD'}, ...
    'RowNames', arrayfun(@(x) sprintf('Seller %d', x), opts.players, 'UniformOutput', false));

for idx = 1:n_players
    iii = opts.players(idx);

    maxvals_iii = maxvals(NGrid*(idx-1)+1 : NGrid*idx, :);
    VV = maxvals_iii(:, 3);
    id_set_index = (VV <= 1e-12);
    ddpars = maxvals_iii(:, 1:2);

    % Identified set bounds
    id_set_points = ddpars(id_set_index, :);
    min_mu = min(id_set_points(:, 1));
    max_mu = max(id_set_points(:, 1));
    min_sigma = min(id_set_points(:, 2));
    max_sigma = max(id_set_points(:, 2));

    %% Cost statistics
    [cost_stats, tot_cost_stats] = df.stages.compute_cost_statistics(...
        ddpars, id_set_index, iii, opts);

    % Reference prices
    paths = df.io.repo_paths();
    [ref_prices_all, ~, ~] = xlsread(fullfile(paths.data, 'Ref_Price_for_Device_Day.xlsx'), ...
        strcat('Seller_', num2str(iii)));
    avg_ref_prices(iii) = mean(ref_prices_all);
    sd_ref_prices(iii) = std(ref_prices_all);

    % Pack per-player results
    results.player(idx).VV = VV;
    results.player(idx).id_set_index = id_set_index;
    results.player(idx).ddpars = ddpars;
    results.player(idx).id_set_points = id_set_points;
    results.player(idx).min_mu = min_mu;
    results.player(idx).max_mu = max_mu;
    results.player(idx).min_sigma = min_sigma;
    results.player(idx).max_sigma = max_sigma;
    results.player(idx).cost_stats = cost_stats;
    results.player(idx).tot_cost_stats = tot_cost_stats;

    % Marginal cost statistics table
    cost_resid_intervals = [min(cost_stats); max(cost_stats)];
    tot_cost_intervals = [min(tot_cost_stats); max(tot_cost_stats)];
    tot_cost_sd(iii, :) = tot_cost_intervals(:, end)';

    mc_stats_table{idx, 1} = {sprintf('[%.1f, %.1f]', min(cost_stats(:,1)), max(cost_stats(:,1)))};
    mc_stats_table{idx, 2} = {sprintf('[%.1f, %.1f]', min(cost_stats(:,5)), max(cost_stats(:,5)))};
end

results.mc_stats_table = mc_stats_table;
results.avg_ref_prices = avg_ref_prices;
results.sd_ref_prices = sd_ref_prices;
results.tot_cost_sd = tot_cost_sd;
results.players = opts.players;
results.n_types = opts.n_types;

end
