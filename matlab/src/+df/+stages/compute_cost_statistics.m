function [cost_stats, tot_cost_stats] = compute_cost_statistics(ddpars, id_set_index, player_idx, opts)
% DF.STAGES.COMPUTE_COST_STATISTICS  Monte Carlo cost statistics from identified set.
%
%   [cost_stats, tot_cost_stats] = df.stages.compute_cost_statistics(ddpars, id_set_index, player_idx, opts)
%
%   Draws parameter vectors from the identified set, samples marginal costs
%   from the implied distribution, and computes summary statistics.
%   Extracted from III_MAIN lines 141-221.
%
%   Inputs:
%     ddpars       — (NGrid x 2) parameter grid [mu, sigma]
%     id_set_index — (NGrid x 1) logical
%     player_idx   — player index (1, 2, ...)
%     opts         — struct with .Dist_file, .Prob_file, .n_types,
%                     .n_draws, .n_samples, .rng_seed
%
%   Outputs:
%     cost_stats     — (n_draws x 5) [mean, median, p25, p75, sd] of residual costs
%     tot_cost_stats — (n_draws x 5) [mean, median, p25, p75, sd] of total costs

n_draws = opts.n_draws;
n_samples = opts.n_samples;
n_types = opts.n_types;

% Load player data
paths = df.io.repo_paths();
[ref_prices_all, ~, ~] = xlsread(fullfile(paths.data, 'Ref_Price_for_Device_Day.xlsx'), ...
    strcat('Seller_', num2str(player_idx)));
[distrib, actions, ~, period] = get_player_data_5acts(player_idx, 'mean', opts.Dist_file, opts.Prob_file);

% Get identified set
id_set = ddpars(id_set_index, :);
n_id_set = sum(id_set_index);

% Initialize
cost_stats = zeros(n_draws, 5);
tot_cost_stats = zeros(n_draws, 5);

rng(opts.rng_seed);

% Draw parameters from identified set
random_indices = randi(n_id_set, n_draws, 1);
drawn_params = id_set(random_indices, :);

for theta = 1:n_draws
    mu_draw = drawn_params(theta, 1);
    sigma2_draw = drawn_params(theta, 2);

    % Construct type space
    P_l = actions(1);
    P_h = actions(5);
    diff_p = P_h - P_l;
    ub = P_h + 0.25 * diff_p;
    lb = P_l - 3 * diff_p;

    type_space_local = cell(1, 1);
    type_space_local{1, 1} = linspace(lb, ub, n_types)';

    % Marginal distribution
    marg_distrib = pdf('Normal', type_space_local{1,1}, mu_draw, sigma2_draw);
    marg_distrib = marg_distrib ./ sum(marg_distrib, 1);

    % Sample costs
    cumulative_dist = cumsum(marg_distrib);
    cum_dist_refp = (1:period+1) / (period+1);
    marginal_costs = zeros(n_samples, 1);
    marginal_costs_tot = zeros(n_samples, 1);

    for i = 1:n_samples
        r = rand();
        r2 = rand();
        idx = find(cumulative_dist >= r, 1, 'first');
        marginal_costs(i) = type_space_local{1}(idx);
        idx2 = find(cum_dist_refp >= r2, 1, 'first');
        marginal_costs_tot(i) = type_space_local{1}(idx) + ref_prices_all(idx2);
    end

    % Statistics
    cost_stats(theta, :) = [mean(marginal_costs), median(marginal_costs), ...
        prctile(marginal_costs, 25), prctile(marginal_costs, 75), std(marginal_costs)];
    tot_cost_stats(theta, :) = [mean(marginal_costs_tot), median(marginal_costs_tot), ...
        prctile(marginal_costs_tot, 25), prctile(marginal_costs_tot, 75), std(marginal_costs_tot)];
end

end
