%% II_RUN_nonparam_revision — Master runner for R1.1.b nonparametric exercises
%
% Runs nonparametric identification for s = {5, 10, 20, 50}, collects
% timing for cost table, generates SVM boundary plots and CDF envelopes.
%
% Strategy: two passes
%   Pass 1 (adaptive/fast): small grid (1000 candidates), all s values,
%          all iteration counts — gives a sense of where things stand.
%   Pass 2 (production):    full grid (10k candidates), all s values.
%          Run only after reviewing Pass 1 results.
%
% See also: df.stages.run_stage_ii_nonparam, df.report.plot_identified_cdfs

%% Setup
clear all; clc; close all;

paths = df_repo_paths();
rng(12345);

% Game parameters (same as parametric Stage II / Bo Feng)
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);

%% Configuration — edit these for Pass 1 vs Pass 2
% Pass 1 (fast exploration):
pass = 1;  % 1 = fast, 2 = production
s_values = [5, 10, 20];  % s=50 in separate run (long)

if pass == 1
    % Fast pass: small grids, all iteration counts
    grid_scale = struct('K_local', 100, 'K_global', 800, 'K_spiky', 100);
    svm_quality = 'draft';
    maxiters_values = [500000, 1000000, 2000000, 4000000];
    tag = 'fast';
elseif pass == 2
    % Production: full grids
    grid_scale = struct('K_local', 1000, 'K_global', 9000, 'K_spiky', 200);
    svm_quality = 'final';
    maxiters_values = [500000, 1000000, 2000000, 4000000];
    tag = 'prod';
end

%% Run for each s value
all_results = cell(numel(s_values), 1);
cost_table = [];

for si = 1:numel(s_values)
    s = s_values(si);
    fprintf('\n========== s = %d ==========\n', s);

    cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

    % Stage opts
    stage_opts = struct();
    stage_opts.maxiters_values = maxiters_values;
    stage_opts.alpha_set = 0.05;
    stage_opts.switch_eps = 1;
    stage_opts.backend = 'fast';

    % Grid parameters — scale down for s>=20 to keep runtime manageable
    if s <= 10
        stage_opts.K_local  = grid_scale.K_local;
        stage_opts.K_global = grid_scale.K_global;
        stage_opts.K_spiky  = grid_scale.K_spiky;
    else
        % For large s, reduce global draws (simplex is sparser)
        stage_opts.K_local  = grid_scale.K_local;
        stage_opts.K_global = max(round(grid_scale.K_global * 10/s), 500);
        stage_opts.K_spiky  = max(round(grid_scale.K_spiky * 10/s), 50);
    end
    stage_opts.spike_mult = 3;
    stage_opts.n_adjacent = min(4, s-1);  % cap at s-1 for small s

    % Run
    t_run = tic;
    results = df.stages.run_stage_ii_nonparam(cfg, stage_opts);
    total_time = toc(t_run);

    all_results{si} = results;

    % Collect cost table rows
    NGrid = size(results.VV_all, 2);
    for mi = 1:numel(maxiters_values)
        n_id = sum(results.VV_all(mi,:) <= 1e-12);
        row = struct();
        row.s = s;
        row.NGrid = NGrid;
        row.maxiters = maxiters_values(mi);
        row.t_learn = results.timing.learn(mi);
        row.t_obj = results.timing.objectives(mi);
        row.t_solve = results.timing.solve(mi);
        row.t_total = results.timing.learn(mi) + results.timing.objectives(mi) + results.timing.solve(mi);
        row.n_identified = n_id;
        row.pct_identified = 100 * n_id / NGrid;
        cost_table = [cost_table; row]; %#ok<AGROW>
    end

    %% Plot: scatter + SVM for each iteration count
    for mi = 1:numel(maxiters_values)
        VV = results.VV_all(mi, :);
        distpars = squeeze(results.distpars_all(mi, :, :));
        id_set_index = (VV <= 1e-12);

        % Skip SVM if too few identified points
        if sum(id_set_index) < 3
            fprintf('  s=%d, iter=%dk: only %d identified — skipping SVM\n', ...
                s, maxiters_values(mi)/1000, sum(id_set_index));
            continue;
        end

        iteration_k = maxiters_values(mi) / 1000;

        % SVM classification
        mu_range = [min(distpars(:,1)), max(distpars(:,1))];
        sig_range = [min(distpars(:,2)), max(distpars(:,2))];
        halton_ranges = [mu_range(1), mu_range(2), sig_range(1), sig_range(2)];

        svm_opts = struct('quality', svm_quality);
        [label, PGx, ~] = df.report.classify_identified_set(...
            distpars, id_set_index(:), halton_ranges, svm_opts);

        % Plot
        plot_opts = struct();
        plot_opts.true_param = [3, 1];
        plot_opts.save_path = fullfile(paths.figures_ii, ...
            sprintf('nonparam_%s_s%d_%dk', tag, s, iteration_k));
        fig = df.report.plot_identified_set(PGx, label, distpars, id_set_index(:), plot_opts);
        title(sprintf('Nonparam s=%d | N=%d grid | %dk iters [%s]', ...
            s, NGrid, iteration_k, tag), 'Interpreter', 'none');
        close(fig);

        %% CDF envelope plot
        df.report.plot_identified_cdfs(results.distribution_parameters, ...
            id_set_index, cfg.type_space{1,1}, ...
            struct('true_distrib', cfg.marg_distrib, ...
                   'save_path', fullfile(paths.figures_ii, ...
                       sprintf('nonparam_CDF_%s_s%d_%dk', tag, s, iteration_k))));
    end
end

%% Write computational cost table
df.report.write_cost_table(cost_table, ...
    fullfile(paths.tables_ii, sprintf('nonparam_cost_%s.tex', tag)));

%% Save workspace
save(fullfile(paths.artifacts, sprintf('nonparam_revision_%s.mat', tag)), ...
    'all_results', 'cost_table', 's_values', 'pass', '-v7.3');

fprintf('\n========== All done (%s pass) ==========\n', tag);
