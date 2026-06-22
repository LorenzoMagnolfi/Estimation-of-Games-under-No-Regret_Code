%% II_RUN_nonparam_revision_noplot_checkpoint
%
% Corrected fast-pass nonparametric runner without figure export. This is a
% server checkpoint for R1.1.b: it preserves the same grid, seed, support
% sizes, horizons, and switch_eps = 10 used by II_RUN_nonparam_revision, but
% writes the table and artifact immediately after the CVX grids finish.

clear all; clc; close all;

paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = eye(NPlayers);

pass = 1;
tag = 'fast_R1_noplot';
switch_eps = 10;
s_values = [5, 10, 20];
maxiters_values = [500000, 1000000, 2000000, 4000000];

grid_scale = struct('K_local', 100, 'K_global', 800, 'K_spiky', 100);

all_results = cell(numel(s_values), 1);
cost_table = struct([]);

fprintf('\n========== Nonparam corrected no-plot checkpoint ==========\n');
fprintf('Tag: %s\n', tag);
fprintf('switch_eps: %d\n', switch_eps);

for si = 1:numel(s_values)
    s = s_values(si);
    fprintf('\n========== s = %d ==========\n', s);

    cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

    stage_opts = struct();
    stage_opts.maxiters_values = maxiters_values;
    stage_opts.alpha_set = 0.05;
    stage_opts.switch_eps = switch_eps;
    stage_opts.backend = 'fast';

    if s <= 10
        stage_opts.K_local = grid_scale.K_local;
        stage_opts.K_global = grid_scale.K_global;
        stage_opts.K_spiky = grid_scale.K_spiky;
    else
        stage_opts.K_local = grid_scale.K_local;
        stage_opts.K_global = max(round(grid_scale.K_global * 10 / s), 500);
        stage_opts.K_spiky = max(round(grid_scale.K_spiky * 10 / s), 50);
    end

    stage_opts.spike_mult = 3;
    stage_opts.n_adjacent = min(4, s - 1);

    t_run = tic;
    results = df.stages.run_stage_ii_nonparam(cfg, stage_opts);
    results.runner_wall_time = toc(t_run);
    all_results{si} = results;

    NGrid = size(results.VV_all, 2);
    for mi = 1:numel(maxiters_values)
        n_id = sum(results.VV_all(mi, :) <= 1e-12);
        row = struct();
        row.s = s;
        row.NGrid = NGrid;
        row.maxiters = maxiters_values(mi);
        row.t_learn = results.timing.learn(mi);
        row.t_obj = results.timing.objectives(mi);
        row.t_solve = results.timing.solve(mi);
        row.t_total = row.t_learn + row.t_obj + row.t_solve;
        row.n_identified = n_id;
        row.pct_identified = 100 * n_id / NGrid;
        cost_table = [cost_table; row]; %#ok<AGROW>
    end

    checkpoint_path = fullfile(paths.artifacts, sprintf('nonparam_revision_%s_s%d_checkpoint.mat', tag, s));
    save(checkpoint_path, 'results', 's', 'pass', 'switch_eps', 'tag', '-v7.3');
    fprintf('Checkpoint artifact: %s\n', checkpoint_path);
end

table_path = fullfile(paths.tables_ii, sprintf('nonparam_cost_%s.tex', tag));
df.report.write_cost_table(cost_table, table_path);

artifact_path = fullfile(paths.artifacts, sprintf('nonparam_revision_%s.mat', tag));
save(artifact_path, 'all_results', 'cost_table', 's_values', 'pass', 'switch_eps', 'tag', '-v7.3');

fprintf('\n========== All done (%s pass) ==========\n', tag);
fprintf('Artifact: %s\n', artifact_path);
fprintf('Table: %s\n', table_path);
