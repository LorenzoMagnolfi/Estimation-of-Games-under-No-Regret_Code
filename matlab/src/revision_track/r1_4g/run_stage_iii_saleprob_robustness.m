function run_stage_iii_saleprob_robustness(grid_m, grid_v, n_draws, n_samples)

if nargin < 1 || isempty(grid_m), grid_m = 100; end
if nargin < 2 || isempty(grid_v), grid_v = 100; end
if nargin < 3 || isempty(n_draws), n_draws = 200; end
if nargin < 4 || isempty(n_samples), n_samples = 2000; end

workspace_root = fileparts(fileparts(mfilename('fullpath')));
code_repo = 'C:\Users\lorem\Dropbox\ClaudeCodeProjects\Estimation-of-Games-under-No-Regret_Code';
src_dir = fullfile(code_repo, 'matlab', 'src');
data_dir = fullfile(code_repo, 'matlab', 'data');
out_dir = fullfile(workspace_root, 'JPE Revision', 'robustness_outputs');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

addpath(src_dir);
addpath(fileparts(mfilename('fullpath')));

setenv('DF_SALEPROB_SHEET', '15sellers');
cleanup_obj = onCleanup(@() setenv('DF_SALEPROB_SHEET', ''));

rng(1);

opts = struct();
opts.Dist_file = fullfile(data_dir, 'SellerDistribution_15_sellers_res1.xlsx');
opts.Prob_file = fullfile(data_dir, 'sale_probability_5bins_res1.xlsx');
opts.players = 1:2;
opts.NGridV = grid_v;
opts.NGridM = grid_m;
opts.n_types = 5;
opts.epsilon_grid = 0.05;
opts.n_draws = n_draws;
opts.n_samples = n_samples;
opts.rng_seed = 12345;

tic;
results = df.stages.run_stage_iii(opts);
elapsed_seconds = toc;

summary = struct();
summary.sheet = '15sellers';
summary.NGridM = grid_m;
summary.NGridV = grid_v;
summary.n_draws = n_draws;
summary.n_samples = n_samples;
summary.elapsed_seconds = elapsed_seconds;
summary.players = cell(1, numel(results.players));

for idx = 1:numel(results.players)
    player = results.players(idx);
    p = results.player(idx);
    summary.players{idx} = struct( ...
        'player', player, ...
        'n_grid_points', size(p.ddpars, 1), ...
        'n_identified', sum(p.id_set_index), ...
        'mu_range', [p.min_mu, p.max_mu], ...
        'sigma_range', [p.min_sigma, p.max_sigma], ...
        'mc_resid_mean_range', [min(p.cost_stats(:,1)), max(p.cost_stats(:,1))], ...
        'mc_resid_sd_range', [min(p.cost_stats(:,5)), max(p.cost_stats(:,5))], ...
        'mc_total_mean_range', [min(p.tot_cost_stats(:,1)), max(p.tot_cost_stats(:,1))], ...
        'mc_total_sd_range', [min(p.tot_cost_stats(:,5)), max(p.tot_cost_stats(:,5))] ...
    );
end

stamp = datestr(now, 'yyyymmdd_HHMMSS');
mat_path = fullfile(out_dir, ['stage3_top15_' stamp '.mat']);
json_path = fullfile(out_dir, ['stage3_top15_' stamp '.json']);
txt_path = fullfile(out_dir, ['stage3_top15_' stamp '.txt']);

save(mat_path, 'results', 'summary');

json_text = jsonencode(summary, PrettyPrint=true);
fid = fopen(json_path, 'w');
fprintf(fid, '%s', json_text);
fclose(fid);

fid = fopen(txt_path, 'w');
fprintf(fid, 'Stage III sale-probability robustness\n');
fprintf(fid, 'sheet=%s\n', summary.sheet);
fprintf(fid, 'grid_m=%d\n', summary.NGridM);
fprintf(fid, 'grid_v=%d\n', summary.NGridV);
fprintf(fid, 'n_draws=%d\n', summary.n_draws);
fprintf(fid, 'n_samples=%d\n', summary.n_samples);
fprintf(fid, 'elapsed_seconds=%.3f\n', summary.elapsed_seconds);
for idx = 1:numel(results.players)
    p = summary.players{idx};
    fprintf(fid, 'player=%d\n', p.player);
    fprintf(fid, 'n_grid_points=%d\n', p.n_grid_points);
    fprintf(fid, 'n_identified=%d\n', p.n_identified);
    fprintf(fid, 'mu_range=[%.6f, %.6f]\n', p.mu_range(1), p.mu_range(2));
    fprintf(fid, 'sigma_range=[%.6f, %.6f]\n', p.sigma_range(1), p.sigma_range(2));
    fprintf(fid, 'mc_resid_mean_range=[%.6f, %.6f]\n', p.mc_resid_mean_range(1), p.mc_resid_mean_range(2));
    fprintf(fid, 'mc_resid_sd_range=[%.6f, %.6f]\n', p.mc_resid_sd_range(1), p.mc_resid_sd_range(2));
    fprintf(fid, 'mc_total_mean_range=[%.6f, %.6f]\n', p.mc_total_mean_range(1), p.mc_total_mean_range(2));
    fprintf(fid, 'mc_total_sd_range=[%.6f, %.6f]\n', p.mc_total_sd_range(1), p.mc_total_sd_range(2));
end
fclose(fid);

disp(txt_path);

end
