%% II_RUN_G1_consistency_L1
%  G1 (regret unit: pooled vs type) redo with the paper-exact BHC-L1 consistency
%  set. Reuses the cached type-conditional (rm) and pooled trajectories from
%  pooled_vs_type_G1.mat (NO re-learning). Re-solves BOTH arms with
%  consistency_slack_kind='L1' on the SAME fixed grid as exact. Obedience radius
%  at alpha_R, L1 consistency radius at alpha_C (split TBD w/ NL). The regret-unit
%  diagnostic is learner-side and unchanged, so it is not recomputed here.
%  Saves exact + L1 areas; NO figures.

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
alpha_set = 0.05; alpha_C = 0.1*alpha_set; alpha_R = 0.9*alpha_set; IDTOL = 1e-8;

g1 = load(fullfile(paths.artifacts, 'pooled_vs_type_G1.mat'));
res_type_exact = g1.res_type; res_pool_exact = g1.res_pool;
fprintf('Loaded G1 exact baseline + cached trajectories.\n');

baseopts = struct('maxiters_values', res_type_exact.maxiters_values, ...
    'alpha_set', alpha_set, 'alpha_R', alpha_R, 'alpha_C', alpha_C, ...
    'switch_eps', 10, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', g1.fixed_gridparamV, 'fixed_gridparamM', g1.fixed_gridparamM);

fprintf('\n===== G1 type-conditional (BCCE) + L1 consistency =====\n');
o = baseopts; o.learning_style = 'rm'; o.precomputed_distY = res_type_exact.distY_time_all;
rng(12345); res_type_L1 = df.stages.run_stage_ii(cfg, o);

fprintf('\n===== G1 pooled (strategic NF) + L1 consistency =====\n');
o = baseopts; o.learning_style = 'pooled'; o.precomputed_distY = res_pool_exact.distY_time_all;
rng(12345); res_pool_L1 = df.stages.run_stage_ii(cfg, o);

nN = numel(res_type_exact.maxiters_values); N = res_type_exact.maxiters_values(:);
[area_type_exact, area_type_L1, area_pool_exact, area_pool_L1] = deal(zeros(nN,1));
for ni = 1:nN
    area_type_exact(ni) = region_area(res_type_exact, ni, IDTOL);
    area_type_L1(ni)    = region_area(res_type_L1, ni, IDTOL);
    area_pool_exact(ni) = region_area(res_pool_exact, ni, IDTOL);
    area_pool_L1(ni)    = region_area(res_pool_L1, ni, IDTOL);
end
summary = table(N, area_type_exact, area_type_L1, area_pool_exact, area_pool_L1);
fprintf('\n=== G1: area vs N, exact vs L1 consistency (type & pooled) ===\n'); disp(summary);

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'pooled_vs_type_G1_L1.csv'));
save(fullfile(paths.artifacts, 'pooled_vs_type_G1_L1.mat'), ...
    'res_type_exact', 'res_pool_exact', 'res_type_L1', 'res_pool_L1', ...
    'summary', 'alpha_R', 'alpha_C', 'alpha_set', 'IDTOL', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'pooled_vs_type_G1_L1.mat'));
fprintf('========== G1 L1 redo complete ==========\n');

function area = region_area(res, ni, IDTOL)
    nV = res.NGridV + 1; nM = res.NGridM + 1;
    dp = squeeze(res.distpars_all(ni, :, :));
    mu_g = reshape(dp(:,1), nV, nM); mu_g = mu_g(2:end, 2:end);
    sg_g = reshape(dp(:,2), nV, nM); sg_g = sg_g(2:end, 2:end);
    VV_g = reshape(res.VV_all(ni, :), nV, nM); VV_g = VV_g(2:end, 2:end);
    n_id = nnz(VV_g <= IDTOL);
    dmu = median(diff(unique(mu_g(:))), 'omitnan');
    dsg = median(diff(unique(sg_g(:))), 'omitnan');
    area = n_id * abs(dmu) * abs(dsg);
end
