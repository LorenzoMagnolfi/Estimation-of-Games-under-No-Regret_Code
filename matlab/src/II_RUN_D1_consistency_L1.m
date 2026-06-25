%% II_RUN_D1_consistency_L1
%  D1 (fixed-eps / confound A) redo with the paper-exact BHC-L1 consistency set.
%  Reuses the cached full-feedback RM trajectory from fixed_eps_panel_D1.mat (NO
%  re-learning). Re-solves BOTH radius policies with consistency_slack_kind='L1':
%    correct: obedience eps(N) + L1 consistency r_N(N)   -- the proper region;
%    frozen : obedience eps(4M) + L1 consistency r_N(N)  -- isolates the obedience
%             radius (consistency r_N is identical in both at each N).
%  Obedience radius at alpha_R, L1 consistency radius at alpha_C (split TBD w/ NL).
%  Area in (mu,sigma^2) on the SAME fixed grid as exact. Saves exact + L1; NO figures.

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
alpha_set = 0.05; alpha_C = 0.1*alpha_set; alpha_R = 0.9*alpha_set; IDTOL = 1e-8;

d1 = load(fullfile(paths.artifacts, 'fixed_eps_panel_D1.mat'));
res_correct_exact = d1.res_correct; res_frozen_exact = d1.res_frozen;
fprintf('Loaded D1 exact baseline + cached trajectory.\n');

baseopts = struct('maxiters_values', res_correct_exact.maxiters_values, ...
    'alpha_set', alpha_set, 'alpha_R', alpha_R, 'alpha_C', alpha_C, ...
    'switch_eps', 10, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'learning_style', 'rm', ...
    'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', d1.fixed_gridparamV, 'fixed_gridparamM', d1.fixed_gridparamM);

fprintf('\n===== D1 correct-eps + L1 consistency =====\n');
o = baseopts; o.precomputed_distY = res_correct_exact.distY_time_all;
rng(12345); res_correct_L1 = df.stages.run_stage_ii(cfg, o);

fprintf('\n===== D1 frozen-eps(4M) + L1 consistency =====\n');
o = baseopts; o.eps_override = d1.eps_4M; o.precomputed_distY = res_correct_exact.distY_time_all;
rng(12345); res_frozen_L1 = df.stages.run_stage_ii(cfg, o);

nN = numel(res_correct_exact.maxiters_values); N = res_correct_exact.maxiters_values(:);
[area_correct_exact, area_correct_L1, area_frozen_exact, area_frozen_L1] = deal(zeros(nN,1));
for ni = 1:nN
    area_correct_exact(ni) = region_area(res_correct_exact, ni, IDTOL);
    area_correct_L1(ni)    = region_area(res_correct_L1, ni, IDTOL);
    area_frozen_exact(ni)  = region_area(res_frozen_exact, ni, IDTOL);
    area_frozen_L1(ni)     = region_area(res_frozen_L1, ni, IDTOL);
end
summary = table(N, area_correct_exact, area_correct_L1, area_frozen_exact, area_frozen_L1);
fprintf('\n=== D1: area vs N, exact vs L1 consistency ===\n'); disp(summary);

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'fixed_eps_panel_D1_L1.csv'));
save(fullfile(paths.artifacts, 'fixed_eps_panel_D1_L1.mat'), ...
    'res_correct_exact', 'res_frozen_exact', 'res_correct_L1', 'res_frozen_L1', ...
    'summary', 'alpha_R', 'alpha_C', 'alpha_set', 'IDTOL', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'fixed_eps_panel_D1_L1.mat'));
fprintf('========== D1 L1 redo complete ==========\n');

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
