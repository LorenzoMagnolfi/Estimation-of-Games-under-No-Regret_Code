%% II_RUN_T1_consistency_L1
%  T1 (sharp-set volume vs N, to 8M) redo with the paper-exact BHC-L1 consistency
%  set. Reuses the cached full-feedback RM trajectory from sharp_set_volume_T1.mat
%  (NO re-learning). Re-solves with consistency_slack_kind='L1' on the SAME fixed
%  grid as exact. This is the exercise where the N^{-1/2} consistency floor may
%  flatten the volume to a positive limit (the partial-ID sharpness message that
%  exact consistency could not deliver). Obedience radius at alpha_R, L1
%  consistency radius at alpha_C (split TBD w/ NL). Saves exact + L1 areas; NO figures.

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
alpha_set = 0.05; alpha_C = 0.1*alpha_set; alpha_R = 0.9*alpha_set; IDTOL = 1e-8;

t1 = load(fullfile(paths.artifacts, 'sharp_set_volume_T1.mat'));
res_exact = t1.res;
fprintf('Loaded T1 exact baseline + cached trajectory (N up to %dM).\n', ...
    max(res_exact.maxiters_values)/1e6);

baseopts = struct('maxiters_values', res_exact.maxiters_values, ...
    'alpha_set', alpha_set, 'alpha_R', alpha_R, 'alpha_C', alpha_C, ...
    'switch_eps', 10, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'learning_style', 'rm', ...
    'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', t1.fixed_gridparamV, 'fixed_gridparamM', t1.fixed_gridparamM);

fprintf('\n===== T1 sharp-set volume + L1 consistency (to 8M) =====\n');
o = baseopts; o.precomputed_distY = res_exact.distY_time_all;
rng(12345); res_L1 = df.stages.run_stage_ii(cfg, o);

nN = numel(res_exact.maxiters_values); N = res_exact.maxiters_values(:);
[area_exact, area_L1] = deal(zeros(nN,1));
for ni = 1:nN
    area_exact(ni) = region_area(res_exact, ni, IDTOL);
    area_L1(ni)    = region_area(res_L1, ni, IDTOL);
end
summary = table(N, area_exact, area_L1);
fprintf('\n=== T1: set volume vs N, exact vs L1 consistency ===\n'); disp(summary);
if nN >= 2 && area_L1(end-1) > 0
    rel_L1 = abs(area_L1(end) - area_L1(end-1)) / area_L1(end-1);
    fprintf('L1 relative area change over last doubling = %.2f%% (exact was 58.8%%)\n', 100*rel_L1);
end

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'sharp_set_volume_T1_L1.csv'));
save(fullfile(paths.artifacts, 'sharp_set_volume_T1_L1.mat'), ...
    'res_exact', 'res_L1', 'summary', 'alpha_R', 'alpha_C', 'alpha_set', 'IDTOL', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'sharp_set_volume_T1_L1.mat'));
fprintf('========== T1 L1 redo complete ==========\n');

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
