%% II_RUN_v05_sigma_check  The variance-convention test at a truth other than one (Niccolò's finding I7(f), 4 November 2026).
%  Canonical lab with the true cost variance sigma2_0 = 2.25 (sd 1.5) instead of 1, Hedge (family F) at N = 4M, direct tail
%  (code 14) and first-moment rectangle (code 16) on the 61x61 grid (multipliers of the truth, so the truth is cell 1 and the
%  candidate axis is a variance). Before the generator fix the learners used 2.25 as the standard deviation, so the data
%  would have come from sd 2.25 while the truth cell stood for sd 1.5; after the fix both are sd 1.5. Reports area, share,
%  truth feasibility and the feasible cells' bounding box in (mu, sigma2). Incremental save; CSV; no figures.
clear; clc; paths = df_repo_paths();
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8]; mu = 3*ones(2,1); sigma2_true = 2.25; s = 5;
IDTOL = 1e-8; NGRID = 60; N = 4000000;
fgM = [1; linspace(0.18, 1.5, NGRID)']; fgV = [1; linspace(0.15, 3.5, NGRID)'];
base = struct('maxiters_values', N, 'alpha_set', 0.05, 'alpha_R', 0.045, 'alpha_C', 0.005, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'consistency_slack_kind', 'L1', 'use_parfor', true, 'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM, 'learning_style', 'hedge');
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2_true * eye(2), s);
fprintf('truth: mu = %g, sigma2 = %g (sd %g); generator prior %s\n', mu(1), sigma2_true, sqrt(sigma2_true), mat2str(cfg.marg_distrib', 4));
ncpu = str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu > 1 && exist('gcp', 'file') == 2 && isempty(gcp('nocreate')), try, parpool('Processes', min(ncpu, 16)); catch, end, end
rng(12345); t0 = tic; [dY, ~, fr] = df.sim.learn_hedge(cfg, 1, N, N, 1, 1, 1, 1); tl = toc(t0);
out = fullfile(paths.artifacts, 'v05_sigma_check.mat'); S = table(); VVs = struct();
for sw = [14 16]
    o = base; o.switch_eps = sw; o.precomputed_distY = {dY};
    rng(12345); t0 = tic; r = df.stages.run_stage_ii(cfg, o); ts = toc(t0);
    nV = r.NGridV + 1; nM = r.NGridM + 1; dp = squeeze(r.distpars_all(1, :, :));
    mu_g = reshape(dp(:,1), nV, nM); sg_g = reshape(dp(:,2), nV, nM); VV_g = reshape(r.VV_all(1,:), nV, nM);
    id = VV_g(2:end, 2:end) <= IDTOL; mu_i = mu_g(2:end, 2:end); sg_i = sg_g(2:end, 2:end);
    dmu = median(diff(unique(mu_i(:)))); dsg = median(diff(unique(sg_i(:))));
    area = nnz(id) * dmu * dsg; share = nnz(id) / numel(id); truth_in = r.VV_all(1,1) <= IDTOL;
    if nnz(id) > 0, box = [min(mu_i(id)), max(mu_i(id)), min(sg_i(id)), max(sg_i(id))]; else, box = nan(1,4); end
    S = [S; table(sw, N, sigma2_true, area, share, nnz(id), truth_in, box(1), box(2), box(3), box(4), max(fr(:)), tl, ts, ...
        'VariableNames', {'switch_eps','N','sigma2_true','area','share','n_cells','truth_feasible','mu_lo','mu_hi','sigma2_lo','sigma2_hi','max_realized_regret','learn_seconds','solve_seconds'})]; %#ok<AGROW>
    VVs.(sprintf('sw%d', sw)) = r.VV_all(1,:);
    fprintf('code %d: area=%.4f share=%.4f cells=%d truth=%d box mu [%.3f, %.3f] sigma2 [%.3f, %.3f] [learn %.0fs, solve %.0fs]\n', sw, area, share, nnz(id), truth_in, box, tl, ts);
    save(out, 'S', 'VVs', 'fgM', 'fgV', 'IDTOL', 'sigma2_true', '-v7.3');
end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(S, fullfile(paths.tables_ii, 'v05_sigma_check.csv'));
fprintf('Artifact: %s\n========== V05 sigma check complete ==========\n', out);
