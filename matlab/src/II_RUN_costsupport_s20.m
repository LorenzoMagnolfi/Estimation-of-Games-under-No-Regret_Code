%% II_RUN_costsupport_s20
%  Cost-support sensitivity at s=20 (S7, R1.4.b), done PARAMETRICALLY: the
%  (mu,sigma^2) identified region with a type support of 20 points, comparable to
%  the s=5 region of Section 3.1. This replaces the nonparametric s=20 run, which is
%  infeasible (the BCCE SOCP scales with s^2 = 400 type profiles, ~16x the s=5 cone,
%  so the ~10,200-candidate nonparam grid would take days) AND poorly posed (a
%  20-dim simplex is far too sparsely covered by random draws). The parametric grid
%  is small, so the big-cone solve is affordable.
%
%  Full-feedback regret matching, corrected Hedge radius (switch_eps=10), BHC-L1
%  consistency (alpha_R=0.045, alpha_C=0.005). Coarser 30x30 fixed grid than the
%  headline (the s=20 cone is expensive); area metric in (mu,sigma^2) is
%  resolution-robust. Per-N incremental save (so a long run keeps the early N).

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 20;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8;

NGRID = 30;  % coarser than the headline: the s=20 cone is ~16x the s=5 cone
fixed_gridparamM = [1; linspace(0.18, 1.5, NGRID)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, NGRID)'];

baseopts = struct( ...
    'alpha_set', 0.05, 'alpha_R', 0.9*0.05, 'alpha_C', 0.1*0.05, ...
    'switch_eps', 10, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'learning_style', 'rm', ...
    'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', fixed_gridparamV, 'fixed_gridparamM', fixed_gridparamM);

Ns = [500000, 1000000, 2000000, 4000000];
out_path = fullfile(paths.artifacts, 'costsupport_s20_L1.mat');
res_by_N = cell(numel(Ns), 1);
area = nan(numel(Ns), 1); share = nan(numel(Ns), 1); hit_boundary = false(numel(Ns), 1);

for k = 1:numel(Ns)
    fprintf('\n===== Cost-support s=20 (parametric) + L1, N = %dM =====\n', Ns(k)/1e6);
    o = baseopts; o.maxiters_values = Ns(k);
    rng(12345);
    r = df.stages.run_stage_ii(cfg, o);
    res_by_N{k} = r;
    [area(k), share(k), hit_boundary(k)] = region_area(r, 1, IDTOL);
    fprintf('  N=%dM: area=%.4f  share=%.4f  hit_boundary=%d\n', Ns(k)/1e6, area(k), share(k), hit_boundary(k));
    save(out_path, 'res_by_N', 'area', 'share', 'hit_boundary', 'Ns', ...
        'fixed_gridparamM', 'fixed_gridparamV', 'IDTOL', '-v7.3');   % incremental
    fprintf('  incremental save through N=%dM\n', Ns(k)/1e6);
end

summary = table(Ns(:), area(:), share(:), hit_boundary(:), ...
    'VariableNames', {'N', 'area', 'share', 'hit_boundary'});
fprintf('\n=== Cost-support s=20 (parametric) area/share vs N (L1, IDTOL=%.0e) ===\n', IDTOL);
disp(summary);
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'costsupport_s20_L1.csv'));
fprintf('\nArtifact: %s\n', out_path);
fprintf('========== Cost-support s=20 complete ==========\n');

function [area, share, hit_boundary] = region_area(res, ni, IDTOL)
    nV = res.NGridV + 1; nM = res.NGridM + 1;
    dp = squeeze(res.distpars_all(ni, :, :));
    mu_g = reshape(dp(:,1), nV, nM); mu_g = mu_g(2:end, 2:end);
    sg_g = reshape(dp(:,2), nV, nM); sg_g = sg_g(2:end, 2:end);
    VV_g = reshape(res.VV_all(ni, :), nV, nM); VV_g = VV_g(2:end, 2:end);
    id_g = VV_g <= IDTOL;
    n_id = nnz(id_g);
    share = n_id / numel(id_g);
    dmu = median(diff(unique(mu_g(:))), 'omitnan');
    dsg = median(diff(unique(sg_g(:))), 'omitnan');
    area = n_id * abs(dmu) * abs(dsg);
    hit_boundary = any(id_g(1,:)) || any(id_g(end,:)) || any(id_g(:,1)) || any(id_g(:,end));
end
