%% II_RUN_nonparam_L1_s20
%  The missing large-support nonparametric run (S4/R1.1.b at s=20): candidate cost
%  distributions as probability vectors on a 20-point support, corrected Hedge
%  radius (switch 10) at alpha_R, paper-exact BHC-L1 consistency set at alpha_C.
%  The pre-correction attempt (job 3707643) ran the per-point CVX lane and was
%  cancelled after ~65 hours; this runner uses the direct-SeDuMi lane with parfor
%  (validated by II_SMOKE_sedumi_direct on the parametric stage and
%  II_SMOKE_sedumi_nonparam on this stage).
%
%  One horizon per stage call, with the artifact re-saved after each, so a
%  cancelled or dead job keeps every completed horizon (the 65-hour lesson).
%  Saves identified shares per N; NO figures (regenerate locally from the CSV).

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 20;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8;
N_grid = [500000, 1000000, 2000000, 4000000];

base = struct( ...
    'alpha_set', 0.05, 'alpha_R', 0.9*0.05, 'alpha_C', 0.1*0.05, ...
    'switch_eps', 10, 'backend', 'fast', 'learning_style', 'rm', ...
    'consistency_slack_kind', 'L1', ...
    'solver_backend', 'sedumi_direct', 'use_parfor', true);

nN = numel(N_grid);
count = nan(nN,1); share = nan(nN,1); res_by_N = cell(nN,1);
out = fullfile(paths.artifacts, 'nonparam_L1_s20.mat');
fprintf('\n========== Nonparametric region (S4) + L1 consistency, s=%d ==========\n', s);
for ni = 1:nN
    o = base; o.maxiters_values = N_grid(ni);
    rng(12345);
    res = df.stages.run_stage_ii_nonparam(cfg, o);
    res_by_N{ni} = res;
    count(ni) = nnz(res.VV_all(1,:) <= IDTOL);
    share(ni) = count(ni) / size(res.VV_all, 2);
    fprintf('[s=20] N=%dk: identified %d/%d (share %.4f)\n', ...
        N_grid(ni)/1000, count(ni), size(res.VV_all,2), share(ni));
    N = N_grid(:); summary = table(N, count, share);
    save(out, 'res_by_N', 'summary', 'IDTOL', '-v7.3');   % checkpoint per horizon
end

fprintf('\n=== Nonparam (s=%d) identified share vs N (L1, IDTOL=%.0e) ===\n', s, IDTOL);
disp(summary);
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'nonparam_L1_s20.csv'));
fprintf('Artifact: %s\n========== Nonparam L1 (s=20) complete ==========\n', out);
