%% II_RUN_nonparam_L1_s10
%  Cost-support sensitivity (S7, R1.4.b): the nonparametric identified set at the
%  larger support size s=10, with the paper-exact BHC-L1 consistency set. Same as
%  II_RUN_nonparam_L1 but s=10 (a richer type support; the SOCP grows with s^2, so
%  this is slower). Full-feedback regret matching, corrected Hedge radius, obedience
%  budget alpha_R, consistency budget alpha_C. Saves identified shares per N; NO
%  figures. (s=20 is deliberately not run here: it is the support size that ran for
%  days and was killed, and needs a coarser-grid rethink to be feasible.)

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 10;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8;

opts = struct( ...
    'maxiters_values', [500000, 1000000, 2000000, 4000000], ...
    'alpha_set', 0.05, 'alpha_R', 0.9*0.05, 'alpha_C', 0.1*0.05, ...
    'switch_eps', 10, 'backend', 'fast', 'learning_style', 'rm', ...
    'consistency_slack_kind', 'L1');

fprintf('\n========== Nonparametric region (S7) + L1 consistency, s=%d ==========\n', s);
res = df.stages.run_stage_ii_nonparam(cfg, opts);

nN = numel(opts.maxiters_values); N = opts.maxiters_values(:);
share = zeros(nN,1); count = zeros(nN,1);
for ni = 1:nN
    count(ni) = nnz(res.VV_all(ni,:) <= IDTOL);
    share(ni) = count(ni) / size(res.VV_all,2);
end
summary = table(N, count, share);
fprintf('\n=== Nonparam (s=%d) identified share vs N (L1, IDTOL=%.0e) ===\n', s, IDTOL);
disp(summary);

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'nonparam_L1_s10.csv'));
save(fullfile(paths.artifacts, 'nonparam_L1_s10.mat'), 'res', 'summary', 'IDTOL', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'nonparam_L1_s10.mat'));
fprintf('========== Nonparam L1 (s=10) complete ==========\n');
