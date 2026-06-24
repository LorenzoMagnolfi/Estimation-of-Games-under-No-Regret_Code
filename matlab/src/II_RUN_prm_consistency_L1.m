%% II_RUN_prm_consistency_L1
%
%  Section 3.1 redo with the PAPER-EXACT consistency constraint: the
%  Bretagnolle-Huber-Carol L1 multinomial set on the (t,theta)-marginal, at the
%  alpha_R + alpha_C budget split (Theorem CR). This replaces the exact-
%  consistency S1/S2 numbers, which treated the type distribution as known.
%
%  Reuses the cached full-feedback (RM, switch_eps=10) and bandit (PRM,
%  switch_eps=11) trajectories from prm_comparison_s5.mat -- NO re-learning. Both
%  arms are re-solved with consistency_slack_kind='L1' on the SAME per-N grid as
%  the exact run, so exact and L1 are directly comparable. The obedience radius is
%  computed at alpha_R (the run_stage_ii budget fix), the L1 consistency radius at
%  alpha_C. Saves exact (from the artifact) + L1 VV for both arms, plus retained
%  shares per N. NO figures (regenerate locally).
%
%  NOTE: the alpha_R/alpha_C split (0.045/0.005) is the working placeholder, TBD
%  with NL. Changing it is a cheap re-solve.

clear; clc;
paths = df_repo_paths();
rng(12345);

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

alpha_set = 0.05;
alpha_C   = 0.1 * alpha_set;    % 0.005  (consistency budget)
alpha_R   = 0.9 * alpha_set;    % 0.045  (obedience budget)
IDTOL     = 1e-8;               % SIM-3 precision-matched

%% Load the cached trajectories + exact-consistency baseline (from J3)
prm_path = fullfile(paths.artifacts, 'prm_comparison_s5.mat');
assert(exist(prm_path, 'file') == 2, ...
    'prm_comparison_s5.mat not found at %s (run II_RUN_prm_comparison first).', prm_path);
S = load(prm_path);
res_rm_exact  = S.results_rm;     % switch_eps=10, exact consistency, full grid
res_prm_exact = S.results_prm;    % switch_eps=11, exact consistency, full grid
fprintf('Loaded exact-consistency baseline + cached trajectories from %s\n', prm_path);

baseopts = struct( ...
    'maxiters_values', res_rm_exact.maxiters_values, ...
    'NGridV', res_rm_exact.NGridV, 'NGridM', res_rm_exact.NGridM, ...
    'alpha_set', alpha_set, 'alpha_R', alpha_R, 'alpha_C', alpha_C, ...
    'backend', 'fast', 'adaptive', false, 'require_corrected', true, ...
    'consistency_slack_kind', 'L1');

out_path = fullfile(paths.artifacts, 'prm_consistency_L1.mat');

%% RM arm (full feedback, Hedge radius) + L1 consistency
fprintf('\n========== RM arm: L1 consistency (switch_eps=10) ==========\n');
o = baseopts; o.switch_eps = 10; o.learning_style = 'rm';
o.precomputed_distY = res_rm_exact.distY_time_all;
rng(12345);
res_rm_L1 = df.stages.run_stage_ii(cfg, o);
save(out_path, 'res_rm_exact', 'res_prm_exact', 'res_rm_L1', ...
    'alpha_R', 'alpha_C', 'IDTOL', '-v7.3');   % incremental: RM safe before PRM
fprintf('  incremental save (RM done): %s\n', out_path);

%% PRM arm (bandit, EXP3 radius) + L1 consistency
fprintf('\n========== PRM arm: L1 consistency (switch_eps=11) ==========\n');
o = baseopts; o.switch_eps = 11; o.learning_style = 'prm';
o.precomputed_distY = res_prm_exact.distY_time_all;
rng(12345);
res_prm_L1 = df.stages.run_stage_ii(cfg, o);

%% Retained shares: exact vs L1, both arms, per N (same grid, same IDTOL)
shr = @(res, ni) nnz(res.VV_all(ni,:) <= IDTOL) / numel(res.VV_all(ni,:));
nN = numel(res_rm_exact.maxiters_values);
N = res_rm_exact.maxiters_values(:);
rm_exact = zeros(nN,1); rm_L1 = zeros(nN,1);
prm_exact = zeros(nN,1); prm_L1 = zeros(nN,1);
for ni = 1:nN
    rm_exact(ni)  = shr(res_rm_exact, ni);   rm_L1(ni)  = shr(res_rm_L1, ni);
    prm_exact(ni) = shr(res_prm_exact, ni);  prm_L1(ni) = shr(res_prm_L1, ni);
end
summary = table(N, rm_exact, rm_L1, prm_exact, prm_L1);
fprintf('\n=== Section 3.1: retained share, exact consistency vs L1 (IDTOL=%.0e) ===\n', IDTOL);
disp(summary);

if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'prm_consistency_L1.csv'));
save(out_path, 'res_rm_exact', 'res_prm_exact', 'res_rm_L1', 'res_prm_L1', ...
    'summary', 'alpha_R', 'alpha_C', 'alpha_set', 'IDTOL', '-v7.3');
fprintf('\nArtifact: %s\n', out_path);
fprintf('========== Section 3.1 L1 redo complete ==========\n');
