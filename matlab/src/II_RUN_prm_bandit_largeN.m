%% II_RUN_prm_bandit_largeN
%  Bandit-feedback response, move 4: does the bandit (proxy-regret matching, EXP3
%  radius) confidence region become informative at the larger sample sizes that
%  high-frequency algorithmic pricing routinely generates? The EXP3 radius is wider
%  than Hedge by ~sqrt(|A|), so bandit needs ~|A| times more data than full feedback
%  to match it; the region should contract from near-total at N=4M toward an
%  informative set by N=40M. Extends the PRM arm to N = 10M and 40M (4M as anchor),
%  under the paper-exact BHC-L1 consistency set (alpha_R/alpha_C split).
%
%  Learns FRESH PRM trajectories (10M/40M are not cached); re-seeds each N so the
%  longer horizons are the same underlying process observed for longer. Fixed grid =
%  the prm-comparison grid (mu in [1.65,4.5], sig2 in [0.15,3.5], 101x101), so the
%  4M point anchors to the Section 3.1 PRM number. Saves retained share vs N
%  INCREMENTALLY (per N, so a kill during the 40M solve keeps 4M and 10M). NO figures.

clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
alpha_set = 0.05; alpha_C = 0.1*alpha_set; alpha_R = 0.9*alpha_set; IDTOL = 1e-8;

% Fixed grid = the prm-comparison large-N grid (mu in [1.65,4.5], sig2 in [0.15,3.5]).
fixed_gridparamM = [1; linspace(0.55, 1.5, 100)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, 100)'];

baseopts = struct( ...
    'alpha_set', alpha_set, 'alpha_R', alpha_R, 'alpha_C', alpha_C, ...
    'switch_eps', 11, 'learning_style', 'prm', ...
    'backend', 'fast', 'adaptive', false, 'require_corrected', true, ...
    'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', fixed_gridparamV, 'fixed_gridparamM', fixed_gridparamM);

Ns = [4000000, 10000000, 40000000];
out_path = fullfile(paths.artifacts, 'prm_bandit_largeN.mat');
res_by_N = cell(numel(Ns), 1);
share = nan(numel(Ns), 1);

for k = 1:numel(Ns)
    fprintf('\n===== PRM (bandit, EXP3) + L1, N = %dM =====\n', Ns(k)/1e6);
    o = baseopts; o.maxiters_values = Ns(k);
    rng(12345);
    r = df.stages.run_stage_ii(cfg, o);
    res_by_N{k} = r;
    share(k) = nnz(r.VV_all(1,:) <= IDTOL) / size(r.VV_all, 2);
    fprintf('  N=%dM: bandit retained share = %.4f\n', Ns(k)/1e6, share(k));
    save(out_path, 'res_by_N', 'share', 'Ns', 'alpha_R', 'alpha_C', 'alpha_set', ...
        'IDTOL', 'fixed_gridparamM', 'fixed_gridparamV', '-v7.3');   % incremental
    fprintf('  incremental save through N=%dM\n', Ns(k)/1e6);
end

summary = table(Ns(:), share(:), 'VariableNames', {'N', 'bandit_share'});
fprintf('\n=== Bandit (PRM) retained share vs N (L1 consistency, IDTOL=%.0e) ===\n', IDTOL);
disp(summary);
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'prm_bandit_largeN.csv'));
fprintf('\nArtifact: %s\n', out_path);
fprintf('========== PRM large-N bandit demonstration complete ==========\n');
