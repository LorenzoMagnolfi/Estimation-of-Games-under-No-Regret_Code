%% II_RUN_consistency_slack_panel
%
%  Consistency-slack confidence-region panels (design item D4), reusing the
%  full-feedback regret-matching trajectory from the PRM comparison (J3) so we
%  do NOT re-learn or re-solve the exact-consistency baseline.
%
%    Panel A (no_slack):  corrected obedience (switch_eps=10), EXACT consistency
%                         == the RM arm of II_RUN_prm_comparison.
%    Panel B (box_slack): corrected obedience + separated consistency slack
%                         (coordinatewise Hoeffding box; alpha_R=0.045, alpha_C=0.005).
%
%  REQUIRES prm_comparison_s5.mat (run AFTER J3 via a Slurm dependency).
%  Saves two .mat artifacts and makes NO figures (regenerated locally from .mat).

clear all; clc;

paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Pass A: load the full-feedback RM result (exact consistency) from J3's artifact
prm_path = fullfile(paths.artifacts, 'prm_comparison_s5.mat');
assert(exist(prm_path, 'file') == 2, ...
    'prm_comparison_s5.mat not found at %s; run AFTER J3 (II_RUN_prm_comparison).', prm_path);
S = load(prm_path);
results_A = S.results_rm;                 % switch_eps=10, exact consistency, full grid
cached_distY = results_A.distY_time_all;   % reuse the RM trajectory (no re-learning)
fprintf('Loaded RM trajectory + exact-consistency Pass A from %s\n', prm_path);

%% Pass B: same trajectory, + box consistency slack, inference-grade full solve
opts = struct();
opts.maxiters_values        = results_A.maxiters_values;
opts.NGridV                 = results_A.NGridV;
opts.NGridM                 = results_A.NGridM;
opts.alpha_set              = 0.05;
opts.alpha_R                = 0.9 * 0.05;   % 0.045
opts.alpha_C                = 0.1 * 0.05;   % 0.005  (split TBD with NL)
opts.switch_eps             = 10;
opts.backend                = 'fast';
opts.adaptive               = false;        % SIM-1b: full SOCP solve, not exploration
opts.require_corrected      = true;         % SIM-5
opts.consistency_slack_kind = 'box';
opts.precomputed_distY      = cached_distY;

fprintf('\n=== Pass B: switch_eps=10 + box consistency slack (alpha_R=%.3f, alpha_C=%.3f), adaptive=false ===\n', ...
    opts.alpha_R, opts.alpha_C);
t0 = tic;
results_B = df.stages.run_stage_ii(cfg, opts);
fprintf('Pass B wall time: %.1f min\n', toc(t0) / 60);

%% Save both passes (NO figures on the server; regenerate locally from these .mat)
out_dir = fullfile(paths.matlab_root, 'output', 'rate_landscape_pilot');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'results_R1_no_slack.mat'),  '-struct', 'results_A');
save(fullfile(out_dir, 'results_R1_box_slack.mat'), '-struct', 'results_B');
fprintf('\nSaved consistency-slack panels to %s\n', out_dir);
fprintf('  results_R1_no_slack.mat  (Panel A: exact consistency, from J3 RM arm)\n');
fprintf('  results_R1_box_slack.mat (Panel B: + box consistency slack)\n');
fprintf('=== Done. ===\n');
