%% II_RUN_pooled_vs_type
%
%  G1 / R2 (regret unit). The paper's identified object (BCCE) rests on
%  CONTEXTUAL no-regret: each firm has no regret type-by-type. A firm that
%  instead minimized POOLED (unconditional) regret -- one regret vector across
%  all periods, ignoring the realized cost type -- would converge to a coarse
%  correlated equilibrium of the strategic normal form, a different object.
%
%  This runner contrasts the two LEARNERS on the canonical lab:
%    - type-conditional regret matching   (learning_style = 'rm', BCCE)
%    - pooled regret matching             (learning_style = 'pooled', strategic-NF)
%  Both are run across the horizon grid with the corrected full-feedback radius
%  (switch_eps = 10) on a FIXED (mu,sigma^2) grid, and their identified-set AREAS
%  are compared. It also reports the regret-unit diagnostic that makes the
%  distinction concrete: each learner drives its OWN regret notion to zero, but
%  the pooled play retains positive TYPE-conditional regret.
%
%  WHY this exercise: the theory memo (item 2) flags that the R2/HST comparison
%  most likely turns on the regret UNIT (type-by-type vs pooled), not on
%  conditional independence. This is the empirical complement to that check.
%
%  *** INTERPRETATION FLAG (read before using the area numbers) ***
%  The COMPUTE here is safe: two learners, same BCCE test, areas measured the
%  same way. But the narrative direction -- "pooled gives the SMALLER,
%  strategic-normal-form set" -- is entangled with the unresolved HST
%  conditional-independence-vs-regret-unit question (memo item 2, gated on the
%  HST source and a likely referee). Treat the area ordering as an empirical
%  finding to be FRAMED only after that theory point is settled, not as a
%  pre-committed claim.
%
%  Saves .mat; makes NO server figures (regenerate locally). New code (new
%  pooled learner) -- smoke before the production run.

clear; clc;

paths = df_repo_paths();
rng(12345);

%% Canonical lab
NPlayers = 2;
alpha = -(1/3);
actions_vec = [4; 5; 6; 7; 8];
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;
confid = 0.05;
switch_eps = 10;
IDTOL = 1e-8;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Fixed (mu,sigma^2) grid (same convention/extent as D1/T1 for cross-exercise comparability)
NGRID = 60;
fixed_gridparamM = [1; linspace(0.18, 1.5, NGRID)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, NGRID)'];

opts = struct();
opts.maxiters_values   = [500000, 1000000, 2000000, 4000000];
opts.alpha_set         = confid;
opts.switch_eps        = switch_eps;
opts.backend           = 'fast';
opts.adaptive          = false;
opts.require_corrected = true;
opts.fixed_gridparamV  = fixed_gridparamV;
opts.fixed_gridparamM  = fixed_gridparamM;

%% Type-conditional learner (BCCE)
fprintf('\n========== G1: type-conditional regret (BCCE) ==========\n');
opts_type = opts; opts_type.learning_style = 'rm';
rng(12345);
res_type = df.stages.run_stage_ii(cfg, opts_type);

%% Pooled learner (strategic normal form)
fprintf('\n========== G1: pooled regret (strategic NF) ==========\n');
opts_pool = opts; opts_pool.learning_style = 'pooled';
rng(12345);
res_pool = df.stages.run_stage_ii(cfg, opts_pool);

%% Area vs N for both learners
nN = numel(opts.maxiters_values);
area_type = zeros(nN, 1); share_type = zeros(nN, 1); bnd_type = false(nN, 1);
area_pool = zeros(nN, 1); share_pool = zeros(nN, 1); bnd_pool = false(nN, 1);
for ni = 1:nN
    [area_type(ni), share_type(ni), bnd_type(ni)] = region_area(res_type, ni, IDTOL);
    [area_pool(ni), share_pool(ni), bnd_pool(ni)] = region_area(res_pool, ni, IDTOL);
end
N = opts.maxiters_values(:);
area_summary = table(N, area_type, share_type, bnd_type, area_pool, share_pool, bnd_pool);
fprintf('\n=== G1 identified-set area: type vs pooled ===\n');
disp(area_summary);

%% Regret-unit diagnostic (direct learner calls at T = 1M, same RNG)
fprintf('\n=== G1 regret-unit diagnostic (T = 1,000,000) ===\n');
T_diag = 1000000;
rng(12345);
[~, ~, type_finalregret] = df.sim.learn(cfg, 1, T_diag, T_diag, 1, 1, 1, 1);
rng(12345);
[~, ~, pool_pooledregret, pool_typeregret] = ...
    df.sim.learn_pooled(cfg, 1, T_diag, T_diag, 1, 1, 1, 1);

regret_diag = struct();
regret_diag.T                       = T_diag;
regret_diag.type_play_max_typeregret = max(type_finalregret(:));      % ~0 (contextual NR holds)
regret_diag.pool_play_max_pooledregret = max(pool_pooledregret(:));   % ~0 (pooled NR holds)
regret_diag.pool_play_max_typeregret   = max(pool_typeregret(:));     % > 0 (pooled FAILS contextual NR)
fprintf('  type play : max type-conditional regret  = %.3e  (contextual no-regret holds)\n', ...
    regret_diag.type_play_max_typeregret);
fprintf('  pooled play: max pooled regret           = %.3e  (pooled no-regret holds)\n', ...
    regret_diag.pool_play_max_pooledregret);
fprintf('  pooled play: max type-conditional regret = %.3e  (contextual no-regret FAILS)\n', ...
    regret_diag.pool_play_max_typeregret);
fprintf(['  -> the two learners minimize DIFFERENT regret notions; the pooled play is\n' ...
         '     not contextually no-regret, so it need not be a BCCE. This is the\n' ...
         '     regret-unit distinction (memo item 2), measured directly.\n']);

if any(bnd_type) || any(bnd_pool)
    fprintf('WARNING: an identified region touches the fixed-grid edge (area truncated).\n');
end

%% Save (single authoritative save; no server figures)
if ~exist(paths.artifacts, 'dir'), mkdir(paths.artifacts); end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(area_summary, fullfile(paths.tables_ii, 'pooled_vs_type_G1.csv'));
save(fullfile(paths.artifacts, 'pooled_vs_type_G1.mat'), ...
    'res_type', 'res_pool', 'area_summary', 'regret_diag', ...
    'opts', 'fixed_gridparamM', 'fixed_gridparamV', 'IDTOL', ...
    'confid', 'switch_eps', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'pooled_vs_type_G1.mat'));
fprintf('NOTE: area-ordering interpretation is gated on the HST regret-unit resolution (memo item 2).\n');
fprintf('========== G1 complete ==========\n');

%% Local helper -------------------------------------------------------------
function [area, share, hit_boundary] = region_area(results, ni, IDTOL)
    nV = results.NGridV + 1;
    nM = results.NGridM + 1;
    distpars = squeeze(results.distpars_all(ni, :, :));
    mu_g = reshape(distpars(:, 1), nV, nM); mu_g = mu_g(2:end, 2:end);
    sg_g = reshape(distpars(:, 2), nV, nM); sg_g = sg_g(2:end, 2:end);
    VV_g = reshape(results.VV_all(ni, :), nV, nM); VV_g = VV_g(2:end, 2:end);
    id_g = VV_g <= IDTOL;
    n_id = nnz(id_g);
    share = n_id / numel(id_g);
    dmu = median(diff(unique(mu_g(:))), 'omitnan');
    dsg = median(diff(unique(sg_g(:))), 'omitnan');
    area = n_id * abs(dmu) * abs(dsg);
    hit_boundary = any(id_g(1, :)) || any(id_g(end, :)) || ...
                   any(id_g(:, 1)) || any(id_g(:, end));
end
