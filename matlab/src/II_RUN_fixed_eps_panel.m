%% II_RUN_fixed_eps_panel
%
%  D1 / confound A: is the identified region shrinking with N because the DATA
%  concentrate (q_N -> q*), or merely because the confidence RADIUS eps(N) ~ 1/sqrt(N)
%  shrinks? The two are observationally similar in a single area-vs-N plot.
%
%  This runner isolates them. It runs full-feedback regret matching once across
%  the horizon grid (corrected Hedge radius, switch_eps = 10), then solves the
%  (mu,sigma^2) confidence region TWICE per horizon on a FIXED grid:
%    (A) correct radius eps(N)       -- the reported region;
%    (B) frozen radius eps(N = 4M)   -- same radius at every N.
%  Both reuse the SAME cached trajectories q_N (precomputed_distY), so the only
%  thing that differs between A and B is the radius. If the region keeps shrinking
%  under (B), the shrinkage is data-driven (q_N concentrating), not a radius
%  artifact -- which is the answer to confound A.
%
%  Set-size metric: AREA of the identified region in (mu,sigma^2) cost space
%  (#identified cells x cell area), comparable across N because the grid is fixed.
%  Saves .mat; makes NO server figures (regenerate locally). New code -- smoke
%  before the production run.

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
switch_eps = 10;              % corrected full-feedback Hedge radius
IDTOL = 1e-8;                 % SIM-3 precision-matched identified-set tolerance

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Fixed (mu,sigma^2) grid -- SAME at every N so areas are comparable.
% [1; linspace] convention: leading 1 is the true-parameter multiplier dropped
% by the (2:end) slice. 60x60 interior grid over mu in [0.54,4.5], sig2 in [0.15,3.5].
NGRID = 60;
fixed_gridparamM = [1; linspace(0.18, 1.5, NGRID)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, NGRID)'];

%% Stage II options
opts = struct();
opts.maxiters_values   = [500000, 1000000, 2000000, 4000000];
opts.alpha_set         = confid;
opts.switch_eps        = switch_eps;
opts.backend           = 'fast';
opts.adaptive          = false;            % inference-grade full SOCP solve
opts.require_corrected = true;             % SIM-5 guard (switch_eps in {10,11})
opts.learning_style    = 'rm';             % full feedback
opts.fixed_gridparamV  = fixed_gridparamV;
opts.fixed_gridparamM  = fixed_gridparamM;

%% Policy A: correct radius eps(N) (learns the trajectories)
fprintf('\n========== D1 Policy A: correct radius eps(N) ==========\n');
rng(12345);
res_correct = df.stages.run_stage_ii(cfg, opts);

%% Policy B: frozen radius eps(4M), reusing the SAME trajectories
fprintf('\n========== D1 Policy B: frozen radius eps(4M) ==========\n');
eps_4M = df.solvers.compute_epsilon(cfg, max(opts.maxiters_values), confid, switch_eps);
fprintf('Frozen radius eps(4M) = [%s]\n', strtrim(sprintf('%.6g ', eps_4M)));
opts_frozen = opts;
opts_frozen.eps_override     = eps_4M;                  % same radius at every N
opts_frozen.precomputed_distY = res_correct.distY_time_all;  % reuse q_N exactly
rng(12345);
res_frozen = df.stages.run_stage_ii(cfg, opts_frozen);

%% Area vs N for both policies
nN = numel(opts.maxiters_values);
area_correct = zeros(nN, 1); share_correct = zeros(nN, 1); bnd_correct = false(nN, 1);
area_frozen  = zeros(nN, 1); share_frozen  = zeros(nN, 1); bnd_frozen  = false(nN, 1);
for ni = 1:nN
    [area_correct(ni), share_correct(ni), bnd_correct(ni)] = region_area(res_correct, ni, IDTOL);
    [area_frozen(ni),  share_frozen(ni),  bnd_frozen(ni)]  = region_area(res_frozen,  ni, IDTOL);
end

N = opts.maxiters_values(:);
summary = table(N, area_correct, share_correct, bnd_correct, ...
                   area_frozen,  share_frozen,  bnd_frozen);
fprintf('\n=== D1 fixed-eps panel ===\n');
disp(summary);
fprintf(['Reading: if area_frozen still falls with N, the region shrinkage is data-driven\n' ...
         '(q_N concentrating), not a radius artifact -- this answers confound A.\n']);
if any(bnd_correct) || any(bnd_frozen)
    fprintf('WARNING: identified region touches the fixed-grid edge at some N (area truncated).\n');
end

%% Save (single authoritative save; no server figures)
if ~exist(paths.artifacts, 'dir'), mkdir(paths.artifacts); end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii, 'fixed_eps_panel_D1.csv'));
save(fullfile(paths.artifacts, 'fixed_eps_panel_D1.mat'), ...
    'res_correct', 'res_frozen', 'summary', 'eps_4M', ...
    'opts', 'fixed_gridparamM', 'fixed_gridparamV', 'IDTOL', ...
    'confid', 'switch_eps', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'fixed_eps_panel_D1.mat'));
fprintf('========== D1 complete ==========\n');

%% Local helper -------------------------------------------------------------
function [area, share, hit_boundary] = region_area(results, ni, IDTOL)
    % Area of the identified region in (mu,sigma^2) space at horizon index ni.
    % Drops the leading true-parameter row/col (multiplier 1), as in the plots.
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
