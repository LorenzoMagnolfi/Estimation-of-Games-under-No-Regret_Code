%% II_RUN_sharp_set_volume
%
%  T1 / E.2, R3.1 (sharpness, partial identification). Demonstrates that the
%  identified set of cost distributions is SHARP in the partial-ID sense: as the
%  horizon N grows (q_N -> q*, eps(N) -> 0), the (mu,sigma^2) confidence region
%  does NOT collapse to a point but converges to a set of POSITIVE volume. That
%  positive-volume limit is the message a partial-ID editor (Molinari) wants:
%  the model set-identifies, and the reported region is the sharp set, not a
%  loose outer bound.
%
%  Method: full-feedback regret matching (corrected Hedge radius, switch_eps=10)
%  across an extended horizon grid up to 8M, solving the (mu,sigma^2) region on a
%  FIXED grid so the AREA (set volume in cost space) is comparable across N. The
%  area-vs-N curve flattening to a positive value is the sharp-volume evidence.
%  The 8M identified mask + grid are saved so the sharp-set BOUNDARY can be drawn
%  locally.
%
%  *** SCOPE FLAG ***
%  This runner delivers the sharp-set VOLUME / boundary (the solid, Molinari-
%  facing object). It deliberately does NOT include the attainability WITNESS
%  figure. The theory memo (item 4) flags that the current witness is a static
%  redraw device, and that the genuine strengthening R3 asked for is a
%  side-signaled CONTEXTUAL no-regret learner that attains a target BCCE point by
%  actual learning. That learner is a separate, careful build (it needs the
%  side-signal/contextual structure and NL's view), so it is scaffolded
%  elsewhere and intentionally not blind-fired here.
%
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
switch_eps = 10;
IDTOL = 1e-8;

cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

%% Fixed (mu,sigma^2) grid (same convention/extent as D1/G1)
NGRID = 60;
fixed_gridparamM = [1; linspace(0.18, 1.5, NGRID)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, NGRID)'];

opts = struct();
opts.maxiters_values   = [500000, 1000000, 2000000, 4000000, 8000000];  % extended to 8M
opts.alpha_set         = confid;
opts.switch_eps        = switch_eps;
opts.backend           = 'fast';
opts.adaptive          = false;            % inference-grade
opts.require_corrected = true;
opts.learning_style    = 'rm';
opts.fixed_gridparamV  = fixed_gridparamV;
opts.fixed_gridparamM  = fixed_gridparamM;

fprintf('\n========== T1: sharp-set volume vs N (up to 8M) ==========\n');
rng(12345);
res = df.stages.run_stage_ii(cfg, opts);

%% Area (set volume) vs N
nN = numel(opts.maxiters_values);
area = zeros(nN, 1); share = zeros(nN, 1); hit_boundary = false(nN, 1);
for ni = 1:nN
    [area(ni), share(ni), hit_boundary(ni)] = region_area(res, ni, IDTOL);
end
N = opts.maxiters_values(:);
volume_summary = table(N, area, share, hit_boundary);
fprintf('\n=== T1 sharp-set volume vs N ===\n');
disp(volume_summary);

% Convergence diagnostic: relative change in area over the last doubling of N.
if nN >= 2 && area(end-1) > 0
    rel_change_last = abs(area(end) - area(end-1)) / area(end-1);
else
    rel_change_last = NaN;
end
fprintf('Final area (N=%dM)            : %.5f (mu,sigma^2 units)\n', N(end)/1e6, area(end));
fprintf('Relative area change last step: %.3f%%\n', 100 * rel_change_last);
if area(end) > 0 && rel_change_last < 0.05
    fprintf('-> area has flattened to a POSITIVE limit: sharp set has positive volume\n');
    fprintf('   (genuine partial identification, not point identification).\n');
else
    fprintf('-> area still moving at 8M (rel change >= 5%%); report the trend / consider larger N.\n');
end
if any(hit_boundary)
    fprintf('WARNING: identified region touches the fixed-grid edge at some N (volume truncated).\n');
end

%% Extract the sharp-set mask at the largest N (for the boundary figure, drawn locally)
nV = res.NGridV + 1; nM = res.NGridM + 1;
distpars_last = squeeze(res.distpars_all(end, :, :));
mu_grid_last = reshape(distpars_last(:, 1), nV, nM); mu_grid_last = mu_grid_last(2:end, 2:end);
sg_grid_last = reshape(distpars_last(:, 2), nV, nM); sg_grid_last = sg_grid_last(2:end, 2:end);
VV_grid_last = reshape(res.VV_all(end, :), nV, nM);  VV_grid_last = VV_grid_last(2:end, 2:end);
sharp_mask_last = VV_grid_last <= IDTOL;

%% Save (single authoritative save; no server figures)
if ~exist(paths.artifacts, 'dir'), mkdir(paths.artifacts); end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(volume_summary, fullfile(paths.tables_ii, 'sharp_set_volume_T1.csv'));
save(fullfile(paths.artifacts, 'sharp_set_volume_T1.mat'), ...
    'res', 'volume_summary', 'rel_change_last', ...
    'mu_grid_last', 'sg_grid_last', 'VV_grid_last', 'sharp_mask_last', ...
    'opts', 'fixed_gridparamM', 'fixed_gridparamV', 'IDTOL', ...
    'confid', 'switch_eps', '-v7.3');
fprintf('\nArtifact: %s\n', fullfile(paths.artifacts, 'sharp_set_volume_T1.mat'));
fprintf('========== T1 complete ==========\n');

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
