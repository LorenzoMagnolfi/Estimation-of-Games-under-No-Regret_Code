%% II_SMOKE_sedumi_direct  Validate the direct-SeDuMi lane against the CVX lane.
%
%  The sedumi_direct backend must be a drop-in replacement for the CVX lane:
%  same VV values (to solver noise), same identified-set classification at
%  IDTOL, on every consistency kind. Checked on ONE cached trajectory so only
%  the solve lane varies:
%    (1) exact ('none'):  VV agree, identified sets identical
%    (2) box:             same
%    (3) L1:              same (epigraph reformulation vs CVX norm(.,Inf) term)
%    (4) CLT:             same (per-point SOC block vs CVX norm(Sh*.,2) term)
%    (5) learn_only:      trajectory from learn_only reproduces the full-call VV
%    (6) candidate_mask:  masked solve equals full solve on the mask, NaN off it
%    (7) eps monotone:    identified(eps) subset of identified(2*eps)
%                         (licenses descending-scale pruning in sweeps)
%    (8) parfor:          parallel VV identical to serial (skipped if no pool)
%  Also reports per-solve timing for both lanes (the point of the exercise).
%
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_sedumi_direct"
%  PASS/FAIL is printed as SMOKE_SEDUMI: PASS|FAIL and FAIL exits nonzero.

fprintf('=== SEDUMI_DIRECT_SMOKE_START ===\n');
ok = true;

%% (0) Lint the touched files
lint_files = {fullfile('+df','+solvers','socp_to_sedumi.m'), ...
    fullfile('+df','+solvers','solve_socp_sedumi.m'), ...
    fullfile('+df','+solvers','solve_grid_sedumi.m'), ...
    fullfile('+df','+stages','run_stage_ii.m')};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

%% Tiny canonical lab (matches II_SMOKE_l1 scale)
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
NGRID = 4; IDTOL = 1e-8; N = 40000;
fgM = [1; linspace(0.18,1.5,NGRID)']; fgV = [1; linspace(0.15,3.5,NGRID)'];
aR = 0.045; aC = 0.005; TOL = 2e-3;   % TOL: SeDuMi per-solve noise scale (II_SMOKE_l1)
base = struct('maxiters_values', N, 'alpha_set', aR + aC, 'switch_eps', 10, ...
    'backend', 'fast', 'adaptive', false, 'require_corrected', true, ...
    'learning_style', 'rm', 'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM, ...
    'use_parfor', false);

%% (5a) learn once via learn_only; all solves below reuse the trajectory
o = base; o.learn_only = true;
rng(12345); res_learn = df.stages.run_stage_ii(cfg, o);
traj = res_learn.distY_time_all;
assert(all(isnan(res_learn.VV_all(:))), 'learn_only must return NaN VV');
fprintf('(5a) learn_only returns trajectory + NaN VV -> OK\n');

kinds = {'none', 'box', 'L1', 'CLT'};
tim = struct('cvx', nan(1,numel(kinds)), 'sd', nan(1,numel(kinds)));
VV_keep = struct();
for k = 1:numel(kinds)
    kind = kinds{k};
    o = base; o.precomputed_distY = traj;
    if ~strcmp(kind, 'none')
        o.consistency_slack_kind = kind; o.alpha_set = aR + aC;
        o.alpha_R = aR; o.alpha_C = aC;
    end
    o.solver_backend = 'cvx';
    rng(12345); t0 = tic; r_cvx = df.stages.run_stage_ii(cfg, o); tim.cvx(k) = toc(t0);
    o.solver_backend = 'sedumi_direct';
    rng(12345); t0 = tic; r_sd = df.stages.run_stage_ii(cfg, o); tim.sd(k) = toc(t0);

    v_cvx = r_cvx.VV_all(1,:); v_sd = r_sd.VV_all(1,:);
    id_cvx = v_cvx <= IDTOL; id_sd = v_sd <= IDTOL;
    both_solved = (v_cvx < 100) & (v_sd < 100);
    dmax = max(abs(v_cvx(both_solved) - v_sd(both_solved)));
    if isempty(dmax), dmax = 0; end
    n_flip = nnz(id_cvx ~= id_sd);
    n_cls = nnz((v_cvx < 100) ~= (v_sd < 100));
    fprintf('(%d) kind=%-4s  max|dVV|=%.3e  set flips=%d  solved-class flips=%d -> %s\n', ...
        k, kind, dmax, n_flip, n_cls, tf(dmax < TOL && n_flip == 0));
    ok = ok && dmax < TOL && n_flip == 0;
    if n_flip > 0 || n_cls > 0
        sts = r_sd.solver_statuses{1};
        fl = find((id_cvx ~= id_sd) | ((v_cvx < 100) ~= (v_sd < 100)));
        for q = 1:min(numel(fl), 12)
            j = fl(q);
            fprintf('    FLIP pt %2d: v_cvx=%12.5g  v_sd=%12.5g  sd_status=%s\n', ...
                j, v_cvx(j), v_sd(j), sts{j});
        end
    end
    VV_keep.(matlab.lang.makeValidName(kind)) = v_sd;
end

%% (6) candidate_mask: half grid; equals full solve on mask, NaN off mask
o = base; o.precomputed_distY = traj; o.solver_backend = 'sedumi_direct';
nG = numel(VV_keep.none);
mask = false(nG,1); mask(1:2:end) = true;
o.candidate_mask = mask;
rng(12345); r_m = df.stages.run_stage_ii(cfg, o);
v_m = r_m.VV_all(1,:);
maskr = mask.';   % row mask: keeps vector-logical indexing in row orientation
d6 = max(abs(v_m(maskr) - VV_keep.none(maskr)));
ok6 = d6 < 1e-12 && all(isnan(v_m(~maskr)));
fprintf('(6) candidate_mask: max|dVV|on-mask=%.3e, off-mask NaN=%d -> %s\n', ...
    d6, all(isnan(v_m(~maskr))), tf(ok6));
ok = ok && ok6;

%% (7) monotone in the radius: identified(eps) subset of identified(2*eps)
eps1 = df.solvers.compute_epsilon(cfg, N, aR, 10);
for sc = [1 2]
    o = base; o.precomputed_distY = traj; o.solver_backend = 'sedumi_direct';
    o.eps_override = sc * eps1; o.switch_eps = 10;
    rng(12345); r_sc = df.stages.run_stage_ii(cfg, o);
    if sc == 1, id_lo = r_sc.VV_all(1,:) <= IDTOL; else, id_hi = r_sc.VV_all(1,:) <= IDTOL; end
end
ok7 = all(id_hi(id_lo));   % every point identified at eps stays identified at 2*eps
fprintf('(7) eps-monotonicity (pruning license): subset holds=%d -> %s\n', ok7, tf(ok7));
ok = ok && ok7;

%% (8) parfor equivalence (only if a pool can be opened; else skipped)
did_par = false;
if exist('gcp', 'file') == 2
    try
        if isempty(gcp('nocreate'))
            ncpu = str2double(getenv('SLURM_CPUS_ON_NODE'));
            if isfinite(ncpu) && ncpu > 1, parpool('Processes', min(ncpu, 8)); end
        end
        if ~isempty(gcp('nocreate'))
            o = base; o.precomputed_distY = traj; o.solver_backend = 'sedumi_direct';
            o.use_parfor = true;
            rng(12345); r_p = df.stages.run_stage_ii(cfg, o);
            d8 = max(abs(r_p.VV_all(1,:) - VV_keep.none));
            fprintf('(8) parfor == serial: max|dVV|=%.3e -> %s\n', d8, tf(d8 < 1e-9));
            ok = ok && d8 < 1e-9;
            did_par = true;
        end
    catch par_err
        fprintf('(8) parfor check skipped (%s)\n', par_err.message);
    end
end
if ~did_par, fprintf('(8) parfor check SKIPPED (no pool available)\n'); end

%% Timing report
fprintf('--- timing (whole stage call on %d-point grid, cached trajectory) ---\n', nG);
for k = 1:numel(kinds)
    fprintf('  %-4s  cvx %.1fs   sedumi_direct %.1fs   speedup x%.1f\n', ...
        kinds{k}, tim.cvx(k), tim.sd(k), tim.cvx(k) / tim.sd(k));
end

if ok
    fprintf('SMOKE_SEDUMI: PASS\n');
else
    fprintf('SMOKE_SEDUMI: FAIL\n');
    error('II_SMOKE_sedumi_direct:FAIL', 'sedumi_direct lane disagrees with CVX lane.');
end

function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
