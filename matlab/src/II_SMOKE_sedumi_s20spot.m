%% II_SMOKE_sedumi_s20spot  Direct-lane spot check at s=20 (nonparametric stage).
%  The lane smokes covered s=5 cones; the s=20 production run (job 3779273) ran
%  on the direct lane. This closes the remaining register caveat: a small s=20
%  candidate cloud, one horizon, CVX lane vs direct lane, identical identified
%  sets required. Also reports the direct lane's status mix at s=20 cone sizes
%  (the share of residual-verified numerr=1 exits is the number to watch).
%
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_sedumi_s20spot"

fprintf('=== S20_SPOT_START ===\n');
ok = true;
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 20;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8; TOL = 2e-3; N = 500000;
base = struct('maxiters_values', N, 'alpha_set', 0.05, ...
    'alpha_R', 0.045, 'alpha_C', 0.005, 'switch_eps', 10, ...
    'backend', 'fast', 'learning_style', 'rm', 'use_parfor', false, ...
    'consistency_slack_kind', 'L1', 'K_local', 20, 'K_global', 30, 'K_spiky', 5);

o = base; o.solver_backend = 'cvx';
rng(12345); t0 = tic; r1 = df.stages.run_stage_ii_nonparam(cfg, o); t1 = toc(t0);
o = base; o.solver_backend = 'sedumi_direct';
rng(12345); t0 = tic; r2 = df.stages.run_stage_ii_nonparam(cfg, o); t2 = toc(t0);

v1 = r1.VV_all(1,:); v2 = r2.VV_all(1,:);
id1 = v1 <= IDTOL; id2 = v2 <= IDTOL; both = (v1 < 100) & (v2 < 100);
dmax = max([0, abs(v1(both) - v2(both))]); nf = nnz(id1 ~= id2);
sts = r2.solver_statuses{1};
nv = nnz(strcmp(sts, 'Solved(numerr=1,verified)'));
nu = nnz(strcmp(sts, 'numerr=1,unverified'));
fprintf(['s=20 spot: max|dVV|=%.3e  set flips=%d  cvx %.1fs  sd %.1fs  ' ...
    'numerr1-verified=%d  numerr1-unverified=%d -> %s\n'], ...
    dmax, nf, t1, t2, nv, nu, tf(dmax < TOL && nf == 0));
if nf > 0
    fl = find(id1 ~= id2);
    for q = 1:min(numel(fl), 10)
        j = fl(q);
        fprintf('    FLIP pt %3d: v_cvx=%12.5g  v_sd=%12.5g  sd_status=%s\n', ...
            j, v1(j), v2(j), sts{j});
    end
end
ok = ok && dmax < TOL && nf == 0;

if ok
    fprintf('S20_SPOT: PASS\n');
else
    fprintf('S20_SPOT: FAIL\n');
    error('II_SMOKE_sedumi_s20spot:FAIL', 's=20 lane disagreement.');
end

function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
