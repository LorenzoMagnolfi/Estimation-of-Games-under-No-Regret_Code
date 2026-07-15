%% II_SMOKE_sedumi_nonparam  Validate the direct-SeDuMi lane on the NONPARAM stage.
%
%  The parametric-stage equivalence is established (II_SMOKE_sedumi_direct, job
%  3778539). The nonparametric stage shares the SOCP layout but builds its own
%  candidate cloud and prior reconstruction, so the lane is validated on THIS
%  stage too before any production run (gate for the s=20 launch):
%    (1) exact consistency: identical identified sets, VV agreement to noise
%    (2) L1 consistency:    same
%  Tiny problem: s=5, ~101-candidate cloud, T=40k, one horizon.
%
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_sedumi_nonparam"
%  Prints SMOKE_SEDUMI_NONPARAM: PASS|FAIL; FAIL exits nonzero.

fprintf('=== SEDUMI_NONPARAM_SMOKE_START ===\n');
ok = true;

lint_files = {fullfile('+df','+stages','run_stage_ii_nonparam.m'), ...
    'II_RUN_nonparam_L1_s20.m'};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8; TOL = 2e-3; N = 40000;
base = struct('maxiters_values', N, 'alpha_set', 0.05, ...
    'alpha_R', 0.045, 'alpha_C', 0.005, 'switch_eps', 10, ...
    'backend', 'fast', 'learning_style', 'rm', 'use_parfor', false, ...
    'K_local', 30, 'K_global', 60, 'K_spiky', 10);

for kindc = {'none', 'L1'}
    kind = kindc{1};
    o = base;
    if ~strcmp(kind, 'none'), o.consistency_slack_kind = 'L1'; end
    o.solver_backend = 'cvx';
    rng(12345); t0 = tic; r_cvx = df.stages.run_stage_ii_nonparam(cfg, o); t_cvx = toc(t0);
    o.solver_backend = 'sedumi_direct';
    rng(12345); t0 = tic; r_sd = df.stages.run_stage_ii_nonparam(cfg, o); t_sd = toc(t0);

    v_cvx = r_cvx.VV_all(1,:); v_sd = r_sd.VV_all(1,:);
    id_cvx = v_cvx <= IDTOL; id_sd = v_sd <= IDTOL;
    both = (v_cvx < 100) & (v_sd < 100);
    dmax = max([0, abs(v_cvx(both) - v_sd(both))]);
    n_flip = nnz(id_cvx ~= id_sd);
    n_cls = nnz((v_cvx < 100) ~= (v_sd < 100));
    fprintf('kind=%-4s  max|dVV|=%.3e  set flips=%d  class flips=%d  cvx %.1fs sd %.1fs -> %s\n', ...
        kind, dmax, n_flip, n_cls, t_cvx, t_sd, tf(dmax < TOL && n_flip == 0));
    if n_flip > 0
        sts = r_sd.solver_statuses{1};
        fl = find(id_cvx ~= id_sd);
        for q = 1:min(numel(fl), 10)
            j = fl(q);
            fprintf('    FLIP pt %3d: v_cvx=%12.5g  v_sd=%12.5g  sd_status=%s\n', ...
                j, v_cvx(j), v_sd(j), sts{j});
        end
    end
    ok = ok && dmax < TOL && n_flip == 0;
end

if ok
    fprintf('SMOKE_SEDUMI_NONPARAM: PASS\n');
else
    fprintf('SMOKE_SEDUMI_NONPARAM: FAIL\n');
    error('II_SMOKE_sedumi_nonparam:FAIL', 'nonparam sedumi_direct disagrees with CVX lane.');
end

function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
