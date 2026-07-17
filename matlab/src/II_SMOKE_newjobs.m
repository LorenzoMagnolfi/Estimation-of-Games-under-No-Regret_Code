%% II_SMOKE_newjobs  Gate for the four queued runners (budget pilot, nested
%  regimes, regret diagnostics, MC quantiles): lint everything touched, then
%  micro-versions of each mechanism at tiny T and grid.
%    (1) budget: rectangle vs budget(scale 1) containment; monotone in scale
%    (2) nested: bandit-Markov region contains FF-Markov region (same traj)
%    (3) regret: all three learners return finite regrets; pooled returns both
%        units with pooled << type
%    (4) mc: two seeds produce finite (trajectory-dependent) areas
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_newjobs"

fprintf('=== NEWJOBS_SMOKE_START ===\n');
ok = true;
lint_files = {fullfile('+df','+solvers','solve_socp_cvx.m'), ...
    fullfile('+df','+solvers','solve_grid_cvx.m'), ...
    fullfile('+df','+stages','run_stage_ii.m'), ...
    'II_RUN_budget_set_pilot.m', 'II_RUN_nested_regimes.m', ...
    'II_RUN_regret_diagnostics.m', 'II_RUN_mc_area_quantiles.m'};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL = 1e-8; N = 40000; NGRID = 4;
fgM = [1; linspace(0.18,1.5,NGRID)']; fgV = [1; linspace(0.15,3.5,NGRID)'];
base = struct('maxiters_values', N, 'alpha_set', 0.05, 'alpha_R', 0.045, ...
    'alpha_C', 0.005, 'backend', 'fast', 'adaptive', false, ...
    'require_corrected', true, 'consistency_slack_kind', 'L1', ...
    'learning_style', 'rm', 'use_parfor', false, ...
    'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM);

o = base; o.switch_eps = 10; o.learn_only = true;
rng(12345); res = df.stages.run_stage_ii(cfg, o); traj = res.distY_time_all;

% (1) budget mechanism
o = base; o.switch_eps = 10; o.precomputed_distY = traj; o.solver_backend = 'cvx';
rng(12345); r_rect = df.stages.run_stage_ii(cfg, o);
sh_rect = nnz(r_rect.VV_all(1,:) <= IDTOL);
o.obed_budget_scale = 1.0; rng(12345); r_b1 = df.stages.run_stage_ii(cfg, o);
sh_b1 = nnz(r_b1.VV_all(1,:) <= IDTOL);
o.obed_budget_scale = 0.3; rng(12345); r_b3 = df.stages.run_stage_ii(cfg, o);
sh_b3 = nnz(r_b3.VV_all(1,:) <= IDTOL);
ok1 = (sh_b1 >= sh_rect) && (sh_b3 <= sh_b1);
% pointwise containment: every rectangle-identified point is budget(1)-identified
cont = all(r_b1.VV_all(1, r_rect.VV_all(1,:) <= IDTOL) <= IDTOL);
fprintf('(1) budget: rect=%d b(1)=%d b(0.3)=%d containment=%d -> %s\n', ...
    sh_rect, sh_b1, sh_b3, cont, tf(ok1 && cont));
ok = ok && ok1 && cont;

% (2) nesting mechanism (direct lane)
o = base; o.switch_eps = 10; o.precomputed_distY = traj;
rng(12345); r_ff = df.stages.run_stage_ii(cfg, o);
o.switch_eps = 11; rng(12345); r_bd = df.stages.run_stage_ii(cfg, o);
id_ff = r_ff.VV_all(1,:) <= IDTOL; id_bd = r_bd.VV_all(1,:) <= IDTOL;
ok2 = all(id_bd(id_ff));
fprintf('(2) nesting: ff=%d bandit=%d, ff subset of bandit=%d -> %s\n', ...
    nnz(id_ff), nnz(id_bd), ok2, tf(ok2));
ok = ok && ok2;

% (3) learner regret outputs
cfg.learning_style = 'rm';
rng(1); [~,~,fr_rm,p1,~] = learn_mod(cfg, 1, N, N, 1, 1, 1, 1);
cfg.learning_style = 'prm';
rng(1); [~,~,fr_prm,~,~] = learn_mod_prm(cfg, 1, N, N, 1, 1, 1, 1);
cfg.learning_style = 'pooled';
rng(1); [~,~,fr_pl,fr_ty,~,~] = learn_mod_pooled(cfg, 1, N, N, 1, 1, 1, 1);
ok3 = all(isfinite([max(fr_rm(:)), max(fr_prm(:)), max(fr_pl(:)), max(fr_ty(:))])) ...
    && ~isempty(p1) && max(fr_pl(:)) < max(fr_ty(:));
fprintf('(3) regrets: rm=%.2e prm=%.2e pooled=%.2e/type=%.2e -> %s\n', ...
    max(fr_rm(:)), max(fr_prm(:)), max(fr_pl(:)), max(fr_ty(:)), tf(ok3));
ok = ok && ok3;

% (4) mc mechanism: two seeds, finite areas
cfg.learning_style = 'rm';
a = nan(2,1);
for k = 1:2
    o = base; o.switch_eps = 10; o.learn_only = true;
    rng(k); rs = df.stages.run_stage_ii(cfg, o);
    o = base; o.switch_eps = 10; o.precomputed_distY = rs.distY_time_all;
    rr = df.stages.run_stage_ii(cfg, o);
    a(k) = nnz(rr.VV_all(1,:) <= IDTOL);
end
ok4 = all(isfinite(a));
fprintf('(4) mc: seed shares=[%d %d] -> %s\n', a(1), a(2), tf(ok4));
ok = ok && ok4;

if ok
    fprintf('SMOKE_NEWJOBS: PASS\n');
else
    fprintf('SMOKE_NEWJOBS: FAIL\n');
    error('II_SMOKE_newjobs:FAIL', 'new-jobs smoke failed.');
end

function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
