%% II_SMOKE_proposals  Validate the new features for the proposal batch:
%   - high-probability epsilon (switch_eps 12/13) must be TIGHTER than Markov (10/11);
%   - CLT consistency: rho=0 reproduces exact, rho>0 relaxes, monotone in rho.
%  Lints the edited solver/driver too. Run: matlab -batch
%    "II_CHECK_corrected_bundle_prereqs; II_SMOKE_proposals"
fprintf('=== PROP_SMOKE_START ===\n'); ok = true;

lint_files = {fullfile('+df','+solvers','solve_socp_cvx.m'), ...
    fullfile('+df','+solvers','solve_grid_cvx.m'), ...
    fullfile('+df','+solvers','compute_epsilon.m'), ...
    fullfile('+df','+stages','run_stage_ii.m')};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
NGRID = 4; IDTOL = 1e-8; N = 40000; aR = 0.045; TOL = 2e-3;
fgM = [1; linspace(0.18,1.5,NGRID)']; fgV = [1; linspace(0.15,3.5,NGRID)'];
base = struct('maxiters_values', N, 'alpha_set', aR, 'switch_eps', 10, 'backend', 'fast', ...
    'adaptive', false, 'require_corrected', true, 'learning_style', 'rm', ...
    'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM);
solve = @(o) getfield(df.stages.run_stage_ii(cfg, o), 'VV_all');  %#ok<GFLD>

% High-probability epsilon must be tighter than Markov
e10 = df.solvers.compute_epsilon(cfg,N,aR,10); e12 = df.solvers.compute_epsilon(cfg,N,aR,12);
e11 = df.solvers.compute_epsilon(cfg,N,aR,11); e13 = df.solvers.compute_epsilon(cfg,N,aR,13);
fprintf('HP eps: Hedge Markov=%.4f HP=%.4f (HP<Markov %d) | EXP3 Markov=%.4f HP=%.4f (HP<Markov %d)\n', ...
    max(e10), max(e12), all(e12<e10), max(e11), max(e13), all(e13<e11));
ok = ok && all(e12 < e10) && all(e13 < e11);

% CLT consistency validation
rng(12345); VV_ex = solve(base); VV_ex = VV_ex(1,:);
o = base; o.consistency_slack_kind='CLT'; o.alpha_R=aR; o.alpha_C=1;       % rho=0 -> exact
rng(12345); VV_c0 = solve(o); VV_c0 = VV_c0(1,:);
o = base; o.consistency_slack_kind='CLT'; o.alpha_R=aR; o.alpha_C=0.005;   % rho>0
rng(12345); VV_c = solve(o); VV_c = VV_c(1,:);
o = base; o.consistency_slack_kind='CLT'; o.alpha_R=aR; o.alpha_C=0.0005;  % larger rho
rng(12345); VV_cb = solve(o); VV_cb = VV_cb(1,:);
o = base; o.consistency_slack_kind='L1'; o.alpha_R=aR; o.alpha_C=0.005;    % L1 comparison
rng(12345); VV_l1 = solve(o); VV_l1 = VV_l1(1,:);

d0 = max(abs(VV_c0 - VV_ex));
fprintf('(1) CLT rho=0 reproduces exact: max|dVV|=%.3e -> %s\n', d0, tf(d0 < 1e-9)); ok = ok && d0 < 1e-9;
v2 = max(VV_c - VV_ex);
fprintf('(2) CLT(rho>0) <= exact: max(VV_CLT-VV_exact)=%.3e -> %s\n', v2, tf(v2 < TOL)); ok = ok && v2 < TOL;
v3 = max(VV_cb - VV_c);
fprintf('(3) larger rho relaxes more: max(dVV)=%.3e -> %s\n', v3, tf(v3 < TOL)); ok = ok && v3 < TOL;
ne = nnz(VV_ex<=IDTOL); ncl = nnz(VV_c<=IDTOL); nl1 = nnz(VV_l1<=IDTOL);
fprintf('identified at alpha_C=0.005: exact=%d  CLT=%d  L1=%d  (which consistency set is sharper at s=5)\n', ne, ncl, nl1);
ok = ok && (ne <= ncl);

fprintf('=== PROP_SMOKE_DONE ALL_PASS=%d ===\n', ok);
function s = tf(b), if b, s='PASS'; else, s='FAIL'; end, end
