%% II_SMOKE_l1  Validate the L1 (Bretagnolle-Huber-Carol) consistency implementation.
%
%  Must-hold properties of a correct L1 relaxation of the consistency equality
%  (checked on one cached trajectory, same grid, only the consistency treatment
%  varies):
%    (1) r=0 reproduces EXACT consistency  (the L1 ball collapses to the point).
%    (2) r>0 weakly RELAXES exact          (the ball contains the exact point, so
%        every candidate stays feasible: VV_L1 <= VV_exact pointwise).
%    (3) larger r relaxes monotonically more (smaller alpha_C -> larger radius).
%    (4) box also relaxes exact            (regression: the box path still works).
%  (Box vs L1 are NOT nested at matched alpha_C: the inf-ball and 1-ball are
%   different shapes, so neither identified set contains the other.)
%
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_l1"

fprintf('=== L1_SMOKE_START ===\n');
ok = true;

%% (0) Lint the edited files (syntax / undefined-var screen)
lint_files = {fullfile('+df','+solvers','solve_socp_cvx.m'), ...
    fullfile('+df','+solvers','solve_grid_cvx.m'), ...
    fullfile('+df','+stages','run_stage_ii.m')};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

%% Tiny canonical lab
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
NGRID = 4; IDTOL = 1e-8; M_C = s^2; N = 40000;
fgM = [1; linspace(0.18,1.5,NGRID)']; fgV = [1; linspace(0.15,3.5,NGRID)'];
base = struct('maxiters_values', N, 'alpha_set', 0.05, 'switch_eps', 10, ...
    'backend', 'fast', 'adaptive', false, 'require_corrected', true, ...
    'learning_style', 'rm', 'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM);

solve = @(o) getfield(df.stages.run_stage_ii(cfg, o), 'VV_all');  %#ok<GFLD>

% exact
rng(12345); VV_ex = solve(base); VV_ex = VV_ex(1,:);
% L1 at r=0 (alpha_C = 2^M_C - 2 makes the BHC radius exactly 0)
o = base; o.consistency_slack_kind = 'L1'; o.alpha_C = 2^M_C - 2; o.alpha_R = 0.05;
rng(12345); VV_l1z = solve(o); VV_l1z = VV_l1z(1,:);
% L1 at alpha_C = 0.005 (r>0)
o = base; o.consistency_slack_kind = 'L1'; o.alpha_C = 0.005; o.alpha_R = 0.045;
rng(12345); VV_l1 = solve(o); VV_l1 = VV_l1(1,:);
% L1 at alpha_C = 0.001 (larger r)
o = base; o.consistency_slack_kind = 'L1'; o.alpha_C = 0.001; o.alpha_R = 0.049;
rng(12345); VV_l1b = solve(o); VV_l1b = VV_l1b(1,:);
% box at alpha_C = 0.005 (regression)
o = base; o.consistency_slack_kind = 'box'; o.alpha_C = 0.005; o.alpha_R = 0.045;
rng(12345); VV_bx = solve(o); VV_bx = VV_bx(1,:);

fprintf('radii @alpha_C=0.005,N=40k: r_L1=%.4f  r_box=%.4f\n', ...
    df.solvers.compute_consistency_slack(M_C,0.005,N,'L1'), ...
    df.solvers.compute_consistency_slack(M_C,0.005,N,'box'));

% (1) r=0 == exact
d0 = max(abs(VV_l1z - VV_ex));
fprintf('(1) L1 r=0 reproduces exact: max|dVV|=%.3e -> %s\n', d0, tf(d0 < 1e-9));
ok = ok && d0 < 1e-9;
% (2) L1 relaxes exact
v2 = max(VV_l1 - VV_ex);
fprintf('(2) L1(r>0) <= exact pointwise: max(VV_L1-VV_exact)=%.3e -> %s\n', v2, tf(v2 < 1e-6));
ok = ok && v2 < 1e-6;
% (3) monotone in r
v3 = max(VV_l1b - VV_l1);
fprintf('(3) larger r relaxes more (alpha_C 0.001<=0.005): max(dVV)=%.3e -> %s\n', v3, tf(v3 < 1e-6));
ok = ok && v3 < 1e-6;
% (4) box relaxes exact
v4 = max(VV_bx - VV_ex);
fprintf('(4) box(r>0) <= exact pointwise: max(VV_box-VV_exact)=%.3e -> %s\n', v4, tf(v4 < 1e-6));
ok = ok && v4 < 1e-6;
% counts
ne = nnz(VV_ex<=IDTOL); nl = nnz(VV_l1<=IDTOL); nb = nnz(VV_l1b<=IDTOL); nx = nnz(VV_bx<=IDTOL);
fprintf('identified counts: exact=%d  L1(.005)=%d  L1(.001)=%d  box(.005)=%d (expect exact<=L1<=L1b)\n', ne, nl, nb, nx);
ok = ok && (ne <= nl) && (nl <= nb);

fprintf('=== L1_SMOKE_DONE ALL_PASS=%d ===\n', ok);

function s = tf(b), if b, s='PASS'; else, s='FAIL'; end, end
