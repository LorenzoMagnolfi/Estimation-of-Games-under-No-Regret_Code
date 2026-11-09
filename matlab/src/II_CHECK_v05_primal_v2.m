%% II_CHECK_v05_primal_v2  Close-out of 9 November 2026 (Niccolò's C1 to C3): the full-coordinate primal check of II_CHECK_v05_primal, with
%  C1  first-moment bounds: b = 0 is a structural zero (r = 0); 0 < b < 1e-13 per period is replaced UPWARD by 1e-13 (a looser, hence
%      conservative, bound) instead of being rounded to zero; the active set is {b > 0} (all ten (player, type) pairs in this lab, so
%      the matched rectangle coincides with the rectangle); replacements are counted per cell;
%  C2  own-count tail: analytic supporting lines, i.e. the tangents of the smooth concave branch K g(m), g(m) = c_F sqrt(m) +
%      sqrt((m + 1) ln(J_N / delta)), at NTAN log-spaced counts (they majorize K g >= H) plus the cap line K m (it majorizes H); the
%      envelope is certified numerically on a fine count grid (minimum margin, maximum relative gap);
%  C3  every program is solved as a minimum-violation LP (min sum of slacks s >= 0 with A x - s <= b, equalities kept): an optimum of
%      zero certifies feasibility (the point is returned and its residual saved), a positive optimum is the L1 distance of the data to
%      the inequality system (a certificate of infeasibility); exit flags other than optimal, or optima in the grey zone, are UNRESOLVED
%      (code 2), counted, and included in an outer region, never labeled infeasible;
%  plus the rule 'momentF_matched' (the rectangle with the budget's active set and weights, eps_t = b_t / (alpha_R w_t)).
%  Same trajectory (seed 12345, N = 4M), same 61 x 61 grid, same other rules as v1. Compares with v1 (v05_primal_check.mat) and the lane.
clear; clc; paths = df_repo_paths();
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8]; mu = 3*ones(2,1); sigma2 = eye(2); s = 5; N = 4000000;
IDTOL = 1e-8; NGRID = 60; alpha_R = 0.045; alpha_C = 0.005; NTAN = 24;
B_FLOOR = 1e-13; FEAS_TOL = 1e-9; INFEAS_TOL = 1e-7;
fgM = [1; linspace(0.18, 1.5, NGRID)']; fgV = [1; linspace(0.15, 3.5, NGRID)'];
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
A = cfg.A; AA = cfg.AA; Pi = cfg.Pi; NA = size(A, 1); d = numel(AA); s2 = s^2; M_R = NPlayers * s; delta = alpha_R / M_R;
rng(12345); t0 = tic; [dY, ~, fr] = df.sim.learn_hedge(cfg, 1, N, N, 1, 1, 1, 1); fprintf('[DGP] Hedge learned %.0fs, max realized regret %.3e\n', toc(t0), max(fr(:)));
qN = dY(:, 1);
T1 = cfg.type_space{1,1}; T2 = cfg.type_space{2,1};
pair_t1 = repmat((1:s)', s, 1); pair_t2 = kron((1:s)', ones(s, 1));          % pair p = t1 + s (t2 - 1)
K = zeros(s, NPlayers); for j = 1:NPlayers, K(:, j) = (max(Pi(:, :, j), [], 1) - min(Pi(:, :, j), [], 1))'; end
dev_idx = zeros(NA, d, NPlayers);
for i = 1:NPlayers
    for a = 1:NA
        for ap = 1:d
            prof = A(a, :); prof(i) = AA(ap); dev_idx(a, ap, i) = find(all(A == prof, 2), 1);
        end
    end
end
nvar = NA * s2; vidx = @(a, p) a + NA * (p - 1);
G = cell(NPlayers, 1);
for i = 1:NPlayers
    Gi = zeros(s, d, nvar);
    for p = 1:s2
        ti = (i == 1) * pair_t1(p) + (i == 2) * pair_t2(p);
        for a = 1:NA
            for ap = 1:d
                Gi(ti, ap, vidx(a, p)) = Pi(dev_idx(a, ap, i), ti, i) - Pi(a, ti, i);
            end
        end
    end
    G{i} = Gi;
end
Aeq_match = zeros(NA, nvar); for a = 1:NA, for p = 1:s2, Aeq_match(a, vidx(a, p)) = 1; end, end
Asum_pair = zeros(s2, nvar); for p = 1:s2, for a = 1:NA, Asum_pair(p, vidx(a, p)) = 1; end, end
Sown = zeros(s, nvar, NPlayers);
for i = 1:NPlayers, for p = 1:s2, ti = (i == 1) * pair_t1(p) + (i == 2) * pair_t2(p); Sown(ti, :, i) = Sown(ti, :, i) + Asum_pair(p, :); end, end
r_L1 = df.solvers.compute_consistency_slack(s2, alpha_C, N, 'L1');
JN = 1 + ceil(log2(N)); cF = 1.25 * sqrt(log(d)); lnJ = log(JN / delta);
g = @(m) cF * sqrt(m) + sqrt((m + 1) * lnJ);                                   % smooth concave branch (per unit K)
dg = @(m) cF ./ (2 * sqrt(m)) + 0.5 * sqrt(lnJ ./ (m + 1));                    % its derivative
Htail = @(m, k) K(k, 1) * min(m, g(m));                                        % cumulative family-F tail
bF = @(m, k) K(k, 1) * min(m, (cF + 0.25) * sqrt(m)) / N;                      % first moment per period
% C2 certificate: the tangent lines of K g at m_grid plus the cap line K m majorize H = K min(m, g(m)) on a fine count grid
m_grid = unique([1, logspace(0, log10(N), NTAN)]); mf = logspace(0, log10(N), 4000); cert = zeros(s, 2);
for k = 1:s
    Kk = K(k, 1); Hf = Kk * min(mf, g(mf));
    lines = Kk * (g(m_grid)' + dg(m_grid)' .* (mf - m_grid'));
    env = min(Kk * mf, min(lines, [], 1)); sc = max(1, max(Hf));
    cert(k, :) = [min(env - Hf) / sc, max((env - Hf) ./ max(Hf, 1e-12 * sc))];
end
assert(all(cert(:, 1) >= -1e-9), 'C2: supporting-line certificate failed');
fprintf('C2 certificate (%d lines plus the cap): min margin %.2e, max relative envelope gap %.2e\n', numel(m_grid), min(cert(:, 1)), max(cert(:, 2)));
% candidate grid and priors
[dp, distribution_parameters] = df.report.build_param_grid(mu, sigma2, fgM, fgV, 'Both'); NGrid = size(dp, 1);
cstr = df.solvers.build_constraints(cfg.type_space, cfg.action_space, cfg.Pi); T_sorted = cstr.T_sorted;
Psi = zeros(s2, NGrid); marg = zeros(s, NGrid);
for nd = 1:NGrid
    Psi(:, nd) = mvnpdf(T_sorted, distribution_parameters{2, nd}, distribution_parameters{3, nd});
    mv = distribution_parameters{2, nd}; mv = mv(1); sv = distribution_parameters{3, nd}; sv = sv(1, 1);
    md = normpdf(cfg.type_space{1,1}, mv, sqrt(sv)); marg(:, nd) = md / sum(md);
end
Psi = Psi ./ sum(Psi, 1);
assert(license('test', 'Optimization_Toolbox') && exist('linprog', 'file') == 2, 'v2 needs linprog');
fprintf('primal LP: %d variables; minimum-violation form; solver linprog\n', nvar);
ncpu = str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu > 1 && exist('gcp', 'file') == 2 && isempty(gcp('nocreate')), try, parpool('Processes', min(ncpu, 16)); catch, end, end
rules = {'momentF', 'momentF_matched', 'tailF', 'jointF', 'tailF_own'}; NR = numel(rules);
feas = zeros(NGrid, NR, 'int8'); dist = nan(NGrid, NR); resid = nan(NGrid, NR); stat = zeros(NGrid, NR); nsmall = zeros(NGrid, 1);
o1 = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex', 'MaxIterations', 2e6, 'OptimalityTolerance', 1e-9, 'ConstraintTolerance', 1e-9);
o2 = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'interior-point', 'MaxIterations', 2000, 'OptimalityTolerance', 1e-9, 'ConstraintTolerance', 1e-9);
parfor nd = 1:NGrid
    psi_nd = Psi(:, nd); phi = marg(:, nd);
    f_nd = zeros(1, NR, 'int8'); d_nd = nan(1, NR); r_nd = nan(1, NR); st_nd = zeros(1, NR);
    % C1: bounds, structural zeros, upward replacement, active set and weights (computed once per cell, before the rules)
    bvec = zeros(NPlayers * s, 1); for i2 = 1:NPlayers, for t2 = 1:s, bvec((i2 - 1) * s + t2) = bF(N * phi(t2), t2); end, end
    small = bvec > 0 & bvec < B_FLOOR; bt = bvec; bt(small) = B_FLOOR; J = bt > 0;
    w = zeros(size(bt)); if any(J), w(J) = 1 / nnz(J); end
    ns_nd = nnz(small);
    for ri = 1:NR
        rule = rules{ri};
        nz = s2; nrho = strcmp(rule, 'jointF') * (NPlayers * s); nv = nvar + nz + nrho;
        Aeq = [Aeq_match, zeros(NA, nz + nrho)]; beq = qN;
        Aineq = [Asum_pair, -eye(nz), zeros(nz, nrho); -Asum_pair, -eye(nz), zeros(nz, nrho); zeros(1, nvar), ones(1, nz), zeros(1, nrho)];
        bineq = [psi_nd; -psi_nd; r_L1];
        for i = 1:NPlayers
            Gi = G{i};
            for t = 1:s
                k = t; kk = (i - 1) * s + t;
                switch rule
                    case 'momentF'
                        eps_t = (M_R / alpha_R) * bvec(kk);
                        for ap = 1:d, Aineq = [Aineq; squeeze(Gi(t, ap, :))', zeros(1, nz + nrho)]; bineq = [bineq; eps_t]; end %#ok<AGROW>
                    case 'momentF_matched'
                        if J(kk), eps_t = bt(kk) / (alpha_R * w(kk)); else, eps_t = 0; end
                        for ap = 1:d, Aineq = [Aineq; squeeze(Gi(t, ap, :))', zeros(1, nz + nrho)]; bineq = [bineq; eps_t]; end %#ok<AGROW>
                    case 'tailF'
                        mcap = N * min(1, phi(t) + r_L1 / 2); eps_t = Htail(mcap, k) / N;
                        for ap = 1:d, Aineq = [Aineq; squeeze(Gi(t, ap, :))', zeros(1, nz + nrho)]; bineq = [bineq; eps_t]; end %#ok<AGROW>
                    case 'jointF'
                        col = nvar + nz + kk;
                        for ap = 1:d, row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))'; row(col) = -1; Aineq = [Aineq; row]; bineq = [bineq; 0]; end %#ok<AGROW>
                    case 'tailF_own'
                        Kk = K(k, 1);
                        for mt = m_grid
                            sl = Kk * dg(mt); ic = Kk * g(mt) - sl * mt;                       % supporting line: K g(mt) + K g'(mt) (m - mt)
                            for ap = 1:d
                                row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))' - sl * Sown(t, :, i);
                                Aineq = [Aineq; row]; bineq = [bineq; ic / N]; %#ok<AGROW>
                            end
                        end
                        for ap = 1:d, row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))' - Kk * Sown(t, :, i); Aineq = [Aineq; row]; bineq = [bineq; 0]; end %#ok<AGROW>
                end
            end
        end
        if strcmp(rule, 'jointF')
            coef = zeros(size(bt)); coef(J) = w(J) ./ bt(J);
            row = zeros(1, nv); row(nvar + nz + 1:end) = coef'; Aineq = [Aineq; row]; bineq = [bineq; 1 / alpha_R]; %#ok<AGROW>
        end
        lb = zeros(nv, 1); ub = inf(nv, 1);
        if strcmp(rule, 'jointF'), ub(nvar + nz + find(~J)) = 0; end
        % C3: minimum-violation LP in (x, slack)
        ni = numel(bineq);
        A2 = [Aineq, -eye(ni)]; Ae2 = [Aeq, zeros(NA, ni)]; c2 = [zeros(nv, 1); ones(ni, 1)]; lb2 = [lb; zeros(ni, 1)]; ub2 = [ub; inf(ni, 1)];
        [x2, fv, ef] = linprog(c2, A2, bineq, Ae2, beq, lb2, ub2, o1);
        if ef ~= 1
            [x2, fv, ef] = linprog(c2, A2, bineq, Ae2, beq, lb2, ub2, o2); ef = 10 * ef;        % interior-point fallback, marked
        end
        st_nd(ri) = ef;
        if ef == 1 || ef == 10
            d_nd(ri) = fv; x = x2(1:nv);
            r_nd(ri) = max([max(Aineq * x - bineq); max(abs(Aeq * x - beq)); max(-x)]);
            if fv <= FEAS_TOL, f_nd(ri) = 1; elseif fv >= INFEAS_TOL, f_nd(ri) = 0; else, f_nd(ri) = 2; end
        else
            f_nd(ri) = 2;
        end
    end
    feas(nd, :) = f_nd; dist(nd, :) = d_nd; resid(nd, :) = r_nd; stat(nd, :) = st_nd; nsmall(nd) = ns_nd;
end
% lane and v1 for comparison
fr_mat = fullfile(paths.artifacts, 'v05_frontier.mat'); lane = struct();
if exist(fr_mat, 'file') == 2
    F = load(fr_mat); names = cellfun(@char, F.name, 'UniformOutput', false);
    for arm = {'hedge_momentF', 'hedge_tailF'}
        ia = find(contains(names, 'hedge') & contains(names, strrep(arm{1}, 'hedge_', '')), 1);
        if ~isempty(ia), lane.(arm{1}) = F.VV{ia}(:) <= IDTOL; end
    end
end
v1 = struct(); v1_mat = fullfile(paths.artifacts, 'v05_primal_check.mat');
if exist(v1_mat, 'file') == 2, V = load(v1_mat); v1.feas = V.feas; v1.rules = V.rules; v1.stat = V.stat; end
% table
names = {'cell', 'mu', 'sigma2', 'n_floor_replaced'}; cols = [(1:NGrid)', dp(:, 1), dp(:, 2), nsmall];
for ri = 1:NR, names = [names, {['feas_' rules{ri}]}]; cols = [cols, double(feas(:, ri))]; end %#ok<AGROW>
for ri = 1:NR, names = [names, {['dist_' rules{ri}]}]; cols = [cols, dist(:, ri)]; end %#ok<AGROW>
for ri = 1:NR, names = [names, {['resid_' rules{ri}]}]; cols = [cols, resid(:, ri)]; end %#ok<AGROW>
for ri = 1:NR, names = [names, {['st_' rules{ri}]}]; cols = [cols, stat(:, ri)]; end %#ok<AGROW>
T = array2table(cols, 'VariableNames', names);
if isfield(lane, 'hedge_momentF') && numel(lane.hedge_momentF) == NGrid, T.lane_momentF = double(lane.hedge_momentF); T.lane_tailF = double(lane.hedge_tailF); end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(T, fullfile(paths.tables_ii, 'v05_primal_check_v2.csv'));
save(fullfile(paths.artifacts, 'v05_primal_check_v2.mat'), 'T', 'feas', 'dist', 'resid', 'stat', 'nsmall', 'cert', 'rules', 'N', 'r_L1', 'NTAN', 'B_FLOOR', 'FEAS_TOL', 'INFEAS_TOL', '-v7.3');
% summaries (the monitor greps these lines)
pure = dp(:, 1) ~= 3 & dp(:, 2) ~= 1;
for ri = 1:NR
    f = feas(:, ri);
    fprintf('%-16s feasible %4d, infeasible %4d, unresolved %d (outer %4d) of %d; pure grid feasible %4d; truth cell %d\n', rules{ri}, nnz(f == 1), nnz(f == 0), nnz(f == 2), nnz(f ~= 0), NGrid, nnz(f == 1 & pure), f(1));
end
fprintf('C1: %d cells had %d bounds replaced upward (of %d bounds)\n', nnz(nsmall > 0), sum(nsmall), NGrid * NPlayers * s);
fprintf('C3: %d LPs needed the interior-point fallback; %d unresolved overall; max residual of feasible points %.2e; grey-zone optima %d\n', nnz(stat >= 10 | (stat ~= 1 & stat < 10 & stat ~= 0)), nnz(feas == 2), max(resid(feas == 1)), nnz(dist > FEAS_TOL & dist < INFEAS_TOL));
if isfield(v1, 'feas')
    for ri = 1:NR
        rn = rules{ri}; ia = find(strcmp(v1.rules, rn), 1);
        if isempty(ia), continue; end
        f2 = feas(:, ri); f1 = v1.feas(:, ia); res = f2 ~= 2;
        chg = find(res & (double(f2) ~= double(f1)));
        fprintf('v1 vs v2 %-12s: resolved %d, agree %d, changed %d', rn, nnz(res), nnz(res & double(f2) == double(f1)), numel(chg));
        if ~isempty(chg), fprintf(' [cells %s]', mat2str(chg(1:min(10, end))')); end
        fprintf('; v1 iteration-limit cells now: %s\n', mat2str(double(f2(v1.stat(:, ia) == 0))'));
    end
end
if isfield(lane, 'hedge_momentF') && numel(lane.hedge_momentF) == NGrid
    for pr = {{'momentF', 'hedge_momentF'}, {'tailF', 'hedge_tailF'}}
        ri = find(strcmp(rules, pr{1}{1}), 1); f2 = feas(:, ri); res = f2 ~= 2; L = lane.(pr{1}{2});
        fprintf('agreement with the lane %-8s: %d of %d resolved cells (lane feasible %d, v2 feasible %d)\n', pr{1}{1}, nnz(res & (double(f2) == double(L))), nnz(res), nnz(L), nnz(f2 == 1));
    end
end
ri_c = find(strcmp(rules, 'tailF'), 1); ri_o = find(strcmp(rules, 'tailF_own'), 1);
fprintf('own-count outer region vs ceiling: own feasible %d, ceiling feasible %d, own \\ ceiling %d, ceiling \\ own %d\n', nnz(feas(:, ri_o) == 1), nnz(feas(:, ri_c) == 1), nnz(feas(:, ri_o) == 1 & feas(:, ri_c) ~= 1), nnz(feas(:, ri_c) == 1 & feas(:, ri_o) ~= 1));
ri_m = find(strcmp(rules, 'momentF'), 1); ri_mm = find(strcmp(rules, 'momentF_matched'), 1); ri_j = find(strcmp(rules, 'jointF'), 1);
fprintf('rectangle %d, matched rectangle %d (active set %d of %d), budget %d (outer %d)\n', nnz(feas(:, ri_m) == 1), nnz(feas(:, ri_mm) == 1), NPlayers * s - 0, NPlayers * s, nnz(feas(:, ri_j) == 1), nnz(feas(:, ri_j) ~= 0));
fprintf('========== V05 primal check v2 complete ==========\n');
