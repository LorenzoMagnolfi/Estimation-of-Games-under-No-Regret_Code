%% II_CHECK_v05_primal  Full-coordinate primal feasibility in the two-seller lab, and the normalized joint budget (Niccolò, 4 Nov 2026: I7(c), M2).
%  One Hedge trajectory at N = 4M (seed 12345, as the frontier). For every cell of the 61x61 grid the direct primal program
%  in the joint distribution nu(a, t) over action profiles and type pairs:
%     sum_t nu(a, t) = q_N(a)                      (action matching)
%     |sum_a nu(a, t) - Psi_lambda(t)| <= z(t), sum_t z(t) <= r_L1          (L1 composition set)
%     Gamma_{i, t_i, a'}(nu) <= eps_{i,t_i}        (obedience; rectangle or ceiling-count tail), or
%     Gamma_{i, t_i, a'}(nu) <= rho_{i,t_i}, sum_k w_k rho_k / b_k <= 1/alpha_R   (normalized joint budget, eq:s4jointbudget)
%  is solved as an LP (linprog if the Optimization Toolbox is licensed, else SeDuMi in standard form). Four rules:
%  'momentF' (rectangle), 'tailF' (direct tail at the count ceiling N min{1, phi + r/2}), 'jointF' (budget on the first
%  moments b_k), 'tailF_own' (direct tail at the witness's own count N s_k(nu) through tangent lines, an outer LP).
%  Compares the primal classification of momentF and tailF with the reduced-direction lane's VV of the frontier run
%  (v05_frontier.mat), cell by cell, and reports the joint-budget and own-count regions. CSV; no figures.
clear; clc; paths = df_repo_paths();
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8]; mu = 3*ones(2,1); sigma2 = eye(2); s = 5; N = 4000000;
IDTOL = 1e-8; NGRID = 60; alpha_R = 0.045; alpha_C = 0.005; NTAN = 24;
fgM = [1; linspace(0.18, 1.5, NGRID)']; fgV = [1; linspace(0.15, 3.5, NGRID)'];
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
A = cfg.A; AA = cfg.AA; Pi = cfg.Pi; NA = size(A, 1); d = numel(AA); s2 = s^2; M_R = NPlayers * s; delta = alpha_R / M_R;
rng(12345); t0 = tic; [dY, ~, fr] = df.sim.learn_hedge(cfg, 1, N, N, 1, 1, 1, 1); fprintf('[DGP] Hedge learned %.0fs, max realized regret %.3e\n', toc(t0), max(fr(:)));
qN = dY(:, 1);
% type pairs in the Stage II order (player 1 fastest), their marginals, and the per-type payoff span
T1 = cfg.type_space{1,1}; T2 = cfg.type_space{2,1};
pair_t1 = repmat((1:s)', s, 1); pair_t2 = kron((1:s)', ones(s, 1));          % pair p = t1 + s (t2 - 1)
K = zeros(s, NPlayers); for j = 1:NPlayers, K(:, j) = (max(Pi(:, :, j), [], 1) - min(Pi(:, :, j), [], 1))'; end
% deviation profiles: for player i and deviation a', the profile index of (a' , a_{-i})
dev_idx = zeros(NA, d, NPlayers);
for i = 1:NPlayers
    for a = 1:NA
        for ap = 1:d
            prof = A(a, :); prof(i) = AA(ap); dev_idx(a, ap, i) = find(all(A == prof, 2), 1);
        end
    end
end
% obedience coefficient matrices: G{i}(t, ap, :) over nu(a, p) with p the pair and t the own type
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
Sown = zeros(s, nvar, NPlayers);                                           % s_k(nu) = mass of own type k of player i
for i = 1:NPlayers, for p = 1:s2, ti = (i == 1) * pair_t1(p) + (i == 2) * pair_t2(p); Sown(ti, :, i) = Sown(ti, :, i) + Asum_pair(p, :); end, end
r_L1 = df.solvers.compute_consistency_slack(s2, alpha_C, N, 'L1');
JN = 1 + ceil(log2(N)); cF = 1.25 * sqrt(log(d));
Htail = @(m, k) K(k, 1) * min(m, cF * sqrt(m) + sqrt((m + 1) * log(JN / delta)));                 % cumulative family-F tail
bF = @(m, k) K(k, 1) * min(m, (cF + 0.25) * sqrt(m)) / N;                                           % first moment per auction
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
use_linprog = license('test', 'Optimization_Toolbox') && exist('linprog', 'file') == 2;
solver_names = {'sedumi', 'linprog'};
fprintf('primal LP: %d variables; solver %s\n', nvar, solver_names{use_linprog + 1});
ncpu = str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu > 1 && exist('gcp', 'file') == 2 && isempty(gcp('nocreate')), try, parpool('Processes', min(ncpu, 16)); catch, end, end
rules = {'momentF', 'tailF', 'jointF', 'tailF_own'};
feas = false(NGrid, numel(rules)); stat = zeros(NGrid, numel(rules));
m_grid = unique([1, logspace(0, log10(N), NTAN)]);
parfor nd = 1:NGrid
    psi_nd = Psi(:, nd); phi = marg(:, nd);
    f_nd = false(1, numel(rules)); st_nd = zeros(1, numel(rules));
    for ri = 1:numel(rules)
        rule = rules{ri};
        % variables: nu (nvar), z (s2), then rho (s per player) for the budget
        nz = s2; nrho = (strcmp(rule, 'jointF')) * (NPlayers * s);
        nv = nvar + nz + nrho;
        % first-moment bounds per (player, type) and the budget's active set; a bound below 1e-13 per auction is treated
        % as zero (r_k = 0, as eq:s4jointbudget prescribes for b_k = 0), which keeps the LP coefficients finite
        bvec = zeros(NPlayers * s, 1); for i2 = 1:NPlayers, for t2 = 1:s, bvec((i2 - 1) * s + t2) = bF(N * phi(t2), t2); end, end
        J = bvec > 1e-13; w = zeros(size(bvec)); if any(J), w(J) = 1 / nnz(J); end
        Aeq = [Aeq_match, zeros(NA, nz + nrho)]; beq = qN;
        Aineq = []; bineq = [];
        % composition: sum_a nu(a,p) - psi <= z,  psi - sum_a nu <= z,  sum z <= r
        Aineq = [Aineq; Asum_pair, -eye(nz), zeros(nz, nrho); -Asum_pair, -eye(nz), zeros(nz, nrho); zeros(1, nvar), ones(1, nz), zeros(1, nrho)];
        bineq = [bineq; psi_nd; -psi_nd; r_L1];
        for i = 1:NPlayers
            Gi = G{i};
            for t = 1:s
                k = t;
                switch rule
                    case 'momentF'
                        eps_t = (M_R / alpha_R) * bF(N * phi(t), k);
                        for ap = 1:d, Aineq = [Aineq; squeeze(Gi(t, ap, :))', zeros(1, nz + nrho)]; bineq = [bineq; eps_t]; end %#ok<AGROW>
                    case 'tailF'
                        mcap = N * min(1, phi(t) + r_L1 / 2); eps_t = Htail(mcap, k) / N;
                        for ap = 1:d, Aineq = [Aineq; squeeze(Gi(t, ap, :))', zeros(1, nz + nrho)]; bineq = [bineq; eps_t]; end %#ok<AGROW>
                    case 'jointF'
                        col = nvar + nz + (i - 1) * s + t;
                        for ap = 1:d, row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))'; row(col) = -1; Aineq = [Aineq; row]; bineq = [bineq; 0]; end %#ok<AGROW>
                    case 'tailF_own'
                        for mt = m_grid
                            h = 1e-3; Hm = Htail(mt, k); dH = (Htail(mt * (1 + h), k) - Htail(mt * (1 - h), k)) / (2 * h * mt);
                            for ap = 1:d
                                row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))' - dH * Sown(t, :, i);
                                Aineq = [Aineq; row]; bineq = [bineq; (Hm - dH * mt) / N]; %#ok<AGROW>
                            end
                        end
                        for ap = 1:d, row = zeros(1, nv); row(1:nvar) = squeeze(Gi(t, ap, :))' - K(k, 1) * Sown(t, :, i); Aineq = [Aineq; row]; bineq = [bineq; 0]; end %#ok<AGROW>
                end
            end
        end
        if strcmp(rule, 'jointF')
            coef = zeros(size(bvec)); coef(J) = w(J) ./ bvec(J);
            row = zeros(1, nv); row(nvar + nz + 1:end) = coef'; Aineq = [Aineq; row]; bineq = [bineq; 1 / alpha_R]; %#ok<AGROW>
        end
        lb = zeros(nv, 1); ub = inf(nv, 1);
        if strcmp(rule, 'jointF'), ub(nvar + nz + find(~J)) = 0; end
        if use_linprog
            o = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
            [~, ~, ef] = linprog(zeros(nv, 1), Aineq, bineq, Aeq, beq, lb, ub, o);
            f_nd(ri) = ef == 1; st_nd(ri) = ef;
        else
            % SeDuMi standard form: [Aeq; Aineq + slack] x = b, x >= 0 (ub = 0 handled by dropping the column)
            keep = ub > 0; nk = nnz(keep); ni = numel(bineq);
            As = [Aeq(:, keep), zeros(NA, ni); Aineq(:, keep), eye(ni)]; bs = [beq; bineq];
            pars = struct('fid', 0, 'eps', 1e-9);
            [~, ~, info] = sedumi(As, bs, zeros(nk + ni, 1), [], pars);
            f_nd(ri) = (info.pinf == 0) && (info.numerr < 2); st_nd(ri) = info.pinf + 10 * info.numerr;
        end
    end
    feas(nd, :) = f_nd; stat(nd, :) = st_nd;
end
% compare with the lane's frontier run
fr_mat = fullfile(paths.artifacts, 'v05_frontier.mat'); lane = struct();
if exist(fr_mat, 'file') == 2
    F = load(fr_mat); names = cellfun(@char, F.name, 'UniformOutput', false);
    for arm = {'hedge_momentF', 'hedge_tailF'}
        ia = find(contains(names, 'hedge') & contains(names, strrep(arm{1}, 'hedge_', '')), 1);
        if ~isempty(ia), lane.(arm{1}) = F.VV{ia}(:) <= IDTOL; end
    end
end
T = table((1:NGrid)', dp(:, 1), dp(:, 2), feas(:, 1), feas(:, 2), feas(:, 3), feas(:, 4), stat(:, 1), stat(:, 2), stat(:, 3), stat(:, 4), ...
    'VariableNames', {'cell', 'mu', 'sigma2', 'primal_momentF', 'primal_tailF', 'primal_jointF', 'primal_tailF_own', 'st_momentF', 'st_tailF', 'st_jointF', 'st_tailF_own'});
if isfield(lane, 'hedge_momentF') && numel(lane.hedge_momentF) == NGrid, T.lane_momentF = lane.hedge_momentF; end
if isfield(lane, 'hedge_tailF') && numel(lane.hedge_tailF) == NGrid, T.lane_tailF = lane.hedge_tailF; end
if ~exist(paths.tables_ii, 'dir'), mkdir(paths.tables_ii); end
writetable(T, fullfile(paths.tables_ii, 'v05_primal_check.csv'));
save(fullfile(paths.artifacts, 'v05_primal_check.mat'), 'T', 'feas', 'stat', 'dp', 'rules', 'N', 'r_L1', 'NTAN', '-v7.3');
fprintf('cells feasible: momentF %d, tailF %d, jointF %d, tailF_own %d (of %d); truth cell 1: %s\n', sum(feas(:, 1)), sum(feas(:, 2)), sum(feas(:, 3)), sum(feas(:, 4)), NGrid, mat2str(feas(1, :)));
if isfield(lane, 'hedge_momentF') && numel(lane.hedge_momentF) == NGrid
    fprintf('agreement with the lane: momentF %d/%d cells, tailF %d/%d cells (lane feasible: %d, %d)\n', sum(feas(:,1) == lane.hedge_momentF), NGrid, sum(feas(:,2) == lane.hedge_tailF), NGrid, sum(lane.hedge_momentF), sum(lane.hedge_tailF));
end
fprintf('solver statuses (nonstandard): %d\n', nnz(stat ~= 1 & stat ~= 0 & use_linprog) + nnz(stat >= 10 & ~use_linprog));
fprintf('========== V05 primal check complete ==========\n');
