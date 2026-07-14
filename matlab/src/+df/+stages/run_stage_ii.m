function results = run_stage_ii(cfg, opts)
% DF.STAGES.RUN_STAGE_II  Stage II: simulation identification across iteration counts.
%
%   results = df.stages.run_stage_ii(cfg, opts)
%
%   For each iteration count in opts.maxiters_values, runs the learning
%   algorithm and solves the BCCE identification problem over a parameter grid.
%
%   Inputs:
%     cfg  — config struct from df.setup.game_simulation (5-action game)
%     opts — struct with fields:
%       .maxiters_values — vector of iteration counts (default: [500000 1000000 2000000 4000000])
%       .NGridV          — variance grid size (default: 100)
%       .NGridM          — mean grid size (default: 100)
%       .alpha_set       — confidence levels (default: 0.05)
%       .switch_eps      — epsilon formula selector (default: 1)
%       .backend         — 'cvx' (default) | 'fast'
%                          'cvx': legacy per-point ComputeBCCE_eps path
%                          'fast': precomputes objectives, uses CVX+SeDuMi batch solver
%       .adaptive        — logical, enable adaptive grid in fast backend (default: true)
%                          NOTE: adaptive is for exploration only, not production/inference
%       .learning_style  — 'rm' (default) | 'prm' (proxy-regret matching)
%       .solver_backend  — 'cvx' (default) | 'sedumi_direct': engine for the
%                          full-grid solve. sedumi_direct builds the canonical
%                          form once and calls sedumi per point (no per-point
%                          CVX overhead, parfor-capable, identical semantics).
%       .learn_only      — logical: learn and return trajectories only;
%                          VV_all is NaN (default: false)
%       .candidate_mask  — logical NGrid x 1: solve only these grid points,
%                          NaN elsewhere (monotone-sweep pruning; default: [])
%
%   Outputs:
%     results — struct with fields:
%       .VV_all                    — (n_iters x NGrid) solver outputs
%       .distpars_all              — (n_iters x NGrid x 2) parameter grids
%       .distribution_parameters   — {n_iters x 1} cell of dist param cells
%       .distY_time_all            — {n_iters x 1} cell of action distributions
%       .maxiters_values           — iteration counts used
%       .gridparamV_all            — {n_iters x 1} cell of variance grid vectors
%       .timing                    — struct with per-iteration timing
%       .cfg                       — config struct (for downstream use)

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'maxiters_values'), opts.maxiters_values = [500000, 1000000, 2000000, 4000000]; end
if ~isfield(opts, 'NGridV'),          opts.NGridV = 100; end
if ~isfield(opts, 'NGridM'),          opts.NGridM = 100; end
if ~isfield(opts, 'alpha_set'),       opts.alpha_set = 0.05; end
if ~isfield(opts, 'switch_eps'),      opts.switch_eps = 1; end
if ~isfield(opts, 'backend'),         opts.backend = 'cvx'; end
if ~isfield(opts, 'use_parfor'),      opts.use_parfor = true; end
if ~isfield(opts, 'adaptive'),        opts.adaptive = true; end
if ~isfield(opts, 'learning_style'),  opts.learning_style = 'rm'; end
if ~isfield(opts, 'require_corrected'), opts.require_corrected = false; end
% SIM-5: guard against the deprecated pre-correction radius silently entering
% revision artifacts. Revision runners set require_corrected = true.
if opts.require_corrected && ~ismember(opts.switch_eps, [10, 11, 12, 13])
    error('run_stage_ii:UncorrectedEps', ...
        ['require_corrected=true but switch_eps=%d. Use 10/11 (Markov Hedge/EXP3) or ' ...
         '12/13 (high-probability Hedge/EXP3); switch_eps=1 is the deprecated radius.'], ...
        opts.switch_eps);
end
% Solver engine for the full-grid solve (within backend='fast', adaptive=false):
%   'cvx'           legacy per-point cvx_begin/cvx_end (modeling overhead per point)
%   'sedumi_direct' build the canonical form once (socp_to_sedumi), call sedumi
%                   per point; identical classification semantics, parfor-capable.
% Validated against 'cvx' by II_SMOKE_sedumi_direct on all consistency kinds.
if ~isfield(opts, 'solver_backend'), opts.solver_backend = 'cvx'; end
if ~ismember(opts.solver_backend, {'cvx', 'sedumi_direct'})
    error('run_stage_ii:BadBackend', ...
        'solver_backend must be ''cvx'' or ''sedumi_direct'' (got ''%s'').', ...
        opts.solver_backend);
end
% learn_only: run the learning stage, return trajectories in distY_time_all,
% and skip the radius + grid solve entirely (VV_all comes back NaN). This is
% the supported way to extract a trajectory for later precomputed_distY reuse.
if ~isfield(opts, 'learn_only'), opts.learn_only = false; end
% candidate_mask: logical NGrid x 1. Solve only the masked grid points; the
% rest return NaN (read as not-identified downstream). Used by monotone
% sweeps (e.g. descending radius scales) to skip points already excluded.
if ~isfield(opts, 'candidate_mask'), opts.candidate_mask = []; end
% Consistency-slack options (default: exact consistency, equivalent to legacy behavior).
%   .consistency_slack_kind  'none' (default) | 'box' | 'L1'
%   .alpha_R, .alpha_C       confidence-budget split (alpha_R + alpha_C = alpha_set);
%                            only used when consistency_slack_kind != 'none'.
%   .precomputed_distY       optional cell array of cached learning trajectories,
%                            one per maxiters_values entry, to skip the learn step
%                            (used for re-solving the LP under different slack/rate).
if ~isfield(opts, 'consistency_slack_kind'), opts.consistency_slack_kind = 'none'; end
if ~isfield(opts, 'alpha_R'), opts.alpha_R = opts.alpha_set; end
if ~isfield(opts, 'alpha_C'), opts.alpha_C = 0; end
if ~isfield(opts, 'precomputed_distY'), opts.precomputed_distY = {}; end
% Fixed-radius mode (D1 fixed-eps panel, T1 sharp-set volume): when nonempty,
% the SAME epsilon vector is used at every N instead of compute_epsilon(maxiters).
% Scalar broadcasts to 1 x s; otherwise must be 1 x s (per-type, as compute_epsilon).
if ~isfield(opts, 'eps_override'), opts.eps_override = []; end
% Fixed-grid mode (area-vs-N comparisons): when nonempty, these (mu,sigma)
% multiplier grids are used for EVERY N so the identified-set AREA is comparable
% across horizons (the default grid narrows with N, which is fine for per-N region
% plots but not for an area-vs-N curve). Follow the [1; linspace(...)'] convention:
% the leading 1 is the true-parameter multiplier dropped by the (2:end) plot slice.
if ~isfield(opts, 'fixed_gridparamV'), opts.fixed_gridparamV = []; end
if ~isfield(opts, 'fixed_gridparamM'), opts.fixed_gridparamM = []; end
% When a fixed grid is supplied, derive the grid sizes from it so the
% preallocation (NGridV+1)*(NGridM+1) matches what build_param_grid returns.
if ~isempty(opts.fixed_gridparamV), opts.NGridV = numel(opts.fixed_gridparamV) - 1; end
if ~isempty(opts.fixed_gridparamM), opts.NGridM = numel(opts.fixed_gridparamM) - 1; end
if ~ismember(opts.consistency_slack_kind, {'none', 'box', 'L1', 'CLT'})
    error('run_stage_ii:BadSlackKind', ...
        'consistency_slack_kind must be ''none'', ''box'', ''L1'', or ''CLT'' (got ''%s'').', ...
        opts.consistency_slack_kind);
end
use_box = strcmp(opts.consistency_slack_kind, 'box');
use_l1  = strcmp(opts.consistency_slack_kind, 'L1');
use_clt = strcmp(opts.consistency_slack_kind, 'CLT');
use_consistency_slack = use_box || use_l1 || use_clt;
% L1 and CLT modify the per-point CVX objective (a support-function term), so they
% need the full grid solve, not the adaptive subgrid.
if (use_l1 || use_clt) && opts.adaptive
    error('run_stage_ii:SlackAdaptive', ...
        'consistency_slack_kind=''L1''/''CLT'' requires adaptive=false (full grid solve).');
end

use_fast = strcmp(opts.backend, 'fast');
if ~isempty(opts.candidate_mask) && (opts.adaptive || ~use_fast)
    error('run_stage_ii:MaskNeedsFullSolve', ...
        'candidate_mask requires backend=''fast'' and adaptive=false.');
end
opts.learning_style = lower(opts.learning_style);
if ~ismember(opts.learning_style, {'rm', 'prm', 'pooled'})
    error('run_stage_ii:UnknownLearningStyle', ...
        'Unknown learning_style "%s". Use "rm", "prm", or "pooled".', opts.learning_style);
end
use_prm = strcmp(opts.learning_style, 'prm');
use_pooled = strcmp(opts.learning_style, 'pooled');

cfg.learning_style = opts.learning_style;
numdst_t = 1;
numdst_t_obs = numdst_t;

mu = cfg.mu;
sigma2 = cfg.sigma2;
type_space = cfg.type_space;
action_space = cfg.action_space;
Pi = cfg.Pi;
s = size(type_space{1,1}, 1);

n_iters = numel(opts.maxiters_values);
num_alpha = numel(opts.alpha_set);

% NGrid computed after grid construction (gridparamV/M include a leading 1,
% so actual size is (NGridV+1)*(NGridM+1))
nV = opts.NGridV + 1;
nM = opts.NGridM + 1;
NGrid = nV * nM;

% Preallocate
VV_all = zeros(n_iters, NGrid);
distpars_all = zeros(n_iters, NGrid, 2);
distribution_parameters_cell = cell(n_iters, 1);
distY_time_all = cell(n_iters, 1);
gridparamV_all = cell(n_iters, 1);
solver_statuses_cell = cell(n_iters, 1);   % per-point statuses (sedumi_direct lane)
timing_all = struct();

% Build constraints ONCE (shared across all iterations and grid points).
% The BOX slack changes the matrix STRUCTURE (consistency equality -> upper/lower
% box inequalities), so it needs the box build with a positive dummy; r_N is
% injected into c_all per iteration below. The L1 slack keeps the EXACT structure
% (consistency stays an equality) and instead adds the L1-ball support function to
% the per-point objective at solve time, so it uses the exact build.
if use_fast
    if use_box
        cstr = df.solvers.build_constraints(type_space, action_space, Pi, 1.0);
    else
        cstr = df.solvers.build_constraints(type_space, action_space, Pi);
    end
    dim_u = cstr.NA - 1;
    a_dim = cstr.a;
    NAg = cstr.NAg;
    s2 = cstr.s2;
    T_sorted = cstr.T_sorted;
    % L1/CLT slack: indices of the consistency-equality dual block in the dual
    % vector x. The objective is ordered [cone(dim_u); action-marginal(NA);
    % consistency(s2); mass(1); obedience], so the consistency duals start after
    % dim_u + NA. (For CLT, p_lambda is read from the same block of c.)
    if use_l1 || use_clt
        cons_idx = dim_u + cstr.NA + (1:s2);
    end
    if strcmp(opts.solver_backend, 'sedumi_direct')
        fprintf('[Stage II] Fast backend: SeDuMi direct (precomputed objectives)');
    else
        fprintf('[Stage II] Fast backend: CVX+SeDuMi (precomputed objectives)');
    end
    if opts.adaptive, fprintf(' + adaptive'); end
    if use_consistency_slack
        fprintf(' + consistency slack (%s, alpha_R=%g, alpha_C=%g)', ...
            opts.consistency_slack_kind, opts.alpha_R, opts.alpha_C);
    end
    fprintf('\n');
end

% Pool policy: the sedumi_direct lane is parfor-safe (no CVX global state).
% Open a pool only when that lane will actually solve with use_parfor, no
% pool is open yet, and the job has cores (SLURM) + the parallel toolbox;
% otherwise everything runs serially in-process (parfor width 0).
if use_fast && ~opts.learn_only && strcmp(opts.solver_backend, 'sedumi_direct') ...
        && opts.use_parfor && exist('gcp', 'file') == 2 && isempty(gcp('nocreate'))
    ncpu = str2double(getenv('SLURM_CPUS_ON_NODE'));
    if isfinite(ncpu) && ncpu > 1
        try
            parpool('Processes', min(ncpu, 16));
        catch pool_err
            fprintf('  (parpool unavailable: %s — solving serially)\n', pool_err.message);
        end
    end
end

for maxiter_index = 1:n_iters
    maxiters = opts.maxiters_values(maxiter_index);
    t_iter = tic;

    N = 1;
    M = maxiters;
    M_obs = maxiters;

    %% Learning  (skip if a precomputed trajectory was provided for this iter)
    if numel(opts.precomputed_distY) >= maxiter_index && ...
            ~isempty(opts.precomputed_distY{maxiter_index})
        fprintf('[Stage II] iter %d/%d: maxiters=%dk, using cached trajectory\n', ...
            maxiter_index, n_iters, maxiters/1000);
        distY_time = opts.precomputed_distY{maxiter_index};
        action_distribution = distY_time;
        distY_time_all{maxiter_index} = distY_time;
        t_learn_val = 0;
    else
        fprintf('[Stage II] iter %d/%d: maxiters=%dk, learning...', ...
            maxiter_index, n_iters, maxiters/1000);
        t_learn = tic;
        if use_prm
            [distY_time, ~] = learn_mod_prm(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
        elseif use_pooled
            [distY_time, ~] = learn_mod_pooled(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
        else
            [distY_time, ~] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
        end
        action_distribution = distY_time;
        distY_time_all{maxiter_index} = distY_time;
        t_learn_val = toc(t_learn);
        fprintf(' %.1fs\n', t_learn_val);
    end

    %% Parameter grid (iteration-dependent variance range, unless fixed grid given)
    if ~isempty(opts.fixed_gridparamV)
        gridparamV = opts.fixed_gridparamV(:);   % same grid at every N (area-vs-N)
    elseif maxiter_index == 1
        gridparamV = [1; linspace(0.15, sigma2(1,1)*10, opts.NGridV)'];
    elseif maxiter_index == 2
        gridparamV = [1; linspace(0.15, sigma2(1,1)*6, opts.NGridV)'];
    else
        gridparamV = [1; linspace(0.15, sigma2(1,1)*3.5, opts.NGridV)'];
    end
    if ~isempty(opts.fixed_gridparamM)
        gridparamM = opts.fixed_gridparamM(:);
    else
        gridparamM = [1; linspace(0.55, mu(1,1)*0.5, opts.NGridM)'];
    end
    gridparamV_all{maxiter_index} = gridparamV;

    [distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV);
    distpars_all(maxiter_index, :, :) = distpars;
    distribution_parameters_cell{maxiter_index} = distribution_parameters;

    %% Learn-only mode: keep the trajectory, skip radius + solve entirely.
    % VV_all must be NaN, not the preallocated 0 (0 would read as identified).
    if opts.learn_only
        VV_all(maxiter_index, :) = NaN;
        t_iter_val = toc(t_iter);
        fprintf('[Stage II] iter %d done (learn only): %.1fs\n\n', maxiter_index, t_iter_val);
        timing_all(maxiter_index).learn = t_learn_val;
        timing_all(maxiter_index).total = t_iter_val;
        continue
    end

    %% Solver
    if use_fast
        %% Fast path: precompute objectives, use coneprog + adaptive + parfor
        fprintf('  Building objectives (%d grid points)...', NGrid);
        t_obj = tic;

        % Epsilon for this iteration count (or fixed-radius override).
        % Use the OBEDIENCE budget alpha_R, not the full alpha: when a consistency
        % slack reserves alpha_C, the regret/obedience radius must be computed at
        % alpha_R = alpha - alpha_C so the union bound delivers 1 - alpha overall
        % (Theorem CR). alpha_R defaults to alpha_set, so exact-consistency runs
        % (alpha_C = 0) are unchanged.
        confid = opts.alpha_R(1);
        if ~isempty(opts.eps_override)
            if isscalar(opts.eps_override)
                eps_vec = opts.eps_override * ones(1, s);
            else
                eps_vec = opts.eps_override(:)';   % expect 1 x s
            end
        else
            eps_vec = df.solvers.compute_epsilon(cfg, maxiters, confid, opts.switch_eps);
        end

        % Psi (joint prior) and marginal distributions
        Psi = zeros(s2, NGrid);
        marg_distrib = zeros(s, NGrid);
        for nd = 1:NGrid
            Psi(:,nd) = mvnpdf(T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
            mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
            sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
            md = normpdf(type_space{1,1}, mu_val, sqrt(sg_val));
            marg_distrib(:,nd) = md / sum(md);
        end
        Psi = Psi ./ sum(Psi, 1);

        % Consistency slack radius (only if enabled)
        if use_box || use_l1
            M_C = s2;  % |T||Theta|; in this simulation Theta is degenerate so M_C = s2
            r_N = df.solvers.compute_consistency_slack( ...
                M_C, opts.alpha_C, maxiters, opts.consistency_slack_kind);
        elseif use_clt
            % CLT chi-square ellipsoid radius rho = sqrt(chi2_{d-1,1-alpha_C}/N).
            % chi2inv(p,k) = 2*gammaincinv(p,k/2) (base MATLAB; no Stats toolbox).
            chi2_q = 2 * gammaincinv(1 - opts.alpha_C, (s2 - 1) / 2);
            clt_rho = sqrt(chi2_q / maxiters);
        end

        % Build all objective vectors
        c_all = zeros(size(cstr.B_EQ, 2), NGrid);
        for nd = 1:NGrid
            bmarg = Psi(:,nd);
            eps_fin = repmat(sqrt(marg_distrib(:,nd))', 1, NAg*a_dim) .* ...
                      repmat(eps_vec, 1, NAg*a_dim);
            if use_box
                % Box constraint order:
                %   Meq:  action marginal (NA), total mass (1)
                %   Mineq: consistency_upper (s2), consistency_lower (s2), obedience
                c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', 1, ...
                               (bmarg + r_N)', (-bmarg + r_N)', eps_fin]';
            else
                % Exact / L1 order: [cone; action_dist; bmarg; 1; eps_fin].
                % L1 keeps this exact objective and adds its dual term at solve time.
                c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', ...
                               bmarg', 1, eps_fin]';
            end
        end
        fprintf(' %.1fs\n', toc(t_obj));

        % Solve (CVX+SeDuMi for accuracy; adaptive grid for speed)
        t_solve = tic;
        if opts.adaptive
            fprintf('  Adaptive grid solve (%dx%d):\n', nV, nM);
            adapt_opts = struct('backend', 'cvx', 'solver', 'sedumi');
            [VV, n_solved, ~] = df.solvers.solve_grid_adaptive(cstr, c_all, nV, nM, adapt_opts);
            fprintf('  Solved %d/%d points (%.1f%%)\n', n_solved, NGrid, 100*n_solved/NGrid);
        else
            % Candidate mask: solve only the requested columns (monotone-sweep
            % pruning); unsolved points come back NaN, which downstream
            % VV<=tol tests read as not-identified.
            if ~isempty(opts.candidate_mask)
                mask = logical(opts.candidate_mask(:));
                if numel(mask) ~= NGrid
                    error('run_stage_ii:BadMask', ...
                        'candidate_mask has %d entries; grid has %d points.', ...
                        numel(mask), NGrid);
                end
            else
                mask = true(NGrid, 1);
            end
            fprintf('  Full grid solve (%d/%d points, %s):\n', ...
                nnz(mask), NGrid, opts.solver_backend);
            gopts = struct('verbose', false, 'solver', 'sedumi');  % overnight: quiet SOCP (VV is saved regardless)
            if use_l1
                gopts.cons_l1_r   = r_N;        % BHC L1 radius (compute_consistency_slack 'L1')
                gopts.cons_l1_idx = cons_idx;   % consistency-equality dual block in x
            elseif use_clt
                gopts.cons_clt_rho = clt_rho;   % chi-square ellipsoid radius
                gopts.cons_clt_idx = cons_idx;
            end
            c_sub = c_all(:, mask);
            if strcmp(opts.solver_backend, 'sedumi_direct')
                gopts.use_parfor = opts.use_parfor;
                [VV_sub, ~, st_sub] = df.solvers.solve_grid_sedumi(cstr, c_sub, gopts);
                st = repmat({'masked'}, NGrid, 1);
                st(mask) = st_sub;
                solver_statuses_cell{maxiter_index} = st;
            else
                [VV_sub, ~] = df.solvers.solve_grid_cvx(cstr, c_sub, gopts);
            end
            VV = nan(NGrid, 1);
            VV(mask) = VV_sub;
        end
        t_solve_val = toc(t_solve);
        fprintf('  Solver: %.1fs\n', t_solve_val);

        VV_all(maxiter_index, :) = VV(:)';

    else
        %% Legacy path: CVX + SeDuMi via ComputeBCCE_eps
        numdist = size(action_distribution, 2);
        maxvals = zeros(numdist, num_alpha, NGrid);

        for ii = 1:numdist
            distrib = action_distribution(:, ii);
            for jj = 1:num_alpha
                confid = opts.alpha_set(jj);
                outs = ComputeBCCE_eps(type_space, action_space, distrib, ...
                    Pi * 0 + Pi, distribution_parameters, maxiters, confid, ...
                    Pi, opts.switch_eps, cfg);
                maxvals(ii, jj, :) = cell2mat(outs);
            end
        end

        VV = squeeze(maxvals);
        VV_all(maxiter_index, :) = VV;
    end

    t_iter_val = toc(t_iter);
    fprintf('[Stage II] iter %d done: %.1fs\n\n', maxiter_index, t_iter_val);
    timing_all(maxiter_index).learn = t_learn_val;
    timing_all(maxiter_index).total = t_iter_val;
end

%% Pack results
results.VV_all = VV_all;
results.distpars_all = distpars_all;
results.distribution_parameters = distribution_parameters_cell;
results.distY_time_all = distY_time_all;
results.maxiters_values = opts.maxiters_values;
results.gridparamV_all = gridparamV_all;
results.alpha_set = opts.alpha_set;
results.switch_eps = opts.switch_eps;
results.solver_backend = opts.solver_backend;
results.solver_statuses = solver_statuses_cell;   % {} entries for cvx-lane iters
results.learn_only = opts.learn_only;
results.eps_override = opts.eps_override;   % [] unless fixed-radius mode (D1/T1)
results.learning_style = opts.learning_style;
results.NGridV = opts.NGridV;
results.NGridM = opts.NGridM;
results.timing = timing_all;
results.cfg = cfg;

end
