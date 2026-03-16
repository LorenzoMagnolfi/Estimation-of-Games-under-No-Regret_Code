function results = run_stage_ii_nonparam(cfg, opts)
% DF.STAGES.RUN_STAGE_II_NONPARAM  Stage II nonparametric: identification over probability vectors.
%
%   results = df.stages.run_stage_ii_nonparam(cfg, opts)
%
%   Nonparametric variant of Stage II. Instead of parameterizing the cost
%   distribution by (mu, sigma) on a regular grid, this version searches
%   over direct probability mass vectors on the type support. The candidate
%   grid includes local perturbations, global simplex draws, and peaked
%   distributions for low-sigma coverage.
%
%   Uses the fast backend by default: builds SOCP constraints once,
%   precomputes all objective vectors, and batch-solves via solve_grid_cvx
%   (CVX+SeDuMi). Falls back to solve_bcce for legacy compatibility.
%
%   This is the refactored version of Bo Feng's II_MAIN_nonparam_simul.m,
%   using the df.* infrastructure (cfg struct, solve_bcce, shared plotting).
%
%   Inputs:
%     cfg  — config struct from df.setup.game_simulation
%     opts — struct with fields:
%       .maxiters_values — vector of iteration counts (default: [500000 1000000 2000000 4000000])
%       .alpha_set       — confidence levels (default: 0.05)
%       .switch_eps      — epsilon formula selector (default: 1)
%       .backend         — 'fast' (default) | 'legacy'
%                          'fast': precomputed objectives + solve_grid_cvx
%                          'legacy': solve_bcce loop (slower, for validation)
%       .K_local         — local perturbations (default: 1000)
%       .K_global        — global simplex draws (default: 9000)
%       .K_spiky         — peaked distributions (default: 200)
%       .local_width     — perturbation scale (default: 20)
%       .spike_mult      — spike multiplier (default: 3)
%       .n_adjacent      — neighbor count for spiky (default: 4)
%       .solver          — CVX solver (default: 'sedumi')
%       .precision       — CVX precision (default: 'default')
%
%   Outputs:
%     results — struct with fields:
%       .VV_all              — (n_iters x NGrid) solver outputs
%       .distpars_all        — (n_iters x NGrid x 2) [mean, variance] per candidate
%       .distribution_parameters — {1 x NGrid} cell of probability vectors
%       .distY_time_all      — {n_iters x 1} cell of action distributions
%       .maxiters_values     — iteration counts used
%       .timing              — struct with per-iteration timing
%       .cfg                 — config struct

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'maxiters_values'), opts.maxiters_values = [500000, 1000000, 2000000, 4000000]; end
if ~isfield(opts, 'alpha_set'),       opts.alpha_set = 0.05; end
if ~isfield(opts, 'switch_eps'),      opts.switch_eps = 1; end
if ~isfield(opts, 'backend'),         opts.backend = 'fast'; end
if ~isfield(opts, 'solver'),          opts.solver = 'sedumi'; end
if ~isfield(opts, 'precision'),       opts.precision = 'default'; end

use_fast = strcmp(opts.backend, 'fast');

% Grid parameters (passed through to build_nonparam_grid)
grid_opts = struct();
grid_fields = {'K_local', 'K_global', 'K_spiky', 'local_width', 'spike_mult', 'n_adjacent'};
for i = 1:numel(grid_fields)
    if isfield(opts, grid_fields{i})
        grid_opts.(grid_fields{i}) = opts.(grid_fields{i});
    end
end

cfg.learning_style = 'rm';

s = cfg.s;
n_iters = numel(opts.maxiters_values);
type_space = cfg.type_space;
action_space = cfg.action_space;
Pi = cfg.Pi;

%% Build nonparametric grid (shared across all iterations)
fprintf('[Stage II nonparam] Building candidate distribution grid...\n');
[distpars, distribution_parameters] = df.report.build_nonparam_grid(...
    cfg.marg_distrib, type_space, grid_opts);
NGrid = size(distpars, 1);
fprintf('  %d candidates: true + local=%d, global=%d, spiky=%d\n', NGrid, ...
    getfield_default(grid_opts, 'K_local', 1000), ...
    getfield_default(grid_opts, 'K_global', 9000), ...
    getfield_default(grid_opts, 'K_spiky', 200));

%% Build constraints ONCE (shared across all iterations and grid points)
if use_fast
    cstr = df.solvers.build_constraints(type_space, action_space, Pi);
    dim_u = cstr.NA - 1;
    a_dim = cstr.a;
    NAg = cstr.NAg;
    s2 = cstr.s2;

    % Precompute Psi (joint prior) and marg_distrib for ALL grid points
    % (grid-invariant across iterations — only depends on candidate distributions)
    fprintf('  Precomputing priors for %d candidates...', NGrid);
    t_psi = tic;
    Psi = zeros(s2, NGrid);
    marg_distrib_grid = zeros(s, NGrid);
    for nd = 1:NGrid
        prob_vec = distribution_parameters{nd};
        prob_vec = prob_vec(:) ./ sum(prob_vec(:));
        marg_distrib_grid(:, nd) = prob_vec;
        % Joint prior from product of marginals (independence)
        joint = zeros(s2, 1);
        for ii = 1:s
            for jj = 1:s
                joint((ii-1)*s + jj) = prob_vec(ii) * prob_vec(jj);
            end
        end
        Psi(:, nd) = joint;
    end
    Psi = Psi ./ sum(Psi, 1);
    fprintf(' %.2fs\n', toc(t_psi));

    fprintf('[Stage II nonparam] Fast backend: CVX+SeDuMi (precomputed objectives)\n');
end

%% Pre-allocate
num_alpha = numel(opts.alpha_set);
VV_all = zeros(n_iters, NGrid);
distpars_all = zeros(n_iters, NGrid, 2);
distY_time_all = cell(n_iters, 1);
timing = struct('learn', zeros(n_iters, 1), 'objectives', zeros(n_iters, 1), ...
    'solve', zeros(n_iters, 1));

%% Main loop
for maxiter_index = 1:n_iters
    maxiters = opts.maxiters_values(maxiter_index);
    t_iter = tic;
    fprintf('[Stage II nonparam] iter %d/%d: maxiters=%dk, learning... ', ...
        maxiter_index, n_iters, maxiters/1000);

    % Learning
    N = 1; M = maxiters; M_obs = maxiters;
    numdst_t = 1; numdst_t_obs = numdst_t;
    [distY_time, ~] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
    timing.learn(maxiter_index) = toc(t_iter);
    fprintf('%.1fs\n', timing.learn(maxiter_index));

    action_distribution = distY_time;
    distY_time_all{maxiter_index} = distY_time;

    if use_fast
        %% Fast path: precompute all objectives, batch solve
        fprintf('  Building objectives (%d grid points)...', NGrid);
        t_obj = tic;

        % Epsilon for this iteration count
        confid = opts.alpha_set(1);
        eps_vec = df.solvers.compute_epsilon(cfg, maxiters, confid, opts.switch_eps);

        % Build all objective vectors
        n_vars = size(cstr.B_EQ, 2);
        c_all = zeros(n_vars, NGrid);
        for nd = 1:NGrid
            bmarg = Psi(:, nd);
            if opts.switch_eps == 1 || opts.switch_eps == 3 || opts.switch_eps == 4
                eps_fin = repmat(sqrt(marg_distrib_grid(:,nd))', 1, NAg*a_dim) .* ...
                          repmat(eps_vec, 1, NAg*a_dim);
            else
                eps_fin = repmat(eps_vec, 1, NAg*a_dim);
            end
            c_all(:, nd) = [zeros(1, dim_u), action_distribution(:,1)', ...
                            bmarg', 1, eps_fin]';
        end
        timing.objectives(maxiter_index) = toc(t_obj);
        fprintf(' %.1fs\n', timing.objectives(maxiter_index));

        % Batch solve via CVX+SeDuMi
        fprintf('  Full grid solve (%d points):\n', NGrid);
        t_solve = tic;
        cvx_opts = struct('verbose', true, 'solver', opts.solver);
        [VV, ~] = df.solvers.solve_grid_cvx(cstr, c_all, cvx_opts);
        timing.solve(maxiter_index) = toc(t_solve);
        fprintf('  Solver: %.1fs, identified: %d/%d (%.1f%%)\n', ...
            timing.solve(maxiter_index), sum(VV <= 1e-12), NGrid, ...
            100*sum(VV <= 1e-12)/NGrid);

        VV_all(maxiter_index, :) = VV(:)';

    else
        %% Legacy path: solve_bcce loop (slower, for validation)
        t_solve = tic;
        numdist = size(action_distribution, 2);

        maxvals = zeros(numdist, num_alpha, NGrid);
        for ii = 1:numdist
            T = ii * (maxiters / numdist);
            distrib = action_distribution(:, ii);
            for jj = 1:num_alpha
                confid = opts.alpha_set(jj);
                eps_info = struct('mode', 'switch', 'eps', ...
                    epsilon_switch(T, confid, opts.switch_eps, cfg));
                solve_opts = struct('switch_eps', opts.switch_eps, ...
                    'solver', opts.solver, 'precision', opts.precision);
                g = df.solvers.solve_bcce(type_space, action_space, distrib, ...
                    cfg.alpha, distribution_parameters, Pi, eps_info, solve_opts);
                maxvals(ii, jj, :) = g(:);
            end
        end

        VV = squeeze(maxvals);
        timing.solve(maxiter_index) = toc(t_solve);
        fprintf('  Solve: %.1fs, identified: %d/%d (%.1f%%)\n', ...
            timing.solve(maxiter_index), sum(VV <= 1e-12), NGrid, ...
            100*sum(VV <= 1e-12)/NGrid);

        VV_all(maxiter_index, :) = VV;
    end

    distpars_all(maxiter_index, :, :) = distpars;
end

%% Pack results
results.VV_all = VV_all;
results.distpars_all = distpars_all;
results.distribution_parameters = distribution_parameters;
results.distY_time_all = distY_time_all;
results.maxiters_values = opts.maxiters_values;
results.alpha_set = opts.alpha_set;
results.timing = timing;
results.cfg = cfg;
results.grid_opts = grid_opts;

end


function val = getfield_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
