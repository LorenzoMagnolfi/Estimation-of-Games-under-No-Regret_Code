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

use_fast = strcmp(opts.backend, 'fast');

cfg.learning_style = 'rm';
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
timing_all = struct();

% Build constraints ONCE (shared across all iterations and grid points)
if use_fast
    cstr = df.solvers.build_constraints(type_space, action_space, Pi);
    dim_u = cstr.NA - 1;
    a_dim = cstr.a;
    NAg = cstr.NAg;
    s2 = cstr.s2;
    T_sorted = cstr.T_sorted;
    fprintf('[Stage II] Fast backend: CVX+SeDuMi (precomputed objectives)');
    if opts.adaptive, fprintf(' + adaptive'); end
    fprintf('\n');
end

for maxiter_index = 1:n_iters
    maxiters = opts.maxiters_values(maxiter_index);
    t_iter = tic;

    N = 1;
    M = maxiters;
    M_obs = maxiters;

    %% Learning
    fprintf('[Stage II] iter %d/%d: maxiters=%dk, learning...', ...
        maxiter_index, n_iters, maxiters/1000);
    t_learn = tic;
    [distY_time, ~] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
    action_distribution = distY_time;
    distY_time_all{maxiter_index} = distY_time;
    t_learn_val = toc(t_learn);
    fprintf(' %.1fs\n', t_learn_val);

    %% Parameter grid (iteration-dependent variance range)
    if maxiter_index == 1
        gridparamV = [1; linspace(0.15, sigma2(1,1)*10, opts.NGridV)'];
    elseif maxiter_index == 2
        gridparamV = [1; linspace(0.15, sigma2(1,1)*6, opts.NGridV)'];
    else
        gridparamV = [1; linspace(0.15, sigma2(1,1)*3.5, opts.NGridV)'];
    end
    gridparamM = [1; linspace(0.55, mu(1,1)*0.5, opts.NGridM)'];
    gridparamV_all{maxiter_index} = gridparamV;

    [distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV);
    distpars_all(maxiter_index, :, :) = distpars;
    distribution_parameters_cell{maxiter_index} = distribution_parameters;

    %% Solver
    if use_fast
        %% Fast path: precompute objectives, use coneprog + adaptive + parfor
        fprintf('  Building objectives (%d grid points)...', NGrid);
        t_obj = tic;

        % Epsilon for this iteration count
        confid = opts.alpha_set(1);
        eps_vec = df.solvers.compute_epsilon(cfg, maxiters, confid, opts.switch_eps);

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

        % Build all objective vectors
        c_all = zeros(size(cstr.B_EQ, 2), NGrid);
        for nd = 1:NGrid
            bmarg = Psi(:,nd);
            eps_fin = repmat(sqrt(marg_distrib(:,nd))', 1, NAg*a_dim) .* ...
                      repmat(eps_vec, 1, NAg*a_dim);
            c_all(:,nd) = [zeros(1, dim_u), action_distribution(:,1)', ...
                           bmarg', 1, eps_fin]';
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
            fprintf('  Full grid solve (%d points):\n', NGrid);
            cvx_opts = struct('verbose', true, 'solver', 'sedumi');
            [VV, ~] = df.solvers.solve_grid_cvx(cstr, c_all, cvx_opts);
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
results.NGridV = opts.NGridV;
results.NGridM = opts.NGridM;
results.timing = timing_all;
results.cfg = cfg;

end
