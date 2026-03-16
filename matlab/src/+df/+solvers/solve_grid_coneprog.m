function [g, timing] = solve_grid_coneprog(cstr, c_all, opts)
% SOLVE_GRID_CONEPROG  Vectorized coneprog solver for parameter grids.
%
%   DEPRECATED — coneprog is numerically unreliable for this problem class at
%   production scale (101x101 grids, 5-action games). At 10,201 grid points,
%   coneprog returns all-negative values with 92% disagreement vs CVX+SeDuMi.
%   Use df.solvers.solve_grid_cvx instead. Retained for benchmarking only.
%
%   [g, timing] = df.solvers.solve_grid_coneprog(cstr, c_all, opts)
%
%   Solves NGrid independent SOCPs using coneprog.  All problems share the
%   same constraint matrices (cstr); only the objective c differs.
%
%   Inputs
%     cstr   constraint struct from build_constraints[_marginal]
%     c_all  (n x NGrid) matrix of objective vectors
%     opts   struct with optional fields:
%       .use_parfor  logical (default: false)
%       .verbose     logical (default: true)
%
%   Outputs
%     g       (NGrid x 1) optimal values (100 if infeasible)
%     timing  struct with .total, .per_solve, .n_solved, .n_feasible

if nargin < 3, opts = struct(); end
use_par = isfield(opts, 'use_parfor') && opts.use_parfor;
verbose = ~isfield(opts, 'verbose') || opts.verbose;

NGrid = size(c_all, 2);
n = size(c_all, 1);

%% Precompute all constant data ONCE
Aeq = cstr.B_EQ;
beq_val = cstr.beq;
Aineq = -cstr.B_INEQ;
bineq = -cstr.b;
lb_val = cstr.lb;
ub_val = cstr.ub;

% SOC constraint object (immutable, safe to share across parfor)
socConstraint = secondordercone(cstr.Mat_NLC, ...
    zeros(size(cstr.Mat_NLC, 1), 1), zeros(n, 1), -1);

cp_options = optimoptions('coneprog', 'Display', 'none');

%% Solve
g = zeros(NGrid, 1);
t_start = tic;

if use_par
    % parfor path — no persistent state, no fprintf inside
    parfor nd = 1:NGrid
        f = c_all(:, nd);
        [~, fval, exitflag] = coneprog(f, socConstraint, ...
            Aineq, bineq, Aeq, beq_val, lb_val, ub_val, cp_options);
        if exitflag == 1
            g(nd) = -fval;
        else
            g(nd) = 100;
        end
    end
else
    % Serial path with progress reporting
    for nd = 1:NGrid
        f = c_all(:, nd);
        [~, fval, exitflag] = coneprog(f, socConstraint, ...
            Aineq, bineq, Aeq, beq_val, lb_val, ub_val, cp_options);
        if exitflag == 1
            g(nd) = -fval;
        else
            g(nd) = 100;
        end
        if verbose && mod(nd, 500) == 0
            elapsed = toc(t_start);
            fprintf('  %d/%d (%.1fs, ETA %.0fs)\n', nd, NGrid, ...
                elapsed, elapsed/nd*(NGrid-nd));
        end
    end
end

timing.total = toc(t_start);
timing.per_solve = timing.total / NGrid;
timing.n_solved = NGrid;
timing.n_feasible = sum(g <= 1e-6);

if verbose
    fprintf('  Done: %d points, %.1fs total (%.3fs/solve), %d feasible\n', ...
        NGrid, timing.total, timing.per_solve, timing.n_feasible);
end

end
