function [g, timing] = solve_grid_cvx(cstr, c_all, opts)
% SOLVE_GRID_CVX  Batch CVX+SeDuMi solver for parameter grids.
%
%   [g, timing] = df.solvers.solve_grid_cvx(cstr, c_all, opts)
%
%   Solves NGrid independent SOCPs using CVX+SeDuMi. All problems share the
%   same constraint matrices (cstr); only the objective c differs.
%
%   Note: CVX uses global state, so parfor is NOT supported. Use serial only.
%   For parfor, use solve_grid_coneprog (but verify numerical accuracy first).
%
%   Inputs
%     cstr   constraint struct from build_constraints[_marginal]
%     c_all  (n x NGrid) matrix of objective vectors
%     opts   struct with optional fields:
%       .solver    'sedumi' (default) | 'sdpt3' | ''
%       .precision 'default' (default) | 'low' | 'medium' | 'high'
%       .verbose   logical (default: true)
%
%   Outputs
%     g       (NGrid x 1) optimal values (100 if infeasible)
%     timing  struct with .total, .per_solve, .n_solved, .n_feasible

if nargin < 3, opts = struct(); end
solver_name = 'sedumi';
if isfield(opts, 'solver'), solver_name = opts.solver; end
precision = 'default';
if isfield(opts, 'precision'), precision = opts.precision; end
verbose = ~isfield(opts, 'verbose') || opts.verbose;

NGrid = size(c_all, 2);

g = zeros(NGrid, 1);
t_start = tic;

for nd = 1:NGrid
    [g(nd), ~] = df.solvers.solve_socp_cvx(cstr, c_all(:,nd), solver_name, precision);
    if verbose && mod(nd, 500) == 0
        elapsed = toc(t_start);
        fprintf('  %d/%d (%.1fs, ETA %.0fs)\n', nd, NGrid, ...
            elapsed, elapsed/nd*(NGrid-nd));
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
