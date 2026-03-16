function [optval, status] = solve_socp_cvx(cstr, c, solver_name, precision)
% SOLVE_SOCP_CVX  Solve a single SOCP instance via CVX.
%
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c)
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c, solver_name)
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c, solver_name, precision)
%
%   Inputs
%     cstr         constraints struct from build_constraints[_marginal]
%     c            objective vector  (maximize  -c'x)
%     solver_name  optional: 'sedumi', 'sdpt3', or '' (default SDPT3)
%     precision    optional: 'default', 'low', 'medium', 'high' (default 'default')
%
%   Outputs
%     optval   optimal value (-c'x*), or 100 if infeasible/failed
%     status   CVX status string

if nargin < 3 || isempty(solver_name)
    solver_name = '';
end
if nargin < 4 || isempty(precision)
    precision = 'default';
end

n = size(c, 1);

cvx_clear
if strcmp(solver_name, 'sedumi')
    cvx_solver sedumi
end
cvx_begin quiet
    if strcmp(precision, 'low')
        cvx_precision low
    elseif strcmp(precision, 'medium')
        cvx_precision medium
    elseif strcmp(precision, 'high')
        cvx_precision high
    end

    variable x(n);

    maximize( -c' * x )

    subject to
    cstr.B_EQ   * x == cstr.beq;
    cstr.B_INEQ * x >= cstr.b;
    cstr.lb <= x <= cstr.ub;
    norm(cstr.Mat_NLC * x) <= 1;
cvx_end

status = cvx_status;
if strcmp(cvx_status, 'Solved')
    optval = cvx_optval;
else
    optval = 100;
end

end
