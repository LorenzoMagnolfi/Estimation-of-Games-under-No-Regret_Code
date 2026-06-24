function [optval, status] = solve_socp_cvx(cstr, c, solver_name, precision, cons_l1)
% SOLVE_SOCP_CVX  Solve a single SOCP instance via CVX.
%
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c)
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c, solver_name)
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c, solver_name, precision)
%   [optval, status] = df.solvers.solve_socp_cvx(cstr, c, solver_name, precision, cons_l1)
%
%   Inputs
%     cstr         constraints struct from build_constraints[_marginal]
%     c            objective vector  (maximize  -c'x)
%     solver_name  optional: 'sedumi', 'sdpt3', or '' (default SDPT3)
%     precision    optional: 'default', 'low', 'medium', 'high' (default 'default')
%     cons_l1      optional struct for the Bretagnolle-Huber-Carol L1 consistency
%                  relaxation: .r (L1 radius) and .idx (indices of the
%                  consistency-equality dual block in x). Relaxing the
%                  consistency equality to the L1 ball ||m_nu - p||_1 <= r adds
%                  the ball's support function, r * ||y_con||_inf, to the dual
%                  objective. With r <= 0 (or empty) the program is the exact-
%                  consistency one, so cons_l1.r = 0 reproduces exact consistency.
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
if nargin < 5
    cons_l1 = [];
end
use_l1 = ~isempty(cons_l1) && cons_l1.r > 0;

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

    if use_l1
        % L1 (Bretagnolle-Huber-Carol) consistency relaxation: subtract the
        % support function of the L1 ball, r * inf-norm of the consistency-
        % equality dual block. Weakly lowers the optimum (enlarges the set).
        maximize( -c' * x - cons_l1.r * norm( x(cons_l1.idx), Inf ) )
    else
        maximize( -c' * x )
    end

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
