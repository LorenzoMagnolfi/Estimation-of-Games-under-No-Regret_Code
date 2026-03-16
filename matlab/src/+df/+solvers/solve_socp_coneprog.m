function [optval, status] = solve_socp_coneprog(cstr, c)
% SOLVE_SOCP_CONEPROG  Solve a single SOCP instance via coneprog (R2020b+).
%
%   [optval, status] = df.solvers.solve_socp_coneprog(cstr, c)
%
%   Maps the CVX problem:
%     maximize  -c'x
%     s.t.  B_EQ * x == beq
%           B_INEQ * x >= b          (note: >= not <=)
%           lb <= x <= ub
%           norm(Mat_NLC * x) <= 1
%
%   to coneprog:
%     minimize  c'x                  (negated objective: max -c'x = min c'x)
%     s.t.  Aeq * x = beq
%           Aineq * x <= bineq       (flip B_INEQ >= b  to  -B_INEQ <= -b)
%           lb <= x <= ub
%           ||Mat_NLC * x - 0|| <= 0'x - (-1)   i.e. ||Mat_NLC * x|| <= 1
%
%   Inputs
%     cstr   constraints struct from build_constraints[_marginal]
%     c      objective vector
%
%   Outputs
%     optval   optimal value (maximize -c'x), or 100 if failed
%     status   'Solved' or error string

persistent cp_options
if isempty(cp_options)
    cp_options = optimoptions('coneprog', 'Display', 'none');
end

% Objective: minimize c'x  (since we want maximize -c'x)
f = c;

% Equality constraints (unchanged)
Aeq = cstr.B_EQ;
beq_val = cstr.beq;

% Inequality constraints: B_INEQ * x >= b  →  -B_INEQ * x <= -b
Aineq = -cstr.B_INEQ;
bineq = -cstr.b;

% Second-order cone constraint: ||A*x - b|| <= d'*x - gamma  (coneprog convention)
%   We need: ||Mat_NLC * x|| <= 1
%   So: A = Mat_NLC, b = 0, d = 0, gamma = -1
%   giving: ||Mat_NLC * x - 0|| <= 0'x - (-1) = 1
n = size(c, 1);
socConstraint = secondordercone(cstr.Mat_NLC, zeros(size(cstr.Mat_NLC, 1), 1), ...
    zeros(n, 1), -1);

% Solve
[x, fval, exitflag, output] = coneprog(f, socConstraint, ...
    Aineq, bineq, Aeq, beq_val, cstr.lb, cstr.ub, cp_options);

if exitflag == 1
    % coneprog minimizes c'x; the CVX "optval" is maximize(-c'x) = -fval
    optval = -fval;
    status = 'Solved';
else
    optval = 100;
    status = sprintf('coneprog exitflag=%d', exitflag);
end

end
