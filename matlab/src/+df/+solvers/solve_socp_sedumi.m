function [optval, status] = solve_socp_sedumi(sd, c_obj, A_override)
% SOLVE_SOCP_SEDUMI  Solve one SOCP instance by calling SeDuMi directly.
%
%   [optval, status] = df.solvers.solve_socp_sedumi(sd, c_obj)
%   [optval, status] = df.solvers.solve_socp_sedumi(sd, c_obj, A_override)
%
%   sd is the precomputed problem data from socp_to_sedumi; c_obj is the
%   per-candidate objective (same vector fed to solve_socp_cvx). No CVX
%   overhead: the canonical form is reused, only the objective changes.
%   A_override replaces sd.A for the CLT lane, where the ellipsoid matrix
%   varies by grid point (assembled in solve_grid_sedumi).
%
%   Outputs
%     optval   optimal value of  max -c_obj'y (- penalty terms),
%              or 100 if infeasible/failed (sentinel, as the CVX lane)
%     status   'Solved' | 'Infeasible' | 'Unbounded' | 'numerr=k'
%
%   Status semantics mirror solve_socp_cvx exactly: the CVX lane counts a
%   point only when cvx_status is 'Solved', which corresponds to SeDuMi
%   numerr == 0; numerr == 1 ('Inaccurate/Solved' under CVX) returns the
%   100 sentinel here too, so the two lanes classify identically.

if nargin < 3 || isempty(A_override)
    A = sd.A;
else
    A = A_override;
end

% Objective: minimize c_s'z with the y block set per candidate; penalty
% coefficients (L1 r on u, CLT rho on w) ride along in c_base.
c_s = sd.c_base;
c_s(sd.o_y + (1:sd.n_orig)) = c_obj;

[x, ~, info] = sedumi(A, sd.b, c_s, sd.K, sd.pars);

if info.pinf == 0 && info.dinf == 0 && info.numerr == 0
    optval = -c_s' * x;   % max -c'y - penalties = -(min c_s'z)
    status = 'Solved';
else
    optval = 100;
    if info.pinf
        status = 'Infeasible';
    elseif info.dinf
        status = 'Unbounded';
    else
        status = sprintf('numerr=%d', info.numerr);
    end
end

end
