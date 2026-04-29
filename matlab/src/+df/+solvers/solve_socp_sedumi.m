function [optval, status] = solve_socp_sedumi(sd, c_obj)
% SOLVE_SOCP_SEDUMI  Solve SOCP by calling SeDuMi directly (no CVX overhead).
%
%   [optval, status] = df.solvers.solve_socp_sedumi(sd, c_obj)
%
%   Inputs
%     sd      precomputed SeDuMi data from socp_to_sedumi
%     c_obj   objective vector (same as CVX version: maximize -c_obj'x)
%
%   Outputs
%     optval   optimal value (maximize -c_obj'x), or 100 if failed
%     status   'Solved' or error string

% Build full objective vector: minimize c_obj'y (y in first n positions)
c_s = zeros(sd.ntot, 1);
c_s(sd.o_y + (1:sd.n_orig)) = c_obj;

% Call SeDuMi directly
[x, y, info] = sedumi(sd.A, sd.b, c_s, sd.K, sd.pars);

% Check solution status
% info.pinf = 1 means primal infeasible, info.dinf = 1 means dual infeasible
% info.numerr: 0 = no error, 1 = warning, 2 = failure
if info.pinf == 0 && info.dinf == 0 && info.numerr < 2
    % Primal optimal value = c_s' * x = c_obj' * x(1:n)
    % But we want maximize -c_obj'x = -(c_obj'*x) = -c_s'*x
    optval = -c_s' * x;
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
