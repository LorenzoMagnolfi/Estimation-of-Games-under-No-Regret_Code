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
%     status   'Solved' | 'Solved(numerr=1,verified)' | 'Infeasible' |
%              'Unbounded' | 'numerr=k' | 'numerr=1,unverified'
%
%   Acceptance rule. These feasibility SOCPs often have optima within ~1e-4
%   of zero, exactly where an interior-point method exhausts its accuracy
%   target and exits with numerr=1 (reduced accuracy). CVX accepts such
%   solves as 'Solved' after its own residual checks; a bare numerr==0 gate
%   therefore rejects ~80% of points the CVX lane solves (smoke 3778532).
%   We mirror CVX with an explicit certificate: a numerr=1 point is accepted
%   only if its normalized primal residual and worst cone violation both
%   clear 1e-6. numerr=2 (genuine failure) is never accepted.

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

clean = (info.pinf == 0 && info.dinf == 0);
accept = clean && info.numerr == 0;
status = 'Solved';
if clean && info.numerr == 1
    rp = full(norm(A * x - sd.b)) / (1 + full(norm(sd.b)));
    cv = cone_viol(x, sd.K);
    if rp < 1e-6 && cv < 1e-6
        accept = true;
        status = 'Solved(numerr=1,verified)';
    else
        status = 'numerr=1,unverified';
    end
end

if accept
    optval = -c_s' * x;   % max -c'y - penalties = -(min c_s'z)
else
    optval = 100;
    if info.pinf
        status = 'Infeasible';
    elseif info.dinf
        status = 'Unbounded';
    elseif info.numerr ~= 1
        status = sprintf('numerr=%d', info.numerr);
    end
end

end

function v = cone_viol(x, K)
% Worst violation of the cone constraints by the returned point:
% negativity in the nonneg block, ||tail|| - head in each SOC block.
v = 0;
off = 0;
if isfield(K, 'f') && K.f > 0
    off = off + K.f;
end
if isfield(K, 'l') && K.l > 0
    v = max(v, max(0, -min(x(off + (1:K.l)))));
    off = off + K.l;
end
if isfield(K, 'q') && ~isempty(K.q)
    for qi = 1:numel(K.q)
        nq = K.q(qi);
        blk = x(off + (1:nq));
        v = max(v, max(0, norm(blk(2:end)) - blk(1)));
        off = off + nq;
    end
end
end
