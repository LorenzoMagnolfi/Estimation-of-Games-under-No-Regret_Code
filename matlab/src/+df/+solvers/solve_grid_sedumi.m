function [g, timing, statuses] = solve_grid_sedumi(cstr, c_all, opts)
% SOLVE_GRID_SEDUMI  Batch direct-SeDuMi solver for parameter grids.
%
%   [g, timing] = df.solvers.solve_grid_sedumi(cstr, c_all, opts)
%
%   Drop-in alternative to solve_grid_cvx: same inputs, same outputs, same
%   feasibility semantics (100 sentinel, tol conventions unchanged). The
%   canonical SeDuMi form is built ONCE (socp_to_sedumi); each grid point
%   then swaps in its objective vector and calls sedumi directly. This
%   removes the per-point CVX modeling overhead, which dominates wall time
%   at these cone sizes, and is parfor-safe (no CVX global state).
%
%   opts fields (all optional):
%     .cons_l1_r / .cons_l1_idx     L1 (BHC) consistency term, as solve_grid_cvx
%     .cons_clt_rho / .cons_clt_idx CLT ellipsoid term, as solve_grid_cvx
%     .use_parfor  logical (default false): use an OPEN parpool if one exists.
%                  Never opens a pool itself (pool policy belongs to the stage).
%     .verbose     logical (default true)
%
%   Outputs
%     g       (NGrid x 1) optimal values (100 if infeasible)
%     timing  struct: .total, .per_solve, .n_solved, .n_feasible, .pool

if nargin < 3, opts = struct(); end
verbose = ~isfield(opts, 'verbose') || opts.verbose;

bopts = struct();
if isfield(opts, 'cons_l1_r') && ~isempty(opts.cons_l1_r) && opts.cons_l1_r > 0
    bopts.cons_l1 = struct('r', opts.cons_l1_r, 'idx', opts.cons_l1_idx);
end
if isfield(opts, 'cons_clt_rho') && ~isempty(opts.cons_clt_rho) && opts.cons_clt_rho > 0
    bopts.cons_clt = struct('rho', opts.cons_clt_rho, 'idx', opts.cons_clt_idx);
end

% CVX's path manager (cvx_clear / cvx_end) prunes directories under the CVX
% root from the path — including the genpath'd sedumi/ folder — so after any
% CVX-lane call the raw sedumi function disappears. Re-secure it here (and on
% any open pool's workers) before solving.
if exist('sedumi', 'file') ~= 2
    cvx_dir = getenv('CVX_DIR');
    sp = fullfile(cvx_dir, 'sedumi');
    if ~isempty(cvx_dir) && exist(sp, 'dir') == 7
        addpath(sp);
        if exist('gcp', 'file') == 2
            p0 = gcp('nocreate');
            if ~isempty(p0)
                wait(parfevalOnAll(p0, @addpath, 0, sp));
            end
        end
    end
end
if exist('sedumi', 'file') ~= 2
    error('solve_grid_sedumi:NoSedumi', ...
        'sedumi is not on the MATLAB path and CVX_DIR is not set to a CVX root with a sedumi/ folder.');
end

sd = df.solvers.socp_to_sedumi(cstr, bopts);
use_clt = ~isempty(sd.clt);

% parfor width: 0 (serial in-process) unless the caller asked for parfor AND
% a pool is already open. parfor(..., 0) runs as a plain for loop, so the
% same body serves both paths and serial/parallel results are identical.
pool_n = 0;
if isfield(opts, 'use_parfor') && opts.use_parfor && exist('gcp', 'file') == 2
    p = gcp('nocreate');
    if ~isempty(p), pool_n = p.NumWorkers; end
end

NGrid = size(c_all, 2);
g = zeros(NGrid, 1);
statuses = cell(NGrid, 1);
t_start = tic;

if ~use_clt
    parfor (nd = 1:NGrid, pool_n)
        [g(nd), statuses{nd}] = df.solvers.solve_socp_sedumi(sd, c_all(:, nd));
    end
else
    % CLT: the ellipsoid matrix Sh = sqrtm(diag(p)-p*p') depends on the grid
    % point through p = c(idx) (the candidate's consistency RHS), so the g-block
    % rows are assembled per point from the cached base triplets.
    clt  = sd.clt;
    trip = sd.trip;
    nci  = clt.n_idx;
    [JJg, IIg] = meshgrid(clt.cols, clt.row0 + (1:nci)');  % row-major pairing
    parfor (nd = 1:NGrid, pool_n)
        c_nd = c_all(:, nd);
        p = c_nd(clt.idx_c);
        Sig = diag(p) - p * p';
        Sig = (Sig + Sig') / 2;
        Sh = real(sqrtm(Sig));
        A_nd = sparse([trip.ii; IIg(:)], [trip.jj; JJg(:)], ...
            [trip.vv; -Sh(:)], trip.m, trip.n);
        [g(nd), statuses{nd}] = df.solvers.solve_socp_sedumi(sd, c_nd, A_nd);
    end
end

timing.total = toc(t_start);
timing.per_solve = timing.total / max(NGrid, 1);
timing.n_solved = NGrid;
timing.n_feasible = sum(g <= 1e-6);
timing.pool = pool_n;

if verbose
    fprintf('  Done (sedumi direct%s): %d points, %.1fs total (%.4fs/solve), %d feasible\n', ...
        tern(pool_n > 0, sprintf(', parfor x%d', pool_n), ''), ...
        NGrid, timing.total, timing.per_solve, timing.n_feasible);
end

end

function s = tern(cond, a, b)
if cond, s = a; else, s = b; end
end
