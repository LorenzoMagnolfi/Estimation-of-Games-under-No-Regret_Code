function [g, n_solved, boundary_idx] = solve_grid_adaptive(cstr, c_all, nV, nM, opts)
% SOLVE_GRID_ADAPTIVE  Adaptive grid refinement for SOCP identification set.
%
%   [g, n_solved, boundary_idx] = df.solvers.solve_grid_adaptive(cstr, c_all, nV, nM)
%   [g, n_solved, boundary_idx] = df.solvers.solve_grid_adaptive(cstr, c_all, nV, nM, opts)
%
%   Instead of solving all nV*nM grid points, uses a coarse-to-fine strategy:
%     1. Solve on a coarse subgrid (every k-th point)
%     2. Identify boundary cells (neighbors disagree on feasibility)
%     3. Solve only the fine-grid points near the boundary
%
%   Inputs
%     cstr     constraint struct from build_constraints
%     c_all    (n × NGrid) matrix of objective vectors
%     nV       number of rows in the grid (gridparamV dimension)
%     nM       number of columns in the grid (gridparamM dimension)
%     opts     struct with optional fields:
%       .backend   'cvx' (default) | 'coneprog'
%       .solver    'sedumi' (default) — CVX solver name
%       .precision 'default' — CVX precision
%
%   Outputs
%     g            (NGrid × 1) objective values (NaN = not solved, inferred)
%     n_solved     number of SOCP solves performed
%     boundary_idx indices of boundary cells

if nargin < 5, opts = struct(); end
use_cvx = ~isfield(opts, 'backend') || strcmp(opts.backend, 'cvx');

NGrid = nV * nM;
assert(size(c_all, 2) == NGrid, 'c_all columns must match nV*nM');

% Feasibility threshold: g <= tol means "in identified set"
tol = 1e-6;

%% Phase 1: Coarse grid (every k-th point in each dimension)
k = max(2, round(min(nV, nM) / 8));  % ~8 points per dimension in coarse grid
coarse_V = unique([1:k:nV, nV]);
coarse_M = unique([1:k:nM, nM]);

coarse_idx = [];
for jm = coarse_M
    for jv = coarse_V
        coarse_idx(end+1) = (jm-1)*nV + jv;
    end
end
coarse_idx = unique(coarse_idx);

g = nan(NGrid, 1);
if use_cvx
    cvx_opts = struct('verbose', false);
    if isfield(opts, 'solver'), cvx_opts.solver = opts.solver; end
    if isfield(opts, 'precision'), cvx_opts.precision = opts.precision; end
    [g_coarse_vals, ~] = df.solvers.solve_grid_cvx(cstr, c_all(:, coarse_idx), cvx_opts);
else
    solve_opts = struct('verbose', false, 'use_parfor', false);
    [g_coarse_vals, ~] = df.solvers.solve_grid_coneprog(cstr, c_all(:, coarse_idx), solve_opts);
end
g(coarse_idx) = g_coarse_vals;
n_solved = numel(coarse_idx);
fprintf('  Phase 1: coarse %d pts, %d feasible\n', n_solved, sum(g_coarse_vals <= tol));

%% Phase 2: Identify boundary region from coarse grid
% A coarse cell straddles the boundary if any neighbor has different feasibility
coarse_feas = nan(nV, nM);
for nd = coarse_idx
    [iv, im] = ind2sub([nV, nM], nd);
    coarse_feas(iv, im) = g(nd) <= tol;
end

% Interpolate coarse feasibility to full grid
% For each fine-grid point, find nearest coarse neighbors
boundary_mask = false(nV, nM);
for im = 1:nM
    for iv = 1:nV
        % Find nearest coarse grid indices
        cv_lo = max(coarse_V(coarse_V <= iv));
        cv_hi = min(coarse_V(coarse_V >= iv));
        cm_lo = max(coarse_M(coarse_M <= im));
        cm_hi = min(coarse_M(coarse_M >= im));

        if isempty(cv_lo), cv_lo = cv_hi; end
        if isempty(cv_hi), cv_hi = cv_lo; end
        if isempty(cm_lo), cm_lo = cm_hi; end
        if isempty(cm_hi), cm_hi = cm_lo; end

        % Gather feasibility of corner neighbors
        corners = [];
        for cv = unique([cv_lo, cv_hi])
            for cm = unique([cm_lo, cm_hi])
                val = coarse_feas(cv, cm);
                if ~isnan(val)
                    corners(end+1) = val;
                end
            end
        end

        % Boundary if neighbors disagree
        if ~isempty(corners) && any(corners ~= corners(1))
            boundary_mask(iv, im) = true;
        end
    end
end

% Dilate boundary by 1 cell in each direction for safety margin
boundary_dilated = boundary_mask;
for im = 1:nM
    for iv = 1:nV
        if boundary_mask(iv, im)
            for di = -1:1
                for dj = -1:1
                    ni = iv + di; nj = im + dj;
                    if ni >= 1 && ni <= nV && nj >= 1 && nj <= nM
                        boundary_dilated(ni, nj) = true;
                    end
                end
            end
        end
    end
end

%% Phase 3: Solve fine grid near boundary
boundary_idx = find(boundary_dilated(:));
unsolved = boundary_idx(isnan(g(boundary_idx)));

if ~isempty(unsolved)
    if use_cvx
        [g_refine_vals, ~] = df.solvers.solve_grid_cvx(cstr, c_all(:, unsolved), cvx_opts);
    else
        [g_refine_vals, ~] = df.solvers.solve_grid_coneprog(cstr, c_all(:, unsolved), solve_opts);
    end
    g(unsolved) = g_refine_vals;
    fprintf('  Phase 3: refine %d pts, %d feasible\n', numel(unsolved), sum(g_refine_vals <= tol));
end
n_solved = n_solved + numel(unsolved);

%% Phase 4: Fill interior/exterior by nearest coarse neighbor
% Points far from boundary: inherit feasibility from coarse grid
for nd = 1:NGrid
    if isnan(g(nd))
        [iv, im] = ind2sub([nV, nM], nd);
        cv = coarse_V(findnearest(coarse_V, iv));
        cm = coarse_M(findnearest(coarse_M, im));
        g(nd) = g((cm-1)*nV + cv);
    end
end

end

function idx = findnearest(arr, val)
    [~, idx] = min(abs(arr - val));
end
