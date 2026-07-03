function sd = socp_to_sedumi(cstr, bopts)
% SOCP_TO_SEDUMI  Convert the SOCP constraint struct to SeDuMi standard form.
%
%   sd = df.solvers.socp_to_sedumi(cstr)
%   sd = df.solvers.socp_to_sedumi(cstr, bopts)
%
%   Precomputes SeDuMi problem data from the constraint struct produced by
%   build_constraints (or build_constraints_marginal). Call ONCE per grid,
%   then feed sd to solve_socp_sedumi with a fresh objective per candidate:
%   the constraints are shared across the whole parameter grid, only the
%   objective vector changes, so all CVX modeling overhead is paid once.
%
%   The SOCP (per grid point, same form as solve_socp_cvx):
%     maximize  -c'y  [ - r*||y(idx_l1)||_inf ]  [ - rho*||Sh*y(idx_clt)||_2 ]
%     s.t.  B_EQ * y   = beq
%           B_INEQ * y >= b
%           lb <= y <= ub
%           ||Mat_NLC * y|| <= 1
%
%   bopts (optional struct):
%     .cons_l1   struct('r', r, 'idx', idx)     L1 (BHC) consistency term.
%                Static: encoded once via an epigraph variable u >= |y_i|,
%                objective picks up + r*u (r is fixed across the grid).
%     .cons_clt  struct('rho', rho, 'idx', idx) CLT ellipsoid consistency term.
%                The ellipsoid matrix Sh = sqrtm(diag(p) - p*p') depends on the
%                candidate's p = c(idx), so it varies BY GRID POINT. This
%                builder lays down the static structure (SOC block [w; g],
%                identity on g, objective rho*w); the per-point -Sh entries are
%                appended by the caller (solve_grid_sedumi) from sd.trip.
%     L1 and CLT are mutually exclusive (as in run_stage_ii).
%
%   SeDuMi primal form used:  min c_s'z  s.t.  A z = b_s,  z in K, with
%     z = [ y (free) | s_ineq, s_lb, s_ub (, u, s1, s2) (nonneg) | t, v (, w, g) (SOC) ]
%   Rows: B_EQ*y=beq; B_INEQ*y-s_ineq=b; y(lb)-s_lb=lb; -y(ub)+s_ub=-(-ub);
%         -Mat_NLC*y+v=0; t=1; [u-y_i-s1_i=0; u+y_i-s2_i=0]; [g-Sh*y_idx=0].
%
%   Output sd fields:
%     A, b, K, pars          prebuilt problem data (A empty when CLT active)
%     trip                   struct(ii, jj, vv, m, n) base triplets (CLT reassembly)
%     c_base                 objective template: r at u / rho at w, zeros elsewhere
%     n_orig, o_y            original variable dimension and offset (0)
%     clt                    [] or struct(row0, cols, o_g, n_idx) for per-point -Sh

if nargin < 2 || isempty(bopts), bopts = struct(); end
cons_l1  = [];
cons_clt = [];
if isfield(bopts, 'cons_l1')  && ~isempty(bopts.cons_l1)  && bopts.cons_l1.r > 0
    cons_l1 = bopts.cons_l1;
end
if isfield(bopts, 'cons_clt') && ~isempty(bopts.cons_clt) && bopts.cons_clt.rho > 0
    cons_clt = bopts.cons_clt;
end
if ~isempty(cons_l1) && ~isempty(cons_clt)
    error('socp_to_sedumi:BothSlacks', 'cons_l1 and cons_clt are mutually exclusive.');
end

n       = size(cstr.B_EQ, 2);
m_eq    = size(cstr.B_EQ, 1);
m_ineq  = size(cstr.B_INEQ, 1);
dim_soc = size(cstr.Mat_NLC, 1);

% Active bounds only (the +/-10000 sentinels never bind; CVX carries them
% inertly, here we drop them so the cone stays small).
tol_bound = 9999;
idx_lb = find(cstr.lb > -tol_bound);
idx_ub = find(cstr.ub <  tol_bound);
n_lb = numel(idx_lb);
n_ub = numel(idx_ub);

n_l1  = 0; n_l1r = 0;
if ~isempty(cons_l1)
    il1   = cons_l1.idx(:);
    n_i   = numel(il1);
    n_l1  = 1 + 2*n_i;    % u, s1, s2
    n_l1r = 2*n_i;        % epigraph rows
end
n_clt = 0; n_cltr = 0;
if ~isempty(cons_clt)
    iclt  = cons_clt.idx(:);
    n_ci  = numel(iclt);
    n_clt = 1 + n_ci;     % w, g  (second SOC block)
    n_cltr = n_ci;        % g identity rows (the -Sh part is per point)
end

% Variable layout (cone order: free, nonneg, SOC blocks)
o_y     = 0;
o_sineq = n;
o_slb   = o_sineq + m_ineq;
o_sub   = o_slb + n_lb;
o_u     = o_sub + n_ub;                 % L1 block start (if any)
o_soc   = o_u + n_l1;                   % [t; v]
o_w     = o_soc + 1 + dim_soc;          % CLT SOC start (if any)
o_g     = o_w + 1;
ntot    = o_soc + 1 + dim_soc + n_clt;

K_out.f = n;
K_out.l = m_ineq + n_lb + n_ub + n_l1;
K_out.q = 1 + dim_soc;
if ~isempty(cons_clt), K_out.q = [K_out.q, 1 + n_cltr]; end

m_total = m_eq + m_ineq + n_lb + n_ub + dim_soc + 1 + n_l1r + n_cltr;

%% Assemble base triplets
ii = cell(0,1); jj = cell(0,1); vv = cell(0,1); bb = cell(0,1);
row = 0;

% (1) B_EQ * y = beq
[ri, ci, vi] = find(cstr.B_EQ);
ii{end+1} = ri + row; jj{end+1} = ci + o_y; vv{end+1} = vi;
bb{end+1} = cstr.beq(:);
row = row + m_eq;

% (2) B_INEQ * y - s_ineq = b
[ri, ci, vi] = find(cstr.B_INEQ);
ii{end+1} = ri + row;          jj{end+1} = ci + o_y;             vv{end+1} = vi;
ii{end+1} = (1:m_ineq)' + row; jj{end+1} = (1:m_ineq)' + o_sineq; vv{end+1} = -ones(m_ineq,1);
bb{end+1} = cstr.b(:);
row = row + m_ineq;

% (3) y(idx_lb) - s_lb = lb
r3 = (1:n_lb)' + row;
ii{end+1} = [r3; r3]; jj{end+1} = [idx_lb(:) + o_y; (1:n_lb)' + o_slb];
vv{end+1} = [ones(n_lb,1); -ones(n_lb,1)];
bb{end+1} = cstr.lb(idx_lb);
row = row + n_lb;

% (4) -y(idx_ub) + s_ub = -ub  (i.e. s_ub = ub - y >= 0)
r4 = (1:n_ub)' + row;
ii{end+1} = [r4; r4]; jj{end+1} = [idx_ub(:) + o_y; (1:n_ub)' + o_sub];
vv{end+1} = [-ones(n_ub,1); ones(n_ub,1)];
bb{end+1} = cstr.ub(idx_ub);
row = row + n_ub;

% (5) -Mat_NLC * y + v = 0
[ri, ci, vi] = find(cstr.Mat_NLC);
ii{end+1} = ri + row;           jj{end+1} = ci + o_y;                 vv{end+1} = -vi;
ii{end+1} = (1:dim_soc)' + row; jj{end+1} = o_soc + 1 + (1:dim_soc)'; vv{end+1} = ones(dim_soc,1);
bb{end+1} = zeros(dim_soc, 1);
row = row + dim_soc;

% (6) t = 1
ii{end+1} = 1 + row; jj{end+1} = o_soc + 1; vv{end+1} = 1;
bb{end+1} = 1;
row = row + 1;

% (7) L1 epigraph: u - y_i - s1_i = 0 ; u + y_i - s2_i = 0
if ~isempty(cons_l1)
    r7a = (1:n_i)' + row;
    ii{end+1} = [r7a; r7a; r7a];
    jj{end+1} = [repmat(o_u+1, n_i, 1); il1 + o_y; o_u + 1 + (1:n_i)'];
    vv{end+1} = [ones(n_i,1); -ones(n_i,1); -ones(n_i,1)];
    bb{end+1} = zeros(n_i,1);
    row = row + n_i;
    r7b = (1:n_i)' + row;
    ii{end+1} = [r7b; r7b; r7b];
    jj{end+1} = [repmat(o_u+1, n_i, 1); il1 + o_y; o_u + 1 + n_i + (1:n_i)'];
    vv{end+1} = [ones(n_i,1); ones(n_i,1); -ones(n_i,1)];
    bb{end+1} = zeros(n_i,1);
    row = row + n_i;
end

% (8) CLT identity on g: g - Sh*y_idx = 0. The +I on g is static; the -Sh
% entries depend on the grid point and are appended by the caller.
clt = [];
if ~isempty(cons_clt)
    r8 = (1:n_ci)' + row;
    ii{end+1} = r8; jj{end+1} = o_g + (1:n_ci)'; vv{end+1} = ones(n_ci,1);
    bb{end+1} = zeros(n_ci,1);
    clt = struct('row0', row, 'cols', iclt + o_y, 'o_g', o_g, ...
        'n_idx', n_ci, 'rho', cons_clt.rho, 'idx_c', iclt);
    row = row + n_ci; %#ok<NASGU>
end

trip = struct('ii', vertcat(ii{:}), 'jj', vertcat(jj{:}), 'vv', vertcat(vv{:}), ...
    'm', m_total, 'n', ntot);
b_s = vertcat(bb{:});

% Objective template: solve_socp_sedumi fills the y block with the per-point
% c; the penalty coefficients (r on u, rho on w) are fixed across the grid.
c_base = zeros(ntot, 1);
if ~isempty(cons_l1),  c_base(o_u + 1) = cons_l1.r;   end
if ~isempty(cons_clt), c_base(o_w + 1) = cons_clt.rho; end

pars.fid = 0;
pars.eps = 1e-8;

A = [];
if isempty(cons_clt)
    A = sparse(trip.ii, trip.jj, trip.vv, m_total, ntot);
end

sd = struct('A', A, 'b', b_s, 'K', K_out, 'pars', pars, 'trip', trip, ...
    'c_base', c_base, 'n_orig', n, 'ntot', ntot, 'o_y', o_y, 'clt', clt);

end
