function sd = socp_to_sedumi(cstr)
% SOCP_TO_SEDUMI  Convert SOCP constraint struct to SeDuMi standard form.
%
%   sd = df.solvers.socp_to_sedumi(cstr)
%
%   Precomputes the SeDuMi problem data from the constraint struct produced
%   by build_constraints (or build_constraints_marginal).  Call this ONCE,
%   then feed sd into solve_socp_sedumi for each objective vector c.
%
%   The SOCP is:
%     maximize  -c'x
%     s.t.  B_EQ * x   = beq
%           B_INEQ * x >= b
%           lb <= x <= ub
%           ||Mat_NLC * x|| <= 1
%
%   SeDuMi solves:  min b'y  s.t.  c_s - At*y ∈ K
%   (dual form:     max c_s'x  s.t.  At*x = b,  x ∈ K)
%
%   We use the primal form with variables in cone K:
%     K.f  = 0            (free variables)
%     K.l  = n_nn         (nonneg: ineq slacks + bound slacks)
%     K.q  = [1 + dim_soc] (second-order cone)

n = size(cstr.B_EQ, 2);    % decision variable dimension
m_eq   = size(cstr.B_EQ, 1);
m_ineq = size(cstr.B_INEQ, 1);
dim_soc = size(cstr.Mat_NLC, 1);  % NA - 1

%% Identify active bounds (skip ±10000 "inactive" bounds)
tol_bound = 9999;
active_lb = cstr.lb > -tol_bound;
active_ub = cstr.ub <  tol_bound;
n_lb = sum(active_lb);
n_ub = sum(active_ub);
idx_lb = find(active_lb);
idx_ub = find(active_ub);

%% Cone structure
% Nonneg variables: ineq slacks (m_ineq) + lb slacks (n_lb) + ub slacks (n_ub)
n_nn = m_ineq + n_lb + n_ub;

% Total SeDuMi variables:  x_nn (n_nn)  +  x_soc (1 + dim_soc)
n_total = n_nn + 1 + dim_soc;

% Cone specification
K.l = n_nn;
K.q = 1 + dim_soc;

%% Build A matrix (equality constraints in terms of SeDuMi variables)
% SeDuMi primal: A * x_s = b_s,  x_s ∈ K
% where x_s = [s_ineq; s_lb; s_ub; t; v]
%
% Equality constraints from original problem:
%   B_EQ * x = beq                     (m_eq rows)
%   B_INEQ * x - s_ineq = b            (m_ineq rows)
%   x(idx_lb) - s_lb = lb(idx_lb)      (n_lb rows)
%   -x(idx_ub) + s_ub = -ub(idx_ub)    (n_ub rows, i.e. x <= ub)
%   Mat_NLC * x - v = 0                (dim_soc rows)
%   t = 1                              (1 row)
%
% But SeDuMi variables are x_s, not x. We need to express x in terms of
% x_s, or use the dual form.
%
% Better approach: use SeDuMi's DUAL form directly.
% SeDuMi dual: max b'y  s.t.  A'y + s = c_sedumi,  s ∈ K
% where y are the dual variables (one per equality constraint in primal).
%
% Our problem: min c'x  s.t.  constraints
% This maps to SeDuMi dual with:
%   y = x (original decision variables, dimension n)
%   Each constraint contributes a cone variable.

% Total equality constraints that become SeDuMi dual variables:
% Actually, let's use the cleaner standard form.
%
% SeDuMi: min c_s' * z  s.t.  A_s * z = b_s,  z ∈ K
%
% Map our problem:
%   z = [s_ineq; s_lb; s_ub; t; v]  (all in cone K)
%
%   From B_INEQ * x - s_ineq = b:     s_ineq = B_INEQ * x - b
%   From x_j - s_lb_j = lb_j:         s_lb_j = x_j - lb_j  (j ∈ idx_lb)
%   From s_ub_j - x_j = -ub_j:        ... this doesn't cleanly eliminate x.
%
% The standard approach: write as min c'x using SeDuMi's (At, b, c, K).
% SeDuMi convention: min b'y, s.t. c - At*y >= 0 (in cone K)
% i.e., c - At*y = s, s ∈ K

% y = x (our n decision variables)
% We need c_sedumi - At_sedumi * y ∈ K

m_total = m_eq;  % equality constraints become rows of At with free vars
                  % ... this is getting complex.

% SIMPLEST APPROACH: use sedumi(At, b, c_s, K) in standard form.
% Standard form: find y to min b'y s.t. At*y + s = c_s, s >= 0 (generalized)
%
% y are our decision variables (dim n)
% s are slack variables in cone K
%
% Constraints to encode:
% (a) B_INEQ * x >= b  →  B_INEQ' column block, RHS = b  →  s_ineq = B_INEQ*y - b >= 0
%     So:  At_ineq * y + s_ineq = c_ineq  →  At_ineq = B_INEQ,  c_ineq = b
%     Wait no: At * y + s = c  →  s = c - At*y
%     We want s = B_INEQ*y - b >= 0
%     So: c = -b, At = -B_INEQ  →  s = -b - (-B_INEQ)*y = B_INEQ*y - b  ✓
%     Actually: At*y + s = c → s = c - At*y. We want s = B_INEQ*y - b.
%     So: c - At*y = B_INEQ*y - b → c = -b, At = -B_INEQ.

% (b) lb <= x  →  x - lb >= 0  →  s_lb = y(idx) - lb(idx) >= 0
%     c_lb(j) - At_lb(j,:)*y = y_j - lb_j  → At_lb = -e_j', c_lb = -lb_j

% (c) x <= ub  →  ub - x >= 0  →  s_ub = ub(idx) - y(idx) >= 0
%     c_ub(j) - At_ub(j,:)*y = ub_j - y_j  → At_ub = e_j', c_ub = ub_j

% (d) ||Mat_NLC * y|| <= 1  →  SOC: (t, Mat_NLC*y) with t=1
%     SeDuMi SOC: s_q = [t; v] ∈ Q, ||v|| <= t
%     s_q(1) = c_q(1) - At_q(1,:)*y = 1  → At_q(1,:) = 0, c_q(1) = 1
%     s_q(2:end) = c_q(2:end) - At_q(2:end,:)*y = Mat_NLC*y
%     → At_q(2:end,:) = -Mat_NLC, c_q(2:end) = 0

% (e) B_EQ * y = beq  →  free variables in K.f
%     This is trickier. SeDuMi handles equalities via K.f (free cone).
%     Or we can use the At/b/c encoding with free slacks.
%     K.f rows: s_free = c_f - At_f * y, s_free is FREE (unrestricted)
%     B_EQ * y = beq → B_EQ * y - beq = 0
%     s_f = c_f - At_f*y = 0 with s_f free → At_f = B_EQ, c_f = beq
%     Actually: we want s_f = beq - B_EQ*y ... no, we want B_EQ*y = beq.
%     Free variable: s_f = c_f - At_f*y can be anything.
%     We want: s_f = 0 always → not enforced by free cone.
%
%     Better: equalities go in b, At with separate handling.
%     SeDuMi actually uses K.f for free variables at the START of the cone.

% Let me use K.f for equality constraints:
% s_f = c_f - At_f * y, where s_f is free → encodes B_EQ * y = beq
% But free means unconstrained, so this doesn't enforce equality!
%
% The right way: use SeDuMi's built-in equality handling.
% SeDuMi: min b'y  s.t.  At*y + s = c,  s ∈ K
% For equalities: put them as K.f = m_eq at the beginning of K.
% Then s_f(1:m_eq) is FREE, and At_f * y + s_f = c_f.
% This means At_f * y = c_f - s_f, with s_f free → not useful.
%
% Actually K.f = 0 means "zero free variables", i.e. all in cone.
% Equalities must be encoded differently.
%
% The standard SeDuMi approach for equalities Ax = b:
% Use K.f rows. The "free" part of the cone means those slack variables
% are unconstrained, which effectively means At_f * y = c_f (since s_f
% can absorb any value... wait, no).
%
% I think I'm overcomplicating this. Let me just encode equalities as
% PAIRS of nonneg constraints: B_EQ * y >= beq AND -B_EQ * y >= -beq.

% Assemble At, c_s, K:
% Rows of [At | c_s]:
%   Free rows (equalities):     K.f = 2*m_eq  (>= and <= pairs)
%   Actually just use K.f for the equality slack:

% REVISED: cleanest approach with K.f
% K.f rows at top: s_f free → At_f * y + s_f = c_f
%   This is NOT the same as At_f * y = c_f.
%   It's redundant (always satisfiable).
%
% So equalities must be via the b vector (RHS of At*y + s = c).
% Wait — SeDuMi's calling convention is:
%   sedumi(A, b, c, K) solves:
%     min c'x  s.t.  Ax = b,  x ∈ K    (primal)
%     max b'y  s.t.  A'y + s = c, s ∈ K (dual)

% Let me use the PRIMAL form directly:
%   min c_s' * x_s  s.t.  A_s * x_s = b_s,  x_s ∈ K
%
% x_s = [s_f; s_nn; s_q]  where:
%   s_f:  free variables for equalities (K.f = m_eq)
%         s_f = beq - B_EQ * y ... hmm, y is not part of x_s in primal.

% OK. I'll use a completely different and cleaner approach.
% Embed everything in one big equality system.

% Decision: use SeDuMi's DUAL form.
% Variables: y ∈ R^n (our original x)
% Constraint: c_s - At' * y ∈ K  (At is transposed relative to convention)
% Objective: max b_s' * y
%
% SeDuMi call: sedumi(At, b_s, c_s, K)
%   where At is m×n, b_s is n×1, c_s is m×1

% Our objective: maximize -c_obj'*x  →  b_s = -c_obj (set per solve)

% Dual cone constraints (m rows total = 2*m_eq + m_ineq + n_lb + n_ub + 1 + dim_soc):

n_rows = 2*m_eq + m_ineq + n_lb + n_ub + 1 + dim_soc;

% At matrix (n_rows × n)
At = zeros(n_rows, n);
c_s = zeros(n_rows, 1);
row = 0;

% (a) Equalities as free cone: K.f = 2*m_eq
% B_EQ * y = beq → s_f1 = beq - B_EQ*y (free)  AND  s_f2 = -beq + B_EQ*y (free)
% Actually for free variables, we just need m_eq rows:
% s_f = arbitrary → At_f * y + s_f = c_f is always satisfiable.
% This doesn't enforce equality.
%
% The correct SeDuMi approach: equalities are encoded via the A matrix
% in primal form, NOT via free cone variables.

% I'll use a simpler, working approach.
% Eliminate equality constraints by substitution, or use SeDuMi's
% built-in support.

% ACTUALLY: the cleanest SeDuMi interface for our problem:
% Use the DUAL form:  max b'y  s.t.  A'y + s = c,  s ∈ K
% Here y are the dual variables and s are primal cone variables.
%
% But we want to solve the PRIMAL: min c'x s.t. Ax=b, x ∈ K.
% Let's just formulate it as primal.
%
% x = [x_free; x_nn; x_soc]  ∈  K = {R^{m_eq}} × {R+^{n_nn}} × {Q^{1+dim_soc}}
%
% where:
%   x_free ∈ R^{m_eq}:  slack for equalities (these are unrestricted)
%       WAIT — this is wrong. K.f means these are FREE, not that they
%       encode equality constraints.

% Let me just take a completely pragmatic approach and call SeDuMi
% through its simplest interface, building the problem cleanly.

% ============================================================
% SeDuMi solves: min c_s'z s.t. A_s z = b_s, z in K
%
% z is the vector of ALL variables (free + nonneg + SOC)
% A_s z = b_s are the equality constraints
% K specifies the cone for z
% ============================================================

% Our original variable is y (dim n). Introduce cone variables:
%
% z = [y; s_ineq; s_lb; s_ub; t; v]
%   y:       n      (free, K.f = n)
%   s_ineq:  m_ineq (nonneg, part of K.l)
%   s_lb:    n_lb   (nonneg, part of K.l)
%   s_ub:    n_ub   (nonneg, part of K.l)
%   [t; v]:  1+dim_soc (SOC, K.q)
%
% n_total = n + m_ineq + n_lb + n_ub + 1 + dim_soc

ntot = n + m_ineq + n_lb + n_ub + 1 + dim_soc;

% Cone
K_out.f = n;           % y is free
K_out.l = n_nn;        % s_ineq, s_lb, s_ub are nonneg
K_out.q = 1 + dim_soc; % SOC

% Equality constraints:
% (1) B_EQ * y = beq                              (m_eq rows)
% (2) B_INEQ * y - s_ineq = b                     (m_ineq rows)
% (3) y(idx_lb) - s_lb = lb(idx_lb)               (n_lb rows)
% (4) -y(idx_ub) + s_ub = -ub(idx_ub)             (n_ub rows)
%     equivalently: ub(idx_ub) - y(idx_ub) = s_ub
% (5) -Mat_NLC * y + v = 0                         (dim_soc rows)
%     equivalently: v = Mat_NLC * y
% (6) t = 1                                        (1 row)

m_total = m_eq + m_ineq + n_lb + n_ub + dim_soc + 1;

% Index offsets in z
o_y      = 0;
o_sineq  = n;
o_slb    = n + m_ineq;
o_sub    = n + m_ineq + n_lb;
o_soc    = n + m_ineq + n_lb + n_ub;  % t is at o_soc+1, v starts at o_soc+2

% Build A_s (sparse, m_total × ntot)
% Using sparse triplets for efficiency
ii = []; jj = []; vv = [];

row_offset = 0;

% (1) B_EQ * y = beq  (m_eq rows)
[ri, ci, vi] = find(cstr.B_EQ);
ii = [ii; ri + row_offset]; jj = [jj; ci + o_y]; vv = [vv; vi];
b_s = cstr.beq(:);
row_offset = row_offset + m_eq;

% (2) B_INEQ * y - I * s_ineq = b  (m_ineq rows)
[ri, ci, vi] = find(cstr.B_INEQ);
ii = [ii; ri + row_offset]; jj = [jj; ci + o_y]; vv = [vv; vi];
% -I for s_ineq
ii = [ii; (1:m_ineq)' + row_offset];
jj = [jj; (1:m_ineq)' + o_sineq];
vv = [vv; -ones(m_ineq, 1)];
b_s = [b_s; cstr.b(:)];
row_offset = row_offset + m_ineq;

% (3) y(idx_lb) - s_lb = lb(idx_lb)  (n_lb rows)
for k = 1:n_lb
    ii = [ii; k + row_offset]; jj = [jj; idx_lb(k) + o_y]; vv = [vv; 1];
    ii = [ii; k + row_offset]; jj = [jj; k + o_slb]; vv = [vv; -1];
end
b_s = [b_s; cstr.lb(idx_lb)];
row_offset = row_offset + n_lb;

% (4) -y(idx_ub) + s_ub = ub(idx_ub) → s_ub = ub - y  (n_ub rows)
% Rewrite: -y(idx_ub) + s_ub = ub(idx_ub) ... wait:
% We want s_ub = ub(idx_ub) - y(idx_ub) >= 0
% So: -y(idx_ub) + s_ub = ub(idx_ub)
for k = 1:n_ub
    ii = [ii; k + row_offset]; jj = [jj; idx_ub(k) + o_y]; vv = [vv; -1];
    ii = [ii; k + row_offset]; jj = [jj; k + o_sub]; vv = [vv; 1];
end
b_s = [b_s; cstr.ub(idx_ub)];
row_offset = row_offset + n_ub;

% (5) -Mat_NLC * y + v = 0  (dim_soc rows)
[ri, ci, vi] = find(cstr.Mat_NLC);
ii = [ii; ri + row_offset]; jj = [jj; ci + o_y]; vv = [vv; -vi];
% v starts at column o_soc + 2 (after t)
for k = 1:dim_soc
    ii = [ii; k + row_offset]; jj = [jj; o_soc + 1 + k]; vv = [vv; 1];
end
b_s = [b_s; zeros(dim_soc, 1)];
row_offset = row_offset + dim_soc;

% (6) t = 1  (1 row)
ii = [ii; 1 + row_offset]; jj = [jj; o_soc + 1]; vv = [vv; 1];
b_s = [b_s; 1];

A_s = sparse(ii, jj, vv, m_total, ntot);

% Objective template: maximize -c_obj'*y = minimize c_obj'*y
% c_s has c_obj in the y-positions and 0 elsewhere.
% (Set per solve in solve_socp_sedumi; here we just store the template.)

% SeDuMi options
pars.fid = 0;  % suppress output
pars.eps = 1e-8;

sd = struct('A', A_s, 'b', b_s, 'K', K_out, 'pars', pars, ...
    'n_orig', n, 'ntot', ntot, 'o_y', o_y);

end
