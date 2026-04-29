function cstr = build_constraints_realization(type_space, action_space, Pi, realized_idx)
% DF.SOLVERS.BUILD_CONSTRAINTS_REALIZATION  Slice-equality SOCP for
% realization-conditional identification.
%
%   cstr = df.solvers.build_constraints_realization(type_space, action_space, Pi, realized_idx)
%
%   Variant of df.solvers.build_constraints for App. SIM-RC: instead of the
%   marginal equality
%       sum_t nu(a, t) = m_N(a)        (data is unconditional marginal)
%   the data constraint is the slice equality
%       nu(a, t_1*, t_2*) = pi(t_1*, t_2*) * m_N(a)        (data is conditional)
%   where (t_1*, t_2*) is the candidate realized type pair, indexed by
%   realized_idx (1..s2) in the Kronecker ordering used by build_constraints.
%
%   All other constraints (consistency, obedience, total mass) are
%   identical to df.solvers.build_constraints.  The output struct has the
%   same shape, with .has_consistency_slack = false, and an additional
%   .realized_idx field for traceability.
%
%   The pi(t_1*, t_2*) factor on the slice rhs is folded into the driver's
%   objective vector c_all via .pi_realized below.

%% Problem dimensions
NAg  = size(type_space, 1);
NA_i = zeros(NAg, 1);
NT_i = zeros(NAg, 1);
for ind = 1:NAg
    NA_i(ind) = size(action_space{ind,1}, 1);
    NT_i(ind) = size(type_space{ind,1}, 1);
end

NA  = prod(NA_i);
s2  = prod(NT_i);
s   = NT_i(1);
a   = NA_i(1);
dv  = s2 * NA;
dineq_obed = NT_i' * NA_i;

if realized_idx < 1 || realized_idx > s2
    error('build_constraints_realization:badIdx', ...
        'realized_idx must be in 1..s2=%d, got %d', s2, realized_idx);
end

deq   = 1 + NA + s2;     % slice (NA) + consistency (s2) + total mass (1)
dineq = dineq_obed;
dM    = deq + dineq;

%% Sorted type profiles (Kronecker product ordering, matches build_constraints)
T_sorted = type_space{1,1};
for ind = 2:NAg
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)), ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

%% Bounds
lb = max([-ones(NA-1,1); -Inf*ones(deq,1); zeros(dineq,1)], -10000);
ub = min([ ones(NA-1,1);  Inf*ones(dM,1)],  10000);

%% Equality constraint matrix Meq
% Slice selection: NA x dv, eye(NA) at columns (k*-1)*NA + (1:NA), zeros elsewhere.
M1eq_slice = sparse(NA, dv);
M1eq_slice(:, (realized_idx - 1) * NA + (1:NA)) = speye(NA);
M1eq_slice = full(M1eq_slice);

% Consistency: same as build_constraints
M2eq = kron(eye(s2), ones(1,NA));

Meq  = [eye(NA),      M1eq_slice; ...
        zeros(s2,NA), M2eq;       ...
        zeros(1,NA),  ones(1,dv)];

%% RHS
beq = zeros(NA, 1);
b   = zeros(dv, 1);

%% Cone
Mat_NLC = [eye(NA-1), zeros(NA-1, dM)];

%% Inequality constraint matrix Mineq (obedience)
C1 = kron(eye(s), ones(1,s));
C2 = kron(ones(1,s), eye(s));

pi_1 = reshape(squeeze(Pi(:,:,1)), NA*s, 1);
pi_2 = reshape(squeeze(Pi(:,:,2)), NA*s, 1);

pi1_res = reshape(pi_1, NA, s);
pi2_res = reshape(pi_2, NA, s);

pi_tilde_1T = reshape(repmat(pi1_res, s, 1), NA*s^2, 1)';
pi_tilde_2T = kron(ones(s,1)', pi_2');

Beta_1 = kron(C1, ones(NA,1)') .* kron(ones(s,1), pi_tilde_1T);
Beta_2 = kron(C2, ones(NA,1)') .* kron(ones(s,1), pi_tilde_2T);

E = eye(a);
pi_1_hat = reshape(pi_1, NA, s);
pi_2_hat = reshape(pi_2, NA, s);

alpha_1 = zeros(s, s^2*NA, a);
alpha_2 = zeros(s, s^2*NA, a);

for j = 1:a
    pi_j_1 = reshape(kron(ones(a,1), kron(E(:,j)', eye(a))) * pi_1_hat, NA*s, []);
    pi_tilde_j_1T = reshape(repmat(reshape(pi_j_1, NA, s), s, 1), NA*s^2, 1)';
    alpha_1(:,:,j) = kron(C1, ones(NA,1)') .* kron(ones(s,1), pi_tilde_j_1T);

    pi_j_2 = reshape(kron(kron(eye(a), E(:,j)') * pi_2_hat, ones(a,1)), NA*s, []);
    pi_tilde_j_2T = kron(ones(s,1)', pi_j_2');
    alpha_2(:,:,j) = kron(C2, ones(NA,1)') .* kron(ones(s,1), pi_tilde_j_2T);
end

alpha_1_dev = reshape(permute(alpha_1, [1 3 2]), [], size(alpha_1, 2));
alpha_2_dev = reshape(permute(alpha_2, [1 3 2]), [], size(alpha_2, 2));

M1ineq = [alpha_1_dev; alpha_2_dev] - [kron(ones(a,1), Beta_1); kron(ones(a,1), Beta_2)];
Mineq  = [zeros(dineq_obed, NA), M1ineq];

%% Dual matrices
M_full  = [Meq; Mineq];
M_prime = M_full';

B_EQ_comp = [eye(NA-1); zeros(1, NA-1)];
B_EQ      = [B_EQ_comp, M_prime(1:NA, :)];
B_INEQ    = [zeros(dv, NA-1), M_prime((NA+1):end, :)];

%% Pack
cstr = struct( ...
    'B_EQ',     B_EQ,     'B_INEQ',   B_INEQ, ...
    'Mat_NLC',  Mat_NLC,  'lb',       lb,      'ub', ub, ...
    'beq',      beq,      'b',        b, ...
    'NA',       NA,       's',        s,       's2', s2, ...
    'dv',       dv,       'deq',      deq,     'dineq', dineq, ...
    'dM',       dM,       'NAg',      NAg,     'NA_i', NA_i, ...
    'NT_i',     NT_i,     'a',        a, ...
    'T_sorted', T_sorted, ...
    'has_consistency_slack', false, ...
    'consistency_slack',     0, ...
    'realized_idx',          realized_idx);
end
