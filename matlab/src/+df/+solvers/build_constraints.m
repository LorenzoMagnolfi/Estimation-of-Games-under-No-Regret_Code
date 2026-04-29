function cstr = build_constraints(type_space, action_space, Pi, consistency_slack)
% BUILD_CONSTRAINTS  Construct SOCP constraint matrices for joint-mode BCCE test.
%
%   cstr = df.solvers.build_constraints(type_space, action_space, Pi)
%   cstr = df.solvers.build_constraints(type_space, action_space, Pi, consistency_slack)
%
%   Builds equality, inequality, bounds, and cone constraint matrices that
%   are INVARIANT across the parameter-grid loop.  Only the objective vector
%   c changes per grid point.
%
%   Inputs
%     type_space        NPlayers x 1 cell of type vectors
%     action_space      NPlayers x 1 cell of action vectors
%     Pi                (NA x s x NPlayers) utility tensor
%     consistency_slack (optional) scalar >= 0.  Default 0 (exact consistency).
%                       When > 0, the s2 consistency-equality rows are moved
%                       to Mineq as a coordinatewise box  |sum_a nu - p| <= r.
%                       The driver must construct c_all to match the new
%                       constraint ordering: see the .has_consistency_slack
%                       flag in the returned cstr.
%
%   Output
%     cstr   struct with fields:
%       B_EQ, B_INEQ, Mat_NLC, lb, ub, beq, b   — constraint matrices / vectors
%       NA, s, s2, dv, deq, dineq, dM, NAg, NA_i, NT_i, a
%       T_sorted                                  — sorted type profiles
%       has_consistency_slack                     — logical flag

if nargin < 4 || isempty(consistency_slack)
    consistency_slack = 0;
end
has_consistency_slack = consistency_slack > 0;

%% Problem dimensions
NAg = size(type_space, 1);
NA_i = zeros(NAg, 1);
NT_i = zeros(NAg, 1);
for ind = 1:NAg
    NA_i(ind) = size(action_space{ind,1}, 1);
    NT_i(ind) = size(type_space{ind,1}, 1);
end

NA  = prod(NA_i);        % action profiles
s2  = prod(NT_i);        % type profiles
s   = NT_i(1);           % types per player (symmetric)
a   = NA_i(1);           % actions per player
dv  = s2 * NA;           % BCE measure dimension
dineq_obed = NT_i' * NA_i;    % obedience inequality constraints

if has_consistency_slack
    deq   = 1 + NA;                    % action marginal + total mass only
    dineq = 2 * s2 + dineq_obed;       % consistency upper + lower + obedience
else
    deq   = 1 + NA + s2;               % action marginal + consistency + total mass
    dineq = dineq_obed;                % obedience only
end
dM  = deq + dineq;

%% Sorted type profiles (Kronecker product ordering)
T_sorted = type_space{1,1};
for ind = 2:NAg
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)), ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

%% Bounds
lb = max([-ones(NA-1,1); -Inf*ones(deq,1); zeros(dineq,1)], -10000);
ub = min([ ones(NA-1,1);  Inf*ones(dM,1)],  10000);

%% Equality constraint matrix  Meq
M1eq = kron(ones(1,s2), eye(NA));
M2eq = kron(eye(s2), ones(1,NA));
if has_consistency_slack
    % Drop consistency rows from Meq; they go into Mineq as upper+lower box.
    Meq = [eye(NA),     M1eq; ...
           zeros(1,NA), ones(1,dv)];
else
    Meq  = [eye(NA),      M1eq; ...
            zeros(s2,NA), M2eq; ...
            zeros(1,NA),  ones(1,dv)];
end

%% RHS vectors
beq = zeros(NA, 1);
b   = zeros(dv, 1);

%% Cone constraint matrix
Mat_NLC = [eye(NA-1), zeros(NA-1, dM)];

%% Inequality constraint matrix  Mineq  (deviation incentive constraints)
C1 = kron(eye(s), ones(1,s));
C2 = kron(ones(1,s), eye(s));

pi_1 = reshape(squeeze(Pi(:,:,1)), NA*s, 1);
pi_2 = reshape(squeeze(Pi(:,:,2)), NA*s, 1);

pi1_res = reshape(pi_1, NA, s);
pi2_res = reshape(pi_2, NA, s);

% pi_tilde vectors
pi_tilde_1T = reshape(repmat(pi1_res, s, 1), NA*s^2, 1)';
pi_tilde_2T = kron(ones(s,1)', pi_2');

% Beta matrices
Beta_1 = kron(C1, ones(NA,1)') .* kron(ones(s,1), pi_tilde_1T);
Beta_2 = kron(C2, ones(NA,1)') .* kron(ones(s,1), pi_tilde_2T);

% Alpha matrices (deviation payoffs)
E = eye(a);
pi_1_hat = reshape(pi_1, NA, s);
pi_2_hat = reshape(pi_2, NA, s);

alpha_1 = zeros(s, s^2*NA, a);
alpha_2 = zeros(s, s^2*NA, a);

for j = 1:a
    % Player 1 deviation
    pi_j_1 = reshape(kron(ones(a,1), kron(E(:,j)', eye(a))) * pi_1_hat, NA*s, []);
    pi_tilde_j_1T = reshape(repmat(reshape(pi_j_1, NA, s), s, 1), NA*s^2, 1)';
    alpha_1(:,:,j) = kron(C1, ones(NA,1)') .* kron(ones(s,1), pi_tilde_j_1T);

    % Player 2 deviation
    pi_j_2 = reshape(kron(kron(eye(a), E(:,j)') * pi_2_hat, ones(a,1)), NA*s, []);
    pi_tilde_j_2T = kron(ones(s,1)', pi_j_2');
    alpha_2(:,:,j) = kron(C2, ones(NA,1)') .* kron(ones(s,1), pi_tilde_j_2T);
end

alpha_1_dev = reshape(permute(alpha_1, [1 3 2]), [], size(alpha_1, 2));
alpha_2_dev = reshape(permute(alpha_2, [1 3 2]), [], size(alpha_2, 2));

M1ineq = [alpha_1_dev; alpha_2_dev] - [kron(ones(a,1), Beta_1); kron(ones(a,1), Beta_2)];
if has_consistency_slack
    % Mineq stack: [consistency_upper; consistency_lower; obedience]
    Mineq = [zeros(s2, NA),         M2eq; ...           % sum_a nu(a,t,θ) <= p_λ + r
             zeros(s2, NA),        -M2eq; ...           % -sum_a nu <= -p_λ + r
             zeros(dineq_obed, NA), M1ineq];            % obedience deviations
else
    Mineq  = [zeros(dineq_obed, NA), M1ineq];
end

%% Assemble dual problem matrices (hoisted from inner loop)
M_full  = [Meq; Mineq];
M_prime = M_full';

B_EQ_comp = [eye(NA-1); zeros(1, NA-1)];
B_EQ      = [B_EQ_comp, M_prime(1:NA, :)];
B_INEQ    = [zeros(dv, NA-1), M_prime((NA+1):end, :)];

%% Pack output
cstr = struct( ...
    'B_EQ',     B_EQ,     'B_INEQ',   B_INEQ, ...
    'Mat_NLC',  Mat_NLC,  'lb',       lb,      'ub', ub, ...
    'beq',      beq,      'b',        b, ...
    'NA',       NA,       's',        s,       's2', s2, ...
    'dv',       dv,       'deq',      deq,     'dineq', dineq, ...
    'dM',       dM,       'NAg',      NAg,     'NA_i', NA_i, ...
    'NT_i',     NT_i,     'a',        a, ...
    'T_sorted', T_sorted, ...
    'has_consistency_slack', has_consistency_slack, ...
    'consistency_slack',     consistency_slack);
end
