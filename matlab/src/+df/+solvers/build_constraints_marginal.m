function cstr = build_constraints_marginal(type_space, action_space, Pi, marg_act_distrib_II)
% BUILD_CONSTRAINTS_MARGINAL  Constraint matrices for marginal-mode BCCE test.
%
%   cstr = df.solvers.build_constraints_marginal(type_space, action_space, Pi, marg_act_distrib_II)
%
%   Marginal mode uses Nactions (individual actions) instead of NA (action profiles)
%   for the decision variable dimension.  The deviation constraints are built
%   from expected payoffs (integrated over opponent actions) rather than from
%   the full joint payoff tensor.
%
%   Inputs
%     type_space            NPlayers x 1 cell of type vectors
%     action_space          NPlayers x 1 cell of action vectors
%     Pi                    (NA x s x NPlayers) utility tensor
%     marg_act_distrib_II   (NA x Nq_N) opponent marginal action distribution
%                           used to compute expected payoffs
%
%   Output
%     cstr   struct — same field names as build_constraints, but with
%            marginal-mode dimensions

%% Problem dimensions
NAg = size(type_space, 1);
NA_i = zeros(NAg, 1);
NT_i = zeros(NAg, 1);
for ind = 1:NAg
    NA_i(ind) = size(action_space{ind,1}, 1);
    NT_i(ind) = size(type_space{ind,1}, 1);
end

Nactions = NA_i(1);       % individual actions (marginal mode)
NA  = prod(NA_i);         % joint action profiles (for Pi indexing)
s2  = prod(NT_i);
s   = NT_i(1);
a   = Nactions;
dv  = s * Nactions;       % marginal BCE measure dimension
deq = 1 + Nactions + s;
dineq = s * Nactions;
dM  = deq + dineq;

%% Sorted type profiles
T_sorted = type_space{1,1};
for ind = 2:NAg
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)), ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

%% Bounds  (Nactions-1 for the u variable, not NA-1)
lb = max([-ones(Nactions-1,1); -Inf*ones(deq,1); zeros(dineq,1)], -10000);
ub = min([ ones(Nactions-1,1);  Inf*ones(dM,1)],  10000);

%% Equality constraint matrix
M1eq = kron(ones(1,s), eye(Nactions));
M2eq = kron(eye(s), ones(1,Nactions));
Meq  = [eye(Nactions),        M1eq; ...
        zeros(s, Nactions),   M2eq; ...
        zeros(1, Nactions),   ones(1, dv)];

%% RHS vectors
beq = zeros(Nactions, 1);
b   = zeros(dv, 1);

%% Cone constraint matrix
Mat_NLC = [eye(Nactions-1), zeros(Nactions-1, dM)];

%% Expected payoffs (integrate over opponent actions using first column of marg_act_distrib_II)
% exp_pi(a_i, t) = sum_{a_{-i}} pi(a_i, a_{-i}, t) * q(a_{-i})
% Use first observation column for the constraint construction
exp_pi = kron(eye(Nactions), marg_act_distrib_II(:,1)') * Pi(:,:,1);

%% Inequality constraints (marginal mode: single player)
C1 = eye(s);

pi1_res   = reshape(exp_pi, Nactions, s);
pi_tilde_1T = exp_pi';

% Beta
Beta_1 = kron(C1, ones(Nactions,1)') .* kron(ones(1,s), pi_tilde_1T);

% Alpha (deviation payoffs, marginal mode)
EE = eye(s);
alpha_1_dev = zeros(s * Nactions, s * Nactions);
aaa = 1;
while aaa <= Nactions
    for jj = 1:s
        alpha_1_dev((aaa-1)*s + jj, :) = pi_tilde_1T(jj, aaa) * kron(EE(jj,:), ones(1, Nactions));
    end
    aaa = aaa + 1;
end

M1ineq = alpha_1_dev - kron(ones(Nactions,1), Beta_1);
Mineq  = [zeros(dineq, Nactions), M1ineq];

%% Assemble dual problem matrices
M_full  = [Meq; Mineq];
M_prime = M_full';

B_EQ_comp = [eye(Nactions-1); zeros(1, Nactions-1)];
B_EQ      = [B_EQ_comp, M_prime(1:Nactions, :)];
B_INEQ    = [zeros(dv, Nactions-1), M_prime((Nactions+1):end, :)];

%% Pack output
cstr = struct( ...
    'B_EQ',     B_EQ,     'B_INEQ',   B_INEQ, ...
    'Mat_NLC',  Mat_NLC,  'lb',       lb,      'ub', ub, ...
    'beq',      beq,      'b',        b, ...
    'NA',       NA,       's',        s,       's2', s2, ...
    'Nactions', Nactions, ...
    'dv',       dv,       'deq',      deq,     'dineq', dineq, ...
    'dM',       dM,       'NAg',      NAg,     'NA_i', NA_i, ...
    'NT_i',     NT_i,     'a',        a, ...
    'T_sorted', T_sorted);
end
