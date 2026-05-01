function cstr = build_constraints_marginal_slack(type_space, action_space, Pi, marg_act_distrib_II, consistency_slack)
% BUILD_CONSTRAINTS_MARGINAL_SLACK  Marginal-mode BCCE LP with separate-slack
% consistency relaxation (per Niccolò's rate-corollaries note Sec. 6.2).
%
%   cstr = df.solvers.build_constraints_marginal_slack(type_space, action_space, Pi,
%          marg_act_distrib_II, consistency_slack)
%
%   Same as df.solvers.build_constraints_marginal but the consistency
%   equality `sum_a nu(a, t) = pi(t)` is replaced by a coordinatewise box
%   relaxation:
%      pi(t) - r_N  <=  sum_a nu(a, t)  <=  pi(t) + r_N      for each t
%   where the slack r_N (a scalar > 0) is supplied by the caller.  The
%   matrix STRUCTURE is independent of r_N's numeric value; the LP
%   driver injects the actual r_N into the rhs vector c_all.
%
%   This mirrors the consistency-slack support that build_constraints.m
%   provides for joint mode.
%
%   Inputs
%     type_space, action_space, Pi, marg_act_distrib_II  — same as
%       df.solvers.build_constraints_marginal
%     consistency_slack  — positive scalar (any value; only the SIGN
%                          determines the structure).  Pass 0 (or omit)
%                          to get the no-slack version, identical to
%                          df.solvers.build_constraints_marginal.
%
%   Output
%     cstr   struct with the same fields as build_constraints_marginal,
%            plus .has_consistency_slack (logical) and .consistency_slack.

if nargin < 5 || isempty(consistency_slack)
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

Nactions = NA_i(1);       % individual actions (marginal mode)
NA  = prod(NA_i);
s2  = prod(NT_i);
s   = NT_i(1);
a   = Nactions;
dv  = s * Nactions;       % marginal BCE measure dimension
dineq_obed = s * Nactions;

if has_consistency_slack
    deq   = 1 + Nactions;                       % action marginal + total mass only
    dineq = 2 * s + dineq_obed;                 % consistency upper + lower + obedience
else
    deq   = 1 + Nactions + s;                   % action marginal + consistency + total mass
    dineq = dineq_obed;
end
dM  = deq + dineq;

%% Sorted type profiles
T_sorted = type_space{1,1};
for ind = 2:NAg
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)), ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

%% Bounds
lb = max([-ones(Nactions-1,1); -Inf*ones(deq,1); zeros(dineq,1)], -10000);
ub = min([ ones(Nactions-1,1);  Inf*ones(dM,1)],  10000);

%% Equality constraint matrix (and consistency block goes to inequality if slack)
M1eq = kron(ones(1,s), eye(Nactions));
M2eq = kron(eye(s), ones(1,Nactions));

if has_consistency_slack
    % Drop M2eq from Meq; it goes into Mineq as upper+lower box.
    Meq = [eye(Nactions),     M1eq; ...
           zeros(1,Nactions), ones(1,dv)];
else
    Meq = [eye(Nactions),        M1eq; ...
           zeros(s, Nactions),   M2eq; ...
           zeros(1, Nactions),   ones(1, dv)];
end

%% RHS vectors
beq = zeros(Nactions, 1);
b   = zeros(dv, 1);

%% Cone constraint matrix
Mat_NLC = [eye(Nactions-1), zeros(Nactions-1, dM)];

%% Expected payoffs (integrate over opponent actions)
% Use first observation column for the constraint construction
exp_pi = kron(eye(Nactions), marg_act_distrib_II(:,1)') * Pi(:,:,1);
pi_tilde_1T = exp_pi';

%% Inequality constraints
C1 = eye(s);
Beta_1 = kron(C1, ones(Nactions,1)') .* kron(ones(1,s), pi_tilde_1T);

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

if has_consistency_slack
    % Mineq stack: [consistency upper (sum_a nu - pi <= r_N);
    %               consistency lower (-sum_a nu + pi <= r_N);
    %               obedience deviations]
    Mineq = [zeros(s, Nactions),         M2eq;       ...
             zeros(s, Nactions),        -M2eq;       ...
             zeros(dineq_obed, Nactions), M1ineq];
else
    Mineq = [zeros(dineq_obed, Nactions), M1ineq];
end

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
    'T_sorted', T_sorted, ...
    'has_consistency_slack', has_consistency_slack, ...
    'consistency_slack',     consistency_slack);
end
