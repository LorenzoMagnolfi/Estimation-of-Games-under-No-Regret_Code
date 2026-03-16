function VP = solve_polytope_lp(cfg, maxiters, conf2, switch_eps)
% SOLVE_POLYTOPE_LP  Compute BCE polytope vertices via linprog.
%
%   VP = df.solvers.solve_polytope_lp(cfg, maxiters, conf2, switch_eps)
%
%   Replaces find_polytope_switch.m + AMPL external solver.  Uses native
%   MATLAB linprog to solve the same LP that was previously dispatched
%   to AMPL/Knitro.
%
%   The LP:
%     max  lambda' * [P_HH, P_HL, P_LH, P_LL]
%     s.t. bce >= 0,  bce <= 1
%          sum(bce) = 1                              (probability)
%          sum_{a1,a2} bce(a1,a2,r1,r2) = Psi(r1,r2) (consistency)
%          IC constraints for each player/action/type (obedience)
%
%   This is the 2-action, 2-player pricing game polytope from Stage I.
%
%   Inputs
%     cfg         game config struct (must have .alpha, .AA, .s, .Egrid,
%                 .Psi, .marg_distrib, .NAct, .NPlayers)
%     maxiters    learning iterations T (for epsilon computation)
%     conf2       confidence level
%     switch_eps  epsilon formula selector
%
%   Output
%     VP   (NG x 4) matrix of [PHH, PHL, PLH, PLL] for each Halton direction

alpha_val = cfg.alpha;
AA   = cfg.AA;
s    = cfg.s;         % number of types per player
s2   = s^2;           % type profiles
Egrid = cfg.Egrid;    % type grid values
Psi  = cfg.Psi;       % joint prior over type profiles
marg_distrib = cfg.marg_distrib;
NAct = cfg.NAct;
NPlayers = cfg.NPlayers;

%% Actions (2-action game)
A1L = AA(1);  A1H = AA(2);
A2L = AA(1);  A2H = AA(2);

%% Logit choice probabilities (same as AMPL .mod file)
prob1HH = exp(alpha_val*A1H) / (1 + exp(alpha_val*A1H) + exp(alpha_val*A2H));
prob1HL = exp(alpha_val*A1H) / (1 + exp(alpha_val*A1H) + exp(alpha_val*A2L));
prob1LH = exp(alpha_val*A1L) / (1 + exp(alpha_val*A1L) + exp(alpha_val*A2H));
prob1LL = exp(alpha_val*A1L) / (1 + exp(alpha_val*A1L) + exp(alpha_val*A2L));
prob2HH = exp(alpha_val*A2H) / (1 + exp(alpha_val*A1H) + exp(alpha_val*A2H));
prob2HL = exp(alpha_val*A2L) / (1 + exp(alpha_val*A1H) + exp(alpha_val*A2L));
prob2LH = exp(alpha_val*A2H) / (1 + exp(alpha_val*A1L) + exp(alpha_val*A2H));
prob2LL = exp(alpha_val*A2L) / (1 + exp(alpha_val*A1L) + exp(alpha_val*A2L));

%% Epsilon (same as find_polytope_switch.m)
eps_vec = epsilon_switch(maxiters, conf2, switch_eps, cfg);
d4 = repmat(marg_distrib, 1, NAct*NPlayers) .* ...
     repmat(sqrt(marg_distrib), 1, NAct*NPlayers) .* ...
     repmat(eps_vec', 1, NAct*NPlayers);
epsil = 1;

%% Decision variable ordering:
%   x(ap, r1, r2) where ap ∈ {1=HH, 2=HL, 3=LH, 4=LL}
%   Linear index: idx(ap, r1, r2) = (ap-1)*s2 + (r1-1)*s + r2
nvar = 4 * s2;

% Helper to get index
idx = @(ap, r1, r2) (ap-1)*s2 + (r1-1)*s + r2;

%% Bounds
lb = zeros(nvar, 1);
ub = ones(nvar, 1);

%% Equality constraints
% 1. Sum of all bce = 1
Aeq_sum = ones(1, nvar);
beq_sum = 1;

% 2. Consistency: for each (r1,r2), sum_{a1,a2} bce = Mass[(r1-1)*s + r2]
%    Mass is Psi (joint prior over type profiles, s2 x 1 vector)
Aeq_consist = zeros(s2, nvar);
beq_consist = Psi(:);  % s2 x 1
for r1 = 1:s
    for r2 = 1:s
        row = (r1-1)*s + r2;
        for ap = 1:4
            Aeq_consist(row, idx(ap, r1, r2)) = 1;
        end
    end
end

Aeq = [Aeq_sum; Aeq_consist];
beq = [beq_sum; beq_consist];

%% Inequality constraints (IC: Ax >= b  →  -Ax <= -b for linprog)
% 4 families of IC constraints, one for each (player, action recommendation)
%
% IC1_H(r1): player 1 told H, prefers H over L
%   (A1H - Egrid(r1)) * sum_{r2} [bce(HL,r1,r2)*prob1HL + bce(HH,r1,r2)*prob1HH]
% - (A1L - Egrid(r1)) * sum_{r2} [bce(HL,r1,r2)*prob1LL + bce(HH,r1,r2)*prob1LH]
%   >= -epsil * d(r1, 2)
%
% IC1_L(r1): player 1 told L, prefers L over H
%   (A1L - Egrid(r1)) * sum_{r2} [bce(LL,r1,r2)*prob1LL + bce(LH,r1,r2)*prob1LH]
% - (A1H - Egrid(r1)) * sum_{r2} [bce(LL,r1,r2)*prob1HL + bce(LH,r1,r2)*prob1HH]
%   >= -epsil * d(r1, 1)
%
% IC2_H(r2): player 2 told H, prefers H over L
%   (A2H - Egrid(r2)) * sum_{r1} [bce(LH,r1,r2)*prob2LH + bce(HH,r1,r2)*prob2HH]
% - (A2L - Egrid(r2)) * sum_{r1} [bce(LH,r1,r2)*prob2LL + bce(HH,r1,r2)*prob2HL]
%   >= -epsil * d(r2, 4)
%
% IC2_L(r2): player 2 told L, prefers L over H
%   (A2L - Egrid(r2)) * sum_{r1} [bce(LL,r1,r2)*prob2LL + bce(HL,r1,r2)*prob2HL]
% - (A2H - Egrid(r2)) * sum_{r1} [bce(LL,r1,r2)*prob2LH + bce(HL,r1,r2)*prob2HH]
%   >= -epsil * d(r2, 3)

n_ic = 4 * s;
A_ic = zeros(n_ic, nvar);
b_ic = zeros(n_ic, 1);
row = 0;

% IC1_H: for each r1
for r1 = 1:s
    row = row + 1;
    markup_H = A1H - Egrid(r1);
    markup_L = A1L - Egrid(r1);
    for r2 = 1:s
        % bce(HH, r1, r2) contributes: markup_H * prob1HH - markup_L * prob1LH
        A_ic(row, idx(1, r1, r2)) = markup_H * prob1HH - markup_L * prob1LH;
        % bce(HL, r1, r2) contributes: markup_H * prob1HL - markup_L * prob1LL
        A_ic(row, idx(2, r1, r2)) = markup_H * prob1HL - markup_L * prob1LL;
        % bce(LH, r1, r2) and bce(LL, r1, r2): zero for player 1 told H
    end
    b_ic(row) = -epsil * d4(r1, 2);
end

% IC1_L: for each r1
for r1 = 1:s
    row = row + 1;
    markup_H = A1H - Egrid(r1);
    markup_L = A1L - Egrid(r1);
    for r2 = 1:s
        % bce(LH, r1, r2) contributes: markup_L * prob1LH - markup_H * prob1HH
        A_ic(row, idx(3, r1, r2)) = markup_L * prob1LH - markup_H * prob1HH;
        % bce(LL, r1, r2) contributes: markup_L * prob1LL - markup_H * prob1HL
        A_ic(row, idx(4, r1, r2)) = markup_L * prob1LL - markup_H * prob1HL;
    end
    b_ic(row) = -epsil * d4(r1, 1);
end

% IC2_H: for each r2
for r2 = 1:s
    row = row + 1;
    markup_H = A2H - Egrid(r2);
    markup_L = A2L - Egrid(r2);
    for r1 = 1:s
        % bce(HH, r1, r2) contributes: markup_H * prob2HH - markup_L * prob2HL
        A_ic(row, idx(1, r1, r2)) = markup_H * prob2HH - markup_L * prob2HL;
        % bce(LH, r1, r2) contributes: markup_H * prob2LH - markup_L * prob2LL
        A_ic(row, idx(3, r1, r2)) = markup_H * prob2LH - markup_L * prob2LL;
    end
    b_ic(row) = -epsil * d4(r2, 4);
end

% IC2_L: for each r2
for r2 = 1:s
    row = row + 1;
    markup_H = A2H - Egrid(r2);
    markup_L = A2L - Egrid(r2);
    for r1 = 1:s
        % bce(HL, r1, r2) contributes: markup_L * prob2HL - markup_H * prob2HH
        A_ic(row, idx(2, r1, r2)) = markup_L * prob2HL - markup_H * prob2HH;
        % bce(LL, r1, r2) contributes: markup_L * prob2LL - markup_H * prob2LH
        A_ic(row, idx(4, r1, r2)) = markup_L * prob2LL - markup_H * prob2LH;
    end
    b_ic(row) = -epsil * d4(r2, 3);
end

% Convert A_ic * x >= b_ic  to  -A_ic * x <= -b_ic for linprog
Aineq = -A_ic;
bineq = -b_ic;

%% Halton directions (same as find_polytope_switch.m)
NG = 100;
pp = haltonset(4, 'Skip', 1e3, 'Leap', 1e2);
PG0 = net(pp, NG);
k = 2;
PG = [PG0(:,1)*k - k/2, PG0(:,2)*k - k/2, PG0(:,3)*k - k/2, PG0(:,4)*k - k/2];
norms = vecnorm(PG, 2, 2);
PGrid = PG ./ norms;

%% Linprog options
options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');

%% Solve for each direction
VP = zeros(NG, 4);
% Extraction matrices: P_ap = sum_{r1,r2} bce(ap, r1, r2)
extract = zeros(4, nvar);
for ap = 1:4
    for r1 = 1:s
        for r2 = 1:s
            extract(ap, idx(ap, r1, r2)) = 1;
        end
    end
end

for i = 1:NG
    lambda = PGrid(i, :);

    % Objective: minimize -lambda' * [PHH; PHL; PLH; PLL]
    %   = minimize -(lambda * extract) * x
    f = -(lambda * extract)';

    [x, ~, exitflag] = linprog(f, Aineq, bineq, Aeq, beq, lb, ub, options);

    if exitflag == 1
        VP(i, :) = (extract * x)';
    else
        warning('solve_polytope_lp: linprog failed on direction %d (exitflag=%d)', i, exitflag);
        VP(i, :) = NaN(1, 4);
    end

    if mod(i, 10) == 0
        fprintf('  Polytope LP: %d/%d\n', i, NG);
    end
end

end
