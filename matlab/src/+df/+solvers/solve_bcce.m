function g = solve_bcce(type_space, action_space, action_distribution, ...
    payoff_parameters, distribution_parameters, Pi, eps_info, opts)
% SOLVE_BCCE  Unified BCE identification solver.
%
%   g = df.solvers.solve_bcce(type_space, action_space, action_distribution,
%       payoff_parameters, distribution_parameters, Pi, eps_info, opts)
%
%   Replaces ComputeBCCE_eps, ComputeBCCE_eps_ApplicationL, and
%   ComputeBCCE_eps_pass with a single entry point that builds constraint
%   matrices ONCE and loops only over the parameter grid.
%
%   Inputs
%     type_space            NPlayers x 1 cell of type vectors
%     action_space          NPlayers x 1 cell of action vectors
%     action_distribution   (dv x Nq_N) observed action distributions
%     payoff_parameters     (NAlpha x ...) demand parameters (rows = alpha values)
%     distribution_parameters  {3 x NGrid} cell: {'Normal'; mu'; sigma'}
%     Pi                    (NA x s x NPlayers) utility tensor
%     eps_info              struct specifying epsilon source (see below)
%     opts                  struct with optional fields (see below)
%
%   eps_info fields:
%     .mode       'switch' | 'pass'
%       If 'switch':
%         .eps     (1 x s) epsilon vector from compute_epsilon
%       If 'pass':
%         .ExpRegr_pass   (s x NPlayers) per-player regret bounds
%
%   opts fields (all optional):
%     .marginal           false (default) | true
%     .marg_act_distrib_I   (Nactions x Nq_N) player 1 marginal (marginal mode)
%     .marg_act_distrib_II  (Nactions x Nq_N) player 2 marginal (marginal mode)
%     .switch_eps         integer, needed for eps_fin weighting rule
%     .solver             'sedumi' (default) | 'sdpt3' | ''  (CVX solver)
%     .precision          'low' (default) | 'default' | 'medium' | 'high'
%     .backend            'cvx' (default) | 'coneprog'
%
%   Output
%     g   objective values:
%         joint mode:    (NAlpha x NGrid x Nq_N)
%         marginal mode: (NGrid x Nq_N)

if nargin < 8, opts = struct(); end
marginal_mode = isfield(opts, 'marginal') && opts.marginal;
solver_name   = 'sedumi';  % SeDuMi is ~2x faster than SDPT3 with same results
if isfield(opts, 'solver'), solver_name = opts.solver; end
precision     = 'default';  % 'low' is 2x faster but changes identification for ~10% of points
if isfield(opts, 'precision'), precision = opts.precision; end
switch_eps_val = 0;
if isfield(opts, 'switch_eps'), switch_eps_val = opts.switch_eps; end

% Backend selection: cvx (default) or coneprog (DEPRECATED, R2020b+)
% coneprog is numerically unreliable at production scale (101x101 grids,
% 5-action games): returns spurious negative values, 92% disagreement with
% CVX+SeDuMi. Use 'cvx' for all production work.
if isfield(opts, 'backend')
    use_coneprog = strcmp(opts.backend, 'coneprog');
else
    use_coneprog = false;
end

%% Grid dimensions
NGrid_lambda = size(distribution_parameters, 2);
if NGrid_lambda == 0, NGrid_lambda = 1; end
Nq_N   = size(action_distribution, 2);
NAlpha = size(payoff_parameters, 1);

%% Build constraints ONCE
if marginal_mode
    cstr = df.solvers.build_constraints_marginal(type_space, action_space, Pi, ...
        opts.marg_act_distrib_II);
    dim_u = cstr.Nactions - 1;   % u-variable dimension
    a_dim = cstr.Nactions;       % per-player actions
else
    cstr = df.solvers.build_constraints(type_space, action_space, Pi);
    dim_u = cstr.NA - 1;
    a_dim = cstr.a;
end

s  = cstr.s;
NAg = cstr.NAg;

%% Compute prior (Psi) and marginal distributions for all grid points
if marginal_mode
    % Marginal mode: prior = marginal distribution
    marg_distrib = zeros(s, NGrid_lambda);
    for nd = 1:NGrid_lambda
        mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
        sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
        md = pdf(distribution_parameters{1,nd}, type_space{1,1}, mu_val, sg_val);
        marg_distrib(:,nd) = md / sum(md);
    end
    Psi = marg_distrib;  % prior IS the marginal in ApplicationL
else
    % Joint mode: prior from mvnpdf on joint type profiles
    T_sorted = cstr.T_sorted;
    s2 = cstr.s2;
    Psi = zeros(s2, NGrid_lambda);
    marg_distrib = zeros(s, NGrid_lambda);
    if size(distribution_parameters,1) == 0
        Psi = ones(s2, 1);
    elseif size(distribution_parameters,1) == 3
        for nd = 1:NGrid_lambda
            if strcmp(distribution_parameters{1,nd}, 'Normal')
                Psi(:,nd) = mvnpdf(T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
            else
                Psi(:,nd) = pdf(distribution_parameters{1,nd}, T_sorted, distribution_parameters{2,nd}, distribution_parameters{3,nd});
            end
        end
    end
    Psi = Psi ./ sum(Psi, 1);

    % Marginal distributions (needed for eps_fin weighting)
    for nd = 1:NGrid_lambda
        mu_val = distribution_parameters{2,nd}; mu_val = mu_val(1);
        sg_val = distribution_parameters{3,nd}; sg_val = sg_val(1,1);
        md = pdf(distribution_parameters{1,nd}, type_space{1,1}, mu_val, sg_val);
        marg_distrib(:,nd) = md / sum(md);
    end
end

%% Prepare epsilon vectors
if strcmp(eps_info.mode, 'pass')
    % Pass-through: per-player regret bounds
    eps_pl1 = eps_info.ExpRegr_pass(:,1)';
    eps_pl2 = eps_info.ExpRegr_pass(:,2)';
else
    eps_vec = eps_info.eps;  % (1 x s) from compute_epsilon
end

%% Solve loop
if marginal_mode
    g = zeros(NGrid_lambda, Nq_N);
else
    g = zeros(NAlpha, NGrid_lambda, Nq_N);
end

for nd = 1:NGrid_lambda
    bmarg = Psi(:, nd);

    % Build eps_fin for this grid point
    if strcmp(eps_info.mode, 'pass')
        % Pass-through: no marg_distrib weighting
        eps_fin = [repmat(eps_pl1, 1, a_dim), repmat(eps_pl2, 1, a_dim)];
    elseif marginal_mode
        % Marginal mode
        if switch_eps_val == 1 || switch_eps_val == 3
            eps_fin = repmat(sqrt(marg_distrib(:,nd))', 1, a_dim) .* repmat(eps_vec, 1, a_dim);
        else
            eps_fin = repmat(eps_vec, 1, a_dim);
        end
    else
        % Joint mode
        if switch_eps_val == 1 || switch_eps_val == 3 || switch_eps_val == 4
            eps_fin = repmat(sqrt(marg_distrib(:,nd))', 1, NAg*a_dim) .* repmat(eps_vec, 1, NAg*a_dim);
        else
            eps_fin = repmat(eps_vec, 1, NAg*a_dim);
        end
    end

    if marginal_mode
        % No NAlpha loop in marginal mode
        for nb = 1:Nq_N
            act_dist = opts.marg_act_distrib_I(:, nb)';
            c = [zeros(1, dim_u), act_dist, bmarg', 1, eps_fin]';
            if use_coneprog
                [g(nd, nb), ~] = df.solvers.solve_socp_coneprog(cstr, c);
            else
                [g(nd, nb), ~] = df.solvers.solve_socp_cvx(cstr, c, solver_name, precision);
            end
            fprintf('  Grid %d/%d, Obs %d/%d\n', nd, NGrid_lambda, nb, Nq_N);
        end
    else
        for np = 1:NAlpha
            for nb = 1:Nq_N
                c = [zeros(1, dim_u), action_distribution(:,nb)', bmarg', 1, eps_fin]';
                if use_coneprog
                    [g(np, nd, nb), ~] = df.solvers.solve_socp_coneprog(cstr, c);
                else
                    [g(np, nd, nb), ~] = df.solvers.solve_socp_cvx(cstr, c, solver_name, precision);
                end
                fprintf('  Grid %d/%d, Alpha %d/%d, Obs %d/%d\n', ...
                    nd, NGrid_lambda, np, NAlpha, nb, Nq_N);
            end
        end
    end
end

end
