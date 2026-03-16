function [distpars, distribution_parameters] = build_nonparam_grid(marg_distrib, type_space, opts)
% DF.REPORT.BUILD_NONPARAM_GRID  Build nonparametric candidate distribution grid.
%
%   [distpars, distribution_parameters] = df.report.build_nonparam_grid(marg_distrib, type_space, opts)
%
%   Generates candidate probability mass vectors over the type support,
%   using a mixture of local perturbations, global simplex draws, and
%   peaked distributions for low-sigma coverage. This is the nonparametric
%   counterpart to build_param_grid (which uses (mu,sigma) parameterization).
%
%   Inputs:
%     marg_distrib  — (s x 1) true marginal distribution
%     type_space    — {NPlayers x 1} cell of type vectors
%     opts          — struct with optional fields:
%       .K_local       — number of local perturbations (default: 1000)
%       .K_global      — number of global simplex draws (default: 9000)
%       .K_spiky       — number of peaked distributions (default: 200)
%       .local_width   — perturbation scale (default: 20)
%       .spike_mult    — spike multiplier (default: 3)
%       .n_adjacent    — neighbor count for spiky (default: 4)
%
%   Outputs:
%     distpars               — (NGrid x 2) [mean, variance] for each candidate
%     distribution_parameters — {1 x NGrid} cell of (s x 1) probability vectors
%                               Row-1 format signals nonparametric mode to solve_bcce

if nargin < 3, opts = struct(); end
if ~isfield(opts, 'K_local'),    opts.K_local = 1000; end
if ~isfield(opts, 'K_global'),   opts.K_global = 9000; end
if ~isfield(opts, 'K_spiky'),    opts.K_spiky = 200; end
if ~isfield(opts, 'local_width'), opts.local_width = 20; end
if ~isfield(opts, 'spike_mult'), opts.spike_mult = 3; end
if ~isfield(opts, 'n_adjacent'), opts.n_adjacent = 4; end

s = numel(marg_distrib);
support = type_space{1,1};  % (s x 1) type support for Player 1

K_local  = opts.K_local;
K_global = opts.K_global;
K_spiky  = opts.K_spiky;
NGrid = 1 + K_local + K_global + K_spiky;  % true + local + global + spiky

distribution_parameters = cell(1, NGrid);

%% 1. True marginal distribution
distribution_parameters{1} = marg_distrib(:);

%% 2. Local perturbations around true marginal
for ind = 2:(1 + K_local)
    candidate = marg_distrib(:) + rand(s, 1) / opts.local_width;
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

%% 3. Global draws over the full simplex (Dirichlet(1) via exponential trick)
for ind = (2 + K_local):(1 + K_local + K_global)
    w = -log(rand(s, 1));
    candidate = w ./ sum(w);
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

%% 4. Spiky distributions for better low-sigma coverage
base = 1 + K_local + K_global;
for ind = (base + 1):(base + K_spiky)
    random_draws = rand(s, 1);
    [~, peak_idx] = max(random_draws);
    candidate = zeros(s, 1);
    candidate(peak_idx) = random_draws(peak_idx) * opts.spike_mult;
    for offset = 1:opts.n_adjacent
        if peak_idx - offset > 0
            candidate(peak_idx - offset) = random_draws(peak_idx - offset);
        end
        if peak_idx + offset <= s
            candidate(peak_idx + offset) = random_draws(peak_idx + offset);
        end
    end
    candidate = max(candidate, 1e-12);
    candidate = candidate ./ sum(candidate);
    distribution_parameters{ind} = candidate;
end

%% Compute (mean, variance) implied by each candidate for plotting
distpars = zeros(NGrid, 2);
for ind = 1:NGrid
    p = distribution_parameters{ind};
    p = p(:) ./ sum(p(:));
    mu1 = sum(p .* support);
    var1 = sum(p .* (support - mu1).^2);
    distpars(ind, 1) = mu1;
    distpars(ind, 2) = var1;
end

end
