function results = run_stage_i(cfg, opts)
% DF.STAGES.RUN_STAGE_I  Stage I: polytope computation + learning convergence.
%
%   results = df.stages.run_stage_i(cfg, opts)
%
%   Computes BCE prediction set polytope vertices for multiple confidence
%   levels, then runs the learning algorithm and computes per-period regrets.
%
%   Inputs:
%     cfg  — config struct from df.setup.game_simulation (2-action game)
%     opts — struct with fields:
%       .maxiters   — number of learning iterations (default: 10000)
%       .conf_set   — confidence levels for polytope (default: [0.1, 0.025, 0.05])
%       .switch_eps — epsilon formula selector (default: 3)
%       .rng_seed_learn — RNG seed for learning phase (default: 11111)
%
%   Outputs:
%     results — struct with fields:
%       .VP          — {n_conf x 1} cell of polytope vertices
%       .conf_set    — confidence levels used
%       .maxiters    — iteration count
%       .distY_time, .distY_time_obs — action distributions
%       .actions     — (N+M x NPlayers) action history
%       .regret      — (N+M x NPlayers) regret history
%       .type_inds   — type indicator matrices
%       .regret_per_period — (N+M x s x NPlayers) regret by type and period

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'maxiters'),       opts.maxiters = 10000; end
if ~isfield(opts, 'conf_set'),       opts.conf_set = [0.1, 0.025, 0.05]; end
if ~isfield(opts, 'switch_eps'),     opts.switch_eps = 3; end
if ~isfield(opts, 'rng_seed_learn'), opts.rng_seed_learn = 11111; end

maxiters = opts.maxiters;
s = cfg.s;
NPlayers = cfg.NPlayers;

%% Polytope computation
VP = cell(numel(opts.conf_set), 1);
for jj = 1:numel(opts.conf_set)
    VP{jj} = df.solvers.solve_polytope_lp(cfg, maxiters, opts.conf_set(jj), opts.switch_eps);
end

%% Learning
cfg.learning_style = 'rm';

Ntrain = 1;
M = maxiters;
M_obs = maxiters;
numdst_t = 2;
numdst_t_obs = numdst_t;

rng(opts.rng_seed_learn);
[distY_time, distY_time_obs, actions, regret, type_inds] = ...
    learn(cfg, Ntrain, M, M_obs, numdst_t, numdst_t_obs);

%% Compute regret per period
regret_per_period = zeros(Ntrain + M, s, NPlayers);
for p = 1:NPlayers
    for jj = 2:(Ntrain + M)
        for ii = 1:s
            if type_inds(jj, ii, p) == 1
                regret_per_period(jj, ii, p) = regret(jj, p);
            else
                regret_per_period(jj, ii, p) = regret_per_period(jj-1, ii, p) * (jj-1) / jj;
            end
        end
    end
end

%% Pack results
results.VP = VP;
results.conf_set = opts.conf_set;
results.maxiters = maxiters;
results.switch_eps = opts.switch_eps;
results.distY_time = distY_time;
results.distY_time_obs = distY_time_obs;
results.actions = actions;
results.regret = regret;
results.type_inds = type_inds;
results.regret_per_period = regret_per_period;

end
