function results = run_stage_ii(cfg, opts)
% DF.STAGES.RUN_STAGE_II  Stage II: simulation identification across iteration counts.
%
%   results = df.stages.run_stage_ii(cfg, opts)
%
%   For each iteration count in opts.maxiters_values, runs the learning
%   algorithm and solves the BCCE identification problem over a parameter grid.
%
%   Inputs:
%     cfg  — config struct from df.setup.game_simulation (5-action game)
%     opts — struct with fields:
%       .maxiters_values — vector of iteration counts (default: [500000 1000000 2000000 4000000])
%       .NGridV          — variance grid size (default: 100)
%       .NGridM          — mean grid size (default: 100)
%       .alpha_set       — confidence levels (default: 0.05)
%       .switch_eps      — epsilon formula selector (default: 1)
%
%   Outputs:
%     results — struct with fields:
%       .VV_all                    — (n_iters x NGrid) solver outputs
%       .distpars_all              — (n_iters x NGrid x 2) parameter grids
%       .distribution_parameters   — {n_iters x 1} cell of dist param cells
%       .distY_time_all            — {n_iters x 1} cell of action distributions
%       .maxiters_values           — iteration counts used
%       .gridparamV_all            — {n_iters x 1} cell of variance grid vectors
%       .cfg                       — config struct (for downstream use)

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'maxiters_values'), opts.maxiters_values = [500000, 1000000, 2000000, 4000000]; end
if ~isfield(opts, 'NGridV'),          opts.NGridV = 100; end
if ~isfield(opts, 'NGridM'),          opts.NGridM = 100; end
if ~isfield(opts, 'alpha_set'),       opts.alpha_set = 0.05; end
if ~isfield(opts, 'switch_eps'),      opts.switch_eps = 1; end

cfg.learning_style = 'rm';
numdst_t = 1;
numdst_t_obs = numdst_t;

mu = cfg.mu;
sigma2 = cfg.sigma2;
type_space = cfg.type_space;
action_space = cfg.action_space;
Pi = cfg.Pi;

n_iters = numel(opts.maxiters_values);
num_alpha = numel(opts.alpha_set);

% NGrid computed after grid construction (gridparamV/M include a leading 1,
% so actual size is (NGridV+1)*(NGridM+1))
NGridV_actual = opts.NGridV + 1;
NGridM_actual = opts.NGridM + 1;
NGrid = NGridV_actual * NGridM_actual;

% Preallocate
VV_all = zeros(n_iters, NGrid);
distpars_all = zeros(n_iters, NGrid, 2);
distribution_parameters_cell = cell(n_iters, 1);
distY_time_all = cell(n_iters, 1);
gridparamV_all = cell(n_iters, 1);

for maxiter_index = 1:n_iters
    maxiters = opts.maxiters_values(maxiter_index);

    N = 1;
    M = maxiters;
    M_obs = maxiters;

    %% Learning
    [distY_time, ~] = learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
    action_distribution = distY_time;
    distY_time_all{maxiter_index} = distY_time;

    %% Parameter grid (iteration-dependent variance range)
    if maxiter_index == 1
        gridparamV = [1; linspace(0.15, sigma2(1,1)*10, opts.NGridV)'];
    elseif maxiter_index == 2
        gridparamV = [1; linspace(0.15, sigma2(1,1)*6, opts.NGridV)'];
    else
        gridparamV = [1; linspace(0.15, sigma2(1,1)*3.5, opts.NGridV)'];
    end
    gridparamM = [1; linspace(0.55, mu(1,1)*0.5, opts.NGridM)'];
    gridparamV_all{maxiter_index} = gridparamV;

    [distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV);
    distpars_all(maxiter_index, :, :) = distpars;
    distribution_parameters_cell{maxiter_index} = distribution_parameters;

    %% Solver
    numdist = size(action_distribution, 2);
    maxvals = zeros(numdist, num_alpha, NGrid);

    for ii = 1:numdist
        distrib = action_distribution(:, ii);
        for jj = 1:num_alpha
            confid = opts.alpha_set(jj);
            outs = ComputeBCCE_eps(type_space, action_space, distrib, ...
                Pi * 0 + Pi, distribution_parameters, maxiters, confid, Pi, opts.switch_eps, cfg);
            maxvals(ii, jj, :) = cell2mat(outs);
        end
    end

    VV = squeeze(maxvals);
    VV_all(maxiter_index, :) = VV;
end

%% Pack results
results.VV_all = VV_all;
results.distpars_all = distpars_all;
results.distribution_parameters = distribution_parameters_cell;
results.distY_time_all = distY_time_all;
results.maxiters_values = opts.maxiters_values;
results.gridparamV_all = gridparamV_all;
results.alpha_set = opts.alpha_set;
results.switch_eps = opts.switch_eps;
results.NGridV = opts.NGridV;
results.NGridM = opts.NGridM;
results.cfg = cfg;

end
