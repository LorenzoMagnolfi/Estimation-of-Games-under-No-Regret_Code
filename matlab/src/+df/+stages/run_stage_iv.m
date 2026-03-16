function results = run_stage_iv(cfg, opts)
% DF.STAGES.RUN_STAGE_IV  Stage IV: bootstrap regrets + identification with empirical epsilon.
%
%   results = df.stages.run_stage_iv(cfg, opts)
%
%   Runs B bootstrap iterations of the learning algorithm, computes
%   theoretical vs empirical regret comparisons, then performs an
%   identification exercise using empirical regret bounds.
%
%   Inputs:
%     cfg  — config struct from df.setup.game_simulation
%     opts — struct with fields:
%       .B              — number of bootstrap iterations (default: 500)
%       .maxiters       — iterations per bootstrap (default: 100000)
%       .NGridV         — variance grid size (default: 100)
%       .NGridM         — mean grid size (default: 100)
%       .alpha_set      — confidence levels (default: 0.05)
%       .use_parfor     — logical, enable parfor for bootstrap (default: false)
%       .Dist_file      — seller distribution file (for Nobs)
%       .Prob_file      — sale probability file (for Nobs)
%
%   Outputs:
%     results — struct with fields:
%       .final_regret    — (s x NPlayers x B) bootstrap regrets
%       .Pl1_regret, .Pl2_regret — per-player regrets
%       .distY_time      — last bootstrap action distribution
%       .avg_exp_regret2 — (s x NPlayers) mean empirical regrets
%       .avg_th_regret   — scalar average theoretical regret
%       .ExpectedRegretComp — theoretical expected regret components
%       .regret_95perc   — 95th percentile regrets
%       .ratio1_Pl1, .ratio2_Pl1, .ratio1_Pl2, .ratio2_Pl2 — ratios
%       .all_regrets     — flattened regret vector
%       .average_epsilon — average epsilon bound
%       .VV, .id_set_index, .ddpars, .distpars — identification outputs
%       .ExpRegr_pass    — passed regret bound

if nargin < 2, opts = struct(); end
if ~isfield(opts, 'B'),         opts.B = 500; end
if ~isfield(opts, 'maxiters'),  opts.maxiters = 100000; end
if ~isfield(opts, 'NGridV'),    opts.NGridV = 100; end
if ~isfield(opts, 'NGridM'),    opts.NGridM = 100; end
if ~isfield(opts, 'alpha_set'), opts.alpha_set = 0.05; end
if ~isfield(opts, 'use_parfor'), opts.use_parfor = false; end

cfg.learning_style = 'rm';

s = cfg.s;
NPlayers = cfg.NPlayers;
type_space = cfg.type_space;
action_space = cfg.action_space;
Pi = cfg.Pi;
marg_distrib = cfg.marg_distrib;
mu = cfg.mu;
sigma2 = cfg.sigma2;
maxiters = opts.maxiters;

N = 1;
M = maxiters;
M_obs = maxiters;
numdst_t = 1;
numdst_t_obs = numdst_t;

%% Load observation counts for application
[~, ~, ~, Nobs_Pl1] = get_player_data_5acts(1, 'median', opts.Dist_file, opts.Prob_file);
[~, ~, ~, Nobs_Pl2] = get_player_data_5acts(2, 'median', opts.Dist_file, opts.Prob_file);

%% Bootstrap
final_regret = zeros(s, NPlayers, opts.B);
Pl1_regret = zeros(s, opts.B);
Pl2_regret = zeros(s, opts.B);

fprintf('[Stage IV] Bootstrap: B=%d, maxiters=%dk', opts.B, maxiters/1000);
if opts.use_parfor, fprintf(' (parfor)'); end
fprintf('\n');
t_boot = tic;

if opts.use_parfor
    % parfor: each bootstrap draw is independent (learn_mod is stateless)
    parfor b = 1:opts.B
        [~, ~, fin, Pl1_EmpRegr, Pl2_EmpRegr] = ...
            learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
        final_regret(:, :, b) = fin;
        Pl1_regret(:, b) = Pl1_EmpRegr;
        Pl2_regret(:, b) = Pl2_EmpRegr;
    end
    % Run one more serial draw for distY_time (needed for identification)
    [distY_time, ~, ~, ~, ~] = ...
        learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
else
    for b = 1:opts.B
        [distY_time, ~, fin, Pl1_EmpRegr, Pl2_EmpRegr] = ...
            learn_mod(cfg, N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
        final_regret(:, :, b) = fin;
        Pl1_regret(:, b) = Pl1_EmpRegr;
        Pl2_regret(:, b) = Pl2_EmpRegr;
    end
end
fprintf('[Stage IV] Bootstrap done: %.1fs\n', toc(t_boot));

%% Theoretical regrets
eps1 = epsilon_switch(maxiters, 1, 1, cfg) .* 0.05;
ExpectedRegretComp = sqrt(marg_distrib) .* eps1';
avg_th_regret = sum(ExpectedRegretComp) / s;

% Application parameters
epsPl1 = epsilon_switch(Nobs_Pl1, 0.05, 1, cfg) .* 0.05;
ExpectedRegretComp_Pl1 = sqrt(marg_distrib) .* epsPl1';
aepsPl1 = NPlayers * s * ExpectedRegretComp_Pl1 / 0.05;

epsPl2 = epsilon_switch(Nobs_Pl2, 0.05, 1, cfg) .* 0.05;
% NOTE: original code uses epsPl1 here (known bug, preserved for baseline fidelity)
ExpectedRegretComp_Pl2 = sqrt(marg_distrib) .* epsPl1';
aepsPl2 = NPlayers * s * ExpectedRegretComp_Pl2 / 0.05;

%% Empirical vs theoretical comparisons
avg_exp_regret2 = mean(final_regret, 3);
avg_exp_regret = mean(avg_exp_regret2(:, 1));

regret_95perc = prctile(final_regret, 95, 3);

regret_995perc = prctile(final_regret, 99.5, 3);

% Per-player comparisons
avg_exp_regret2_Pl1 = mean(Pl1_regret, 2);
regret_95perc_Pl1 = prctile(Pl1_regret, 100*(1 - (0.05/(NPlayers*s))), 2);
ratio1_Pl1 = ExpectedRegretComp_Pl1 ./ avg_exp_regret2_Pl1;
ratio2_Pl1 = aepsPl1 ./ regret_95perc_Pl1;

avg_exp_regret2_Pl2 = mean(Pl2_regret, 2);
regret_95perc_Pl2 = prctile(Pl2_regret, 100*(1 - (0.05/(NPlayers*s))), 2);
ratio1_Pl2 = ExpectedRegretComp_Pl2 ./ avg_exp_regret2_Pl2;
ratio2_Pl2 = aepsPl2 ./ regret_95perc_Pl2;

%% Histogram data
all_regrets = reshape(final_regret, [], 1);
average_epsilon = mean(sqrt(marg_distrib) .* mean(epsilon_switch(maxiters, 1, 1, cfg)));

%% Identification exercise
action_distribution = distY_time;

gridparamV = [1; linspace(0.15, sigma2(1,1)*2, opts.NGridV)'];
gridparamM = [1; linspace(0.55, mu(1,1)*0.5, opts.NGridM)'];

[distpars, distribution_parameters] = df.report.build_param_grid(mu, sigma2, gridparamM, gridparamV);

num_alpha = numel(opts.alpha_set);
nV = opts.NGridV + 1;  % gridparamV includes leading 1
nM = opts.NGridM + 1;  % gridparamM includes leading 1
NGrid = nV * nM;
numdist = size(action_distribution, 2);
maxvals = zeros(numdist, num_alpha, NGrid);

fprintf('[Stage IV] Identification exercise (%d grid points)...\n', NGrid);
t_ident = tic;
for ii = 1:numdist
    T = ii * (maxiters / numdist);
    distrib = action_distribution(:, ii);
    for jj = 1:num_alpha
        confid = opts.alpha_set(jj);
        regret_comp = avg_exp_regret2 * (s * NPlayers);
        ExpRegr_pass = regret_comp ./ confid;

        % Use unified solver (builds constraints once, loops over grid)
        eps_info = struct('mode', 'pass', 'ExpRegr_pass', ExpRegr_pass);
        solve_opts = struct('switch_eps', 0);
        g = df.solvers.solve_bcce(type_space, action_space, distrib, ...
            cfg.alpha, distribution_parameters, Pi, eps_info, solve_opts);
        maxvals(ii, jj, :) = g(:);
    end
end
fprintf('[Stage IV] Identification done: %.1fs\n', toc(t_ident));

VV = squeeze(maxvals);
id_set_index = (VV <= 1e-12);
ddpars = repmat(distpars, 1, num_alpha, 1)';

%% Pack results
results.final_regret = final_regret;
results.Pl1_regret = Pl1_regret;
results.Pl2_regret = Pl2_regret;
results.distY_time = distY_time;
results.B = opts.B;
results.maxiters = maxiters;
results.Nobs_Pl1 = Nobs_Pl1;
results.Nobs_Pl2 = Nobs_Pl2;

results.avg_exp_regret2 = avg_exp_regret2;
results.avg_th_regret = avg_th_regret;
results.ExpectedRegretComp = ExpectedRegretComp;
results.regret_95perc = regret_95perc;
results.ratio1_Pl1 = ratio1_Pl1;
results.ratio2_Pl1 = ratio2_Pl1;
results.ratio1_Pl2 = ratio1_Pl2;
results.ratio2_Pl2 = ratio2_Pl2;

results.all_regrets = all_regrets;
results.average_epsilon = average_epsilon;

results.VV = VV;
results.id_set_index = id_set_index;
results.ddpars = ddpars;
results.distpars = distpars;
results.ExpRegr_pass = ExpRegr_pass;
results.distribution_parameters = distribution_parameters;

end
