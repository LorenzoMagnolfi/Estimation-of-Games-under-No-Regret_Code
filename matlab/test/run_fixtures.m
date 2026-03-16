%% run_fixtures.m — Reduced-scale fixture runner for baseline capture
%
% PURPOSE:  Exercise the same computational code paths as the four MAIN
%           scripts but at reduced scale, saving key intermediates to
%           matlab/test/fixtures/ for regression testing after refactoring.
%
% USAGE:    >> cd matlab/src; run('../test/run_fixtures.m')
%           Or from any directory with matlab/src on the MATLAB path.
%
% TIMING:   Estimated ~15-25 min total (vs ~7.6 h at full scale).
%
% STAGES COVERED:
%   Stage I  — Polytope (SKIPPED: requires AMPL; will be replaced by linprog)
%   Stage II — Simulation: learn_mod + ComputeBCCE_eps
%   Stage III— Application: Identification_Pricing_Game_ApplicationL
%   Stage IV — Empirical regret distribution: bootstrap learn_mod + ComputeBCCE_eps_pass
%
% REDUCED-SCALE CONFIG (vs production):
%   Stage II:  maxiters = [5000]         (vs [500k, 1M, 2M, 4M])
%              NGridV = NGridM = 20      (vs 100)
%              s = 5                     (same as production)
%   Stage III: NGridV = NGridM = 20      (vs 100)
%              n_types = 5              (same as production)
%   Stage IV:  B = 10                    (vs 500)
%              maxiters = 5000           (vs 100000)
%              NGridV = NGridM = 20      (vs 100)

clear all; clc; close all;
clear global;

%% Path setup
% Ensure matlab/src is on the path (needed when running from test/ or batch mode)
this_file = mfilename('fullpath');
test_dir = fileparts(this_file);
matlab_root = fileparts(test_dir);
src_dir = fullfile(matlab_root, 'src');
addpath(src_dir);

paths = df_repo_paths();
fixture_dir = fullfile(paths.matlab_root, 'test', 'fixtures');
if ~exist(fixture_dir, 'dir'), mkdir(fixture_dir); end

timing = struct();
fprintf('=== Fixture Runner: Baseline Capture ===\n');
fprintf('Fixture output: %s\n\n', fixture_dir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STAGE I — Polytope (SKIPPED)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[Stage I] SKIPPED — requires AMPL. Will be replaced by linprog.\n');
fprintf('  When linprog replacement is ready, add Stage I fixture capture here.\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STAGE II — Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[Stage II] Simulation: learn_mod + ComputeBCCE_eps\n');
t_start = tic;

% --- Globals (same as II_MAIN_simul.m) ---
global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps
global learning_style

% --- RNG seed (matches II_MAIN_simul.m) ---
rng(12345);

% --- Game parameters ---
NPlayers = 2;
alpha = -(1/3);
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);

% Action space: 5 actions (same as production)
for ind=1:NPlayers
    action_space{ind,1} = [4;5;6;7;8];
end

AA = action_space{1,1};
A = allcomb(action_space{1,:},action_space{2,:});
NAct = size(action_space{1,:},1);
NActPr = size(A,1);

% Type space: s=5 (same as production)
s = 5;
[type_space, marg_distrib] = marginal_cost_draws_v5(mu, sigma2, s);
s2 = s^2;
Egrid = type_space{1,1}';

% Joint type space
AEnum = 1;
T_sorted = type_space{1,1};
for ind=2:NPlayers
    AEnum = [size(T_sorted,1), AEnum];
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)) ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

% Joint prior Psi
Psi = zeros(s2,1);
for ii = 1:s
    for jj = 1:s
        kk = (ii-1)*s+jj;
        Psi(kk) = marg_distrib(ii)*marg_distrib(jj);
    end
end

% Payoff matrix Pi
cp = zeros(NActPr,NPlayers);
tps = [type_space{1,:},type_space{2,:}];
r_sq = size(tps,1);
Pi = zeros(NActPr,r_sq,NPlayers);
for j = 1:NPlayers
    for aa = 1:NActPr
        cp(aa,j) = exp(alpha*A(aa,j))/(1 + sum(exp(alpha*A(aa,:)),2));
        for tt = 1:s
            Pi(aa,tt,j) = cp(aa,j).*(A(aa,j)-tps(tt,j));
        end
    end
end

% --- Learning ---
learning_style = 'rm';
numdst_t = 1;
numdst_t_obs = numdst_t;

% REDUCED: single maxiters value
maxiters_values = [5000];

for maxiter_index = 1:length(maxiters_values)
    maxiters = maxiters_values(maxiter_index);
    N = 1;
    M = maxiters;
    M_obs = maxiters;

    fprintf('  Learning: maxiters=%d ...', maxiters);
    t_learn = tic;
    [distY_time, ~] = learn_mod(N, M, M_obs, numdst_t, numdst_t_obs, 1, 1);
    fprintf(' %.1fs\n', toc(t_learn));

    action_distribution = distY_time;

    % REDUCED: 20x20 grid (vs 100x100)
    NGridV = 20;
    NGridM = 20;
    NGrid = NGridV*NGridM;

    plot_param = 'Both';

    gridparamV = [1; linspace(0.15, sigma2(1,1)*10, NGridV)'];
    gridparamM = [1; linspace(0.55, mu(1,1)*0.5, NGridM)'];

    for ind1 = 1:NGridM
        for ind2 = 1:NGridV
            distribution_parameters{1,(ind1-1)*NGridM+ind2} = 'Normal';
            distribution_parameters{2,(ind1-1)*NGridM+ind2} = gridparamM(ind1)*mu';
            distribution_parameters{3,(ind1-1)*NGridM+ind2} = gridparamV(ind2)*sigma2';
            distpars((ind1-1)*NGridM+ind2,:) = [gridparamM(ind1)*mu(1,1), gridparamV(ind2)*sigma2(1,1)];
        end
    end

    switch_eps = 1;
    alpha_set = [0.05];
    numdist = size(action_distribution,2);
    num_alpha = length(alpha_set);
    maxvals = zeros(numdist, num_alpha, NGrid);

    fprintf('  Solver: %d grid points ...', NGrid);
    t_solve = tic;
    for ii = 1:numdist
        T = ii*(maxiters/numdist);
        distrib = action_distribution(:,ii);
        for jj = 1:num_alpha
            confid = alpha_set(jj);
            outs = ComputeBCCE_eps(type_space, action_space, distrib, alpha, ...
                distribution_parameters, maxiters, confid, Pi, switch_eps);
            maxvals(ii,jj,:) = cell2mat(outs);
        end
    end
    fprintf(' %.1fs\n', toc(t_solve));

    VV = squeeze(maxvals);
    VV_all(maxiter_index,:) = VV;

    % Save fixture
    save(fullfile(fixture_dir, sprintf('fixture_stage_ii_iter_%dk.mat', maxiters/1000)), ...
        'distY_time', 'action_distribution', 'VV', 'distpars', 'maxiters', ...
        'distribution_parameters', 'switch_eps', 'NGridV', 'NGridM');
end

save(fullfile(fixture_dir, 'fixture_stage_ii_solver_all.mat'), ...
    'VV_all', 'maxiters_values', 'NGridV', 'NGridM');

timing.stage_ii = toc(t_start);
fprintf('[Stage II] Done: %.1fs\n\n', timing.stage_ii);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STAGE III — Application (Identification)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[Stage III] Application: Identification_Pricing_Game_ApplicationL\n');
t_start = tic;

% Re-seed (matches III_MAIN)
rng(1);

% Data files
Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');

% Check data files exist
if ~isfile(Dist_file) || ~isfile(Prob_file)
    fprintf('  WARNING: Data files not found. Skipping Stage III.\n');
    fprintf('  Expected: %s\n', Dist_file);
    fprintf('  Expected: %s\n\n', Prob_file);
    timing.stage_iii = 0;
else
    epsilon_grid = [0.05];
    players = 1:2;

    % REDUCED: 20x20 grid (vs 100x100)
    NGridV_iii = 20;
    NGridM_iii = 20;
    n_types = 5;

    fprintf('  Solver: %d grid points x %d players ...', NGridV_iii*NGridM_iii, length(players));
    t_solve = tic;
    [outputs] = Identification_Pricing_Game_ApplicationL(epsilon_grid, Dist_file, Prob_file, ...
        players, NGridV_iii, NGridM_iii, n_types);
    maxvals_iii = cell2mat(outputs);
    fprintf(' %.1fs\n', toc(t_solve));

    save(fullfile(fixture_dir, 'fixture_stage_iii_solver_raw.mat'), ...
        'maxvals_iii', 'epsilon_grid', 'players', 'NGridV_iii', 'NGridM_iii', 'n_types');

    % Per-player identification
    NGrid_iii = NGridV_iii * NGridM_iii;
    for iii = players
        maxvals_player = maxvals_iii(NGrid_iii*(iii-1)+1:NGrid_iii*iii,:);
        VV_iii = maxvals_player(:,3);
        id_set_index_iii = (VV_iii <= 1e-12);
        ddpars_iii = maxvals_player(:,1:2);
        id_set_points_iii = ddpars_iii(id_set_index_iii, :);

        if ~isempty(id_set_points_iii)
            min_mu_iii = min(id_set_points_iii(:,1));
            max_mu_iii = max(id_set_points_iii(:,1));
            min_sigma_iii = min(id_set_points_iii(:,2));
            max_sigma_iii = max(id_set_points_iii(:,2));
        else
            min_mu_iii = NaN; max_mu_iii = NaN;
            min_sigma_iii = NaN; max_sigma_iii = NaN;
        end

        save(fullfile(fixture_dir, sprintf('fixture_stage_iii_player_%d.mat', iii)), ...
            'VV_iii', 'id_set_index_iii', 'ddpars_iii', 'id_set_points_iii', ...
            'min_mu_iii', 'max_mu_iii', 'min_sigma_iii', 'max_sigma_iii');
    end

    timing.stage_iii = toc(t_start);
    fprintf('[Stage III] Done: %.1fs\n\n', timing.stage_iii);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STAGE IV — Empirical Distribution of Regrets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('[Stage IV] Empirical regret distribution\n');
t_start = tic;

% Reset globals to simulation game (same as II, matching IV_MAIN)
clear global
global NAct alpha NPlayers A AA s Egrid Psi Pi marg_distrib mu sigma2 type_space tps
global learning_style

rng(1);

NPlayers = 2;
alpha = -(1/3);
mu = 3*ones(NPlayers,1);
sigma2 = 1*eye(NPlayers);

for ind=1:NPlayers
    action_space_iv{ind,1} = [4;5;6;7;8];
end
AA = action_space_iv{1,1};
A = allcomb(action_space_iv{1,:}, action_space_iv{2,:});
NAct = size(action_space_iv{1,:},1);
NActPr = size(A,1);

s = 5;
[type_space, marg_distrib] = marginal_cost_draws_v5(mu, sigma2, s);
s2 = s^2;
Egrid = type_space{1,1}';

AEnum = 1;
T_sorted = type_space{1,1};
for ind=2:NPlayers
    AEnum = [size(T_sorted,1), AEnum];
    T_sorted = [kron(type_space{ind,1}, ones(size(T_sorted,1),1)) ...
                kron(ones(size(type_space{ind,1},1),1), T_sorted)];
end

Psi = zeros(s2,1);
for ii = 1:s
    for jj = 1:s
        kk = (ii-1)*s+jj;
        Psi(kk) = marg_distrib(ii)*marg_distrib(jj);
    end
end

cp = zeros(NActPr,NPlayers);
tps = [type_space{1,:},type_space{2,:}];
r_sq = size(tps,1);
Pi = zeros(NActPr,r_sq,NPlayers);
for j = 1:NPlayers
    for aa = 1:NActPr
        cp(aa,j) = exp(alpha*A(aa,j))/(1 + sum(exp(alpha*A(aa,:)),2));
        for tt = 1:s
            Pi(aa,tt,j) = cp(aa,j).*(A(aa,j)-tps(tt,j));
        end
    end
end

learning_style = 'rm';

% REDUCED: maxiters and bootstrap count
maxiters_iv = 5000;   % vs 100000
B = 10;               % vs 500

N = 1;
M = maxiters_iv;
M_obs = maxiters_iv;
numdst_t = 1;
numdst_t_obs = numdst_t;

% Load Nobs for players (same as IV_MAIN)
Dist_file = fullfile(paths.data, 'SellerDistribution_15_sellers_res1.xlsx');
Prob_file = fullfile(paths.data, 'sale_probability_5bins_res1.xlsx');

if ~isfile(Dist_file) || ~isfile(Prob_file)
    fprintf('  WARNING: Data files not found. Skipping Stage IV.\n\n');
    timing.stage_iv = 0;
else
    [~, ~, ~, Nobs_Pl1] = get_player_data_5acts(1, 'median', Dist_file, Prob_file);
    [~, ~, ~, Nobs_Pl2] = get_player_data_5acts(2, 'median', Dist_file, Prob_file);

    % Bootstrap
    final_regret_iv = zeros(s, NPlayers, B);
    Pl1_regret_iv = zeros(s, B);
    Pl2_regret_iv = zeros(s, B);

    fprintf('  Bootstrap: B=%d, maxiters=%d ...', B, maxiters_iv);
    t_boot = tic;
    for b = 1:B
        [distY_time_iv, ~, fin, Pl1_EmpRegr, Pl2_EmpRegr] = ...
            learn_mod(N, M, M_obs, numdst_t, numdst_t_obs, Nobs_Pl1, Nobs_Pl2);
        final_regret_iv(:,:,b) = fin;
        Pl1_regret_iv(:,b) = Pl1_EmpRegr;
        Pl2_regret_iv(:,b) = Pl2_EmpRegr;
    end
    fprintf(' %.1fs\n', toc(t_boot));

    save(fullfile(fixture_dir, 'fixture_stage_iv_bootstrap.mat'), ...
        'final_regret_iv', 'Pl1_regret_iv', 'Pl2_regret_iv', 'distY_time_iv', ...
        'B', 'maxiters_iv', 'Nobs_Pl1', 'Nobs_Pl2');

    % Theoretical regrets (same formulas as IV_MAIN)
    eps1 = epsilon_switch(maxiters_iv, 1, 1) .* 0.05;
    ExpectedRegretComp_iv = sqrt(marg_distrib) .* eps1';
    avg_th_regret_iv = sum(ExpectedRegretComp_iv) / s;

    avg_exp_regret2_iv = mean(final_regret_iv, 3);
    regret_95perc_iv = prctile(final_regret_iv(:,:,:), 95, 3);

    % Application-specific regret ratios
    epsPl1 = epsilon_switch(Nobs_Pl1, 0.05, 1) .* 0.05;
    ExpectedRegretComp_Pl1 = sqrt(marg_distrib) .* epsPl1';
    aepsPl1 = NPlayers*s*ExpectedRegretComp_Pl1/0.05;

    epsPl2 = epsilon_switch(Nobs_Pl2, 0.05, 1) .* 0.05;
    ExpectedRegretComp_Pl2 = sqrt(marg_distrib) .* epsPl1';  % Note: matches IV_MAIN (uses epsPl1, not epsPl2)
    aepsPl2 = NPlayers*s*ExpectedRegretComp_Pl2/0.05;

    avg_exp_regret2_Pl1 = mean(Pl1_regret_iv, 2);
    regret_95perc_Pl1 = prctile(Pl1_regret_iv, 100*(1-(0.05/NPlayers*s)), 2);
    ratio1_Pl1_iv = ExpectedRegretComp_Pl1 ./ avg_exp_regret2_Pl1;
    ratio2_Pl1_iv = aepsPl1 ./ regret_95perc_Pl1;

    avg_exp_regret2_Pl2 = mean(Pl2_regret_iv, 2);
    regret_95perc_Pl2 = prctile(Pl2_regret_iv, 100*(1-(0.05/NPlayers*s)), 2);
    ratio1_Pl2_iv = ExpectedRegretComp_Pl2 ./ avg_exp_regret2_Pl2;
    ratio2_Pl2_iv = aepsPl2 ./ regret_95perc_Pl2;

    save(fullfile(fixture_dir, 'fixture_stage_iv_regret_comparison.mat'), ...
        'avg_exp_regret2_iv', 'avg_th_regret_iv', 'ExpectedRegretComp_iv', ...
        'regret_95perc_iv', 'ratio1_Pl1_iv', 'ratio2_Pl1_iv', ...
        'ratio1_Pl2_iv', 'ratio2_Pl2_iv');

    % Identification exercise using empirical regrets (ComputeBCCE_eps_pass)
    action_distribution_iv = distY_time_iv;

    NGridV_iv = 20;
    NGridM_iv = 20;
    NGrid_iv = NGridV_iv * NGridM_iv;

    gridparamV_iv = [1; linspace(0.15, sigma2(1,1)*2, NGridV_iv)'];
    gridparamM_iv = [1; linspace(0.55, mu(1,1)*0.5, NGridM_iv)'];

    clear distribution_parameters_iv distpars_iv
    for ind1 = 1:NGridM_iv
        for ind2 = 1:NGridV_iv
            distribution_parameters_iv{1,(ind1-1)*NGridM_iv+ind2} = 'Normal';
            distribution_parameters_iv{2,(ind1-1)*NGridM_iv+ind2} = gridparamM_iv(ind1)*mu';
            distribution_parameters_iv{3,(ind1-1)*NGridM_iv+ind2} = gridparamV_iv(ind2)*sigma2';
            distpars_iv((ind1-1)*NGridM_iv+ind2,:) = [gridparamM_iv(ind1)*mu(1,1), gridparamV_iv(ind2)*sigma2(1,1)];
        end
    end

    switch_eps_iv = 0;
    alpha_set_iv = [0.05];
    numdist_iv = size(action_distribution_iv, 2);
    num_alpha_iv = length(alpha_set_iv);
    maxvals_iv = zeros(numdist_iv, num_alpha_iv, NGrid_iv);

    fprintf('  Identification solver (eps_pass): %d grid points ...', NGrid_iv);
    t_solve = tic;
    for ii = 1:numdist_iv
        T_iv = ii*(maxiters_iv/numdist_iv);
        distrib_iv = action_distribution_iv(:,ii);
        for jj = 1:num_alpha_iv
            confid_iv = alpha_set_iv(jj);
            regret_comp_iv = avg_exp_regret2_iv * (s*NPlayers);
            ExpRegr_pass_iv = regret_comp_iv ./ confid_iv;
            outs_iv = ComputeBCCE_eps_pass(type_space, action_space_iv, distrib_iv, alpha, ...
                distribution_parameters_iv, T_iv, confid_iv, Pi, switch_eps_iv, ExpRegr_pass_iv);
            maxvals_iv(ii,jj,:) = cell2mat(outs_iv);
        end
    end
    fprintf(' %.1fs\n', toc(t_solve));

    VV_iv = squeeze(maxvals_iv);
    id_set_index_iv = (VV_iv <= 1e-12);
    ddpars_iv = repmat(distpars_iv, 1, num_alpha_iv, 1)';

    save(fullfile(fixture_dir, 'fixture_stage_iv_identification.mat'), ...
        'VV_iv', 'id_set_index_iv', 'ddpars_iv', 'distpars_iv', 'ExpRegr_pass_iv', ...
        'NGridV_iv', 'NGridM_iv');

    timing.stage_iv = toc(t_start);
    fprintf('[Stage IV] Done: %.1fs\n\n', timing.stage_iv);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('=== Fixture Runner Complete ===\n');
fn = fieldnames(timing);
total = 0;
for k = 1:numel(fn)
    fprintf('  %-12s %7.1fs\n', fn{k}, timing.(fn{k}));
    total = total + timing.(fn{k});
end
fprintf('  %-12s %7.1fs\n', 'TOTAL', total);
fprintf('\nFixtures saved to: %s\n', fixture_dir);

% Save timing metadata
save(fullfile(fixture_dir, 'fixture_timing.mat'), 'timing');

% List all fixture files
fprintf('\nFixture files:\n');
d = dir(fullfile(fixture_dir, 'fixture_*.mat'));
for k = 1:numel(d)
    fprintf('  %s  (%.1f KB)\n', d(k).name, d(k).bytes/1024);
end
