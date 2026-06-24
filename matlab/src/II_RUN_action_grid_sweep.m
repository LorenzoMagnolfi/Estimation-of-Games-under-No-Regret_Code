%% II_RUN_action_grid_sweep
%
%  S6 / confound B: identified-set SIZE vs action-grid resolution |A|.
%  Tests the claim that the action grid |A| (not the horizon T) is the binding
%  identification floor: the (mu,sigma^2) confidence region should keep shrinking
%  as |A| grows, while it saturates in T.
%
%  Full-feedback regret matching, corrected Hedge radius (switch_eps=10),
%  parametric (mu,sigma^2) grid, adaptive=false (inference-grade full SOCP solve).
%  Action grid = linspace(4,8,|A|): a finer discretization of the SAME [4,8] price
%  range. |A| capped at 20.
%
%  Set-size metric: AREA of the identified region in (mu,sigma^2) space
%  (#identified cells x cell area) -- a defensible cost-space measure, NOT a
%  grid-fraction of an arbitrary candidate cloud. Reports a (|A|, N) table.
%  Saves .mat incrementally; makes NO server figures (regenerate locally).
%
%  STATUS: pending pre-flight review before its cluster run (new code).

clear all; clc;

paths = df_repo_paths();
rng(12345);

NPlayers = 2;
alpha = -(1/3);
mu = 3 * ones(NPlayers, 1);
sigma2 = 1 * eye(NPlayers);
s = 5;

A_values = [5, 10, 20];                                  % capped at 20
maxiters_values = [500000, 1000000, 2000000, 4000000];
switch_eps = 10;
% (mu,sigma) grid scales DOWN with |A|, set per-|A| in the loop below. The SOCP
% cost explodes with |A| (smoke: |A|=20 ~16.5s/solve vs |A|=10 ~0.8s), so a full
% 100x100 grid at |A|=20 would take ~days. A coarser grid at large |A| keeps each
% cell feasible overnight; the area metric is in (mu,sigma^2) units so it stays
% comparable across grid resolutions.
IDTOL = 1e-8;     % SIM-3 precision-matched identified-set tolerance (SeDuMi ~1e-8)

n_A = numel(A_values);
n_N = numel(maxiters_values);
results_by_A = cell(n_A, 1);
sweep_rows = [];

for ai = 1:n_A
    A = A_values(ai);
    actions_vec = linspace(4, 8, A)';
    switch A
        case 5,  NGridV = 60; NGridM = 60;
        case 10, NGridV = 40; NGridM = 40;
        otherwise, NGridV = 20; NGridM = 20;   % |A| >= 20: SOCP very slow, coarsen
    end
    fprintf('\n===== |A| = %d (prices %.3f .. %.3f, step %.3f) =====\n', ...
        A, actions_vec(1), actions_vec(end), actions_vec(2) - actions_vec(1));

    cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

    opts = struct();
    opts.maxiters_values   = maxiters_values;
    opts.NGridV            = NGridV;
    opts.NGridM            = NGridM;
    opts.alpha_set         = 0.05;
    opts.switch_eps        = switch_eps;
    opts.backend           = 'fast';
    opts.adaptive          = false;       % inference-grade
    opts.require_corrected = true;        % SIM-5 guard
    opts.learning_style    = 'rm';        % full feedback

    results = df.stages.run_stage_ii(cfg, opts);
    results_by_A{ai} = results;

    for ni = 1:n_N
        % (mu,sigma^2) grid for this N; drop the prepended out-of-order row/col.
        distpars = squeeze(results.distpars_all(ni, :, :));
        mu_g = reshape(distpars(:, 1), NGridV + 1, NGridM + 1); mu_g = mu_g(2:end, 2:end);
        sg_g = reshape(distpars(:, 2), NGridV + 1, NGridM + 1); sg_g = sg_g(2:end, 2:end);
        VV_g = reshape(results.VV_all(ni, :), NGridV + 1, NGridM + 1); VV_g = VV_g(2:end, 2:end);

        id_g = VV_g <= IDTOL;
        n_id = nnz(id_g);

        % Cost-space area = #identified cells x cell area; grid spacing taken from
        % the unique axis values so it is robust to grid orientation.
        dmu = median(diff(unique(mu_g(:))), 'omitnan');
        dsg = median(diff(unique(sg_g(:))), 'omitnan');
        area = n_id * abs(dmu) * abs(dsg);

        % Flag if the identified region touches the grid edge (area truncated).
        hit_boundary = any(id_g(1, :)) || any(id_g(end, :)) || ...
                       any(id_g(:, 1)) || any(id_g(:, end));

        sweep_rows = [sweep_rows; struct( ...
            'A', A, 'N', maxiters_values(ni), 'n_identified', n_id, ...
            'area_musigma', area, 'hit_boundary', hit_boundary)]; %#ok<AGROW>
    end

    % Incremental save (data first; no figures on the server).
    save(fullfile(paths.artifacts, 'action_grid_sweep_R1.mat'), ...
        'results_by_A', 'sweep_rows', 'A_values', 'maxiters_values', ...
        'switch_eps', 'IDTOL', '-v7.3');
    fprintf('  saved incremental artifact (through |A| = %d)\n', A);
end

fprintf('\n===== Action-grid sweep done =====\n');
disp(struct2table(sweep_rows));
